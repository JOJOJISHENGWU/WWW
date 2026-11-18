from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch.utils.data import DataLoader, TensorDataset

from models.SCOUT import TrafficForecastModel
from utils.community_clustering import spectral_cluster


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(y_true - y_pred)).item())


def metric_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item())


def metric_mape(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-5) -> float:
    denom = torch.clamp(torch.abs(y_true), min=eps)
    return float((torch.mean(torch.abs((y_true - y_pred) / denom)) * 100.0).item())


def load_cfg(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML configuration: {config_path}")
    return cfg


def _resolve_repo_path(raw_path: str, repo_root: Path) -> Path:
    """Resolve paths used by both current and earlier YAML configurations."""
    path = Path(raw_path)
    candidates = [path if path.is_absolute() else repo_root / path]
    if not path.is_absolute():
        # Old configs used ../DATA/... although DATA is inside the repository.
        normalized_parts = [part for part in path.parts if part not in ("..", ".")]
        candidates.append(repo_root.joinpath(*normalized_parts))
        candidates.append(repo_root / "DATA" / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_dataset_array(cfg: Dict, repo_root: Optional[Path] = None) -> np.ndarray:
    repo_root = repo_root or Path(__file__).resolve().parent
    raw_path = cfg["dataset"]["data_path"]
    data_file = _resolve_repo_path(raw_path, repo_root)
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    if data_file.suffix.lower() == ".npz":
        with np.load(data_file, allow_pickle=False) as npz:
            keys = list(npz.files)
            preferred = next((key for key in ("data", "x", "arr_0") if key in npz), None)
            arr = npz[preferred] if preferred is not None else npz[keys[0]]
    elif data_file.suffix.lower() == ".npy":
        arr = np.load(data_file, mmap_mode=None, allow_pickle=False)
    else:
        raise ValueError(f"Unsupported dataset format {data_file.suffix}: {data_file}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected data shape (T,N,C) or (T,N), got {arr.shape} from {data_file}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Dataset contains NaN or Inf values: {data_file}")
    return arr


def _read_graph_file(path: Path, num_nodes: int) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as npz:
            key = next((k for k in ("adj", "matrix", "data", "arr_0") if k in npz), npz.files[0])
            arr = npz[key]
    elif suffix in (".csv", ".txt", ".tsv"):
        # Try common numeric matrix/edge-list encodings in order. Avoid silently
        # accepting a malformed parse (for example, a whitespace file read as a
        # single comma-delimited column).
        delimiters = ["\t", ",", None] if suffix == ".tsv" else [",", None, "\t"]
        arr = None
        parse_errors = []
        for delimiter in delimiters:
            for skip_header in (0, 1):
                try:
                    candidate = np.genfromtxt(path, delimiter=delimiter, dtype=np.float64,
                                              skip_header=skip_header, invalid_raise=False)
                    candidate = np.asarray(candidate)
                    if candidate.ndim == 2 and candidate.shape[1] >= 2 and np.isfinite(candidate).any():
                        # Accept only plausible dense adjacency or edge-list shapes.
                        if candidate.shape == (num_nodes, num_nodes) or candidate.shape[1] in (2, 3):
                            arr = candidate
                            break
                except Exception as exc:
                    parse_errors.append(str(exc))
            if arr is not None:
                break
        if arr is None:
            raise ValueError(f"Could not parse graph matrix/edge list in {path}: {parse_errors[:2]}")
    elif suffix in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            obj = pickle.load(f, encoding="latin1")
        if isinstance(obj, (tuple, list)) and len(obj) >= 3:
            arr = obj[2]
        elif isinstance(obj, dict):
            arr = obj.get("adj", obj.get("matrix", obj.get("data")))
        else:
            arr = obj
    else:
        raise ValueError(f"Unsupported graph file format: {path}")

    arr = np.asarray(arr)
    if arr.shape == (num_nodes, num_nodes):
        return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # Edge-list support: columns are source, destination, optional weight.
    if arr.ndim == 2 and arr.shape[1] in (2, 3) and arr.shape[0] > num_nodes:
        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for row in arr:
            src, dst = int(row[0]), int(row[1])
            weight = float(row[2]) if arr.shape[1] == 3 else 1.0
            if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                adjacency[src, dst] = weight
                adjacency[dst, src] = weight
        if np.count_nonzero(adjacency) > 0:
            return adjacency
    raise ValueError(
        f"Graph file {path} has shape {arr.shape}; expected ({num_nodes},{num_nodes}) "
        "or an edge list with 2/3 columns."
    )


def load_adjacency(cfg: Dict, num_nodes: int, repo_root: Optional[Path] = None,
                   override_path: Optional[str] = None) -> np.ndarray:
    repo_root = repo_root or Path(__file__).resolve().parent
    dataset_cfg = cfg["dataset"]
    candidates: List[Path] = []
    configured = override_path or dataset_cfg.get("adj_path")
    if configured:
        candidates.append(_resolve_repo_path(configured, repo_root))

    data_file = _resolve_repo_path(dataset_cfg["data_path"], repo_root)
    data_dir = data_file.parent
    for name in (
        "adj.npy", "matrix.npy", "adj_mx.npy", "adjacency.npy", "adj.npz",
        "adj.csv", "graph.csv", "distance.csv", "adj.txt", "graph.txt",
        "adj_mx.pkl", f"{dataset_cfg.get('name', '').upper()}.txt",
        f"{dataset_cfg.get('name', '').upper()}.csv",
    ):
        candidate = data_dir / name
        if candidate not in candidates:
            candidates.append(candidate)

    errors = []
    for path in candidates:
        if path.exists():
            try:
                adj = _read_graph_file(path, num_nodes)
                if adj.shape != (num_nodes, num_nodes):
                    continue
                if np.any(adj < 0):
                    raise ValueError("Adjacency contains negative weights")
                if not np.any(adj > 0):
                    raise ValueError("Adjacency contains no positive edges")
                print(f"Loaded road graph: {path}")
                return adj
            except Exception as exc:
                errors.append(f"{path}: {exc}")
    detail = "\n".join(errors) if errors else "No candidate graph file was found."
    raise FileNotFoundError(
        f"A valid {num_nodes}x{num_nodes} road adjacency matrix is required. "
        f"Set dataset.adj_path to the correct graph file. Checked:\n{detail}"
    )


def average_shortest_path_cost(adj: np.ndarray) -> np.ndarray:
    """Per-node mean finite shortest-path cost used by PCS-ViM Eq. (13)."""
    graph = (np.asarray(adj) > 0).astype(np.float64)
    np.fill_diagonal(graph, 0.0)
    dist = shortest_path(csr_matrix(graph), directed=False, unweighted=True)
    finite = np.isfinite(dist)
    out = np.zeros(dist.shape[0], dtype=np.float32)
    finite_values = dist[finite]
    disconnected_penalty = float(finite_values.max()) if finite_values.size else 1.0
    for node in range(dist.shape[0]):
        row = dist[node]
        valid = np.isfinite(row)
        valid[node] = False
        if valid.any():
            # Unreachable nodes receive a conservative finite penalty.
            unreachable = (~np.isfinite(row)).sum()
            out[node] = (row[valid].sum() + unreachable * disconnected_penalty) / max(dist.shape[0] - 1, 1)
        else:
            out[node] = disconnected_penalty
    return out


def build_windows(
    arr: np.ndarray,
    seq_len: int,
    pred_len: int = 1,
    feature_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build multi-step windows: X=(B,T,N), Y=(B,H,N), not one target point."""
    if arr.ndim != 3:
        raise ValueError(f"Expected arr=(time,nodes,features), got {arr.shape}")
    if seq_len <= 0 or pred_len <= 0:
        raise ValueError("seq_len and pred_len must be positive")
    if not 0 <= feature_idx < arr.shape[-1]:
        raise ValueError(f"feature_idx={feature_idx} is out of range for C={arr.shape[-1]}")
    x_list, y_list = [], []
    total = arr.shape[0]
    for i in range(total - seq_len - pred_len + 1):
        x_list.append(arr[i:i + seq_len, :, feature_idx])
        y_list.append(arr[i + seq_len:i + seq_len + pred_len, :, feature_idx])
    if not x_list:
        raise ValueError(f"Not enough time steps ({total}) for seq_len={seq_len}, pred_len={pred_len}")
    return np.stack(x_list).astype(np.float32), np.stack(y_list).astype(np.float32)


def chronological_split(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-6):
        raise ValueError("train/val/test ratios must sum to 1.0")
    n = x.shape[0]
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)
    if train_end == 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Chronological split produced an empty dataset partition")
    return (x[:train_end], y[:train_end]), (x[train_end:val_end], y[train_end:val_end]), (x[val_end:], y[val_end:])


def build_chronological_partitions(
    arr: np.ndarray,
    seq_len: int,
    pred_len: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    feature_idx: int = 0,
):
    """Split the raw timeline first; each later split receives past context only.

    This prevents multi-step target windows from crossing train/validation/test
    boundaries while still allowing validation/test windows to use historical
    observations immediately preceding their own target interval.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-6):
        raise ValueError("train/val/test ratios must sum to 1.0")
    total = arr.shape[0]
    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)
    if train_end <= seq_len + pred_len or val_end - train_end <= pred_len or total - val_end <= pred_len:
        raise ValueError("Raw chronological split is too small for the requested input/prediction windows")
    train_arr = arr[:train_end]
    val_arr = arr[max(0, train_end - seq_len):val_end]
    test_arr = arr[max(0, val_end - seq_len):]
    train_xy = build_windows(train_arr, seq_len, pred_len, feature_idx)
    val_xy = build_windows(val_arr, seq_len, pred_len, feature_idx)
    test_xy = build_windows(test_arr, seq_len, pred_len, feature_idx)
    return train_xy, val_xy, test_xy


def normalize_by_train(x_train: np.ndarray, x_other: np.ndarray):
    # Per-node normalization, estimated on training inputs only.
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = np.clip(x_train.std(axis=(0, 1), keepdims=True), 1e-6, None)
    return (x_train - mean) / std, (x_other - mean) / std, (mean, std)


def normalize_targets(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # mean/std: (1,1,N); target: (B,H,N)
    return (y - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)


def to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False) -> DataLoader:
    # x: (B,T,N) -> (B,1,N,T); y remains (B,H,N).
    x_t = torch.as_tensor(x, dtype=torch.float32).permute(0, 2, 1).unsqueeze(1)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    if y_t.ndim == 2:
        y_t = y_t.unsqueeze(1)
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=shuffle)


def spectral_consistency_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Equation (25): compare future prediction/target spectra along horizon."""
    if y_pred.shape != y_true.shape or y_pred.ndim != 3:
        raise ValueError(
            f"Expected matching (B,H,N) prediction/target, got {tuple(y_pred.shape)} and {tuple(y_true.shape)}"
        )
    pred_amp = torch.fft.rfft(y_pred, dim=1).abs()
    true_amp = torch.fft.rfft(y_true, dim=1).abs()
    return F.l1_loss(pred_amp, true_amp)


def regularization_loss(
    output: Dict[str, torch.Tensor],
    lambda_sep: float,
    lambda_orth: float,
) -> torch.Tensor:
    """Equation (26): routing variance maximization + kernel orthogonality."""
    rho = output["rho_comm"]
    routing_term = -float(lambda_sep) * rho.var(unbiased=False)
    w_low = output["W_low"].reshape(1, -1)
    w_high = output["W_high"].reshape(1, -1)
    cosine = F.cosine_similarity(w_high, w_low, dim=-1, eps=1e-8).squeeze(0)
    orthogonal_term = float(lambda_orth) * cosine.square()
    return routing_term + orthogonal_term


def _to_device_graph(adj: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(adj, dtype=torch.float32, device=device)


def _horizon_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, interval_minutes: int,
                     requested_horizons: Optional[List[int]] = None) -> Dict[str, Dict[str, float]]:
    """Return metrics at requested forecast lead times when available."""
    metrics = {}
    horizons = requested_horizons or [interval_minutes * (i + 1) for i in range(y_true.shape[1])]
    for minutes in horizons:
        index = int(round(minutes / interval_minutes)) - 1
        if 0 <= index < y_true.shape[1]:
            yt = y_true[:, index, :]
            yp = y_pred[:, index, :]
            metrics[f"{minutes}min"] = {
                "MAE": metric_mae(yt, yp),
                "RMSE": metric_rmse(yt, yp),
                "MAPE": metric_mape(yt, yp),
            }
    return metrics


def train_one_run(
    cfg: Dict,
    config_name: str,
    adj_override: Optional[str] = None,
    pred_len_override: Optional[int] = None,
) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parent
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    loss_cfg = cfg.get("loss", {})
    runtime_cfg = cfg.get("runtime", {})
    dataset_name = str(dataset_cfg["name"])

    set_seed(int(train_cfg.get("seed", 42)))
    requested_device = str(runtime_cfg.get("device", "cuda")).lower()
    device = torch.device("cuda" if requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    arr = load_dataset_array(cfg, repo_root)
    num_nodes = int(dataset_cfg.get("num_nodes", arr.shape[1]))
    if arr.shape[1] != num_nodes:
        raise ValueError(f"Dataset has N={arr.shape[1]} nodes but config says {num_nodes}")

    seq_len = int(model_cfg.get("seq_len", 12))
    pred_len = int(pred_len_override if pred_len_override is not None else model_cfg.get("pred_len", 1))
    feature_idx = int(dataset_cfg.get("feature_idx", 0))
    split_cfg = cfg.get("split", {})
    ratios = (float(split_cfg.get("train", 0.7)), float(split_cfg.get("val", 0.2)), float(split_cfg.get("test", 0.1)))
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = build_chronological_partitions(
        arr, seq_len=seq_len, pred_len=pred_len,
        train_ratio=ratios[0], val_ratio=ratios[1], test_ratio=ratios[2],
        feature_idx=feature_idx,
    )

    x_train_norm, x_val_norm, norm_stats = normalize_by_train(x_train, x_val)
    mean, std = norm_stats
    x_test_norm = (x_test - mean) / std
    y_train_norm = normalize_targets(y_train, mean, std)
    y_val_norm = normalize_targets(y_val, mean, std)
    y_test_norm = normalize_targets(y_test, mean, std)

    batch_size = int(train_cfg.get("batch_size", 32))
    train_loader = to_loader(x_train_norm, y_train_norm, batch_size=batch_size, shuffle=True)
    val_loader = to_loader(x_val_norm, y_val_norm, batch_size=batch_size, shuffle=False)
    test_loader = to_loader(x_test_norm, y_test_norm, batch_size=batch_size, shuffle=False)

    adj_np = load_adjacency(cfg, num_nodes, repo_root, override_path=adj_override)
    num_communities = int(model_cfg.get("num_communities", 4))
    node2comm_np = spectral_cluster(adj_np, num_communities=num_communities, seed=int(train_cfg.get("seed", 42)))
    avg_sp = average_shortest_path_cost(adj_np)
    adj_t = _to_device_graph(adj_np, device)

    model = TrafficForecastModel(
        in_channels=1,
        hidden_channels=int(model_cfg.get("d_model", 128)),
        num_nodes=num_nodes,
        out_horizon=pred_len,
        seq_len=seq_len,
        num_communities=num_communities,
        topk_freq=int(model_cfg.get("top_k", model_cfg.get("topk_freq", 6))),
        cheb_order_local=int(model_cfg.get("cheb_order_local", 1)),
        cheb_order_global=int(model_cfg.get("cheb_order_global", 3)),
        num_morphology_prototypes=int(model_cfg.get("num_morphology_prototypes", 8)),
        num_state_prototypes=int(model_cfg.get("num_state_prototypes", 8)),
        warmup_steps=int(train_cfg.get("warmup_steps", 1000)),
        node2comm=torch.as_tensor(node2comm_np, dtype=torch.long),
        adj=adj_t,
        avg_shortest_path_cost=torch.as_tensor(avg_sp, dtype=torch.float32),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_cfg.get("scheduler_factor", 0.5)),
        patience=int(train_cfg.get("scheduler_patience", 4)),
        min_lr=float(train_cfg.get("min_lr", 1e-5)),
    )
    criterion_l1 = nn.L1Loss()
    criterion_smooth = nn.SmoothL1Loss(beta=1.0)

    log_dir = repo_root / "logs"
    ckpt_dir = repo_root / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{dataset_name.lower()}_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = log_dir / f"{run_id}.log"
    csv_path = log_dir / f"{run_id}_metrics.csv"
    ckpt_path = ckpt_dir / f"{dataset_name.lower()}_best.pth"

    best_val_mae = float("inf")
    best_state = None
    early_stop_wait = 0
    early_stop_patience = int(train_cfg.get("early_stop_patience", 5))
    epochs = int(train_cfg.get("epochs", 300))
    global_step = 0
    interval_minutes = int(dataset_cfg.get("interval_minutes", 5))
    requested_horizons = cfg.get("evaluation", {}).get("horizons_minutes", [15, 30, 60])

    with open(log_path, "w", encoding="utf-8") as log_file, open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", "split", "horizon", "MAE", "RMSE", "MAPE", "LR"])
        writer.writeheader()
        log_file.write(f"Run ID: {run_id}\nDataset: {dataset_name}\nConfig: {config_name}\nDevice: {device}\n")
        log_file.write(f"Model d_model={model_cfg.get('d_model',128)}, seq_len={seq_len}, pred_len={pred_len}\n")
        log_file.write(f"Split train/val/test = {ratios[0]:.2f}/{ratios[1]:.2f}/{ratios[2]:.2f}\n")
        log_file.write(f"Graph: {dataset_cfg.get('adj_path', adj_override or 'auto-discovered')}\n")
        log_file.write("Metric space: de-normalized original scale\n")
        log_file.flush()

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(bx, step=global_step, inference=False)
                y_hat = output["y_hat"]

                loss_pred = (
                    float(loss_cfg.get("pred_l1", 1.0)) * criterion_l1(y_hat, by)
                    + float(loss_cfg.get("pred_smoothl1", 0.3)) * criterion_smooth(y_hat, by)
                )
                loss_spec = float(loss_cfg.get("spectral_consistency", 0.02)) * spectral_consistency_loss(y_hat, by)
                lambda_sep = float(loss_cfg.get("lambda_sep", abs(float(loss_cfg.get("routing_var_reg", 0.02)))))
                lambda_orth = float(loss_cfg.get("lambda_orth", loss_cfg.get("gcn_orthogonality", 0.01)))
                loss_reg = regularization_loss(output, lambda_sep=lambda_sep, lambda_orth=lambda_orth)
                loss = loss_pred + loss_spec + loss_reg
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={global_step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg.get("grad_clip_norm", 0.5)))
                optimizer.step()
                running_loss += float(loss.detach().item())
                global_step += 1

            model.eval()
            val_true, val_pred = [], []
            with torch.no_grad():
                for bx, by in val_loader:
                    output = model(bx.to(device), inference=True)
                    val_true.append(by)
                    val_pred.append(output["y_hat"].cpu())
            y_true_val = torch.cat(val_true, dim=0)
            y_pred_val = torch.cat(val_pred, dim=0)
            mean_t = torch.as_tensor(mean.reshape(1, 1, -1), dtype=torch.float32)
            std_t = torch.as_tensor(std.reshape(1, 1, -1), dtype=torch.float32)
            y_true_val_denorm = y_true_val * std_t + mean_t
            y_pred_val_denorm = y_pred_val * std_t + mean_t
            val_mae = metric_mae(y_true_val_denorm, y_pred_val_denorm)
            val_rmse = metric_rmse(y_true_val_denorm, y_pred_val_denorm)
            val_mape = metric_mape(y_true_val_denorm, y_pred_val_denorm)
            horizon_values = _horizon_metrics(y_true_val_denorm, y_pred_val_denorm, interval_minutes, requested_horizons)

            scheduler.step(val_mae)
            lr_now = optimizer.param_groups[0]["lr"]
            msg = (
                f"[{dataset_name}] Epoch {epoch:03d} | loss={running_loss / max(len(train_loader),1):.5f} "
                f"| val MAE={val_mae:.4f} RMSE={val_rmse:.4f} MAPE={val_mape:.2f}% | lr={lr_now:.2e}"
            )
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()
            writer.writerow({"epoch": epoch, "split": "val", "horizon": "all", "MAE": f"{val_mae:.6f}", "RMSE": f"{val_rmse:.6f}", "MAPE": f"{val_mape:.6f}", "LR": f"{lr_now:.8f}"})
            for horizon, values in horizon_values.items():
                writer.writerow({"epoch": epoch, "split": "val", "horizon": horizon, **{k: f"{v:.6f}" for k, v in values.items()}, "LR": f"{lr_now:.8f}"})
            csv_file.flush()

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                early_stop_wait = 0
            else:
                early_stop_wait += 1
                if early_stop_wait >= early_stop_patience:
                    stop_message = f"EarlyStopping: no validation MAE improvement for {early_stop_patience} epochs."
                    print(stop_message)
                    log_file.write(stop_message + "\n")
                    break

        if best_state is None:
            raise RuntimeError("Training finished without producing a valid checkpoint")
        torch.save(best_state, ckpt_path)
        # Strict loading prevents silent partial checkpoint/model mismatches.
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
        model.eval()

        test_true, test_pred = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                output = model(bx.to(device), inference=True)
                test_true.append(by)
                test_pred.append(output["y_hat"].cpu())
        y_true_test = torch.cat(test_true, dim=0)
        y_pred_test = torch.cat(test_pred, dim=0)
        y_true_test_denorm = y_true_test * std_t + mean_t
        y_pred_test_denorm = y_pred_test * std_t + mean_t
        test_mae = metric_mae(y_true_test_denorm, y_pred_test_denorm)
        test_rmse = metric_rmse(y_true_test_denorm, y_pred_test_denorm)
        test_mape = metric_mape(y_true_test_denorm, y_pred_test_denorm)
        test_horizons = _horizon_metrics(y_true_test_denorm, y_pred_test_denorm, interval_minutes, requested_horizons)
        for horizon, values in test_horizons.items():
            writer.writerow({"epoch": "best", "split": "test", "horizon": horizon, **{k: f"{v:.6f}" for k, v in values.items()}, "LR": f"{optimizer.param_groups[0]['lr']:.8f}"})
        final_message = (
            f"[{dataset_name}-Test] MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, MAPE={test_mape:.2f}%\n"
            f"Best-Val-MAE={best_val_mae:.4f}\nCheckpoint={ckpt_path}\nLog={log_path}\nMetricsCSV={csv_path}"
        )
        print(final_message)
        log_file.write(final_message + "\n")

    return {
        "dataset": dataset_name,
        "best_val_mae": float(best_val_mae),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_mape": float(test_mape),
        "test_horizons": test_horizons,
        "checkpoint": str(ckpt_path),
        "log": str(log_path),
        "metrics_csv": str(csv_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Training entrypoint for the proposed traffic prediction framework")
    parser.add_argument("--config", type=str, default="configs/pems08.yaml", help="YAML config path")
    parser.add_argument("--adj-path", type=str, default=None, help="Optional explicit road adjacency file")
    parser.add_argument("--pred-len", type=int, default=None,
                        help="Override prediction length in 5-minute steps (e.g. 12 for 60 minutes)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    config_path = _resolve_repo_path(args.config, repo_root)
    cfg = load_cfg(str(config_path))
    config_name = config_path.stem
    result = train_one_run(
        cfg, config_name, adj_override=args.adj_path, pred_len_override=args.pred_len
    )
    print("=" * 72)
    for key, value in result.items():
        print(f"{key}: {value}")
    print("=" * 72)


if __name__ == "__main__":
    main()
