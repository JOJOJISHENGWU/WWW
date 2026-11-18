from __future__ import annotations

import argparse
import csv
import os
import random
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from models.SCOUT import SCOUT


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
        return yaml.safe_load(f)


def load_dataset_array(cfg: Dict) -> np.ndarray:
    data_path = cfg["dataset"]["data_path"]
    abs_path = os.path.normpath(os.path.join(os.path.dirname(__file__), data_path))

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Dataset not found: {abs_path}")

    if abs_path.endswith(".npz"):
        npz = np.load(abs_path)
        if "data" in npz:
            arr = npz["data"]
        elif "x" in npz:
            arr = npz["x"]
        else:
            arr = list(npz.values())[0]
    elif abs_path.endswith(".npy"):
        arr = np.load(abs_path)
    else:
        raise ValueError(f"Unsupported dataset format: {abs_path}")

    # expected shape (T, N, C) or (T, N)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


def build_windows(arr: np.ndarray, seq_len: int, pred_len: int = 1, feature_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    x_list, y_list = [], []
    total = arr.shape[0]

    for i in range(total - seq_len - pred_len + 1):
        x_list.append(arr[i: i + seq_len, :, feature_idx])
        y_list.append(arr[i + seq_len + pred_len - 1, :, feature_idx])

    x = np.stack(x_list)  # (B, T, N)
    y = np.stack(y_list)  # (B, N)
    return x, y


def chronological_split(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    n = x.shape[0]
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:val_end], y[train_end:val_end]
    x_test, y_test = x[val_end:], y[val_end:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def normalize_by_train(x_train: np.ndarray, x_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.clip(std, 1e-6, None)
    return (x_train - mean) / std, (x_other - mean) / std, (mean, std)


def to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False) -> DataLoader:
    # x: (B,T,N) -> (B,1,N,T)
    x_t = torch.tensor(x, dtype=torch.float32).permute(0, 2, 1).unsqueeze(1)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (B,1,N)
    ds = TensorDataset(x_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def spectral_consistency_loss(x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # Light-weight surrogate for paper-level L_spec, no proprietary details.
    x_fft = torch.fft.rfft(x.squeeze(1), dim=-1)
    y_fft = torch.fft.rfft(y_pred, dim=-1)
    x_amp = torch.abs(x_fft).mean(dim=-1)
    y_amp = torch.abs(y_fft).mean(dim=-1)
    return F.l1_loss(y_amp, x_amp)


def train_one_run(cfg: Dict, config_name: str) -> Dict[str, str | float]:
    dataset_name = cfg["dataset"]["name"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    loss_cfg = cfg.get("loss", {})
    runtime_cfg = cfg.get("runtime", {})

    set_seed(int(train_cfg.get("seed", 42)))

    device_name = runtime_cfg.get("device", "cuda")
    device = torch.device("cuda" if (device_name == "cuda" and torch.cuda.is_available()) else "cpu")

    arr = load_dataset_array(cfg)
    x, y = build_windows(arr, seq_len=int(model_cfg["seq_len"]), pred_len=int(model_cfg.get("pred_len", 1)))

    split_cfg = cfg.get("split", {})
    train_ratio = float(split_cfg.get("train", 0.7))
    val_ratio = float(split_cfg.get("val", 0.2))
    test_ratio = float(split_cfg.get("test", 0.1))

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = chronological_split(
        x,
        y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    x_train_norm, x_val_norm, norm_stats = normalize_by_train(x_train, x_val)
    mean, std = norm_stats
    x_test_norm = (x_test - mean) / std

    y_train_norm = (y_train - mean.squeeze(0).squeeze(0)) / std.squeeze(0).squeeze(0)
    y_val_norm = (y_val - mean.squeeze(0).squeeze(0)) / std.squeeze(0).squeeze(0)
    y_test_norm = (y_test - mean.squeeze(0).squeeze(0)) / std.squeeze(0).squeeze(0)

    batch_size = int(train_cfg.get("batch_size", 32))
    train_loader = to_loader(x_train_norm, y_train_norm, batch_size=batch_size, shuffle=True)
    val_loader = to_loader(x_val_norm, y_val_norm, batch_size=batch_size, shuffle=False)
    test_loader = to_loader(x_test_norm, y_test_norm, batch_size=batch_size, shuffle=False)

    model = SCOUT(
        in_channels=1,
        hidden_channels=int(model_cfg["d_model"]),
        num_nodes=int(cfg["dataset"]["num_nodes"]),
        out_horizon=int(model_cfg.get("pred_len", 1)),
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

    os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "checkpoints"), exist_ok=True)

    run_id = f"{dataset_name.lower()}_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = os.path.join(os.path.dirname(__file__), "logs", f"{run_id}.log")
    csv_path = os.path.join(os.path.dirname(__file__), "logs", f"{run_id}_metrics.csv")
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", f"{dataset_name.lower()}_best.pth")

    best_val_mae = float("inf")
    best_state = None
    es_wait = 0
    es_patience = int(train_cfg.get("early_stop_patience", 5))

    max_epochs = int(train_cfg.get("epochs", 300))

    with open(log_path, "w", encoding="utf-8") as lf, open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["epoch", "split", "MAE", "RMSE", "MAPE", "LR"])
        writer.writeheader()

        lf.write(f"Run ID: {run_id}\n")
        lf.write(f"Dataset: {dataset_name}\n")
        lf.write(f"Config: {config_name}\n")
        lf.write(f"Device: {device}\n")
        lf.write(f"Model d_model={model_cfg['d_model']}, seq_len={model_cfg['seq_len']}, pred_len={model_cfg.get('pred_len', 1)}\n")
        lf.write(f"Split train/val/test = {train_ratio:.2f}/{val_ratio:.2f}/{test_ratio:.2f}\n")
        lf.write("Metric space: de-normalized original scale\n")
        lf.flush()

        global_step = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss = 0.0

            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)

                optimizer.zero_grad(set_to_none=True)
                out = model(bx, step=global_step, inference=False)

                y_hat = out["y_hat"]
                loss_time = (
                    float(loss_cfg.get("pred_l1", 1.0)) * criterion_l1(y_hat, by)
                    + float(loss_cfg.get("pred_smoothl1", 0.3)) * criterion_smooth(y_hat, by)
                )
                loss_spec = float(loss_cfg.get("spectral_consistency", 0.02)) * spectral_consistency_loss(bx, y_hat)

                # Lightweight regularizer placeholder for paper-level L_reg.
                reg = 0.0
                if "rho_tilde" in out and "rho_comm" in out:
                    reg = float(loss_cfg.get("semantic_propagation", 0.01)) * F.mse_loss(out["rho_tilde"], out["rho_comm"].detach())

                loss = loss_time + loss_spec + reg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg.get("grad_clip_norm", 0.5)))
                optimizer.step()

                train_loss += float(loss.item())
                global_step += 1

            # Validation
            model.eval()
            y_true_val, y_pred_val = [], []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx, inference=True)
                    y_true_val.append(by.detach().cpu())
                    y_pred_val.append(out["y_hat"].detach().cpu())

            y_true_val = torch.cat(y_true_val, dim=0)
            y_pred_val = torch.cat(y_pred_val, dim=0)

            # De-normalize back to original scale for metric consistency with paper.
            mean_t = torch.tensor(mean.squeeze(0).squeeze(0), dtype=torch.float32).view(1, 1, -1)
            std_t = torch.tensor(std.squeeze(0).squeeze(0), dtype=torch.float32).view(1, 1, -1)
            y_true_val_denorm = y_true_val * std_t + mean_t
            y_pred_val_denorm = y_pred_val * std_t + mean_t

            val_mae = metric_mae(y_true_val_denorm, y_pred_val_denorm)
            val_rmse = metric_rmse(y_true_val_denorm, y_pred_val_denorm)
            val_mape = metric_mape(y_true_val_denorm, y_pred_val_denorm)

            scheduler.step(val_mae)
            lr_now = optimizer.param_groups[0]["lr"]

            msg = (
                f"[{dataset_name}] Epoch {epoch:03d} | train_loss={train_loss/max(1,len(train_loader)):.4f} "
                f"| val MAE={val_mae:.4f} RMSE={val_rmse:.4f} MAPE={val_mape:.2f}% | lr={lr_now:.2e}"
            )
            print(msg)
            lf.write(msg + "\n")
            lf.flush()

            writer.writerow({"epoch": epoch, "split": "val", "MAE": f"{val_mae:.6f}", "RMSE": f"{val_rmse:.6f}", "MAPE": f"{val_mape:.6f}", "LR": f"{lr_now:.8f}"})
            cf.flush()

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                es_wait = 0
            else:
                es_wait += 1
                if es_wait >= es_patience:
                    stop_msg = f"EarlyStopping: no val MAE improvement for {es_patience} epochs."
                    print(stop_msg)
                    lf.write(stop_msg + "\n")
                    break

        # Save best checkpoint
        if best_state is not None:
            torch.save(best_state, ckpt_path)
        else:
            torch.save(model.state_dict(), ckpt_path)

        # Test evaluation (using best checkpoint)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()

        y_true_test, y_pred_test = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx, inference=True)
                y_true_test.append(by.detach().cpu())
                y_pred_test.append(out["y_hat"].detach().cpu())

        y_true_test = torch.cat(y_true_test, dim=0)
        y_pred_test = torch.cat(y_pred_test, dim=0)

        # De-normalize back to original scale for metric consistency with paper.
        mean_t = torch.tensor(mean.squeeze(0).squeeze(0), dtype=torch.float32).view(1, 1, -1)
        std_t = torch.tensor(std.squeeze(0).squeeze(0), dtype=torch.float32).view(1, 1, -1)
        y_true_test_denorm = y_true_test * std_t + mean_t
        y_pred_test_denorm = y_pred_test * std_t + mean_t

        test_mae = metric_mae(y_true_test_denorm, y_pred_test_denorm)
        test_rmse = metric_rmse(y_true_test_denorm, y_pred_test_denorm)
        test_mape = metric_mape(y_true_test_denorm, y_pred_test_denorm)

        final_msg = (
            f"[{dataset_name}-Test] MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, MAPE={test_mape:.2f}%\n"
            f"Best-Val-MAE={best_val_mae:.4f}\n"
            f"Checkpoint={ckpt_path}\n"
            f"Log={log_path}\n"
            f"MetricsCSV={csv_path}"
        )
        print(final_msg)
        lf.write(final_msg + "\n")

    return {
        "dataset": dataset_name,
        "best_val_mae": float(best_val_mae),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_mape": float(test_mape),
        "checkpoint": ckpt_path,
        "log": log_path,
        "metrics_csv": csv_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SCOUT training entrypoint (review package)")
    parser.add_argument("--config", type=str, default="configs/pems08.yaml", help="YAML config path")
    args = parser.parse_args()

    config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), args.config))
    cfg = load_cfg(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    result = train_one_run(cfg, config_name)
    print("=" * 72)
    for k, v in result.items():
        print(f"{k}: {v}")
    print("=" * 72)


if __name__ == "__main__":
    main()

