"""Propagation-Compiled Scanning VisionMamba (PCS-ViM).

The module retains the original repository interface, but replaces the MLP
scan placeholders with dual-axis selective state-space scans. When the
``mamba_ssm`` package is installed, its Mamba block is used; otherwise a
self-contained diagonal selective-SSM implementation is used (not an MLP).
Input convention: H_spatial=(B,C,N,T); output residual=(B,H,N).
"""
from __future__ import annotations

from typing import Dict, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Optional optimized Mamba implementation.
    from mamba_ssm import Mamba as _Mamba
except Exception:  # Package may be absent or CUDA extensions unavailable.
    _Mamba = None


class _SelectiveSSMFallback(nn.Module):
    """Diagonal selective state-space block with input-dependent time steps.

    This keeps the scan state-space-based when mamba_ssm is unavailable. It is
    deliberately explicit and does not claim to be a fused CUDA Mamba kernel.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=d_conv, groups=d_model, padding=d_conv - 1)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.b_proj = nn.Linear(d_model, d_state)
        self.c_proj = nn.Linear(d_model, d_state)
        self.a_log = nn.Parameter(torch.zeros(d_model, d_state))
        self.d_skip = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, sequence, channels)
        residual = x
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        u = F.silu(u)
        delta = F.softplus(self.dt_proj(u)).clamp(max=10.0)
        b_t = torch.tanh(self.b_proj(u))
        c_t = torch.tanh(self.c_proj(u))
        a = -torch.exp(self.a_log).unsqueeze(0)  # (1,D,D_state)
        state = u.new_zeros(u.shape[0], self.d_model, self.d_state)
        outputs = []
        for step in range(x.shape[1]):
            dt = delta[:, step, :].unsqueeze(-1)
            a_bar = torch.exp(dt * a)
            # Exact zero-order-hold discretization for diagonal A.
            b_bar = ((a_bar - 1.0) / a) * b_t[:, step, :].unsqueeze(1)
            state = a_bar * state + b_bar * u[:, step, :].unsqueeze(-1)
            y = (state * c_t[:, step, :].unsqueeze(1)).sum(dim=-1) + self.d_skip * u[:, step, :]
            outputs.append(y)
        y = torch.stack(outputs, dim=1) if outputs else u
        y = self.out_proj(y * torch.sigmoid(gate))
        return self.norm(residual + y)


class SelectiveSSMBlock(nn.Module):
    """Mamba block wrapper with a native selective-SSM fallback."""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.fallback = _SelectiveSSMFallback(d_model=d_model, d_state=d_state, d_conv=d_conv)
        self.official = None
        if _Mamba is not None:
            try:
                self.official = _Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            except Exception:
                self.official = None
        self.is_official_mamba = self.official is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CUDA is the supported execution path for most mamba_ssm builds.
        if self.official is not None and x.is_cuda:
            try:
                return self.official(x)
            except Exception:
                # Fall back rather than failing on an incompatible CUDA extension.
                pass
        return self.fallback(x)


class PCS_ViM(nn.Module):
    """Topology-informed morphology encoder with dual-axis SSM scans."""
    def __init__(
        self,
        d_model: int,
        out_horizon: int = 12,
        num_prototypes: int = 8,
        num_nodes: Optional[int] = None,
        num_communities: int = 4,
        node2comm: Optional[torch.Tensor] = None,
        avg_shortest_path_cost: Optional[torch.Tensor] = None,
        tau_d: float = 1.0,
        eta: float = 1.0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.out_horizon = int(out_horizon)
        self.tau_d = max(float(tau_d), 1e-6)
        self.eta = float(eta)

        if node2comm is None:
            if num_nodes is None:
                raise ValueError("num_nodes is required when node2comm is not supplied")
            node2comm = torch.zeros(num_nodes, dtype=torch.long)
            num_communities = 1
        node2comm = torch.as_tensor(node2comm, dtype=torch.long).reshape(-1)
        self.num_nodes = int(node2comm.numel())
        self.num_communities = max(int(num_communities), int(node2comm.max().item()) + 1)
        self.register_buffer("node2comm", node2comm)

        if avg_shortest_path_cost is None:
            avg_shortest_path_cost = torch.zeros(self.num_nodes, dtype=torch.float32)
        avg_shortest_path_cost = torch.as_tensor(avg_shortest_path_cost, dtype=torch.float32).reshape(-1)
        if avg_shortest_path_cost.numel() != self.num_nodes:
            raise ValueError("avg_shortest_path_cost must contain one value per node")
        self.register_buffer("avg_shortest_path_cost", avg_shortest_path_cost)

        # Period-aware multi-scale pseudo-image projections.
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(d_model, d_model, kernel_size=(1, k), padding=(0, k // 2), groups=1)
            for k in (3, 5, 7)
        ])
        self.scale_norm = nn.BatchNorm2d(d_model)

        # Community-potential prototypes and calibration term in Eq. (13).
        self.community_prototypes = nn.Parameter(torch.randn(self.num_communities, d_model) * 0.02)
        self.community_potential = nn.Sequential(nn.Linear(d_model, d_model // 2 or 1), nn.Tanh(), nn.Linear(d_model // 2 or 1, 1))
        self.priority_scale = nn.Parameter(torch.tensor(1.0))

        self.ssm_time = SelectiveSSMBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.ssm_node = SelectiveSSMBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.lambda_scan_logit = nn.Parameter(torch.tensor(0.0))

        # Prototype-aligned morphology injection, Eq. (18).
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model) * 0.02)
        self.proto_proj = nn.Linear(d_model, d_model)
        self.priority_gate = nn.Linear(1, 1)
        self.out_proj = nn.Linear(d_model, out_horizon)
        self.lambda_img_logit = nn.Parameter(torch.tensor(0.0))

    def _estimate_node_priority(self, h: torch.Tensor) -> torch.Tensor:
        """q_n = community activity + calibrated community potential + travel decay."""
        # h: (B,C,N,T); activity prior is averaged within each spectral community.
        node_activity = h.abs().mean(dim=(0, 1, 3))  # (N,)
        community_activity = h.new_zeros(self.num_communities)
        community_count = h.new_zeros(self.num_communities)
        community_activity.index_add_(0, self.node2comm, node_activity)
        community_count.index_add_(0, self.node2comm, torch.ones_like(node_activity))
        community_activity = community_activity / community_count.clamp_min(1.0)
        calibrated_potential = self.community_potential(self.community_prototypes).squeeze(-1)
        q = community_activity[self.node2comm] + calibrated_potential[self.node2comm]
        travel_decay = torch.exp(-self.avg_shortest_path_cost.to(h.device) / self.tau_d)
        q = q + self.eta * travel_decay
        return self.priority_scale * q

    def _build_pseudo_image(self, h: torch.Tensor) -> torch.Tensor:
        # Multi-scale temporal projections, retaining the sensor-node axis.
        multi = sum(conv(h) for conv in self.scale_convs) / float(len(self.scale_convs))
        return F.gelu(self.scale_norm(h + multi)).permute(0, 2, 3, 1)  # (B,N,T,C)

    def _dual_ssm_scan(self, x_btnc: torch.Tensor) -> torch.Tensor:
        """Dual-axis selective SSM: temporal scan then node-axis scan."""
        b, n, t, c = x_btnc.shape
        time_seq = x_btnc.reshape(b * n, t, c)
        h_time = self.ssm_time(time_seq).reshape(b, n, t, c)
        node_seq = x_btnc.permute(0, 2, 1, 3).reshape(b * t, n, c)
        h_node = self.ssm_node(node_seq).reshape(b, t, n, c).permute(0, 2, 1, 3)
        return 0.5 * (h_time + h_node)

    def _prototype_align(self, h_img: torch.Tensor) -> torch.Tensor:
        b, n, t, c = h_img.shape
        flat = h_img.reshape(b * n * t, c)
        prototypes = F.normalize(self.prototypes, dim=-1)
        query = F.normalize(flat, dim=-1)
        attention = torch.softmax((query @ prototypes.t()) / math.sqrt(c), dim=-1)
        proto_features = attention @ self.prototypes
        proto_features = self.proto_proj(proto_features).reshape(b, n, t, c)
        return h_img + proto_features

    def forward(
        self,
        H_spatial: torch.Tensor,
        node_priority_q: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if H_spatial.ndim != 4:
            raise ValueError(f"Expected H_spatial=(B,C,N,T), got {tuple(H_spatial.shape)}")
        b, c, n, t = H_spatial.shape
        if n != self.num_nodes:
            raise ValueError(f"Expected N={self.num_nodes}, got N={n}")

        h_base_input = self._build_pseudo_image(H_spatial)
        q = self._estimate_node_priority(H_spatial) if node_priority_q is None else node_priority_q.to(H_spatial.device)
        if q.shape != (n,):
            raise ValueError(f"node_priority_q must have shape ({n},), got {tuple(q.shape)}")

        # Eq. (14): Pi = Argsort(q). A single scalar ordering is a heuristic,
        # not a guarantee that every directed edge is traversed consistently.
        perm_pi = torch.argsort(q, descending=False)
        h_scan_perm = self._dual_ssm_scan(h_base_input[:, perm_pi, :, :])
        inv_perm = torch.empty_like(perm_pi)
        inv_perm[perm_pi] = torch.arange(n, device=perm_pi.device)
        h_scan = h_scan_perm[:, inv_perm, :, :]

        # The unpermuted branch preserves absolute spatial cues (Eq. 17).
        h_base = self._dual_ssm_scan(h_base_input)
        lambda_scan = torch.sigmoid(self.lambda_scan_logit)
        h_img = h_base + lambda_scan * (h_scan - h_base)
        h_img = self._prototype_align(h_img)

        q_norm = (q - q.mean()) / q.std(unbiased=False).clamp_min(1e-6)
        q_norm = q_norm.view(1, n, 1).expand(b, -1, -1)
        gate = torch.sigmoid(self.priority_gate(q_norm)).unsqueeze(2)  # (B,N,1,1)
        pooled = (h_img * gate).mean(dim=2)  # (B,N,C)
        delta_y_img = self.out_proj(pooled).transpose(1, 2) * torch.sigmoid(self.lambda_img_logit)

        return {
            "h_scan": h_scan,
            "h_img": h_img,
            "delta_y_img": delta_y_img,
            "node_priority_q": q,
            "perm_pi": perm_pi,
            "priority_gate": gate.squeeze(-1),
        }
