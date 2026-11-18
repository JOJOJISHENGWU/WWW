"""
Pseudo-code level implementation of PCS-ViM module.

Paper-aligned Stage-2 (Morphological Correction):
  1) Build pseudo-image feature map H from Stage-1 output
  2) Reorder node axis by diffusion priority Pi = Argsort(q)
  3) Dual-axis SSM scan to obtain h_scan
  4) Prototype-guided image branch fusion
  5) Produce bounded residual DeltaY_img
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCS_ViM(nn.Module):
    """
    Propagation-Compiled Scanning Vision-Mamba (pseudocode form).

    Input:
        H_spatial: (B, C, N, T)
    Output dict:
        h_scan:          (B, N, T, C)
        h_img:           (B, N, T, C)
        delta_y_img:     (B, horizon, N)
        node_priority_q: (N,)
        perm_pi:         (N,)
    """

    def __init__(self, d_model: int, out_horizon: int = 12, num_prototypes: int = 8):
        super().__init__()
        self.d_model = d_model
        self.out_horizon = out_horizon

        # Dual-axis scan placeholders (token-mixing + channel-mixing).
        self.scan_time = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.scan_node = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Prototype bank for morphology residual injection.
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model) * 0.02)
        self.proto_proj = nn.Linear(d_model, d_model)

        self.delta_head = nn.Conv2d(d_model, out_horizon, kernel_size=(1, 1))

    @staticmethod
    def _estimate_node_priority(h: torch.Tensor) -> torch.Tensor:
        """
        Estimate diffusion-style node priority q.
        h: (B, C, N, T)
        q: (N,)
        """
        # Priority proxy: temporal fluctuation energy per node.
        node_energy = h.diff(dim=-1).abs().mean(dim=(0, 1, 3))  # (N,)
        return node_energy

    def _dual_ssm_scan(self, x_btnc: torch.Tensor) -> torch.Tensor:
        """
        x_btnc: (B, N, T, C)
        returns h_scan: (B, N, T, C)
        """
        # Time-axis scan
        h_t = self.scan_time(x_btnc)

        # Node-axis scan (swap N and T to mimic second-axis pass)
        h_n = self.scan_node(x_btnc.transpose(1, 2)).transpose(1, 2)

        return x_btnc + 0.5 * (h_t + h_n)

    def _prototype_inject(self, h_scan: torch.Tensor) -> torch.Tensor:
        """
        Prototype-guided morphology enhancement.
        h_scan: (B, N, T, C)
        """
        B, N, T, C = h_scan.shape
        flat = h_scan.reshape(B * N * T, C)

        # Similarity to morphology prototypes.
        logits = flat @ self.prototypes.t()  # (BNT, P)
        attn = torch.softmax(logits, dim=-1)
        proto_feat = attn @ self.prototypes  # (BNT, C)
        proto_feat = self.proto_proj(proto_feat).reshape(B, N, T, C)

        return h_scan + proto_feat

    def forward(self, H_spatial: torch.Tensor, node_priority_q: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Paper-aligned morphological residual branch."""
        B, C, N, T = H_spatial.shape

        # 1) pseudo-image tensor: (B, N, T, C)
        h_btnc = H_spatial.permute(0, 2, 3, 1)

        # 2) diffusion-consistent order Pi
        if node_priority_q is None:
            node_priority_q = self._estimate_node_priority(H_spatial)
        perm_pi = torch.argsort(node_priority_q, descending=True)

        h_reordered = h_btnc[:, perm_pi, :, :]

        # 3) dual-axis scan
        h_scan = self._dual_ssm_scan(h_reordered)

        # 4) image branch prototype fusion
        h_img = self._prototype_inject(h_scan)

        # restore original node order
        inv_perm = torch.empty_like(perm_pi)
        inv_perm[perm_pi] = torch.arange(N, device=perm_pi.device)
        h_img_restore = h_img[:, inv_perm, :, :]

        # 5) bounded image residual DeltaY_img
        h_for_head = h_img_restore.permute(0, 3, 1, 2)  # (B, C, N, T)
        pooled = h_for_head.mean(dim=-1, keepdim=True)
        delta_y_img = self.delta_head(pooled).squeeze(-1)  # (B, horizon, N)

        return {
            "h_scan": h_scan,
            "h_img": h_img_restore,
            "delta_y_img": delta_y_img,
            "node_priority_q": node_priority_q,
            "perm_pi": perm_pi,
        }
