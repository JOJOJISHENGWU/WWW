"""
Pseudo-code level implementation of SCOUT.

SCOUT paper-aligned tri-modal pipeline:
  Stage 1: FCR-ST (spectral-topological baseline)
  Stage 2: PCS-ViM (morphological bounded correction)
  Stage 3: TESC-EoS (semantic dynamic modulation)

This file intentionally emphasizes algorithmic mapping and symbol consistency
with the paper (Y_FCR, DeltaY_img, DeltaY_sem, rho_comm, rho_tilde) rather than
full engineering reproducibility.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .FCR_ST import FCR_ST
from .PCS_ViM import PCS_ViM
from .TESC_EoS import TESC_EoS


class SCOUT(nn.Module):
    """
    Spectral-Community Orchestrated Unified Tri-modal model (pseudocode form).

    Args:
        in_channels: input feature channels
        hidden_channels: latent width
        num_nodes: graph nodes
        out_horizon: prediction horizon
        num_communities: spectral communities

    Forward returns a dictionary for transparent stage-wise interpretation.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_nodes: int,
        out_horizon: int,
        num_communities: int = 8,
    ) -> None:
        super().__init__()
        self.out_horizon = out_horizon

        # Stage-1: baseline backbone
        self.fcr_st = FCR_ST(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_nodes=num_nodes,
            num_communities=num_communities,
        )

        # Baseline readout Y_FCR
        self.baseline_head = nn.Conv2d(hidden_channels, out_horizon, kernel_size=(1, 1))

        # Stage-2: morphology residual branch
        self.pcs_vim = PCS_ViM(d_model=hidden_channels, out_horizon=out_horizon)

        # Stage-3: semantic residual branch + route modulation
        self.tesc_eos = TESC_EoS(
            d_model=hidden_channels,
            num_communities=num_communities,
            out_horizon=out_horizon,
        )

    @staticmethod
    def _warmup_weight(step: Optional[int], total: int = 10_000) -> float:
        """Simple linear warmup surrogate for lambda_img/lambda_sem schedules."""
        if step is None:
            return 1.0
        return float(min(1.0, max(0.0, step / float(total))))

    def forward(
        self,
        x: torch.Tensor,
        *,
        adj: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        inference: bool = False,
    ) -> Dict[str, torch.Tensor | list[str]]:
        """
        Args:
            x: (B, C_in, N, T)
            adj: optional graph adjacency placeholder
            step: global optimization step for warmup scheduling
            inference: if True, fix lambda=1 and use semantic cache path

        Returns:
            A dict with final prediction and intermediate paper symbols.
        """
        # -------------------------
        # Stage 1: FCR-ST
        # -------------------------
        stage1 = self.fcr_st(x, adj=adj)
        H_spatial = stage1["H_spatial"]
        rho_comm = stage1["rho_comm"]

        # Baseline prediction Y_FCR
        y_fcr = self.baseline_head(H_spatial.mean(dim=-1, keepdim=True)).squeeze(-1)  # (B, horizon, N)

        # -------------------------
        # Stage 2: PCS-ViM
        # -------------------------
        stage2 = self.pcs_vim(H_spatial)
        delta_y_img = stage2["delta_y_img"]

        # -------------------------
        # Stage 3: TESC-EoS
        # -------------------------
        stage3 = self.tesc_eos(H_spatial, rho_comm, use_cache=inference)
        rho_tilde = stage3["rho_tilde"]
        delta_y_sem = stage3["delta_y_sem"]

        # -------------------------
        # Aggregation
        # -------------------------
        if inference:
            lambda_img = 1.0
            lambda_sem = 1.0
        else:
            lambda_img = self._warmup_weight(step, total=8_000)
            lambda_sem = self._warmup_weight(step, total=12_000)

        y_hat = y_fcr + lambda_img * delta_y_img + lambda_sem * delta_y_sem

        return {
            "y_hat": y_hat,
            "y_fcr": y_fcr,
            "delta_y_img": delta_y_img,
            "delta_y_sem": delta_y_sem,
            "lambda_img": torch.tensor(lambda_img, device=x.device),
            "lambda_sem": torch.tensor(lambda_sem, device=x.device),
            "rho_comm": rho_comm,
            "rho_tilde": rho_tilde,
            "M_comm": stage1["M_comm"],
            "event_tokens": stage3["event_tokens"],
            "node_priority_q": stage2["node_priority_q"],
        }
