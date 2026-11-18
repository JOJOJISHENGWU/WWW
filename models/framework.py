"""Overall model wrapper for the anonymous proposed traffic forecasting method.

The filename is retained for compatibility with the existing repository
layout, but the model is referred to generically in code and documentation.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .FCR_ST import FCR_ST
from .PCS_ViM import PCS_ViM
from .TSC import TSC


class TrafficForecastModel(nn.Module):
    """Integrate FCR-ST, PCS-ViM and MLP-based TSC."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_nodes: int,
        out_horizon: int = 12,
        seq_len: int = 12,
        num_communities: int = 4,
        topk_freq: int = 6,
        cheb_order_local: int = 1,
        cheb_order_global: int = 3,
        num_morphology_prototypes: int = 8,
        num_state_prototypes: int = 8,
        warmup_steps: int = 1000,
        node2comm: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        avg_shortest_path_cost: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if adj is None:
            raise ValueError("A road adjacency matrix is required by the graph-based framework")
        if node2comm is None:
            raise ValueError("Offline spectral community labels (node2comm) must be supplied")
        self.num_nodes = int(num_nodes)
        self.hidden_channels = int(hidden_channels)
        self.seq_len = int(seq_len)
        self.out_horizon = int(out_horizon)
        self.num_communities = max(int(num_communities), int(torch.as_tensor(node2comm).max().item()) + 1)

        self.fcr_st = FCR_ST(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_nodes=num_nodes,
            num_communities=self.num_communities,
            topk_freq=topk_freq,
            cheb_order_local=cheb_order_local,
            cheb_order_global=cheb_order_global,
            node2comm=node2comm,
            adj=adj,
        )
        self.pcs_vim = PCS_ViM(
            d_model=hidden_channels,
            out_horizon=out_horizon,
            num_prototypes=num_morphology_prototypes,
            num_nodes=num_nodes,
            num_communities=self.num_communities,
            node2comm=node2comm,
            avg_shortest_path_cost=avg_shortest_path_cost,
        )
        self.tsc = TSC(
            d_model=hidden_channels,
            num_communities=self.num_communities,
            out_horizon=out_horizon,
            num_prototypes=num_state_prototypes,
            warmup_steps=warmup_steps,
        )

        # Linear prediction head: project the historical time axis to H steps,
        # then map hidden features to one traffic value per node and horizon.
        self.temporal_projection = nn.Linear(seq_len, out_horizon)
        self.channel_projection = nn.Linear(hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        step: int = 0,
        inference: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Input x=(B,1,N,T); return y_hat=(B,H,N)."""
        if x.ndim != 4:
            raise ValueError(f"Expected x=(B,C,N,T), got {tuple(x.shape)}")
        if x.shape[2] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.shape[2]}")
        if x.shape[-1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.shape[-1]}")

        fcr = self.fcr_st(x)
        h_mod = fcr["H_mod"]
        rho_comm = fcr["rho_comm"]
        routing_logits = fcr["routing_logits"]
        h_spatial_base = fcr["H_spatial"]

        # PCS-ViM provides the topology-informed morphology residual branch.
        pcs = self.pcs_vim(h_spatial_base)

        # TSC first computes state and corrected routing. The corrected spatial
        # feature map is then recomputed with rho_tilde, as specified in Eq. (23).
        x_raw = x.squeeze(1).transpose(1, 2)  # (B,T,N)
        tsc_state = self.tsc(
            X=x_raw,
            rho_comm=rho_comm,
            S=None,
            t=step,
            routing_logits=routing_logits,
            inference=inference,
        )
        rho_tilde = tsc_state["rho_tilde"]
        h_spatial = self.fcr_st.route_features(h_mod, rho_tilde)

        # S ≡ H_spatial in (B,T,N,D) layout; apply bounded feature correction.
        spatial_features = h_spatial.permute(0, 3, 2, 1)
        delta_h_corr = self.tsc.apply_feature_correction(spatial_features, tsc_state["beta_eff"])
        fused_features = spatial_features + delta_h_corr

        # Y_hat_FCR = Linear(H_spatial + Delta H_corr), followed by image residual.
        temporal_input = fused_features.permute(0, 2, 3, 1)  # (B,N,D,T)
        temporal_hidden = self.temporal_projection(temporal_input)  # (B,N,D,H)
        per_node_horizon = self.channel_projection(temporal_hidden.permute(0, 1, 3, 2)).squeeze(-1)
        y_fcr = per_node_horizon.transpose(1, 2)  # (B,H,N)
        y_hat = y_fcr + pcs["delta_y_img"]

        w_low, w_high = self.fcr_st.regularization_weights()
        return {
            "y_hat": y_hat,
            "y_fcr": y_fcr,
            "delta_y_img": pcs["delta_y_img"],
            "H_spatial": h_spatial,
            "H_mod": h_mod,
            "M_comm": fcr["M_comm"],
            "rho_comm": rho_comm,
            "routing_logits": routing_logits,
            "rho_tilde": rho_tilde,
            "s_ctrl": tsc_state["s_ctrl"],
            "beta_eff": tsc_state["beta_eff"],
            "delta_H_corr": delta_h_corr,
            "node_priority_q": pcs["node_priority_q"],
            "perm_pi": pcs["perm_pi"],
            "W_low": w_low,
            "W_high": w_high,
        }

