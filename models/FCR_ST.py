"""
Pseudo-code level implementation of FCR-ST module.

This file is intentionally written as *method-aligned pseudocode* rather than
fully reproduced training code. The objective is to mirror the paper logic:

Stage-1 (FCR-ST):
  1) RFFT-based periodic decomposition
  2) community spectral aggregation M_comm (Eq.1)
  3) top-k periodic feature extraction
  4) community affine modulation (Eq.2)
  5) dual-kernel routing fusion with rho_comm (Eq.3)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCR_ST(nn.Module):
    """
    Frequency-Community Routed Spatio-Temporal backbone.

    Input:
        x: (B, C_in, N, T)
    Output dict:
        H_spatial: (B, C_hid, N, T)
        H_mod:     (B, C_hid, N, T)
        M_comm:    (B, C_comm, K)
        rho_comm:  (B, C_comm)

    Notes:
    - This pseudocode is designed for reviewer readability and paper alignment.
    - Community assignment is represented by `node2comm` buffer.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_nodes: int,
        num_communities: int = 8,
        topk_freq: int = 6,
        cheb_order_local: int = 1,
        cheb_order_global: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_nodes = num_nodes
        self.num_communities = num_communities
        self.topk_freq = topk_freq
        self.cheb_order_local = cheb_order_local
        self.cheb_order_global = cheb_order_global

        # Community assignment placeholder (can be replaced by offline spectral partition).
        node2comm = torch.arange(num_nodes) % num_communities
        self.register_buffer("node2comm", node2comm.long())

        # Periodic branch (Top-k frequency to latent temporal features).
        self.period_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1))
        self.temporal_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 3), padding=(0, 1))

        # Community-aware affine recalibration (Eq.2 style).
        self.gamma = nn.Parameter(torch.ones(num_communities, hidden_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(num_communities, hidden_channels, 1, 1))

        # Dual-kernel routing gate rho_comm = sigma(MLP(M_comm)).
        self.routing_mlp = nn.Sequential(
            nn.Linear(topk_freq, max(16, topk_freq * 2)),
            nn.GELU(),
            nn.Linear(max(16, topk_freq * 2), 1),
        )

        # Local/global graph kernels are abstracted as depthwise temporal operators.
        self.kernel_local = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 3), padding=(0, 1), groups=hidden_channels)
        self.kernel_global = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 5), padding=(0, 2), groups=hidden_channels)

    def _community_spectrum_aggregate(self, amp: torch.Tensor) -> torch.Tensor:
        """
        Aggregate node-level frequency amplitudes to community-level spectrum M_comm.

        Args:
            amp: (B, C_in, N, F) from RFFT amplitude
        Returns:
            M_comm: (B, C_comm, K) where K = topk_freq
        """
        B, C_in, N, Freq = amp.shape
        K = min(self.topk_freq, Freq)

        # First aggregate along channels for compact spectrum descriptor.
        amp_mean = amp.mean(dim=1)  # (B, N, F)

        # Community pooling: mean over nodes in each community.
        M_comm = []
        for c in range(self.num_communities):
            node_mask = self.node2comm == c
            if node_mask.any():
                c_spec = amp_mean[:, node_mask, :].mean(dim=1)  # (B, F)
            else:
                c_spec = amp_mean.mean(dim=1)  # fallback
            M_comm.append(c_spec[:, :K])
        return torch.stack(M_comm, dim=1)  # (B, C_comm, K)

    def _topk_periodic_reconstruct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Keep top-k frequency components and reconstruct periodic signal.

        Returns:
            periodic_signal: (B, C_in, N, T)
            amp:             (B, C_in, N, F)
        """
        fft = torch.fft.rfft(x, dim=-1)
        amp = torch.abs(fft)

        K = min(self.topk_freq, amp.shape[-1])
        topk_idx = torch.topk(amp, k=K, dim=-1).indices
        mask = torch.zeros_like(fft, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        fft_sparse = torch.where(mask, fft, torch.zeros_like(fft))

        periodic_signal = torch.fft.irfft(fft_sparse, n=x.shape[-1], dim=-1)
        return periodic_signal, amp

    def _community_affine_modulation(self, h: torch.Tensor) -> torch.Tensor:
        """Apply community-specific affine modulation to hidden features (Eq.2 style)."""
        B, C_h, N, T = h.shape
        out = torch.empty_like(h)
        for c in range(self.num_communities):
            node_mask = self.node2comm == c
            if not node_mask.any():
                continue
            out[:, :, node_mask, :] = self.gamma[c] * h[:, :, node_mask, :] + self.beta[c]
        return out

    def _dual_kernel_route(self, h_mod: torch.Tensor, rho_comm: torch.Tensor) -> torch.Tensor:
        """
        H_spatial = rho * K_local(h_mod) + (1-rho) * K_global(h_mod), per community.
        """
        local = self.kernel_local(h_mod)
        global_ = self.kernel_global(h_mod)

        out = torch.empty_like(local)
        for c in range(self.num_communities):
            node_mask = self.node2comm == c
            if not node_mask.any():
                continue
            rho = rho_comm[:, c].view(-1, 1, 1, 1)  # (B,1,1,1)
            out[:, :, node_mask, :] = rho * local[:, :, node_mask, :] + (1.0 - rho) * global_[:, :, node_mask, :]
        return out

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Pseudocode path matching paper Algorithm Stage-1.

        `adj` is kept for interface consistency; this pseudocode abstracts graph ops.
        """
        periodic_signal, amp = self._topk_periodic_reconstruct(x)
        M_comm = self._community_spectrum_aggregate(amp)

        # Build periodic hidden representation Z.
        z = self.period_proj(periodic_signal)
        z = F.gelu(self.temporal_conv(z))

        # Community affine recalibration -> H_mod.
        H_mod = self._community_affine_modulation(z)

        # rho_comm = sigmoid(MLP(M_comm)).
        rho_comm = torch.sigmoid(self.routing_mlp(M_comm).squeeze(-1))  # (B, C_comm)

        # Dual-kernel routing -> H_spatial.
        H_spatial = self._dual_kernel_route(H_mod, rho_comm)

        return {
            "H_spatial": H_spatial,
            "H_mod": H_mod,
            "M_comm": M_comm,
            "rho_comm": rho_comm,
        }
