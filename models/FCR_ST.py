"""Frequency-Community Routed Spatio-Temporal backbone.

This is a minimally extended implementation of the original repository module,
with the paper's graph-conditioned operations made explicit:
  (i) community-level RFFT descriptors,
  (ii) community-conditioned affine modulation,
  (iii) local/global Chebyshev graph filters and community routing.

Tensor convention throughout this module: x is (B, C_in, N, T).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """Return L = I - D^{-1/2} A D^{-1/2}."""
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be square (N,N), got {tuple(adj.shape)}")
    a = torch.as_tensor(adj, dtype=torch.float32)
    # The paper uses a road graph; symmetrization makes the spectral operator
    # numerically well-defined when the stored adjacency is slightly asymmetric.
    a = 0.5 * (a + a.transpose(0, 1))
    a = a.clamp_min(0)
    degree = a.sum(dim=1).clamp_min(1e-8)
    d_inv = torch.diag(degree.rsqrt())
    eye = torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    return eye - d_inv @ a @ d_inv


def scaled_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """Return the scaled Laplacian L_tilde = 2 L / lambda_max - I."""
    lap = normalized_laplacian(adj)
    if lap.numel() == 0:
        return lap
    lambda_max = torch.linalg.eigvalsh(lap).amax().clamp_min(1e-6)
    eye = torch.eye(lap.shape[0], dtype=lap.dtype, device=lap.device)
    return (2.0 / lambda_max) * lap - eye


class ChebyshevGraphConv(nn.Module):
    """K-order Chebyshev graph convolution for (B,C,N,T) features."""

    def __init__(self, in_channels: int, out_channels: int, order: int) -> None:
        super().__init__()
        if order < 0:
            raise ValueError("Chebyshev order must be non-negative")
        self.order = int(order)
        self.weight = nn.Parameter(torch.empty(self.order + 1, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, laplacian_tilde: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x=(B,C,N,T), got {tuple(x.shape)}")
        if laplacian_tilde.shape != (x.shape[2], x.shape[2]):
            raise ValueError(
                f"Laplacian shape {tuple(laplacian_tilde.shape)} does not match N={x.shape[2]}"
            )
        t0 = x
        supports = [t0]
        if self.order >= 1:
            t1 = torch.einsum("nm,bcmt->bcnt", laplacian_tilde, x)
            supports.append(t1)
        for _ in range(2, self.order + 1):
            tk = 2.0 * torch.einsum("nm,bcmt->bcnt", laplacian_tilde, supports[-1]) - supports[-2]
            supports.append(tk)
        out = x.new_zeros(x.shape[0], self.weight.shape[-1], x.shape[2], x.shape[3])
        for k, support in enumerate(supports):
            out = out + torch.einsum("bcnt,co->bont", support, self.weight[k])
        return out + self.bias.view(1, -1, 1, 1)


class FCR_ST(nn.Module):
    """Frequency-community routed spatio-temporal backbone."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_nodes: int,
        num_communities: int = 4,
        topk_freq: int = 6,
        cheb_order_local: int = 1,
        cheb_order_global: int = 3,
        node2comm: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_nodes = int(num_nodes)
        self.topk_freq = int(topk_freq)

        if adj is not None:
            adj_t = torch.as_tensor(adj, dtype=torch.float32)
            if adj_t.shape != (num_nodes, num_nodes):
                raise ValueError(f"adj must have shape ({num_nodes},{num_nodes})")
            lap_tilde = scaled_laplacian(adj_t)
            self.has_graph = True
        else:
            # Kept for backwards-compatible construction/testing only. The
            # training entrypoint deliberately requires an actual road graph.
            warnings.warn("FCR_ST was created without adj; using identity graph placeholder.", RuntimeWarning)
            lap_tilde = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
            self.has_graph = False

        if node2comm is None:
            if adj is not None:
                # Offline spectral partition: normalized-Laplacian eigenspace + KMeans.
                try:
                    import numpy as np
                    from sklearn.cluster import KMeans
                    lap_np = normalized_laplacian(adj_t).cpu().numpy().astype(np.float64)
                    _, eigvecs = np.linalg.eigh(lap_np)
                    k = max(1, min(int(num_communities), num_nodes))
                    node2comm = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(eigvecs[:, :k])
                    node2comm = torch.as_tensor(node2comm, dtype=torch.long)
                except Exception as exc:
                    raise RuntimeError("Spectral community assignment failed; install scikit-learn or pass node2comm.") from exc
            else:
                num_communities = 1
                node2comm = torch.zeros(num_nodes, dtype=torch.long)
        else:
            node2comm = torch.as_tensor(node2comm, dtype=torch.long)
            if node2comm.numel() != num_nodes:
                raise ValueError("node2comm must contain one community id per node")
            if node2comm.min().item() < 0:
                raise ValueError("Community ids must be non-negative")
            num_communities = max(int(num_communities), int(node2comm.max().item()) + 1)

        self.num_communities = int(num_communities)
        self.register_buffer("node2comm", node2comm.reshape(-1).long())
        self.register_buffer("laplacian_tilde", lap_tilde)

        # Top-k RFFT reconstruction followed by multi-kernel temporal encoding.
        self.period_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.temporal_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 3), padding=(0, 1))
        self.temporal_conv_5 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 5), padding=(0, 2))
        self.temporal_conv_7 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 7), padding=(0, 3))

        # Community embedding -> (gamma_c, beta_c), matching Eq. (9).
        self.community_embedding = nn.Embedding(self.num_communities, hidden_channels)
        self.community_affine = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 2 * hidden_channels),
        )
        nn.init.zeros_(self.community_affine[-1].weight)
        nn.init.zeros_(self.community_affine[-1].bias)

        # M_comm has a fixed top-k descriptor dimension per community.
        self.routing_mlp = nn.Sequential(
            nn.Linear(self.topk_freq, max(16, self.topk_freq * 2)),
            nn.GELU(),
            nn.Linear(max(16, self.topk_freq * 2), 1),
        )

        # Local (short-range) and global (broader) Chebyshev graph filters.
        self.kernel_local = ChebyshevGraphConv(hidden_channels, hidden_channels, cheb_order_local)
        self.kernel_global = ChebyshevGraphConv(hidden_channels, hidden_channels, cheb_order_global)

    def _community_spectrum_aggregate(self, amp: torch.Tensor) -> torch.Tensor:
        """Aggregate RFFT amplitudes to M_comm: (B,C_comm,K)."""
        # amp: (B,C_in,N,F); first average input channels, retaining nodes/frequencies.
        amp_mean = amp.mean(dim=1)
        descriptors = []
        k = min(self.topk_freq, amp_mean.shape[-1])
        for community_id in range(self.num_communities):
            mask = self.node2comm == community_id
            if mask.any():
                community_spectrum = amp_mean[:, mask, :].mean(dim=1)
            else:
                community_spectrum = amp_mean.mean(dim=1)
            top_values = torch.topk(community_spectrum, k=k, dim=-1, largest=True, sorted=True).values
            if k < self.topk_freq:
                top_values = F.pad(top_values, (0, self.topk_freq - k))
            descriptors.append(top_values)
        return torch.stack(descriptors, dim=1)

    def _topk_periodic_reconstruct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fft = torch.fft.rfft(x, dim=-1)
        amp = fft.abs()
        k = min(self.topk_freq, amp.shape[-1])
        # Select dominant frequencies per sample/channel/node, including DC only
        # when it is genuinely among the strongest components.
        indices = torch.topk(amp, k=k, dim=-1).indices
        mask = torch.zeros_like(fft, dtype=torch.bool)
        mask.scatter_(-1, indices, True)
        periodic = torch.fft.irfft(torch.where(mask, fft, torch.zeros_like(fft)), n=x.shape[-1], dim=-1)
        return periodic, amp

    def _period_aware_encode(self, periodic_signal: torch.Tensor, amp: torch.Tensor) -> torch.Tensor:
        """Build period-specific intra-period views before multi-kernel Conv2D.

        For each sample, dominant non-DC frequencies define p_i ~= T/f_i.
        The signal is folded into cycles of p_i, averaged across cycles, then
        interpolated back to T so the parallel convolution branches can be
        aggregated without changing the public tensor shape.
        """
        b, _, _, t = periodic_signal.shape
        latent = self.period_proj(periodic_signal)
        spectrum = amp.mean(dim=(1, 2))  # (B,F)
        encoded = []
        for batch_index in range(b):
            spec = spectrum[batch_index].clone()
            if spec.numel() > 1:
                spec[0] = -torch.inf  # exclude the zero-frequency component from period selection
            k = min(self.topk_freq, int(torch.isfinite(spec).sum().item()))
            if k <= 0:
                sample_view = latent[batch_index:batch_index + 1]
                multi = (self.temporal_conv(sample_view) + self.temporal_conv_5(sample_view) + self.temporal_conv_7(sample_view)) / 3.0
                encoded.append(F.gelu(multi))
                continue
            freq_indices = torch.topk(spec, k=k, largest=True, sorted=True).indices
            period_views = []
            sample_latent = latent[batch_index:batch_index + 1]
            for freq_index in freq_indices.tolist():
                period = max(1, min(t, int(round(t / max(int(freq_index), 1)))))
                cycles = int(math.ceil(t / period))
                padded_len = cycles * period
                padded = F.pad(sample_latent, (0, padded_len - t))
                folded = padded.reshape(1, self.hidden_channels, self.num_nodes, cycles, period).mean(dim=3)
                folded_nodes_first = folded.permute(0, 2, 1, 3).reshape(self.num_nodes, self.hidden_channels, 1, period)
                unfolded = F.interpolate(folded_nodes_first, size=(1, t), mode="bilinear", align_corners=False)
                period_views.append(unfolded.reshape(1, self.num_nodes, self.hidden_channels, t).permute(0, 2, 1, 3))
            period_feature = torch.stack(period_views, dim=0).mean(dim=0)
            multi = (self.temporal_conv(period_feature) + self.temporal_conv_5(period_feature) + self.temporal_conv_7(period_feature)) / 3.0
            encoded.append(F.gelu(multi))
        return torch.cat(encoded, dim=0)

    def _community_affine_modulation(self, h: torch.Tensor) -> torch.Tensor:
        # community affine parameters are generated from learnable community embeddings.
        affine = self.community_affine(self.community_embedding.weight)
        gamma, beta = affine.chunk(2, dim=-1)
        gamma_node = gamma[self.node2comm].transpose(0, 1).view(1, self.hidden_channels, self.num_nodes, 1)
        beta_node = beta[self.node2comm].transpose(0, 1).view(1, self.hidden_channels, self.num_nodes, 1)
        return (1.0 + gamma_node) * h + beta_node

    def _dual_kernel_route(self, h_mod: torch.Tensor, rho_comm: torch.Tensor) -> torch.Tensor:
        local = self.kernel_local(h_mod, self.laplacian_tilde)
        global_ = self.kernel_global(h_mod, self.laplacian_tilde)
        rho_node = rho_comm[:, self.node2comm].view(h_mod.shape[0], 1, self.num_nodes, 1)
        return rho_node * local + (1.0 - rho_node) * global_

    def route_features(self, h_mod: torch.Tensor, rho_comm: torch.Tensor) -> torch.Tensor:
        """Re-run graph propagation with state-conditioned community gates."""
        return self._dual_kernel_route(h_mod, rho_comm)

    def regularization_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return order-aggregated local/global coefficient matrices for Eq. (26)."""
        w_low = self.kernel_local.weight.sum(dim=0)
        w_high = self.kernel_global.weight.sum(dim=0)
        return w_low, w_high

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        rho_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"FCR_ST expects x=(B,C,N,T), got {tuple(x.shape)}")
        if x.shape[2] != self.num_nodes:
            raise ValueError(f"Expected N={self.num_nodes}, got N={x.shape[2]}")
        if adj is not None:
            adj_tilde = scaled_laplacian(torch.as_tensor(adj, device=x.device, dtype=x.dtype))
        else:
            adj_tilde = self.laplacian_tilde.to(device=x.device, dtype=x.dtype)

        periodic_signal, amp = self._topk_periodic_reconstruct(x)
        m_comm = self._community_spectrum_aggregate(amp)
        z = self._period_aware_encode(periodic_signal, amp)
        h_mod = self._community_affine_modulation(z)
        routing_logits = self.routing_mlp(m_comm).squeeze(-1)
        rho_comm = torch.sigmoid(routing_logits) if rho_override is None else rho_override
        if rho_comm.shape != (x.shape[0], self.num_communities):
            raise ValueError(
                f"rho_comm must be {(x.shape[0], self.num_communities)}, got {tuple(rho_comm.shape)}"
            )

        # Use the supplied graph for this pass if one was explicitly provided.
        if adj is not None:
            local = self.kernel_local(h_mod, adj_tilde)
            global_ = self.kernel_global(h_mod, adj_tilde)
            rho_node = rho_comm[:, self.node2comm].view(x.shape[0], 1, self.num_nodes, 1)
            h_spatial = rho_node * local + (1.0 - rho_node) * global_
        else:
            h_spatial = self._dual_kernel_route(h_mod, rho_comm)
        return {
            "H_spatial": h_spatial,
            "H_mod": h_mod,
            "M_comm": m_comm,
            "rho_comm": rho_comm,
            "routing_logits": routing_logits,
        }

