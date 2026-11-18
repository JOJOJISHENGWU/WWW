"""Traffic State Controller (TSC).

Implements the five-dimensional evidence vector, MLP state induction,
prototype anchoring, warmup-gated routing-logit modulation and feature-level
residual correction in the current paper. No language-model branch is used.
"""
from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TSC(nn.Module):
    """Evidence-driven traffic-state controller."""

    def __init__(
        self,
        d_model: int,
        num_communities: int,
        out_horizon: int = 12,
        num_prototypes: int = 8,
        tau_p: float = 0.1,
        warmup_steps: int = 1000,
        beta_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_communities = int(num_communities)
        self.out_horizon = int(out_horizon)
        self.tau_p = max(float(tau_p), 1e-6)
        self.warmup_steps = max(int(warmup_steps), 1)

        self.mlp_encoder = nn.Sequential(
            nn.Linear(5, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model) * 0.02)
        self.W_rho = nn.Linear(d_model, num_communities)
        self.W_beta = nn.Linear(d_model, d_model)
        self.beta_base = nn.Parameter(torch.tensor(float(beta_init)))

    def _build_evidence_vector(self, X: torch.Tensor) -> torch.Tensor:
        """Equation (20): [shock, trend, recovery, volatility, magnitude].

        X is (B,T,N), with the latest temporal difference at index -1.
        """
        if X.ndim != 3:
            raise ValueError(f"X must have shape (B,T,N), got {tuple(X.shape)}")
        if X.shape[1] < 2:
            raise ValueError("TSC requires at least two historical time steps")
        delta_x = X.diff(dim=1)
        delta_last = delta_x[:, -1, :]  # (B,N)
        shock = delta_last.mean(dim=-1)
        trend = delta_x.mean(dim=(1, 2))
        latest_change_magnitude = delta_last.abs().mean(dim=-1)
        recovery = -latest_change_magnitude * trend.sign()
        volatility = delta_x.std(dim=(1, 2), unbiased=False)
        magnitude = delta_x.abs().mean(dim=(1, 2))
        return torch.stack([shock, trend, recovery, volatility, magnitude], dim=-1)

    def _warmup_gate(self, step: int, inference: bool = False) -> float:
        if inference:
            return 1.0
        return min(1.0, max(0.0, float(step) / float(self.warmup_steps)))

    @staticmethod
    def _logits_from_probability(rho_comm: torch.Tensor) -> torch.Tensor:
        rho = rho_comm.clamp(min=1e-6, max=1.0 - 1e-6)
        return torch.log(rho) - torch.log1p(-rho)

    def apply_feature_correction(
        self,
        spatial_features: torch.Tensor,
        beta_eff: torch.Tensor,
    ) -> torch.Tensor:
        """Equation (23): Delta H_corr = beta_eff ⊙ S.

        Accepts S=(B,N,D) or S=(B,T,N,D), and broadcasts beta_eff=(B,D).
        """
        if spatial_features.shape[-1] != self.d_model:
            raise ValueError(
                f"Last dimension of spatial_features must be d_model={self.d_model}, "
                f"got {spatial_features.shape[-1]}"
            )
        shape = [beta_eff.shape[0]] + [1] * (spatial_features.ndim - 2) + [self.d_model]
        return beta_eff.view(*shape) * spatial_features

    def forward(
        self,
        X: torch.Tensor,
        rho_comm: torch.Tensor,
        S: Optional[torch.Tensor] = None,
        t: int = 0,
        routing_logits: Optional[torch.Tensor] = None,
        inference: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if rho_comm.ndim != 2 or rho_comm.shape[-1] != self.num_communities:
            raise ValueError(
                f"rho_comm must have shape (B,{self.num_communities}), got {tuple(rho_comm.shape)}"
            )
        v_t = self._build_evidence_vector(X)
        s_state = self.mlp_encoder(v_t)

        # Equation (22): soft assignment to the learnable state-prototype bank.
        logits = (s_state @ self.prototypes.transpose(0, 1)) / self.tau_p
        pi_t = torch.softmax(logits, dim=-1)
        s_ctrl = pi_t @ self.prototypes

        lam = self._warmup_gate(t, inference=inference)
        base_logits = routing_logits if routing_logits is not None else self._logits_from_probability(rho_comm)
        if base_logits.shape != rho_comm.shape:
            raise ValueError("routing_logits and rho_comm must have the same shape")

        # Equation (23): perturb logits before sigmoid to keep gates in (0,1).
        routing_delta = torch.tanh(self.W_rho(s_ctrl))
        rho_tilde = torch.sigmoid(base_logits + lam * routing_delta)

        # Equation (23): beta_eff = beta * (1 + lambda * sigmoid(W_beta s_ctrl)).
        beta_eff = self.beta_base * (1.0 + lam * torch.sigmoid(self.W_beta(s_ctrl)))
        result = {
            "v_t": v_t,
            "s_state": s_state,
            "pi_t": pi_t,
            "s_ctrl": s_ctrl,
            "rho_tilde": rho_tilde,
            "beta_eff": beta_eff,
            "lambda_t": X.new_tensor(lam),
        }
        if S is not None:
            result["delta_H_corr"] = self.apply_feature_correction(S, beta_eff)
        return result
