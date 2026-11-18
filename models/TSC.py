"""
Pseudo-code level implementation of TSC module.

Paper-aligned Stage-3 (State-Guided Modulation):
  1) Compute first-order difference statistics to build 5D evidence vector v_t (Eq.20)
  2) Induce latent control state s_state via MLP encoder (Eq.21)
  3) Soft prototype anchoring to produce s_ctrl (Eq.22)
  4) Warmup-gated bounded modulation of routing coefficients rho -> rho_tilde (Eq.24)
  5) Produce bounded correction residual delta_y_corr (Eq.24)
  Optional: frozen LLM branch (TESC-EoS) provides additive s_aux, disabled by default.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class TSC(nn.Module):
    """
    Traffic State Controller (pseudocode form).

    Inputs:
        X:         (B, T, N)        raw input sequence for evidence extraction
        rho_comm:  (B, C_comm)      community routing weights from FCR-ST
        S:         (B, N, d_model)  spatial feature map for residual correction
        t:         int              current training step for warmup schedule
    Output dict:
        s_ctrl:       (B, d_model)
        rho_tilde:    (B, C_comm)
        delta_y_corr: (B, horizon, N)
    """

    def __init__(
        self,
        d_model: int,
        num_communities: int,
        out_horizon: int = 12,
        num_prototypes: int = 8,
        tau_p: float = 0.1,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_communities = num_communities
        self.out_horizon = out_horizon
        self.tau_p = tau_p
        self.warmup_steps = warmup_steps

        # MLP encoder: 5D evidence vector -> latent state (Eq.21)
        self.mlp_encoder = nn.Sequential(
            nn.Linear(5, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Learnable prototype bank (Eq.22)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))

        # Routing modulation: s_ctrl -> delta rho (Eq.24)
        self.W_rho = nn.Linear(d_model, num_communities)

        # Residual correction strength: s_ctrl -> beta (Eq.24)
        self.W_beta = nn.Linear(d_model, d_model)

        # Correction head applied to spatial feature map S
        self.corr_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_horizon),
        )

        # Optional frozen-LLM branch (disabled by default)
        self.use_llm = False
        self.llm_gamma = nn.Parameter(torch.zeros(1))

    def _build_evidence_vector(self, X: torch.Tensor) -> torch.Tensor:
        """
        Construct 5D evidence vector v_t from input differences (Eq.20).
        v_t = [shock, trend, recovery, volatility, magnitude]
        X: (B, T, N)
        """
        delta_X = X.diff(dim=1)                        # (B, T-1, N)
        delta_last = delta_X[:, -1, :].mean(dim=-1)    # shock:    µ(∆X_last)
        delta_tau = delta_X.mean(dim=(1, 2))            # trend:    µ(∆X_τ)
        s = delta_last
        u = delta_tau
        recovery = -(s * u.sign())                      # recovery: -s·sgn(u)
        volatility = delta_X.std(dim=(1, 2))            # volatility: σ(∆X_τ)
        magnitude = delta_X.abs().mean(dim=(1, 2))      # magnitude: µ(|∆X_τ|)
        v_t = torch.stack(
            [delta_last, delta_tau, recovery, volatility, magnitude], dim=-1
        )  # (B, 5)
        return v_t

    def _warmup_gate(self, t: int) -> float:
        """Warmup schedule λ(t): linearly ramps from 0 to 1 over warmup_steps."""
        return min(1.0, t / max(self.warmup_steps, 1))

    def forward(
        self,
        X: torch.Tensor,
        rho_comm: torch.Tensor,
        S: torch.Tensor,
        t: int = 1000,
    ) -> Dict[str, torch.Tensor]:
        """Paper-aligned state-guided control branch."""
        B, T, N = X.shape

        # 1) Build 5D evidence vector v_t (Eq.20)
        v_t = self._build_evidence_vector(X)            # (B, 5)

        # 2) MLP state induction (Eq.21)
        s_state = self.mlp_encoder(v_t)                 # (B, d_model)

        # 3) Soft prototype anchoring -> s_ctrl (Eq.22)
        logits = s_state @ self.prototypes.T / self.tau_p   # (B, K)
        pi_t = torch.softmax(logits, dim=-1)                # (B, K)
        s_ctrl = pi_t @ self.prototypes                     # (B, d_model)

        # Optional LLM additive enhancement (Eq.23), disabled by default
        if self.use_llm:
            s_ctrl = s_ctrl + self.llm_gamma.clamp(0, 1) * self._llm_branch(X)

        # 4) Warmup gate λ(t)
        lam = self._warmup_gate(t)

        # 5) Bounded routing modulation: rho_tilde (Eq.24)
        rho_tilde = rho_comm + lam * torch.tanh(self.W_rho(s_ctrl))

        # 6) Bounded residual correction: beta_eff, delta_y_corr (Eq.24)
        beta_eff = torch.sigmoid(self.W_beta(s_ctrl))      # (B, d_model)
        # S: (B, N, d_model)
        corrected = beta_eff.unsqueeze(1) * S              # (B, N, d_model)
        delta_y_corr = self.corr_head(corrected).permute(0, 2, 1)  # (B, horizon, N)

        return {
            "s_ctrl": s_ctrl,
            "rho_tilde": rho_tilde,
            "delta_y_corr": delta_y_corr,
        }
