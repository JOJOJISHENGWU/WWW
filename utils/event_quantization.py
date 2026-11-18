"""
Event quantization helper (review package version).

Maps traffic volatility statistics to coarse event tokens used by semantic branch.
"""

from __future__ import annotations

from typing import List

import torch


def quantize_event_tokens(h_spatial: torch.Tensor) -> List[str]:
    """
    Args:
        h_spatial: Tensor shaped (B, C, N, T)
    Returns:
        list of event tokens length B
    """
    if h_spatial.dim() != 4:
        raise ValueError(f"Expected (B,C,N,T), got shape={tuple(h_spatial.shape)}")

    vol = h_spatial.diff(dim=-1).abs().mean(dim=(1, 2, 3))
    trend = h_spatial.mean(dim=(1, 2, 3))

    tokens: List[str] = []
    for v, t in zip(vol.tolist(), trend.tolist()):
        if v > 0.25 and t > 0:
            tokens.append("event:surge")
        elif v > 0.25 and t <= 0:
            tokens.append("event:shock")
        elif v > 0.10:
            tokens.append("event:fluctuation")
        else:
            tokens.append("event:stable")
    return tokens
