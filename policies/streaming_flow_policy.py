"""Backward-compatible streaming flow exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from policies.streaming_flow_unet_hybrid_image_policy import (
    StreamingFlowUnetHybridImagePolicy,
)

__all__ = ["StreamingFlowConfig", "StreamingFlowPolicy"]


@dataclass(frozen=True)
class StreamingFlowConfig:
    flow_mode: str = "stochastic"
    sigma0: float = 0.0
    sigma1: float = 0.1
    num_integration_steps: int = 100
    initial_action_mode: str = "auto"
    initial_action_keys: Optional[Sequence[str]] = None


StreamingFlowPolicy = StreamingFlowUnetHybridImagePolicy
