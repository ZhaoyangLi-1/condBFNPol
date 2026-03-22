"""Streaming Flow training workspace.

Streaming Flow fits the same train/eval contract as the BFN image policies, so
we reuse that workspace directly and keep a dedicated class only for config and
script compatibility.
"""

from __future__ import annotations

from workspaces.train_bfn_workspace import TrainBFNWorkspace

__all__ = ["TrainStreamingFlowWorkspace"]


class TrainStreamingFlowWorkspace(TrainBFNWorkspace):
    pass
