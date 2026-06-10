from __future__ import annotations

from typing import Protocol

from ..ingest.trajectory import Step
from .gdj import GoalSpec, ScoredStep


class Featuriser(Protocol):
    """
    Interface shared by every feature-extraction strategy.

    A featuriser maps a declared purpose to a feature basis (decompose) and scores
    each trajectory step against that basis (score_trajectory). The GoalDecompositionJudge
    is the semantic LLM implementation; StructuralFeaturiser is a deterministic,
    model-free implementation; EnsembleJudge averages several judges.

    The harness and feature builder depend only on this protocol, so featurisers
    are interchangeable.
    """

    def decompose(self, declared_purpose: str, n_goals: int = 5) -> GoalSpec: ...

    def score_trajectory(self, steps: list[Step], goal_spec: GoalSpec) -> list[ScoredStep]: ...
