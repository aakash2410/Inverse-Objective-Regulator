from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ingest.trajectory import Trajectory
from ..inference.gdj import GoalSpec
from ..divergence.locator import DivergenceResult


@dataclass
class GymResult:
    """Result of running the IOR pipeline on a gym agent.

    planted_weights:   the known ground-truth objective weights (normalised).
    recovered:         the DivergenceResult from the full pipeline.
    recovery_correlation: Pearson r between planted and recovered divergence vectors.
    """

    agent_id: str
    planted_weights: np.ndarray
    recovered: DivergenceResult
    recovery_correlation: float


class GymHarness:
    """
    Reproducible evaluation harness for planted-divergence reference agents (R-ENV-01/02).

    Runs the full IOR pipeline on a gym agent trajectory and measures how well
    the recovered divergence correlates with the known planted divergence.
    A seed is stored in the Trajectory so results are deterministically replayable.
    """

    def evaluate(self, trajectory: Trajectory, planted_weights: np.ndarray) -> GymResult:
        raise NotImplementedError(
            "Gym harness evaluation requires a fully wired pipeline (Phase 0 milestone)."
        )
