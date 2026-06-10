from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ingest.trajectory import Trajectory
from ..inference.birl import BayesianIRL
from ..inference.features import build_feature_matrix
from ..inference.gdj import GoalDecompositionJudge, GoalSpec
from ..divergence.locator import DivergenceLocator, DivergenceResult
from .agents.multi_agent_pair import JointTrajectory


def _to_simplex(v: np.ndarray) -> np.ndarray:
    v = v - v.min()
    s = v.sum()
    return v / s if s > 0 else np.ones(len(v)) / len(v)


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

    def __init__(
        self,
        judge: GoalDecompositionJudge,
        birl: BayesianIRL | None = None,
        locator: DivergenceLocator | None = None,
        n_goals: int = 5,
    ) -> None:
        self._judge = judge
        self._birl = birl or BayesianIRL()
        self._locator = locator or DivergenceLocator()
        self._n_goals = n_goals

    def evaluate(
        self,
        trajectory: Trajectory,
        planted_weights: np.ndarray,
        goal_spec: GoalSpec | None = None,
    ) -> GymResult:
        """Run the full pipeline on a planted-divergence trajectory.

        Pass goal_spec to pin the feature basis to known canonical sub-goals,
        isolating BIRL recovery from decomposer variability. Leave it None to
        let the GDJ decompose the declared purpose end-to-end.
        """
        if goal_spec is None:
            goal_spec = self._judge.decompose(trajectory.declared_purpose, n_goals=self._n_goals)
        feature_matrix = build_feature_matrix(trajectory, goal_spec, self._judge)
        birl_result = self._birl.fit(feature_matrix, seed=trajectory.seed or 0)
        divergence = self._locator.compute(birl_result, goal_spec)

        recovered_magnitudes = self._magnitudes_in_goal_order(divergence)
        correlation = self._recovery_correlation(planted_weights, recovered_magnitudes)

        return GymResult(
            agent_id=trajectory.agent_id,
            planted_weights=_to_simplex(planted_weights),
            recovered=divergence,
            recovery_correlation=correlation,
        )

    def evaluate_joint(self, joint: JointTrajectory, planted_weights: np.ndarray) -> GymResult:
        """Audit a cooperating agent pair as one system (R-ENV-04).

        The joint revealed objective is expected to diverge from either agent's
        individual declared purpose; recovery is measured against the planted
        joint divergence.
        """
        return self.evaluate(joint.interleaved, planted_weights)

    @staticmethod
    def _magnitudes_in_goal_order(divergence: DivergenceResult) -> np.ndarray:
        ordered = sorted(divergence.dimensions, key=lambda d: d.index)
        return np.array([d.magnitude for d in ordered], dtype=float)

    @staticmethod
    def _recovery_correlation(planted: np.ndarray, recovered: np.ndarray) -> float:
        """Pearson r between the planted divergence profile and the recovered one.

        The planted weights are mapped to a divergence profile by measuring each
        dimension's absolute deviation from a uniform declared objective, matching
        how DivergenceLocator computes magnitudes.
        """
        planted_simplex = _to_simplex(planted)
        d = len(planted_simplex)
        planted_divergence = np.abs(planted_simplex - np.ones(d) / d)
        if planted_divergence.std() == 0 or recovered.std() == 0:
            return 0.0
        return float(np.corrcoef(planted_divergence, recovered)[0, 1])
