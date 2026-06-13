from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..inference.birl import BIRLResult
from ..inference.gdj import GoalSpec


@dataclass
class DivergenceDimension:
    """Divergence along a single feature dimension (one sub-goal)."""

    index: int
    sub_goal: str
    declared_weight: float
    inferred_mean: float
    inferred_std: float
    magnitude: float           # |declared_weight - inferred_mean| on the simplex


@dataclass
class DivergenceResult:
    """Ranked divergence between inferred objective and declared purpose.

    dimensions is sorted by magnitude descending, satisfying R-DIV-01's
    requirement for ranked divergence dimensions with per-dimension scalar magnitude.
    """

    dimensions: list[DivergenceDimension]
    scalar_divergence: float   # total variation distance between the two distributions, in [0, 1]


class DivergenceLocator:
    """
    Computes divergence delta = R_hat vs declared purpose over the shared feature basis.

    Declared purpose is grounded into the feature basis as a weight vector theta_D.
    The default (R-INF-04 not yet implemented) treats all sub-goals as equally
    important: theta_D = uniform(d). R-INF-04 will replace this with NL grounding.

    Both theta_hat and theta_D are mapped to the probability simplex by a softmax
    before comparison. Softmax is used rather than min-subtraction so that no
    dimension is forced to exactly zero, which would inflate apparent divergence on
    the least-weighted sub-goal. The per-dimension magnitude is the absolute
    difference of the two distributions; the scalar divergence is the total
    variation distance (half the L1 distance), a standard distance between
    distributions that lies in [0, 1].
    """

    def __init__(self, declared_weights: Optional[np.ndarray] = None) -> None:
        self._declared_weights = declared_weights

    def compute(self, birl_result: BIRLResult, goal_spec: GoalSpec) -> DivergenceResult:
        d = len(goal_spec.sub_goals)
        theta_hat = birl_result.theta_map
        samples = birl_result.theta_samples

        declared = (
            self._declared_weights
            if self._declared_weights is not None
            else np.ones(d) / d
        )

        theta_norm = self._to_simplex(theta_hat)
        declared_norm = self._to_simplex(declared)

        magnitudes = np.abs(theta_norm - declared_norm)
        stds = samples.std(axis=0)

        dims = [
            DivergenceDimension(
                index=i,
                sub_goal=goal_spec.sub_goals[i],
                declared_weight=float(declared_norm[i]),
                inferred_mean=float(theta_norm[i]),
                inferred_std=float(stds[i]),
                magnitude=float(magnitudes[i]),
            )
            for i in range(d)
        ]
        dims.sort(key=lambda x: x.magnitude, reverse=True)

        return DivergenceResult(
            dimensions=dims,
            scalar_divergence=float(0.5 * np.sum(np.abs(theta_norm - declared_norm))),
        )

    @staticmethod
    def _to_simplex(v: np.ndarray) -> np.ndarray:
        """Map a real weight vector to the probability simplex via softmax."""
        z = v - np.max(v)
        e = np.exp(z)
        return e / e.sum()
