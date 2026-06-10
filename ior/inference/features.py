from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .gdj import GoalDecompositionJudge, GoalSpec, ScoredStep
from ..ingest.trajectory import Trajectory


@dataclass
class FeatureMatrix:
    """Feature matrix for a trajectory, ready for BIRL.

    observed:         (T, d) -- GDJ scores for each observed (s, a, o) step.
    counterfactual:   (T, K, d) -- GDJ scores for K alternative actions at each
                      step, or None when fast mode is used.
    goal_spec:        the GoalSpec that defines the d feature dimensions.
    confidence:       (T, d) -- GDJ confidence scores for observed features.
    """

    observed: np.ndarray
    goal_spec: GoalSpec
    confidence: np.ndarray
    counterfactual: Optional[np.ndarray] = None


def build_feature_matrix(
    trajectory: Trajectory,
    goal_spec: GoalSpec,
    judge: GoalDecompositionJudge,
) -> FeatureMatrix:
    """Build a FeatureMatrix from a trajectory using the GDJ critic.

    Calls the GDJ once per step. Counterfactual features are not computed here;
    pass the result to BayesianIRL which handles fast-mode approximation.
    """
    scored: list[ScoredStep] = judge.score_trajectory(trajectory.steps, goal_spec)
    observed = np.array([s.scores for s in scored], dtype=float)       # (T, d)
    confidence = np.array([s.confidence for s in scored], dtype=float) # (T, d)
    return FeatureMatrix(observed=observed, goal_spec=goal_spec, confidence=confidence)
