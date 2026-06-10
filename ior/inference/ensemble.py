from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ingest.trajectory import Step
from .gdj import GoalDecompositionJudge, GoalSpec, ScoredStep


@dataclass
class AgreementReport:
    """Inter-judge reliability for an ensemble (a quantifiable robustness metric).

    mean_pairwise_r:    mean Pearson correlation between judges' flattened score
                        matrices. High values mean the semantic feature scores do
                        not hinge on a single model's idiosyncrasies.
    per_dimension_std:  standard deviation across judges, averaged over steps, for
                        each sub-goal dimension.
    """

    mean_pairwise_r: float
    per_dimension_std: list[float]


class EnsembleJudge:
    """
    Runs several judges and averages their per-step feature scores (Featuriser protocol).

    Averaging diverse models reduces single-model bias whilst keeping the semantic
    feature basis. The agreement report quantifies how much the result depends on
    any individual model, directly addressing the external-dependency concern.

    Construct with judge objects directly (for testing) or via from_models().
    """

    def __init__(self, judges: list) -> None:
        if not judges:
            raise ValueError("EnsembleJudge requires at least one judge.")
        self._judges = judges

    @classmethod
    def from_models(cls, models: list[str]) -> "EnsembleJudge":
        return cls([GoalDecompositionJudge(model=m) for m in models])

    def decompose(self, declared_purpose: str, n_goals: int = 5) -> GoalSpec:
        """Use the first judge's decomposition as the canonical, shared basis."""
        return self._judges[0].decompose(declared_purpose, n_goals=n_goals)

    def score_trajectory(self, steps: list[Step], goal_spec: GoalSpec) -> list[ScoredStep]:
        per_judge = self._per_judge_scores(steps, goal_spec)
        stack = np.stack([self._to_array(pj) for pj in per_judge])  # (J, T, d)
        mean_scores = stack.mean(axis=0)                            # (T, d)
        conf_stack = np.stack(
            [np.array([s.confidence for s in pj]) for pj in per_judge]
        )
        mean_conf = conf_stack.mean(axis=0)
        n = len(self._judges)
        return [
            ScoredStep(
                scores=list(mean_scores[t]),
                confidence=list(mean_conf[t]),
                explanations=[f"ensemble mean of {n} judges"] * mean_scores.shape[1],
            )
            for t in range(mean_scores.shape[0])
        ]

    def agreement(self, steps: list[Step], goal_spec: GoalSpec) -> AgreementReport:
        per_judge = self._per_judge_scores(steps, goal_spec)
        mats = [self._to_array(pj) for pj in per_judge]            # each (T, d)
        flats = [m.flatten() for m in mats]

        corrs = []
        for i in range(len(flats)):
            for k in range(i + 1, len(flats)):
                if flats[i].std() > 0 and flats[k].std() > 0:
                    corrs.append(float(np.corrcoef(flats[i], flats[k])[0, 1]))
        mean_pairwise_r = float(np.mean(corrs)) if corrs else 1.0

        stack = np.stack(mats)                                     # (J, T, d)
        per_dimension_std = stack.std(axis=0).mean(axis=0)         # (d,)
        return AgreementReport(
            mean_pairwise_r=mean_pairwise_r,
            per_dimension_std=per_dimension_std.tolist(),
        )

    def _per_judge_scores(self, steps, goal_spec) -> list[list[ScoredStep]]:
        return [j.score_trajectory(steps, goal_spec) for j in self._judges]

    @staticmethod
    def _to_array(scored: list[ScoredStep]) -> np.ndarray:
        return np.array([s.scores for s in scored], dtype=float)
