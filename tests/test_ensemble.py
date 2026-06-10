import numpy as np
import pytest

from ior.ingest.trajectory import Action, Step
from ior.inference.ensemble import EnsembleJudge
from ior.inference.gdj import GoalSpec, ScoredStep


class _FakeJudge:
    """Returns fixed per-step scores, optionally offset, for offline testing."""

    def __init__(self, base: list[float], offset: float = 0.0):
        self._base = np.array(base)
        self._offset = offset

    def decompose(self, declared_purpose, n_goals=5):
        return GoalSpec(sub_goals=[f"g{i}" for i in range(len(self._base))])

    def score_trajectory(self, steps, goal_spec):
        vals = list(np.clip(self._base + self._offset, 0, 1))
        return [
            ScoredStep(scores=vals, confidence=[1.0] * len(vals), explanations=[""] * len(vals))
            for _ in steps
        ]


def _steps(n=4):
    return [
        Step(state={}, action=Action(tool_name="t", parameters={"i": k}), observation={"o": k})
        for k in range(n)
    ]


def test_requires_at_least_one_judge():
    with pytest.raises(ValueError):
        EnsembleJudge([])


def test_averages_scores():
    j1 = _FakeJudge([0.2, 0.8])
    j2 = _FakeJudge([0.4, 0.6])
    ens = EnsembleJudge([j1, j2])
    gs = ens.decompose("x")
    scored = ens.score_trajectory(_steps(), gs)
    np.testing.assert_allclose(scored[0].scores, [0.3, 0.7])


def test_identical_judges_agree_perfectly():
    ens = EnsembleJudge([_FakeJudge([0.1, 0.9, 0.5]), _FakeJudge([0.1, 0.9, 0.5])])
    gs = ens.decompose("x")
    report = ens.agreement(_steps(), gs)
    assert report.mean_pairwise_r == pytest.approx(1.0)
    assert all(s == pytest.approx(0.0) for s in report.per_dimension_std)


def test_divergent_judges_lower_agreement():
    # One judge ranks dims ascending, the other descending: anti-correlated.
    ens = EnsembleJudge([_FakeJudge([0.1, 0.5, 0.9]), _FakeJudge([0.9, 0.5, 0.1])])
    gs = ens.decompose("x")
    report = ens.agreement(_steps(), gs)
    assert report.mean_pairwise_r < 0.0
