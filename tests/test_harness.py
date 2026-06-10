import numpy as np

from ior.gym.agents.cost_minimiser import CostMinimiserAgent, PLANTED_THETA_RAW
from ior.gym.harness import GymHarness, GymResult
from ior.inference.gdj import GoalSpec, ScoredStep


class _FakeJudge:
    """Offline GDJ stand-in. Scores divergent steps high on the planted dimension."""

    def __init__(self, planted_theta):
        self._planted = np.array(planted_theta)
        self._d = len(planted_theta)

    def decompose(self, declared_purpose, n_goals=5):
        return GoalSpec(sub_goals=[f"sub_goal_{i}" for i in range(self._d)])

    def score_trajectory(self, steps, goal_spec):
        scored = []
        weights = np.clip(self._planted - self._planted.min(), 0, None)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(self._d) / self._d
        for step in steps:
            redundant = step.observation.get("cached") is False and step.action.tool_name in (
                "web_search", "read_cache",
            )
            base = weights if redundant else np.ones(self._d) / self._d
            scored.append(
                ScoredStep(
                    scores=list(base),
                    confidence=[1.0] * self._d,
                    explanations=[""] * self._d,
                )
            )
        return scored


def test_harness_end_to_end_offline():
    trajectory = CostMinimiserAgent().run(seed=42)
    judge = _FakeJudge(PLANTED_THETA_RAW)
    harness = GymHarness(judge=judge, n_goals=len(PLANTED_THETA_RAW))

    result = harness.evaluate(trajectory, np.array(PLANTED_THETA_RAW))

    assert isinstance(result, GymResult)
    assert result.agent_id == "gym/cost_minimiser"
    assert len(result.recovered.dimensions) == len(PLANTED_THETA_RAW)
    assert -1.0 <= result.recovery_correlation <= 1.0
