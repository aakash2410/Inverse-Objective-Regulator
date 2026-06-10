import numpy as np
import pytest

from ior.inference.birl import BIRLResult
from ior.inference.gdj import GoalSpec
from ior.divergence.locator import DivergenceLocator, DivergenceResult


def _make_birl_result(theta: np.ndarray, noise: float = 0.05, seed: int = 0) -> BIRLResult:
    d = len(theta)
    rng = np.random.default_rng(seed)
    cov = np.eye(d) * noise
    samples = rng.multivariate_normal(theta, cov, size=200)
    return BIRLResult(
        theta_map=theta,
        theta_samples=samples,
        posterior_cov=cov,
        log_marginal=0.0,
    )


def _make_goal_spec(d: int) -> GoalSpec:
    return GoalSpec(sub_goals=[f"sub_goal_{i}" for i in range(d)])


def test_zero_divergence_when_inferred_equals_declared():
    d = 4
    theta = np.ones(d) / d
    result = BIRLResult(
        theta_map=theta,
        theta_samples=np.tile(theta, (100, 1)),
        posterior_cov=np.eye(d) * 1e-6,
        log_marginal=0.0,
    )
    locator = DivergenceLocator(declared_weights=theta)
    div = locator.compute(result, _make_goal_spec(d))
    assert div.scalar_divergence < 1e-6


def test_dimensions_sorted_by_magnitude():
    theta = np.array([0.9, 0.05, 0.03, 0.02])
    declared = np.array([0.25, 0.25, 0.25, 0.25])
    div = DivergenceLocator(declared_weights=declared).compute(
        _make_birl_result(theta), _make_goal_spec(4)
    )
    magnitudes = [d.magnitude for d in div.dimensions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_largest_divergence_matches_planted():
    """The dimension with the highest planted divergence should rank first."""
    theta = np.array([0.9, 0.05, 0.03, 0.02])   # sub-goal 0 is dominant
    declared = np.ones(4) / 4                     # equal weights
    div = DivergenceLocator(declared_weights=declared).compute(
        _make_birl_result(theta), _make_goal_spec(4)
    )
    assert div.dimensions[0].index == 0


def test_result_has_d_dimensions():
    d = 5
    theta = np.random.default_rng(0).random(d)
    div = DivergenceLocator().compute(_make_birl_result(theta), _make_goal_spec(d))
    assert len(div.dimensions) == d


def test_scalar_divergence_is_nonnegative():
    theta = np.array([0.6, 0.3, 0.1])
    div = DivergenceLocator().compute(_make_birl_result(theta), _make_goal_spec(3))
    assert div.scalar_divergence >= 0.0
