import numpy as np
import pytest

from ior.inference.birl import BayesianIRL, BIRLResult
from ior.inference.features import FeatureMatrix
from ior.inference.gdj import GoalSpec


def _make_feature_matrix(theta_true: np.ndarray, T: int = 30, seed: int = 0) -> FeatureMatrix:
    """Generate a synthetic FeatureMatrix consistent with theta_true."""
    rng = np.random.default_rng(seed)
    d = len(theta_true)
    observed = np.clip(
        rng.random((T, d)) * 0.3 + np.outer(np.ones(T), np.clip(theta_true, 0, 1)) * 0.7,
        0.0,
        1.0,
    )
    goal_spec = GoalSpec(sub_goals=[f"sub_goal_{i}" for i in range(d)])
    confidence = np.ones((T, d))
    return FeatureMatrix(observed=observed, goal_spec=goal_spec, confidence=confidence)


def test_birl_returns_result():
    fm = _make_feature_matrix(np.array([0.8, 0.1, 0.1]))
    result = BayesianIRL().fit(fm, n_samples=100, seed=0)
    assert isinstance(result, BIRLResult)


def test_birl_map_shape():
    d = 4
    fm = _make_feature_matrix(np.ones(d) / d)
    result = BayesianIRL().fit(fm)
    assert result.theta_map.shape == (d,)


def test_birl_samples_shape():
    fm = _make_feature_matrix(np.array([0.5, 0.3, 0.2]))
    result = BayesianIRL().fit(fm, n_samples=200)
    assert result.theta_samples.shape == (200, 3)


def test_birl_posterior_covariance_is_psd():
    fm = _make_feature_matrix(np.array([0.6, 0.4]))
    result = BayesianIRL().fit(fm)
    eigenvalues = np.linalg.eigvalsh(result.posterior_cov)
    assert np.all(eigenvalues >= -1e-8), "Posterior covariance must be positive semi-definite."


def test_birl_recovers_planted_direction():
    """MAP estimate should rank the dominant sub-goal above the weak sub-goal."""
    theta_true = np.array([0.9, 0.1, 0.0])
    fm = _make_feature_matrix(theta_true, T=50, seed=7)
    result = BayesianIRL(beta=2.0).fit(fm, seed=7)
    dominant_idx = int(np.argmax(result.theta_map))
    assert dominant_idx == 0, (
        f"Expected sub-goal 0 to dominate; got {dominant_idx}. "
        f"theta_map = {result.theta_map}"
    )


def test_birl_fast_mode_vs_provided_counterfactuals():
    """Fast mode and explicit neutral counterfactuals should agree closely."""
    fm = _make_feature_matrix(np.array([0.7, 0.2, 0.1]), T=20, seed=1)
    result_fast = BayesianIRL().fit(fm, seed=0)

    K = 5
    cf = np.full((20, K, 3), 0.5)
    fm_cf = FeatureMatrix(
        observed=fm.observed, goal_spec=fm.goal_spec, confidence=fm.confidence, counterfactual=cf
    )
    result_cf = BayesianIRL().fit(fm_cf, seed=0)

    np.testing.assert_allclose(result_fast.theta_map, result_cf.theta_map, atol=1e-8)
