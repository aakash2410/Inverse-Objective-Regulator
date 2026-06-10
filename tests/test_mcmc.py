import numpy as np

from ior.inference.birl_mcmc import MCMCBayesianIRL
from ior.inference.features import FeatureMatrix
from ior.inference.gdj import GoalSpec


def _make_feature_matrix(theta_true: np.ndarray, T: int = 40, seed: int = 0) -> FeatureMatrix:
    rng = np.random.default_rng(seed)
    d = len(theta_true)
    observed = np.clip(
        rng.random((T, d)) * 0.3 + np.outer(np.ones(T), np.clip(theta_true, 0, 1)) * 0.7,
        0.0, 1.0,
    )
    goal_spec = GoalSpec(sub_goals=[f"sub_goal_{i}" for i in range(d)])
    return FeatureMatrix(observed=observed, goal_spec=goal_spec, confidence=np.ones((T, d)))


def test_mcmc_returns_samples():
    fm = _make_feature_matrix(np.array([0.8, 0.1, 0.1]))
    result = MCMCBayesianIRL(burn_in=200).fit(fm, n_samples=300, seed=0)
    assert result.theta_samples.shape == (300, 3)


def test_mcmc_recovers_planted_direction():
    theta_true = np.array([0.9, 0.05, 0.05])
    fm = _make_feature_matrix(theta_true, T=60, seed=3)
    result = MCMCBayesianIRL(beta=2.0, burn_in=500).fit(fm, n_samples=500, seed=3)
    assert int(np.argmax(result.theta_map)) == 0


def test_mcmc_covariance_shape():
    fm = _make_feature_matrix(np.array([0.6, 0.4]))
    result = MCMCBayesianIRL(burn_in=100).fit(fm, n_samples=200, seed=0)
    assert result.posterior_cov.shape == (2, 2)
