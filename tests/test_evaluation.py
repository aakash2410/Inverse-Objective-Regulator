import numpy as np

from ior.evaluation import behaviour_prediction_lift, split_feature_matrix
from ior.inference.birl import BayesianIRL
from ior.inference.features import FeatureMatrix
from ior.inference.gdj import GoalSpec


def _make_feature_matrix(theta_true: np.ndarray, T: int = 50, seed: int = 0) -> FeatureMatrix:
    rng = np.random.default_rng(seed)
    d = len(theta_true)
    observed = np.clip(
        rng.random((T, d)) * 0.3 + np.outer(np.ones(T), np.clip(theta_true, 0, 1)) * 0.7,
        0.0, 1.0,
    )
    return FeatureMatrix(
        observed=observed,
        goal_spec=GoalSpec(sub_goals=[f"g{i}" for i in range(d)]),
        confidence=np.ones((T, d)),
    )


def test_split_sizes():
    fm = _make_feature_matrix(np.array([0.5, 0.5]), T=10)
    train, test = split_feature_matrix(fm, train_frac=0.8)
    assert train.observed.shape[0] == 8
    assert test.observed.shape[0] == 2


def test_lift_positive_when_behaviour_is_structured():
    """A trajectory consistent with a strong theta should beat the uniform baseline."""
    fm = _make_feature_matrix(np.array([0.9, 0.05, 0.05]), T=60, seed=5)
    train, test = split_feature_matrix(fm)
    lift = behaviour_prediction_lift(BayesianIRL(beta=2.0), train, test, seed=5)
    assert lift > 0.0


def test_lift_is_finite():
    fm = _make_feature_matrix(np.array([0.4, 0.3, 0.3]), T=40)
    train, test = split_feature_matrix(fm)
    lift = behaviour_prediction_lift(BayesianIRL(), train, test)
    assert np.isfinite(lift)
