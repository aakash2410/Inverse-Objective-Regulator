from __future__ import annotations

import numpy as np

from .inference.birl import BayesianIRL
from .inference.features import FeatureMatrix


def split_feature_matrix(
    fm: FeatureMatrix, train_frac: float = 0.8
) -> tuple[FeatureMatrix, FeatureMatrix]:
    """Split a FeatureMatrix into train and test by step order."""
    T = fm.observed.shape[0]
    cut = max(1, int(round(T * train_frac)))
    cut = min(cut, T - 1)

    def _slice(lo: int, hi: int) -> FeatureMatrix:
        return FeatureMatrix(
            observed=fm.observed[lo:hi],
            goal_spec=fm.goal_spec,
            confidence=fm.confidence[lo:hi],
            counterfactual=None if fm.counterfactual is None else fm.counterfactual[lo:hi],
        )

    return _slice(0, cut), _slice(cut, T)


def behaviour_prediction_lift(
    birl: BayesianIRL,
    train_fm: FeatureMatrix,
    test_fm: FeatureMatrix,
    seed: int = 0,
) -> float:
    """Relative log-loss reduction on held-out actions vs a uniform baseline (G1).

    Fits BIRL on train_fm, then measures how much better it predicts held-out
    actions than a uniform predictor over the (1 + K) action candidates. Positive
    values mean the inferred objective carries predictive signal. This metric is
    comparable across featurisers, since it only concerns action prediction, not
    semantic divergence.
    """
    result = birl.fit(train_fm, seed=seed)
    log_probs = birl.action_log_probs(test_fm, result.theta_map)
    model_loss = float(-log_probs.mean())

    K = (
        birl.n_counterfactuals
        if test_fm.counterfactual is None
        else test_fm.counterfactual.shape[1]
    )
    uniform_loss = float(np.log(1 + K))
    if uniform_loss == 0:
        return 0.0
    return (uniform_loss - model_loss) / uniform_loss
