from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .divergence.locator import DivergenceLocator
from .inference.birl import BayesianIRL
from .inference.features import FeatureMatrix


def split_feature_matrix(
    fm: FeatureMatrix, train_frac: float = 0.8, shuffle: bool = True, seed: int = 0
) -> tuple[FeatureMatrix, FeatureMatrix]:
    """Split a FeatureMatrix into train and test.

    The split is shuffled by default. Step-ordered splits are unsafe for the gym
    agents, whose trajectories front-load divergent steps and place honest steps
    last; an ordered split then trains on one behavioural phase and tests on the
    other, producing distribution shift rather than a clean held-out prediction
    measurement. Shuffling (seeded for reproducibility) mixes both phases into
    train and test, which is the intended behaviour for the G1 metric.
    """
    T = fm.observed.shape[0]
    cut = max(1, int(round(T * train_frac)))
    cut = min(cut, T - 1)

    order = np.arange(T)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    train_idx, test_idx = order[:cut], order[cut:]

    def _select(idx: np.ndarray) -> FeatureMatrix:
        return FeatureMatrix(
            observed=fm.observed[idx],
            goal_spec=fm.goal_spec,
            confidence=fm.confidence[idx],
            counterfactual=None if fm.counterfactual is None else fm.counterfactual[idx],
        )

    return _select(train_idx), _select(test_idx)


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


def _divergence_magnitudes(weights: np.ndarray, declared: np.ndarray) -> np.ndarray:
    p = DivergenceLocator._to_simplex(weights)
    q = DivergenceLocator._to_simplex(declared)
    return np.abs(p - q)


def mean_baseline_magnitudes(
    feature_matrix: FeatureMatrix, declared_weights: np.ndarray | None = None
) -> np.ndarray:
    """Divergence profile from a plain column-mean of the critic scores.

    This is the ablation the reviewer asks for: it skips Bayesian IRL entirely and
    treats the per-dimension average feature score as the revealed objective, then
    measures its divergence from the declared objective. If this recovers the same
    profile as the full BIRL posterior, the IRL machinery adds little beyond
    uncertainty quantification.
    """
    phi_bar = feature_matrix.observed.mean(axis=0)
    d = phi_bar.shape[0]
    declared = declared_weights if declared_weights is not None else np.ones(d) / d
    return _divergence_magnitudes(phi_bar, declared)


def birl_baseline_magnitudes(
    feature_matrix: FeatureMatrix,
    birl: BayesianIRL,
    declared_weights: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Divergence profile from the BIRL MAP estimate, for comparison with the mean."""
    result = birl.fit(feature_matrix, seed=seed)
    d = feature_matrix.observed.shape[1]
    declared = declared_weights if declared_weights is not None else np.ones(d) / d
    return _divergence_magnitudes(result.theta_map, declared)


def compare_birl_vs_mean(
    feature_matrix: FeatureMatrix,
    birl: BayesianIRL,
    declared_weights: np.ndarray | None = None,
    seed: int = 0,
) -> dict:
    """Quantify how much BIRL's divergence ranking departs from a column-mean.

    Returns the two profiles, their Spearman rank correlation, and whether they
    agree on the top divergence dimension. A high correlation is evidence that the
    IRL point estimate is largely reproducible by a simple mean under fast mode.
    """
    birl_mag = birl_baseline_magnitudes(feature_matrix, birl, declared_weights, seed)
    mean_mag = mean_baseline_magnitudes(feature_matrix, declared_weights)
    rho = float(spearmanr(birl_mag, mean_mag).statistic) if len(birl_mag) > 1 else 1.0
    return {
        "birl_magnitudes": birl_mag.tolist(),
        "mean_magnitudes": mean_mag.tolist(),
        "spearman": rho,
        "same_top_dimension": int(np.argmax(birl_mag)) == int(np.argmax(mean_mag)),
    }
