from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .features import FeatureMatrix


@dataclass
class BIRLResult:
    """Output of one BIRL inference run.

    theta_map:      (d,) MAP estimate of reward weights.
    theta_samples:  (n_samples, d) Laplace posterior samples, satisfying R-INF-03's
                    requirement for a distribution over objectives, not a point estimate.
    posterior_cov:  (d, d) Laplace approximation to the posterior covariance.
    log_marginal:   Laplace approximation to the log marginal likelihood.
    """

    theta_map: np.ndarray
    theta_samples: np.ndarray
    posterior_cov: np.ndarray
    log_marginal: float


class BayesianIRL:
    """
    Bayesian IRL via Laplace approximation for black-box agent trajectories.

    Likelihood model (Boltzmann rationality):
        P(a_t | s_t, theta) = softmax(beta * Phi_t @ theta)[a_obs]
    where Phi_t is a (1 + K, d) matrix stacking the observed action features
    (row 0) against K counterfactual action features (rows 1..K).

    Prior:
        theta ~ N(0, prior_std^2 * I)

    Fast mode (counterfactual=None):
        Counterfactual features are approximated as 0.5 * ones(d), representing
        a neutral action with no preference over any sub-goal. This is cheap but
        slightly conservative: it underestimates divergence magnitude.

    Parameters
    ----------
    beta : float
        Inverse temperature. Higher values assume the agent is more rational.
    prior_std : float
        Standard deviation of the Gaussian prior over theta.
    n_counterfactuals : int
        Number of neutral counterfactuals to use in fast mode.
    """

    def __init__(
        self,
        beta: float = 1.0,
        prior_std: float = 1.0,
        n_counterfactuals: int = 5,
    ) -> None:
        self.beta = beta
        self.prior_std = prior_std
        self.n_counterfactuals = n_counterfactuals

    def fit(
        self,
        feature_matrix: FeatureMatrix,
        n_samples: int = 500,
        seed: Optional[int] = 0,
    ) -> BIRLResult:
        """Fit the BIRL model and return a posterior over reward weights."""
        observed = feature_matrix.observed  # (T, d)
        cf = feature_matrix.counterfactual  # (T, K, d) or None
        T, d = observed.shape

        if cf is None:
            K = self.n_counterfactuals
            cf = np.full((T, K, d), 0.5)

        def neg_log_post(theta: np.ndarray) -> float:
            return -(self._log_likelihood(theta, observed, cf) + self._log_prior(theta))

        def neg_log_post_grad(theta: np.ndarray) -> np.ndarray:
            return -(self._log_likelihood_grad(theta, observed, cf) + self._log_prior_grad(theta))

        result = minimize(
            neg_log_post,
            x0=np.zeros(d),
            jac=neg_log_post_grad,
            method="L-BFGS-B",
        )
        theta_map: np.ndarray = result.x

        hessian = self._neg_log_post_hessian(theta_map, observed, cf)
        try:
            c, low = cho_factor(hessian)
            post_cov = cho_solve((c, low), np.eye(d))
        except np.linalg.LinAlgError:
            post_cov = np.linalg.pinv(hessian)

        rng = np.random.default_rng(seed)
        samples = rng.multivariate_normal(theta_map, post_cov, size=n_samples)

        log_marginal = (
            -result.fun
            - 0.5 * float(np.linalg.slogdet(hessian)[1])
            + 0.5 * d * np.log(2 * np.pi)
        )

        return BIRLResult(
            theta_map=theta_map,
            theta_samples=samples,
            posterior_cov=post_cov,
            log_marginal=log_marginal,
        )

    def action_log_probs(self, feature_matrix: FeatureMatrix, theta: np.ndarray) -> np.ndarray:
        """Per-step log P(observed action | theta), used for behaviour prediction.

        Returns one log-probability per step under the Boltzmann model, against the
        same neutral counterfactuals fast mode uses when none are supplied.
        """
        observed = feature_matrix.observed
        cf = feature_matrix.counterfactual
        T, d = observed.shape
        if cf is None:
            cf = np.full((T, self.n_counterfactuals, d), 0.5)
        out = np.empty(T)
        for t in range(T):
            r_obs = self.beta * (observed[t] @ theta)
            r_cf = self.beta * (cf[t] @ theta)
            all_r = np.concatenate([[r_obs], r_cf])
            out[t] = r_obs - (np.max(all_r) + np.log(np.sum(np.exp(all_r - np.max(all_r)))))
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_likelihood(
        self,
        theta: np.ndarray,
        observed: np.ndarray,
        cf: np.ndarray,
    ) -> float:
        beta = self.beta
        log_lik = 0.0
        for t in range(observed.shape[0]):
            r_obs = beta * (observed[t] @ theta)
            r_cf = beta * (cf[t] @ theta)                      # (K,)
            all_r = np.concatenate([[r_obs], r_cf])
            log_lik += r_obs - (np.max(all_r) + np.log(np.sum(np.exp(all_r - np.max(all_r)))))
        return log_lik

    def _log_likelihood_grad(
        self,
        theta: np.ndarray,
        observed: np.ndarray,
        cf: np.ndarray,
    ) -> np.ndarray:
        beta = self.beta
        d = theta.shape[0]
        grad = np.zeros(d)
        for t in range(observed.shape[0]):
            r_obs = beta * (observed[t] @ theta)
            r_cf = beta * (cf[t] @ theta)
            all_r = np.concatenate([[r_obs], r_cf])
            w = np.exp(all_r - np.max(all_r))
            w /= w.sum()
            all_phi = np.vstack([observed[t], cf[t]])          # (1+K, d)
            expected_phi = w @ all_phi
            grad += beta * (observed[t] - expected_phi)
        return grad

    def _neg_log_post_hessian(
        self,
        theta: np.ndarray,
        observed: np.ndarray,
        cf: np.ndarray,
    ) -> np.ndarray:
        beta = self.beta
        d = theta.shape[0]
        H = np.eye(d) / self.prior_std ** 2
        for t in range(observed.shape[0]):
            r_obs = beta * (observed[t] @ theta)
            r_cf = beta * (cf[t] @ theta)
            all_r = np.concatenate([[r_obs], r_cf])
            w = np.exp(all_r - np.max(all_r))
            w /= w.sum()
            all_phi = np.vstack([observed[t], cf[t]])
            E_phi = w @ all_phi
            cov_phi = (all_phi - E_phi).T @ np.diag(w) @ (all_phi - E_phi)
            H += beta ** 2 * cov_phi
        return H

    def _log_prior(self, theta: np.ndarray) -> float:
        return float(-0.5 * np.sum(theta ** 2) / self.prior_std ** 2)

    def _log_prior_grad(self, theta: np.ndarray) -> np.ndarray:
        return -theta / self.prior_std ** 2
