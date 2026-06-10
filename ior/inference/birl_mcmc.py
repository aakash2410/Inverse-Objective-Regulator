from __future__ import annotations

import numpy as np

from .birl import BayesianIRL, BIRLResult
from .features import FeatureMatrix


class MCMCBayesianIRL(BayesianIRL):
    """
    Bayesian IRL via Metropolis-Hastings sampling.

    An alternative to the Laplace approximation in BayesianIRL, exposing the same
    fit() interface so the two can be compared on sparse, short trajectories (Q1).
    Where the Laplace approximation assumes a locally Gaussian posterior, MCMC
    samples the true (unnormalised) posterior and is more faithful when that
    assumption fails, at the cost of more compute.

    Parameters extend BayesianIRL with a random-walk proposal scale.
    """

    def __init__(
        self,
        beta: float = 1.0,
        prior_std: float = 1.0,
        n_counterfactuals: int = 5,
        proposal_std: float = 0.1,
        burn_in: int = 1000,
    ) -> None:
        super().__init__(beta=beta, prior_std=prior_std, n_counterfactuals=n_counterfactuals)
        self.proposal_std = proposal_std
        self.burn_in = burn_in

    def fit(
        self,
        feature_matrix: FeatureMatrix,
        n_samples: int = 500,
        seed: int = 0,
    ) -> BIRLResult:
        observed = feature_matrix.observed
        cf = feature_matrix.counterfactual
        T, d = observed.shape
        if cf is None:
            cf = np.full((T, self.n_counterfactuals, d), 0.5)

        def log_post(theta: np.ndarray) -> float:
            return self._log_likelihood(theta, observed, cf) + self._log_prior(theta)

        rng = np.random.default_rng(seed)
        theta = np.zeros(d)
        current_lp = log_post(theta)
        chain = np.empty((n_samples, d))
        accepted = 0

        total = self.burn_in + n_samples
        for i in range(total):
            proposal = theta + rng.normal(0.0, self.proposal_std, size=d)
            proposal_lp = log_post(proposal)
            if np.log(rng.random()) < proposal_lp - current_lp:
                theta = proposal
                current_lp = proposal_lp
                accepted += 1
            if i >= self.burn_in:
                chain[i - self.burn_in] = theta

        theta_mean = chain.mean(axis=0)
        post_cov = np.cov(chain, rowvar=False) if n_samples > 1 else np.eye(d) * 1e-6
        if post_cov.ndim == 0:
            post_cov = post_cov.reshape(1, 1)

        return BIRLResult(
            theta_map=theta_mean,
            theta_samples=chain,
            posterior_cov=np.atleast_2d(post_cov),
            log_marginal=float("nan"),
        )
