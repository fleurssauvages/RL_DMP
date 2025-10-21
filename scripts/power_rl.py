#usr/bin/env python3
"""
A PoWER-style EM parameter update with importance sampling.
This implementation is kept clear and general:
 - It assumes policies are parametrized as flat vectors (numpy arrays)
 - Exploration is Gaussian in parameter space
 - Keeps history of past rollouts for importance sampling
"""
import numpy as np

class PowerRL:
    def __init__(self, init_params, exploration_std=0.1, reuse_top_n=6):
        """
        init_params: np.array (P,) initial parameter vector
        exploration_std: scalar or vector (P,) std dev for Gaussian exploration
        reuse_top_n: number of best rollouts to keep for importance sampling (Kormushev used 6)
        """
        self.theta = init_params.copy()
        if np.isscalar(exploration_std):
            self.expl_std = np.ones_like(self.theta) * exploration_std
        else:
            self.expl_std = np.asarray(exploration_std)
        self.history_params = []  # list of param vectors used in rollouts
        self.history_returns = []  # list of returns for each rollout
        self.reuse_top_n = reuse_top_n
        
    def update_exploration(self, new_std):
        """Update exploration std dev (can be scalar or vector)"""
        if np.isscalar(new_std):
            self.expl_std = np.ones_like(self.theta) * new_std
        else:
            self.expl_std = np.asarray(new_std)

    def sample_policy(self):
        """Sample an exploratory policy params theta_k ~ N(theta, diag(expl_std^2))"""
        noise = np.random.randn(*self.theta.shape) * self.expl_std
        return self.theta + noise

    def add_rollout(self, params, ret):
        self.history_params.append(params.copy())
        self.history_returns.append(float(ret))

    def update(self):
        """
        PoWER update as in eq (2):
        theta_{n+1} = theta_n + ( < (theta_k - theta_n) R(theta_k) >_w ) / ( < R(theta_k) >_w )
        where < . >_w means importance sampling over the top-N past rollouts.
        We'll select top-N rollouts by return and compute the update.
        """
        if len(self.history_returns) == 0:
            return self.theta  # nothing to update

        idx_sorted = np.argsort(self.history_returns)[::-1]
        top_idx = idx_sorted[:min(self.reuse_top_n, len(idx_sorted))]

        # Stack parameters and returns
        th_k = np.stack([self.history_params[i] for i in top_idx])     # shape (N, D)
        Rk = np.maximum(np.array(self.history_returns)[top_idx], 0.0)  # shape (N,)

        # Vectorized numerator and denominator
        numer = np.sum((th_k - self.theta) * Rk[:, None], axis=0)
        denom = np.sum(Rk)

        if denom > 0:
            self.theta = self.theta + numer / denom
        return self.theta

    def reset_history(self):
        self.history_params = []
        self.history_returns = []