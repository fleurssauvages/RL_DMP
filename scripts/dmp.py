#usr/bin/env python3
"""
Mixture-of-PD primitives DMP-like controller.
Implements:
  x_ddot_hat = sum_i h_i(t) [ Kp_i (X_i - x) - V x_dot ]
Where:
 - Kp_i : (D x D) coordination matrices (full matrices allowed)
 - X_i  : (D,) attractor for primitive i
 - h_i(t) are time-dependent normalized Gaussian weights
"""
import numpy as np
from numba import njit

class MixturePD:
    def __init__(self, D, K, duration, kp_diag=50.0, vel_gain=2.0):
        """
        D: dimension of task-space (e.g. 3)
        K: number of primitives
        duration: rollout duration in seconds
        kp_diag: default stiffness for diagonal initialization
        vel_gain: damping gain (scalar) -> V = vel_gain * np.eye(D)
        """
        self.D = D
        self.K = K
        self.duration = duration
        self.V = vel_gain * np.eye(D)
        # Policy parameters
        # Kp_i list of (D,D) matrices
        self.Kp = [kp_diag * np.eye(D) for _ in range(K)]
        # Attractors X_i
        self.X = [np.zeros(D) for _ in range(K)]
        # Time-basis parameters (centers and widths)
        self.centers = np.linspace(0.0, duration, K)
        self.sigmas = np.full(K, (duration/(K*1.5))**2)

    def desired_acceleration(self, x, x_dot, t):
        """Compute desired acceleration (D,) at time t"""
        exps = np.exp(-0.5 * ((t - self.centers)**2) / (self.sigmas + 1e-12))
        denom = np.sum(exps) + 1e-12
        hs = exps / denom  # (K,)
        X = np.stack(self.X)         # (K, D)
        Kp = np.stack(self.Kp)       # (K, D, D)
        diff = X - x                 # (K, D)
        term = np.einsum('kij,kj->ki', Kp, diff) - self.V.dot(x_dot)  # (K, D)
        acc = np.tensordot(hs, term, axes=1)
        return acc     

    def get_flat_params(self):
        """
        Flatten Kp matrices and X vectors into 1D array for RL parameterization.
        Order: [Kp_0 (D*D), X_0 (D), Kp_1, X_1, ...]
        """
        parts = []
        for i in range(self.K):
            parts.append(self.Kp[i].reshape(-1))
            parts.append(self.X[i].reshape(-1))
        return np.concatenate(parts)

    def set_flat_params(self, flat):
        """Set Kp and X from flattened parameter vector."""
        idx = 0
        for i in range(self.K):
            n_kp = self.D * self.D
            self.Kp[i] = flat[idx:idx+n_kp].reshape(self.D, self.D)
            idx += n_kp
            n_x = self.D
            self.X[i] = flat[idx:idx+n_x].reshape(self.D,)
            idx += n_x

    def n_params(self):
        return self.K * (self.D*self.D + self.D)