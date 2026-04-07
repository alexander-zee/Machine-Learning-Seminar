import numpy as np
from .base import BaseKernel


class UniformKernel(BaseKernel):
    """
    Uniform (equal) weighting — reproduces the original baseline exactly.

    Every training month receives weight 1/T regardless of state.
    Equivalent to K(s - S_{t-1}; h) = 1/T for all t, which makes the
    kernel-weighted mu and sigma identical to the standard sample mean
    and covariance used by Bryzgalova et al. (2025).

    bandwidth_grid returns [None] — one run, no bandwidth to tune.
    state_train and state_current are accepted but ignored.
    """

    def weights(self, state_train: np.ndarray, state_current: float) -> np.ndarray:
        T = len(state_train)
        return np.ones(T) / T

    @classmethod
    def bandwidth_grid(cls):
        """One bandwidth only — uniform weighting has no h to tune."""
        return [None]

    def __repr__(self):
        return "UniformKernel()"