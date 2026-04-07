import numpy as np
from .base import BaseKernel


class ExponentialKernel(BaseKernel):
    """
    One-sided exponential kernel for time as the state variable.

    More recent months receive higher weight, older months receive
    exponentially decaying weight. Nests Bryzgalova's rolling window
    as a special case (uniform within window, zero outside).

    Weight for a month j periods in the past:
        w_j = lambda^j * (1 - lambda) / (1 - lambda^m)   if j < m
        w_j = 0                                            if j >= m

    state_train and state_current are not used — ordering of rows
    determines recency. The kernel is purely time-based.

    Following Kim & Oh (2025), lambda is searched in [0.98, 0.9999].
    """

    # Candidate lambda values to search over
    default_lambdas = [0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999]

    def __init__(self, lam: float, m: int):
        if not (0 < lam < 1):
            raise ValueError(f"Lambda must be in (0, 1), got {lam}")
        if m <= 0:
            raise ValueError(f"Window length m must be positive, got {m}")
        self.lam = lam
        self.m   = m

    def weights(self, state_train: np.ndarray, state_current: float) -> np.ndarray:
        T = len(state_train)
        j = np.arange(T - 1, -1, -1)
        w = np.where(
            j < self.m,
            self.lam ** j * (1 - self.lam) / (1 - self.lam ** self.m),
            0.0
        )
        total = w.sum()
        if total == 0:
            return np.ones(T) / T
        return w / total

    @classmethod
    def bandwidth_grid(cls, m: int = 240, lambdas=None):
        """
        Return candidate lambda values.

        Parameters
        ----------
        m       : window length in months (default: full training window)
        lambdas : override default_lambdas if provided
        """
        return lambdas if lambdas is not None else cls.default_lambdas

    def __repr__(self):
        return f"ExponentialKernel(lam={self.lam}, m={self.m})"