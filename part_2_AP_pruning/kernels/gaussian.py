import numpy as np
from .base import BaseKernel


class GaussianKernel(BaseKernel):
    """
    Gaussian kernel for continuous stochastic state variables (svar, DEF, TMS).
 
    Weight for training month t:
        w_t = exp(-(s_current - s_t)^2 / 2h^2)
    normalized to sum to 1.
 
    Following Kim & Oh (2025), bandwidth h is set as a multiple of the
    rolling std of the state variable over the training window:
        h = multiplier * sigma_S
 
    The multiplier is selected on the validation set by maximising the
    out-of-sample Sharpe ratio. The candidate multipliers are hardcoded
    in default_multipliers and used via bandwidth_grid(sigma_s).
    """
    # Range of multipliers to search over
    # who search h in [0.05*sigma_S, 5*sigma_S].
    # Use multiplier_grid(n) to get n log-evenly-spaced values in this range.
    multiplier_min = 0.05
    multiplier_max = 5.0
    default_n_multipliers = 5
 
    @classmethod
    def multiplier_grid(cls, n=None):
        """
        Return n log-evenly-spaced multipliers between multiplier_min and multiplier_max.
        Defaults to default_n_multipliers if n is not given.
        """
        n = n if n is not None else cls.default_n_multipliers
        return list(np.geomspace(cls.multiplier_min, cls.multiplier_max, n))

    def __init__(self, h: float):
        if h <= 0:
            raise ValueError(f"Bandwidth h must be positive, got {h}")
        self.h = h

    def weights(self, state_train: np.ndarray, state_current: float) -> np.ndarray:
        diff  = state_current - state_train
        w     = np.exp(-(diff ** 2) / (2 * self.h ** 2))
        total = w.sum()
        if total == 0:
            # Fallback to uniform if all weights collapse to zero
            return np.ones(len(state_train)) / len(state_train)
        return w / total

    @classmethod
    def bandwidth_grid(cls, sigma_s: float, n=None, multipliers=None):
        """
        Return candidate h values as multiples of the state variable std.
 
        Parameters
        ----------
        sigma_s     : std of the state variable over the training window.
        n           : number of bandwidth candidates. Defaults to
                      default_n_multipliers. Ignored if multipliers is given.
        multipliers : explicit list of multipliers — overrides n entirely.
 
        Returns
        -------
        list of h floats, one per bandwidth candidate.
        """
        mults = multipliers if multipliers is not None else cls.multiplier_grid(n)
        return [c * sigma_s for c in mults]

    @classmethod
    def bandwidth_grid_from_state(cls, state, n_train_valid: int, n=None):
        sigma_s = state.iloc[:n_train_valid].std()
        return cls.bandwidth_grid(sigma_s, n=n)

    def __repr__(self):
        return f"GaussianKernel(h={self.h:.6f})"