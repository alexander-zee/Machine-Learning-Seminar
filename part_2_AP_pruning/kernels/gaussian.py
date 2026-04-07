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

    # Multipliers of sigma_S to search over, following Kim & Oh (2025)
    # who search h in [0.05*sigma_S, 5*sigma_S]
    default_multipliers = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

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
    def bandwidth_grid(cls, sigma_s: float, multipliers=None):
        """
        Return candidate h values as multiples of the state variable std.

        Parameters
        ----------
        sigma_s     : rolling std of the state variable over training window.
                      Computed in AP_Pruning from state.iloc[:n_train_valid].
        multipliers : override default_multipliers if provided.

        Returns
        -------
        list of h floats — one per bandwidth candidate.
        Enumerate this list in lasso_valid_full to get h_idx.
        """
        mults = multipliers if multipliers is not None else cls.default_multipliers
        return [c * sigma_s for c in mults]

    def __repr__(self):
        return f"GaussianKernel(h={self.h:.6f})"