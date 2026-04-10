from abc import ABC, abstractmethod
import numpy as np


class BaseKernel(ABC):
    """
    Abstract base class for all kernels.

    Every kernel must implement weights(), which takes the state variable
    values for all training months and the current state value, and returns
    a normalized weight vector of shape (T_train,) summing to 1.

    Every kernel must also implement bandwidth_grid() as a classmethod,
    returning the list of candidate bandwidth values to search over.
    For UniformKernel this is [None] (one run, no bandwidth to tune).
    For GaussianKernel this is [c * sigma_s for c in multipliers].
    """

    @abstractmethod
    def weights(self, state_train: np.ndarray, state_current: float) -> np.ndarray:
        """
        Compute normalized kernel weights for all training months.

        Parameters
        ----------
        state_train   : (T_train,) state variable values for each training month
        state_current : scalar — the current state value S_{t*-1}

        Returns
        -------
        w : (T_train,) normalized weights summing to 1
        """
        pass

    @classmethod
    @abstractmethod
    def bandwidth_grid(cls, *args, **kwargs):
        """
        Return the list of candidate bandwidth values for this kernel.
        Called by AP_Pruning to construct the search grid.
        """
        pass

    @classmethod
    def bandwidth_grid_from_state(cls, state, n_train_valid: int):
        """
        Derive the bandwidth grid using the training portion of the state variable.

        This is the single consistent interface called by AP_Pruning — no
        if/else logic needed there.  Each kernel overrides this to extract
        whatever it needs from state (e.g. std for Gaussian, window length
        for Exponential).  UniformKernel uses the default below.

        Parameters
        ----------
        state         : pd.Series (T_total,) or None
        n_train_valid : int — number of training+validation months

        Returns
        -------
        list of bandwidth candidates (passed to lasso_valid_full as bandwidths)
        """
        # Default: no state needed — delegates to bandwidth_grid() with no args.
        # Correct for UniformKernel; override in kernels that need state.
        return cls.bandwidth_grid()