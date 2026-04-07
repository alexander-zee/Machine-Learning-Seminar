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