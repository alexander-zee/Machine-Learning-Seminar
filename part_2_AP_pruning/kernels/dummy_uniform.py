import numpy as np
from .base import BaseKernel

class DummyUniformKernel(BaseKernel):
    def __init__(self, h=None):
        self.h = h

    def weights(self, state_train, state_current):
        return np.ones(len(state_train)) / len(state_train)

    @classmethod
    def bandwidth_grid(cls, sigma_s=None):
        """sigma_s is ignored; return a non-None list."""
        return [1.0]

    def __repr__(self):
        return "DummyUniformKernel()"