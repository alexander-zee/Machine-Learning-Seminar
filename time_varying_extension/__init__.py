"""
Plug-in time-varying kernel extension (one triplet).

Inspired by the dual interpretation of ML forecasts (weights on historical
analogues): kernel weights over past months play the role of analogue weights.

This package does not modify the baseline AP-tree pipeline; import and call
``workflow_one_triplet.run_time_varying_one_triplet`` when outputs exist.
"""

from .workflow_one_triplet import run_time_varying_one_triplet

__all__ = ["run_time_varying_one_triplet"]
