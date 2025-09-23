"""
hybrid.preprocess
======================

Temporal preprocessing steps that affect the signal used for analysis.

Modules
-------
detrend : Baseline estimation (e.g., temporal Gaussian) and trend removal.
dff     : ΔF/F0 normalization using either a provided baseline or a rolling
          percentile baseline.

Guidelines
----------
- Perform detrending after motion correction (and optional downsampling).
- Use ΔF/F0 for analysis-ready signals; keep raw/detrended stacks as well.
"""

# Re-exports for short imports like:
#   from hybrid.preprocess import gaussian_baseline, detrend
#   from hybrid.preprocess import dff_from_baseline, dff_percentile
from .detrend import gaussian_baseline, detrend
from .dff import dff_from_baseline, dff_percentile

import importlib as _importlib
detrend_mod = _importlib.import_module(".detrend", __name__)
dff = _importlib.import_module(".dff", __name__)

__all__ = [
    # functions
    "gaussian_baseline",
    "detrend",
    "dff_from_baseline",
    "dff_percentile",
    # modules
    "detrend_mod",
    "dff",
]