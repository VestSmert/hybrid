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

__all__ = ["detrend", "dff"]
