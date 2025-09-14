"""
hybrid.summary
===================

Quality-control maps and lightweight visualizations for inspection.

Modules
-------
qc_maps : Correlation image and PNR (Peak-to-Noise Ratio) image generators.
viz     : Display-only transforms (CLAHE, gamma) and small preview panels.

Policy
------
- Summary products are for QC/visualization; they should not alter the data
  used for quantitative analysis.
"""

__all__ = ["qc_maps", "viz"]
