"""
hybrid.summary
======================

Quality-control maps and lightweight visualizations for inspection.

Modules
-------
qc_maps : Correlation image and PNR (Peak-to-Noise Ratio) image generators.
viz     : Display-only transforms (CLAHE, gamma) and small preview panels.

Guidelines
----------
- Summary products are for QC/visualization; they should not alter the data
  used for quantitative analysis.
"""

# Re-exports for short imports like:
#   from hybrid.summary import correlation_image, pnr_image
#   from hybrid.summary import grid_before_after, show_qc_maps, montage
from .qc_maps import correlation_image, pnr_image
from .viz import grid_before_after, show_qc_maps, montage

import importlib as _importlib
qc_maps = _importlib.import_module(".qc_maps", __name__)
viz = _importlib.import_module(".viz", __name__)

__all__ = [
    # functions
    "correlation_image",
    "pnr_image",
    "grid_before_after",
    "show_qc_maps",
    "montage",
    # modules
    "qc_maps",
    "viz",
]
