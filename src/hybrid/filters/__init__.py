"""
hybrid.filters
===================

Spatial filtering utilities used primarily to stabilize motion estimation
and to correct illumination.

Modules
-------
dog        : Difference-of-Gaussians (DoG) bandpass for registration prep.
flatfield  : Illumination field estimation and flat-field correction.
contrast   : Local contrast enhancement (e.g., CLAHE) for visualization/segmentation.

Design
------
- DoG is applied only on a working copy for shift estimation; we then apply
  the computed shifts to the raw (or flat-field corrected) data.
- Flat-field divides by a smooth illumination field estimated from a median
  frame; it should be done early in the pipeline.
- Contrast utilities (e.g., CLAHE) are applied at the very end to enhance
  local contrast of a 2D image prepared for downstream tools (Cellpose, QA).

Typical defaults
----------------
- DoG: sigma_low ≈ 1.2 px, sigma_high ≈ 12–20 px.
- Flat-field: gaussian sigma ≈ 40–80 px (depends on FOV).
- CLAHE: clip_limit ≈ 2.0–3.0, tile size ≈ 16×16.
"""

# Short imports for public API
from .dog import dog_bandpass
from .flatfield import estimate_flatfield, apply_flatfield
from .contrast import clahe_u8

# Modules export (dog, flatfield, contrast)
import importlib as _importlib
dog = _importlib.import_module(".dog", __name__)
flatfield = _importlib.import_module(".flatfield", __name__)
contrast = _importlib.import_module(".contrast", __name__)

__all__ = [
    # functions
    "dog_bandpass",
    "estimate_flatfield",
    "apply_flatfield",
    "clahe_u8",
    # modules
    "dog",
    "flatfield",
    "contrast",
]