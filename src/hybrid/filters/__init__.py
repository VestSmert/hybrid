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
from .dog import dog_bandpass          # DoG bandpass (2D/3D)  :contentReference[oaicite:4]{index=4}
from .flatfield import (                # Flat-field estimation & apply  :contentReference[oaicite:5]{index=5}
    estimate_flatfield,
    apply_flatfield,
)
from .contrast import clahe_u8         # CLAHE convenience wrapper  :contentReference[oaicite:6]{index=6}

__all__ = [
    "dog_bandpass",
    "estimate_flatfield",
    "apply_flatfield",
    "clahe_u8",
]
