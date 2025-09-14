# --- file: hybrid/filters/flatfield.py ---
"""
Flat-field estimation & correction.

Goal
----
Estimate a smooth illumination field from the movie and divide frames by it
to remove vignetting / uneven illumination.

Typical usage
-------------
>>> from hybrid.preprocess.flatfield import estimate_flatfield, apply_flatfield
>>> flat = estimate_flatfield(movie, sigma_px=60.0)
>>> movie_ff = apply_flatfield(movie, flat, renormalize="robust")

Notes
-----
- Use this early in the pipeline (before motion correction).
- The estimator uses a median frame smoothed by a large Gaussian.
"""

from __future__ import annotations
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter

__all__ = ["estimate_flatfield", "apply_flatfield"]


def estimate_flatfield(
    movie: np.ndarray,
    sigma_px: float = 60.0,
    percentile_floor: float = 1.0,
    normalize_mean: bool = True,
) -> np.ndarray:
    """
    Estimate a smooth illumination field from the movie.

    Parameters
    ----------
    movie : np.ndarray
        Stack with shape (T, Y, X) or (Y, X). If 2D, treated as a single frame.
    sigma_px : float
        Gaussian sigma in pixels (large -> smoother field). Typical 40–80 px.
    percentile_floor : float
        Prevent division by very small values by clamping the field to at least
        the given percentile of the field (default 1.0).
    normalize_mean : bool
        If True, scale the field to have mean == 1.0 (recommended).

    Returns
    -------
    np.ndarray
        Float32 2D array (Y, X) representing the illumination field.
    """
    a = np.asarray(movie, dtype=np.float32)
    if a.ndim == 3:
        med = np.median(a, axis=0)  # (Y, X)
    elif a.ndim == 2:
        med = a
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {a.shape}.")

    flat = gaussian_filter(med, sigma=sigma_px).astype(np.float32, copy=False)

    # avoid zeros/very small values
    floor = np.percentile(flat, percentile_floor)
    if not np.isfinite(floor) or floor <= 0:
        floor = np.finfo(np.float32).eps
    flat = np.maximum(flat, float(floor))

    if normalize_mean:
        m = float(np.mean(flat))
        if m > 0 and np.isfinite(m):
            flat = flat / m

    return flat.astype(np.float32, copy=False)


def apply_flatfield(
    movie: np.ndarray,
    flat: np.ndarray,
    renormalize: Literal["robust", "none"] = "robust",
    p_low: float = 1.0,
    p_high: float = 99.5,
) -> np.ndarray:
    """
    Apply flat-field correction.

    Parameters
    ----------
    movie : np.ndarray
        Input stack (T, Y, X) or frame (Y, X).
    flat : np.ndarray
        Illumination field (Y, X), typically from `estimate_flatfield`.
    renormalize : {'robust','none'}
        After division, optionally rescale intensities.
        - 'robust' -> percentile [p_low, p_high] to [0,1] (good for viewing/MC).
        - 'none'   -> just return the divided data (keeps native scale).
    p_low, p_high : float
        Percentiles for robust rescale if renormalize='robust'.

    Returns
    -------
    np.ndarray
        Corrected array in float32; same shape as input.
    """
    a = np.asarray(movie, dtype=np.float32)
    if a.ndim == 2:
        a = a[None, ...]  # promote to (1, Y, X) for uniform handling

    if a.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {a.shape}.")

    if flat.ndim != 2:
        raise ValueError(f"Flat-field must be 2D (Y, X). Got shape {flat.shape}.")

    # apply division
    out = (a / flat[None, ...]).astype(np.float32, copy=False)

    if renormalize == "robust":
        lo, hi = np.percentile(out, [p_low, p_high])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(out))
            hi = float(np.max(out)) if float(np.max(out)) > lo else lo + 1.0
        out = np.clip((out - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)
    elif renormalize == "none":
        pass
    else:
        raise ValueError("renormalize must be one of {'robust','none'}")

    # return with original dimensionality
    return out if movie.ndim == 3 else out[0]
