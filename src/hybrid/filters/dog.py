# --- file: hybrid/filters/dog.py ---
"""
Difference-of-Gaussians (DoG) bandpass filter for registration prep.

- Applies a spatial bandpass:  DoG = G(sigma_low) - G(sigma_high),  with sigma_high > sigma_low
- Supports 2D frames (Y, X) and 3D stacks (T, Y, X).
- For 3D input, filtering is done per-frame (no temporal blur) using sigma=(0, σ, σ).

Typical usage
-------------
>>> from hybrid.filters.dog import dog_bandpass
>>> dog = dog_bandpass(movie, sigma_low=1.2, sigma_high=14.0)

Notes
-----
- Output is float32 and typically zero-centered (mean ~ 0).
- Use this on a working copy for **shift estimation** (motion correction),
  then apply the estimated shifts to the raw/flat-fielded data.
"""

from __future__ import annotations
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter

__all__ = ["dog_bandpass"]


def dog_bandpass(
    arr: np.ndarray,
    sigma_low: float = 1.2,
    sigma_high: float = 14.0,
    mode: Literal["nearest", "reflect", "mirror", "wrap", "constant"] = "nearest",
    zero_center: bool = True,
) -> np.ndarray:
    """
    Apply spatial Difference-of-Gaussians bandpass.

    Parameters
    ----------
    arr : np.ndarray
        Input image or stack. Shape (Y, X) or (T, Y, X).
    sigma_low : float
        Small (narrow) Gaussian sigma in pixels. Must be > 0 and < sigma_high.
    sigma_high : float
        Large (broad) Gaussian sigma in pixels. Must be > sigma_low.
    mode : {'nearest','reflect','mirror','wrap','constant'}
        Boundary handling for gaussian_filter.
    zero_center : bool
        If True, subtract per-frame mean after DoG to stabilize the zero level
        (helps phase correlation on some datasets).

    Returns
    -------
    np.ndarray
        Filtered array (float32) with the same shape as `arr`.

    Raises
    ------
    ValueError
        If input dimensionality is not 2D or 3D, or if sigmas are invalid.
    """
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}.")

    if not (sigma_low > 0 and sigma_high > 0 and sigma_high > sigma_low):
        raise ValueError(
            f"Invalid sigmas: require 0 < sigma_low < sigma_high, got "
            f"sigma_low={sigma_low}, sigma_high={sigma_high}"
        )

    a = np.asarray(arr, dtype=np.float32, order="C")

    if a.ndim == 2:
        g1 = gaussian_filter(a, sigma=sigma_low, mode=mode)
        g2 = gaussian_filter(a, sigma=sigma_high, mode=mode)
        dog = g1 - g2
        if zero_center:
            dog -= float(dog.mean())
        return dog.astype(np.float32, copy=False)

    # a.ndim == 3  -> per-frame spatial filtering without temporal blur
    # Use sigma=(0, σ, σ) to avoid smoothing across time axis.
    g1 = gaussian_filter(a, sigma=(0.0, sigma_low, sigma_low), mode=mode)
    g2 = gaussian_filter(a, sigma=(0.0, sigma_high, sigma_high), mode=mode)
    dog = g1 - g2
    if zero_center:
        # subtract mean per-frame to keep zero-centered statistics
        means = dog.reshape(dog.shape[0], -1).mean(axis=1).astype(np.float32)
        dog -= means[:, None, None]
    return dog.astype(np.float32, copy=False)
