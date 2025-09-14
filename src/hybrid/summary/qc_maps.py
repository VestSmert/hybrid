# --- file: hybrid/summary/qc_maps.py ---

from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter

__all__ = [
    "correlation_image",
    "pnr_image",
]


def _as_TYX(movie: np.ndarray, time_axis: int) -> np.ndarray:
    """
    Move the time axis to axis 0 if needed and return a contiguous float32 view.
    """
    if time_axis != 0:
        movie = np.moveaxis(movie, time_axis, 0)
    # Work in float32 for numerical stability / speed; do not mutate input
    return np.asarray(movie, dtype=np.float32, order="C")


def correlation_image(
    movie: np.ndarray,
    *,
    time_axis: int = 0,
    eps: float = 1e-8,
    fill_border: float | None = np.nan,
) -> np.ndarray:
    """
    Compute a *mean local temporal correlation* map.

    For each pixel, we z-score the temporal trace and compute the mean
    Pearson correlation with its 8-neighborhood (N, S, W, E, and diagonals).
    The final value is the average over the 8 correlations.

    Parameters
    ----------
    movie : np.ndarray
        3D movie of shape (T, H, W) by default (or any, if `time_axis` is given).
    time_axis : int, optional
        Which axis is time. Default is 0.
    eps : float, optional
        Small constant to avoid division by zero in the z-score denominator.
    fill_border : float | None, optional
        Value to fill the 1-pixel image border where a full 8-neighborhood
        is not available. If None, the border is left as-is (zeros).

    Returns
    -------
    np.ndarray
        Correlation image of shape (H, W), dtype float32. Border pixels are
        set to `fill_border` (NaN by default).

    Notes
    -----
    This implementation avoids recomputing means per neighbor by first
    z-scoring the movie along time and then using time-wise dot products.
    """
    x = _as_TYX(movie, time_axis)  # (T, H, W) float32
    T, H, W = x.shape

    if T < 2:
        raise ValueError("correlation_image requires at least two time points.")

    # Z-score along time: (x - mean) / std
    mu = x.mean(axis=0, dtype=np.float32)
    var = ((x - mu[None, ...]) ** 2).mean(axis=0, dtype=np.float32)
    std = np.sqrt(var + eps)
    z = (x - mu[None, ...]) / std[None, ...]  # (T, H, W), zero-mean unit-var

    # Helper to compute mean over time of elementwise product of z with a shifted version
    def mean_prod(shift_y: int, shift_x: int) -> np.ndarray:
        ys = slice(1, H - 1)
        xs = slice(1, W - 1)
        y2 = slice(1 + shift_y, H - 1 + shift_y)
        x2 = slice(1 + shift_x, W - 1 + shift_x)
        # (T, h, w)
        a = z[:, ys, xs]
        b = z[:, y2, x2]
        return (a * b).mean(axis=0, dtype=np.float32)

    # 8-neighborhood
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),            (0, 1),
               (1, -1),  (1, 0),   (1, 1)]

    acc = np.zeros((H - 2, W - 2), dtype=np.float32)
    for dy, dx in offsets:
        acc += mean_prod(dy, dx)

    C = np.empty((H, W), dtype=np.float32)
    if fill_border is None:
        C[:] = 0.0
    else:
        C[:] = fill_border
    C[1:H-1, 1:W-1] = acc / len(offsets)
    return C


def pnr_image(
    movie: np.ndarray,
    *,
    time_axis: int = 0,
    hp_window: int = 7,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute a Peak-to-Noise Ratio (PNR) map.

    Steps
    -----
    1) High-pass along time by subtracting a running mean (uniform window).
    2) Estimate per-pixel noise sigma using a robust MAD on the high-passed signal.
    3) Convert each timepoint to a z-score and take the temporal maximum.

    Parameters
    ----------
    movie : np.ndarray
        3D movie of shape (T, H, W) by default (or any, if `time_axis` is given).
    time_axis : int, optional
        Which axis is time. Default is 0.
    hp_window : int, optional
        Odd window length for the running mean (time domain). Must be >= 1.
        Typical values: 5– nine-ish (depends on frame rate / expected kinetics).
    eps : float, optional
        Small constant to avoid division by zero in sigma.

    Returns
    -------
    np.ndarray
        PNR image of shape (H, W), dtype float32.

    Notes
    -----
    - Uniform filtering along time is done with `mode='nearest'` to avoid edge
      transients. If memory becomes a concern, replace with a cumulative sum
      implementation of a rolling mean.
    """
    if hp_window < 1:
        raise ValueError("hp_window must be >= 1")
    if hp_window % 2 == 0:
        hp_window += 1  # make it odd

    x = _as_TYX(movie, time_axis)  # (T, H, W) float32
    # High-pass via running mean subtraction
    hp = x - uniform_filter(x, size=(hp_window, 1, 1), mode="nearest")

    # Robust per-pixel sigma via MAD
    med = np.median(hp, axis=0)
    mad = np.median(np.abs(hp - med[None, ...]), axis=0)
    sigma = 1.4826 * mad + eps

    # Z-score and take temporal maximum
    z = (hp - med[None, ...]) / sigma[None, ...]
    pnr = np.max(z, axis=0)
    return pnr.astype(np.float32, copy=False)
