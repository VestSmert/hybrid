# --- file: hybrid/preprocess/detrend.py ---
"""
Temporal baseline estimation & detrending.

Goal
----
Estimate a smooth per-pixel baseline along the time axis (frame dimension)
using a 1‑D Gaussian filter, then subtract it to obtain a high‑pass /
zero‑mean (approximately) movie for downstream analysis.

Typical usage
-------------
>>> from hybrid.preprocess.detrend import gaussian_baseline, detrend
>>> F0 = gaussian_baseline(movie, sigma_t=100.0)
>>> F, F0 = detrend(movie, sigma_t=100.0)

Notes
-----
- Works on arrays with shape (T, H, W). If you pass (H, W, T), set `time_axis=-1`.
- The Gaussian `sigma_t` is expressed in **frames**. Choose it larger than
  the timescale of neural transients, smaller than slow drifts.
- Computation is done in float32; integer inputs are converted.
- For very long movies, you can process in chunks to reduce peak memory.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter1d

__all__ = ["gaussian_baseline", "detrend"]


def _ensure_float32(x: np.ndarray) -> np.ndarray:
    """Return a float32 view/copy of `x`.

    Integer inputs are cast to float32. Float64 is downcast to float32 to
    keep memory usage predictable across functions.
    """
    if x.dtype == np.float32:
        return x
    return x.astype(np.float32, copy=False)


def gaussian_baseline(
    movie: np.ndarray,
    sigma_t: float = 100.0,
    time_axis: int = 0,
    mode: str = "reflect",
    chunk: Optional[int] = None,
) -> np.ndarray:
    """Estimate per-pixel baseline by temporal Gaussian smoothing.

    Parameters
    ----------
    movie : np.ndarray
        Movie array. By default expects shape (T, H, W). You can override the
        time dimension via `time_axis`.
    sigma_t : float
        Temporal Gaussian sigma **in frames** (not seconds). Increase to capture
        slower trends in the baseline.
    time_axis : int
        Axis index corresponding to time. Defaults to 0.
    mode : str
        Border handling for the Gaussian filter. See SciPy's `gaussian_filter1d`.
    chunk : Optional[int]
        If given and `movie` is 3‑D, process along time in blocks of `chunk` frames
        to reduce peak memory (useful for very large datasets). Overlaps are handled
        by padding inside `gaussian_filter1d`.

    Returns
    -------
    F0 : np.ndarray
        Baseline with the same shape as `movie`, dtype float32.
    """
    if movie.ndim < 1:
        raise ValueError("movie must be an array")

    x = _ensure_float32(movie)

    # Fast path: no chunking
    if chunk is None or x.ndim == 1:
        return gaussian_filter1d(x, sigma=sigma_t, axis=time_axis, mode=mode).astype(
            np.float32, copy=False
        )

    # Chunked path for (T, H, W) style data
    # We iterate along time axis and filter each chunk separately. For large sigma_t,
    # boundaries are handled by the filter's padding mode.
    F0 = np.empty_like(x, dtype=np.float32)

    # Reorder axes so time is first for simpler slicing
    x_tfirst = np.moveaxis(x, time_axis, 0)
    F0_tfirst = np.moveaxis(F0, time_axis, 0)
    T = x_tfirst.shape[0]

    for start in range(0, T, chunk):
        stop = min(T, start + chunk)
        F0_tfirst[start:stop] = gaussian_filter1d(
            x_tfirst[start:stop], sigma=sigma_t, axis=0, mode=mode
        ).astype(np.float32, copy=False)

    # Move axes back
    return np.moveaxis(F0_tfirst, 0, time_axis)


def detrend(
    movie: np.ndarray,
    sigma_t: float = 100.0,
    time_axis: int = 0,
    mode: str = "reflect",
    chunk: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract a smooth per-pixel baseline from the movie.

    Parameters
    ----------
    movie : np.ndarray
        Input stack (T, H, W) by default, or use `time_axis` to adapt.
    sigma_t : float
        Temporal Gaussian sigma in frames.
    time_axis : int
        Time axis index.
    mode : str
        Border handling for Gaussian.
    chunk : Optional[int]
        Process in blocks of `chunk` frames along time.

    Returns
    -------
    F, F0 : (np.ndarray, np.ndarray)
        Detrended movie and the baseline, both float32 and same shape as input.
    """
    F0 = gaussian_baseline(movie, sigma_t=sigma_t, time_axis=time_axis, mode=mode, chunk=chunk)
    F = _ensure_float32(movie) - F0
    return F.astype(np.float32, copy=False), F0
