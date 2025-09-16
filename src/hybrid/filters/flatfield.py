# --- file: hybrid/filters/flatfield.py ---
"""
Flat-field estimation & correction.

Goal
----
Estimate a smooth illumination field from the movie and divide frames by it
to remove vignetting / uneven illumination.

Typical usage
-------------
>>> from hybrid.filters import estimate_flatfield, apply_flatfield
>>> flat = estimate_flatfield(movie, sigma_px=60.0, device="auto")
>>> movie_ff = apply_flatfield(movie, flat, renormalize="robust", device="auto")

Notes
-----
- Use this early in the pipeline (before motion correction).
- The estimator uses a median frame smoothed by a large Gaussian.
"""

from __future__ import annotations
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter as np_gaussian_filter

# --- Optional GPU stack (CuPy) in the same spirit as pwrigid/dog ---
HAVE_CUPY = False
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
    HAVE_CUPY = True
except Exception:  # CuPy is optional
    cp = None  # type: ignore
    cp_gaussian_filter = None  # type: ignore

__all__ = ["estimate_flatfield", "apply_flatfield"]


def _choose_backend(device: Literal["auto", "cpu", "gpu"]):
    """Return (xp, use_gpu: bool) where xp is np or cp."""
    if device == "gpu":
        if not (HAVE_CUPY and cp.cuda.runtime.getDeviceCount() > 0):  # type: ignore[attr-defined]
            raise RuntimeError("GPU requested but CuPy/CUDA device is not available.")
        print("[GPU] Using CuPy backend")
        return cp, True  # type: ignore[return-value]
    if device == "auto" and HAVE_CUPY:
        try:
            if cp.cuda.runtime.getDeviceCount() > 0:  # type: ignore[attr-defined]
                print("[GPU-auto] CuPy available → using GPU backend")
                return cp, True  # type: ignore[return-value]
        except Exception:
            pass
    return np, False


def estimate_flatfield(
    movie: np.ndarray,
    sigma_px: float = 60.0,
    percentile_floor: float = 1.0,
    normalize_mean: bool = True,
    *,
    device: Literal["auto", "cpu", "gpu"] = "auto",
    progress: bool = False,  # kept for parity; estimation is single-shot
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
        Clamp the field to at least the given percentile to avoid tiny values.
    normalize_mean : bool
        If True, scale the field to have mean == 1.0.
    device : {'auto','cpu','gpu'}
        Backend selection (CuPy on GPU when available).
    progress : bool
        Present for API symmetry (no per-frame loop here).

    Returns
    -------
    np.ndarray
        Float32 2D array (Y, X) representing the illumination field.
    """
    xp, use_gpu = _choose_backend(device)

    a = np.asarray(movie, dtype=np.float32)  # keep host array for shape logic
    if a.ndim == 3:
        if use_gpu:
            a_dev = cp.asarray(a, dtype=cp.float32)  # type: ignore[name-defined]
            med = cp.median(a_dev, axis=0)
            flat = cp_gaussian_filter(med, sigma=sigma_px)
            # floor & normalization on device
            floor = cp.percentile(flat, percentile_floor)
            floor = cp.maximum(floor, cp.asarray(np.finfo(np.float32).eps, dtype=cp.float32))
            flat = cp.maximum(flat, floor)
            if normalize_mean:
                m = cp.mean(flat)
                flat = flat / cp.clip(m, 1e-8, cp.inf)
            flat = cp.asnumpy(flat).astype(np.float32, copy=False)
        else:
            med = np.median(a, axis=0)  # (Y, X)
            flat = np_gaussian_filter(med, sigma=sigma_px).astype(np.float32, copy=False)
            # floor & normalization on host
            floor = np.percentile(flat, percentile_floor)
            if not np.isfinite(floor) or floor <= 0:
                floor = np.finfo(np.float32).eps
            flat = np.maximum(flat, float(floor))
            if normalize_mean:
                m = float(np.mean(flat))
                if m > 0 and np.isfinite(m):
                    flat = flat / m
    elif a.ndim == 2:
        if use_gpu:
            fr = cp.asarray(a, dtype=cp.float32)  # type: ignore[name-defined]
            flat = cp_gaussian_filter(fr, sigma=sigma_px)
            floor = cp.percentile(flat, percentile_floor)
            floor = cp.maximum(floor, cp.asarray(np.finfo(np.float32).eps, dtype=cp.float32))
            flat = cp.maximum(flat, floor)
            if normalize_mean:
                m = cp.mean(flat)
                flat = flat / cp.clip(m, 1e-8, cp.inf)
            flat = cp.asnumpy(flat).astype(np.float32, copy=False)
        else:
            flat = np_gaussian_filter(a, sigma=sigma_px).astype(np.float32, copy=False)
            floor = np.percentile(flat, percentile_floor)
            if not np.isfinite(floor) or floor <= 0:
                floor = np.finfo(np.float32).eps
            flat = np.maximum(flat, float(floor))
            if normalize_mean:
                m = float(np.mean(flat))
                if m > 0 and np.isfinite(m):
                    flat = flat / m
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {a.shape}.")

    return flat.astype(np.float32, copy=False)


def apply_flatfield(
    movie: np.ndarray,
    flat: np.ndarray,
    renormalize: Literal["robust", "none"] = "robust",
    p_low: float = 1.0,
    p_high: float = 99.5,
    *,
    device: Literal["auto", "cpu", "gpu"] = "auto",
    progress: bool = False,  # vectorized; bar обычно не нужен
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
    device : {'auto','cpu','gpu'}
        Backend selection (CuPy on GPU when available).
    progress : bool
        Present for API symmetry (операции векторизованы).

    Returns
    -------
    np.ndarray
        Corrected array in float32; same shape as input.
    """
    xp, use_gpu = _choose_backend(device)

    a = np.asarray(movie, dtype=np.float32)
    if a.ndim == 2:
        a = a[None, ...]  # promote to (1, Y, X)
    if a.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {a.shape}.")
    if flat.ndim != 2:
        raise ValueError(f"Flat-field must be 2D (Y, X). Got shape {flat.shape}.")

    if use_gpu:
        a_dev = cp.asarray(a, dtype=cp.float32)            # type: ignore[name-defined]
        flat_dev = cp.asarray(flat, dtype=cp.float32)      # type: ignore[name-defined]
        out = (a_dev / flat_dev[None, ...]).astype(cp.float32, copy=False)

        if renormalize == "robust":
            lo = cp.percentile(out, p_low)
            hi = cp.percentile(out, p_high)
            hi = cp.maximum(hi, lo + 1e-6)
            out = cp.clip((out - lo) / (hi - lo), 0.0, 1.0)
        elif renormalize == "none":
            pass
        else:
            raise ValueError("renormalize must be one of {'robust','none'}")

        out_host = cp.asnumpy(out).astype(np.float32, copy=False)
        return out_host if movie.ndim == 3 else out_host[0]

    # --- CPU path (NumPy) ---
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

    return out if movie.ndim == 3 else out[0]
