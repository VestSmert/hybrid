# -*- coding: utf-8 -*-
"""
Flat-field estimation (mask-aware, GPU/CPU, tqdm progress-friendly).

The estimator:
1) takes the temporal median frame,
2) applies a normalized blur inside a (possibly eroded) core mask,
3) floors tiny values and optionally normalizes the mean to 1.0.

Works with masked and non-masked data:
- If mask is None -> core is the entire frame.
- If mask is given (soft [0..1] or binary), we optionally erode it by `erode_px`
  (safe shrink) and use it as support for the normalized blur.
"""

from __future__ import annotations
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion

# ---- Optional GPU (CuPy) ----
HAVE_CUPY = False
try:
    import cupy as cp  # type: ignore
    HAVE_CUPY = True
except Exception:
    cp = None  # type: ignore


def _choose_backend(device: Literal["auto", "cpu", "gpu"]) -> Tuple[object, bool]:
    """Return (xp, use_gpu) where xp is numpy or cupy module."""
    if device == "gpu":
        if not (HAVE_CUPY and cp.cuda.runtime.getDeviceCount() > 0):  # type: ignore[attr-defined]
            raise RuntimeError("GPU requested but CuPy/CUDA is unavailable.")
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


__all__ = ["estimate_flatfield"]


def _to_numpy(a):
    return a.get() if HAVE_CUPY and isinstance(a, cp.ndarray) else a  # type: ignore


def estimate_flatfield(
    movie: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,   # (H,W) soft/binary or None
    blur_px: float = 60.0,               # typical 40–80 px
    erode_px: int = 0,                   # shrink mask support before smoothing
    floor_pct: float = 1.0,              # robust floor percentile inside core
    normalize: Literal["mean", "none"] = "mean",
    device: Literal["auto", "cpu", "gpu"] = "auto",
    progress: bool = False,              # kept for parity; single-shot op
) -> Dict[str, np.ndarray]:
    """
    Estimate a smooth illumination field from a stack or a frame.

    Parameters
    ----------
    movie : (T,Y,X) or (Y,X) float/uint array
    mask  : optional (H,W) in [0..1] or {0,1}. If None -> full frame.
    blur_px : Gaussian sigma for the normalized blur.
    erode_px : integer pixel erosion applied to the *binary* core mask.
    floor_pct : percentile floor within the core (avoids tiny divisors).
    normalize : 'mean' -> divide by mean so ⟨flat⟩≈1 ; 'none' -> raw flat.
    device : 'auto'|'cpu'|'gpu'
    progress : reserved (no loop here).

    Returns
    -------
    dict with keys:
      'flat'      : (H,W) float32 raw flat (before mean-normalization)
      'flat_norm' : (H,W) float32 (after mean-normalization if requested)
      'core_mask' : (H,W) float32 core used for smoothing (0..1)
    """
    xp, use_gpu = _choose_backend(device)

    a = np.asarray(movie, dtype=np.float32)
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {a.shape}.")

    T, H, W = a.shape

    # --- core mask (on CPU; simple & robust) ---
    if mask is None:
        core = np.ones((H, W), np.float32)
    else:
        core = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    # binarize + optional erosion
    core_bin = core > 0.5
    if erode_px > 0:
        core_bin = binary_erosion(core_bin, iterations=int(erode_px))
    core_mask = core_bin.astype(np.float32)

    # --- temporal median (CPU or GPU) ---
    if use_gpu:
        a_dev = cp.asarray(a)  # type: ignore
        med = cp.median(a_dev, axis=0).astype(cp.float32)  # (H,W)
        med_h = med.get()
    else:
        med_h = np.median(a, axis=0).astype(np.float32)

    # --- normalized blur inside the core (CPU math for stability) ---
    num = gaussian_filter(med_h * core_mask, sigma=float(blur_px))
    den = gaussian_filter(core_mask,       sigma=float(blur_px))
    den = np.maximum(den, 1e-6).astype(np.float32)
    flat = (num / den).astype(np.float32)
    # outside core — set to 1 to avoid division artifacts
    flat[core_mask == 0.0] = 1.0

    # --- robust floor inside the core ---
    core_vals = flat[core_mask > 0.0]
    if core_vals.size:
        floor = float(np.percentile(core_vals, floor_pct))
        if not np.isfinite(floor) or floor <= 0:
            floor = float(np.finfo(np.float32).eps)
        flat = np.maximum(flat, floor, dtype=np.float32)

    flat_norm = flat.astype(np.float32, copy=True)
    if normalize == "mean":
        m = float(np.mean(flat_norm[core_mask > 0.0])) if np.any(core_mask) else float(np.mean(flat_norm))
        if np.isfinite(m) and m > 0:
            flat_norm /= m

    return {
        "flat": flat.astype(np.float32, copy=False),
        "flat_norm": flat_norm.astype(np.float32, copy=False),
        "core_mask": core_mask.astype(np.float32, copy=False),
    }
