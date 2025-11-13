# -*- coding: utf-8 -*-
"""
Flat-field application (mask-aware, gain cap, GPU/CPU, tqdm progress).
"""

from __future__ import annotations
from typing import Literal, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

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


__all__ = ["apply_flatfield", "apply_from_model"]


def _robust_rescale(x: np.ndarray, p_low: float, p_high: float, mask: Optional[np.ndarray]) -> np.ndarray:
    """Map percentiles to [0,1] (CPU helper used for both paths)."""
    if mask is not None:
        valid = mask > 0.0
        vals = x[..., valid] if x.ndim == 3 else x[valid]
        if vals.size == 0:
            return x
        lo, hi = np.percentile(vals, [p_low, p_high])
    else:
        lo, hi = np.percentile(x, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(x)); hi = float(np.max(x)) if float(np.max(x)) > lo else lo + 1.0
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)


def apply_flatfield(
    movie: np.ndarray,
    flat: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,              # (H,W) soft/binary; applied after division
    gain_cap: Optional[float] = None,               # e.g., 1.7 to limit amplification
    renormalize: Literal["robust", "none"] = "none",
    p_low: float = 1.0,
    p_high: float = 99.0,
    device: Literal["auto", "cpu", "gpu"] = "auto",
    progress: bool = True,
    chunk: int = 64,                                 # CPU batching for big stacks
) -> np.ndarray:
    """
    Divide frames by flat-field; optional gain cap and post renormalization.

    Returns a float32 array with the same shape as input.
    """
    xp, use_gpu = _choose_backend(device)

    a = np.asarray(movie, dtype=np.float32)
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {a.shape}.")
    if flat.ndim != 2:
        raise ValueError(f"Flat-field must be 2D (H,W). Got shape {flat.shape}.")
    H, W = a.shape[1:]
    if flat.shape != (H, W):
        raise ValueError(f"Flat shape {flat.shape} != (H,W)=({H},{W}).")

    # ---- compute gain = 1/flat; optional cap ----
    eps = np.finfo(np.float32).eps
    gain = 1.0 / np.maximum(flat.astype(np.float32, copy=False), eps)
    if gain_cap is not None:
        gain = np.minimum(gain, float(gain_cap)).astype(np.float32, copy=False)

    m = None if mask is None else np.clip(mask.astype(np.float32, copy=False), 0.0, 1.0)

    if use_gpu:
        it = range(0, a.shape[0], chunk)
        if progress:
            it = tqdm(it, total=(a.shape[0] + chunk - 1) // chunk, desc="[flatfield apply]")
    
        out = np.empty_like(a, dtype=np.float32)
        gain_d = cp.asarray(gain)  # (H,W) on device
        m_d = None if m is None else cp.asarray(m)
    
        for i in it:
            sl = slice(i, min(i + chunk, a.shape[0]))         # (B,H,W)
            a_d = cp.asarray(a[sl])                            # host->device
            tmp = (a_d * gain_d[None, ...]).astype(cp.float32, copy=False)
            if m_d is not None:
                tmp *= m_d[None, ...]
            out[sl] = tmp.get().astype(np.float32, copy=False) # device->host
    
        a = out  # используем общий путь ниже для renormalize
    else:
        out = np.empty_like(a, dtype=np.float32)
        iterator = range(0, a.shape[0], chunk)
        if progress:
            iterator = tqdm(iterator, total=(a.shape[0] + chunk - 1) // chunk, desc="[flatfield apply]")
        for i in iterator:
            sl = slice(i, min(i + chunk, a.shape[0]))
            tmp = (a[sl] * gain[None, ...]).astype(np.float32, copy=False)
            if m is not None:
                tmp *= m[None, ...]
            out[sl] = tmp

    # ---- optional robust rescale to [0,1] (for visualization only) ----
    if renormalize == "robust":
        out = _robust_rescale(out, p_low, p_high, m)
    elif renormalize == "none":
        pass
    else:
        raise ValueError("renormalize must be one of {'robust','none'}")

    return out if movie.ndim == 3 else out[0]


def apply_from_model(
    movie: np.ndarray,
    model: dict,
    *,
    mask: Optional[np.ndarray] = None,
    gain_cap: Optional[float] = None,
    renormalize: Literal["robust", "none"] = "none",
    p_low: float = 1.0,
    p_high: float = 99.0,
    device: Literal["auto", "cpu", "gpu"] = "auto",
    progress: bool = True,
    chunk: int = 64,
) -> np.ndarray:
    """
    Convenience wrapper to apply using the dict returned by `estimate_flatfield`.
    Uses `model['flat_norm']` if present, else `model['flat']`.
    """
    flat = model.get("flat_norm", None)
    if flat is None:
        flat = model.get("flat", None)
    if flat is None:
        raise ValueError("Model dict must contain 'flat' or 'flat_norm'.")
    return apply_flatfield(
        movie, flat,
        mask=mask,
        gain_cap=gain_cap,
        renormalize=renormalize,
        p_low=p_low, p_high=p_high,
        device=device, progress=progress, chunk=chunk,
    )
