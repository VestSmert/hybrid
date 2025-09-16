"""
Difference-of-Gaussians (DoG) bandpass filter for registration prep.

- Applies a spatial bandpass:  DoG = G(sigma_low) - G(sigma_high),  with sigma_high > sigma_low
- Supports 2D frames (Y, X) and 3D stacks (T, Y, X).
- For 3D input, filtering is done per-frame (no temporal blur) using sigma=(0, σ, σ).

Typical usage
-------------
>>> from hybrid.filters import dog_bandpass
>>> dog = dog_bandpass(movie, sigma_low=1.2, sigma_high=14.0, device="auto", progress=True)

Notes
-----
- Output is float32 and typically zero-centered (mean ~ 0) if `zero_center=True`.
- Use this on a working copy for **shift estimation** (motion correction),
  then apply the estimated shifts to the raw/flat-fielded data.
"""

from __future__ import annotations
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter as np_gaussian_filter

# --- Optional GPU stack (CuPy) like pwrigid.py ---
HAVE_CUPY = False
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
    HAVE_CUPY = True
except Exception:  # CuPy is optional
    cp = None  # type: ignore
    cp_gaussian_filter = None  # type: ignore

__all__ = ["dog_bandpass"]


def _dog2d(a2: np.ndarray, s_low: float, s_high: float, mode: str, xp=np):
    """DoG for a single 2D frame on CPU (NumPy) or GPU (CuPy)."""
    if xp is np:
        g1 = np_gaussian_filter(a2, sigma=s_low, mode=mode)
        g2 = np_gaussian_filter(a2, sigma=s_high, mode=mode)
        return (g1 - g2).astype(np.float32, copy=False)
    # GPU
    fr = cp.asarray(a2, dtype=cp.float32)
    g1 = cp_gaussian_filter(fr, sigma=s_low, mode=mode)
    g2 = cp_gaussian_filter(fr, sigma=s_high, mode=mode)
    return (g1 - g2).astype(cp.float32, copy=False)


def dog_bandpass(
    arr: np.ndarray,
    sigma_low: float = 1.2,
    sigma_high: float = 14.0,
    mode: Literal["nearest", "reflect", "mirror", "wrap", "constant"] = "nearest",
    zero_center: bool = True,
    *,
    progress: bool = True,
    device: Literal["auto", "cpu", "gpu"] = "auto",
) -> np.ndarray:
    """
    Apply spatial Difference-of-Gaussians bandpass.

    Parameters
    ----------
    arr : np.ndarray
        Input image or stack. Shape (Y, X) or (T, Y, X). Any numeric dtype.
    sigma_low : float
        Small (narrow) Gaussian sigma in pixels. Must be > 0 and < sigma_high.
    sigma_high : float
        Large (broad) Gaussian sigma in pixels. Must be > sigma_low.
    mode : {'nearest','reflect','mirror','wrap','constant'}
        Boundary handling for gaussian_filter.
    zero_center : bool
        If True, subtract mean (per-frame for stacks) after DoG to stabilize zero level.
    progress : bool
        Show tqdm progress bar for 3D input.
    device : {'auto','cpu','gpu'}
        Backend selection. 'auto' picks GPU when CuPy & a CUDA device are available.

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

    # ---- choose backend (align with pwrigid.py approach) ----
    xp = np
    use_gpu = False
    if device == "gpu":
        if not (HAVE_CUPY and cp.cuda.runtime.getDeviceCount() > 0):
            raise RuntimeError("GPU requested but CuPy/CUDA device is not available.")
        xp = cp; use_gpu = True
        print("[GPU] Using CuPy backend for DoG")
    elif device == "auto":
        if HAVE_CUPY and cp.cuda.runtime.getDeviceCount() > 0:
            xp = cp; use_gpu = True
            print("[GPU-auto] CuPy available → using GPU backend for DoG")
        else:
            # CPU path: silent
            pass

    a = np.asarray(arr, dtype=np.float32, order="C")  # keep host copy for final return

    # -------------------- 2D path --------------------
    if a.ndim == 2:
        out = _dog2d(a, sigma_low, sigma_high, mode=mode, xp=(cp if use_gpu else np))
        if zero_center:
            if use_gpu:
                out = out - out.mean()
                out = cp.asnumpy(out)
            else:
                out -= float(out.mean())
        else:
            out = cp.asnumpy(out) if use_gpu else out
        return out.astype(np.float32, copy=False)

    # -------------------- 3D path (T,Y,X) --------------------
    T = a.shape[0]
    out_host = np.empty_like(a, dtype=np.float32)

    # tqdm only on CPU host (чтобы не ломать вывод при CuPy)
    rng = range(T)
    if progress:
        import sys
        from tqdm import tqdm
        title = "DoG (GPU)" if use_gpu else "DoG (CPU)"
        rng = tqdm(rng, desc=title, file=sys.stdout)

    if not use_gpu:
        # CPU: покадрово, без временного блюра
        for t in rng:
            fr = a[t]
            g1 = np_gaussian_filter(fr, sigma=(sigma_low), mode=mode)
            g2 = np_gaussian_filter(fr, sigma=(sigma_high), mode=mode)
            d = (g1 - g2).astype(np.float32, copy=False)
            if zero_center:
                d -= float(d.mean())
            out_host[t] = d
        return out_host

    # GPU: вычисляем кадры на устройстве
    for t in rng:
        fr = cp.asarray(a[t], dtype=cp.float32)
        g1 = cp_gaussian_filter(fr, sigma=sigma_low, mode=mode)
        g2 = cp_gaussian_filter(fr, sigma=sigma_high, mode=mode)
        d = (g1 - g2).astype(cp.float32, copy=False)
        if zero_center:
            d = d - d.mean()
        out_host[t] = cp.asnumpy(d)

    return out_host
