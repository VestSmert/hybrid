# --- file: hybrid/io/tif.py ---
"""
TIFF I/O helpers (TYX stacks)

Functions
---------
read_tiff_stack(
    path,
    normalize=False,
    p_low=1.0,
    p_high=99.5,
    dtype=np.float32,
    *,
    normalize_mode: {"none","global","percentile"} = "none",
)
    Read a multi-page TIFF into shape (T, Y, X). Optionally apply robust
    normalization to [0,1] and cast to desired dtype (default float32).

    Notes on normalization:
      - normalize_mode="none": keep original scale (recommended for QC / dF/F).
      - normalize_mode="global": min/max over the whole stack → scale to [0,1].
      - normalize_mode="percentile": robust scaling using p_low/p_high percentiles.
      - The legacy flag `normalize=True` is treated as normalize_mode="percentile"
        for backward compatibility (see Parameters below).

write_tiff_stack(path, arr, dtype="uint16", scale="auto", bigtiff=True, compress=False)
    Save a (T, Y, X) array to TIFF. If dtype != input dtype, data is scaled/
    clipped appropriately. Adds metadata axes='TYX' for downstream tools.

Notes
-----
- Keep normalization optional: use normalize=False (or normalize_mode="none")
  when you need raw intensities (e.g., for ΔF/F calculations based on original scale).
- For large stacks use BigTIFF to avoid 4GB limit.
- If `imagecodecs` is not available/compatible, compression is silently
  disabled (safe fallback).
"""

from __future__ import annotations
from typing import Literal

import warnings
import numpy as np
from tifffile import imread, imwrite

__all__ = ["read_tiff_stack", "write_tiff_stack"]

# --- optional codec backend (for compression) ---------------------------------
try:
    import imagecodecs  # noqa: F401
    _HAVE_IMAGECODECS = True
except Exception:
    _HAVE_IMAGECODECS = False


# --- normalization helpers ----------------------------------------------------
def _normalize_global(a: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Global min/max scaling to [0,1] over the whole stack."""
    a = a.astype(np.float32, copy=False)
    lo = np.min(a).astype(np.float32)
    hi = np.max(a).astype(np.float32)
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + np.float32(1.0)
    a -= lo
    a /= (hi - lo + np.float32(eps))
    np.clip(a, 0.0, 1.0, out=a)
    return a


def _normalize_percentile(a: np.ndarray, p_low: float, p_high: float, eps: float = 1e-6) -> np.ndarray:
    """Robust percentile scaling to [0,1] over the whole stack."""
    a = a.astype(np.float32, copy=False)
    lo = np.percentile(a, p_low).astype(np.float32)
    hi = np.percentile(a, p_high).astype(np.float32)
    if not np.isfinite(hi) or hi <= lo:
        lo = np.min(a).astype(np.float32)
        hi_val = np.max(a)
        hi = (hi_val if hi_val > lo else lo + np.float32(1.0)).astype(np.float32)
    a -= lo
    a /= (hi - lo + np.float32(eps))
    np.clip(a, 0.0, 1.0, out=a)
    return a


def read_tiff_stack(
    path: str,
    normalize: bool = False,
    p_low: float = 1.0,
    p_high: float = 99.5,
    dtype: type = np.float32,
    *,
    normalize_mode: Literal["none", "global", "percentile"] = "none",
) -> np.ndarray:
    """
    Read a TIFF stack as (T, Y, X).

    Parameters
    ----------
    path : str
        Path to a multi-page TIFF.
    normalize : bool
        Legacy switch kept for backward compatibility:
          - True  -> behaves like normalize_mode="percentile".
          - False -> behaves like normalize_mode="none" (unless normalize_mode is set).
        Prefer using `normalize_mode` directly.
    p_low, p_high : float
        Percentiles for robust normalization (used only if normalize_mode="percentile").
    dtype : numpy dtype
        Output dtype (default float32).
    normalize_mode : {"none","global","percentile"}, optional (keyword-only)
        Select normalization strategy:
          - "none":     no scaling, keep original intensity scale.
          - "global":   global min/max → [0,1].
          - "percentile": robust p_low/p_high → [0,1].

    Returns
    -------
    np.ndarray
        Array of shape (T, Y, X) with the requested dtype.

    Notes
    -----
    - If both `normalize` and `normalize_mode` are provided, `normalize_mode`
      takes precedence unless `normalize=True` and `normalize_mode="none"`,
      in which case percentile normalization is applied to preserve legacy behavior.
    """
    arr = imread(path)  # supports BigTIFF, tiled, etc.
    if arr.ndim == 2:
        # Promote single frame to (1, Y, X) for consistency
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D stack (T,Y,X). Got shape {arr.shape}.")

    # Decide normalization path with backward compatibility
    mode = normalize_mode
    if normalize and (normalize_mode == "none"):
        # legacy behavior: normalize=True -> percentile scaling
        mode = "percentile"

    a = np.asarray(arr)

    if mode == "none":
        out = a.astype(dtype, copy=False)
    elif mode == "global":
        out = _normalize_global(a).astype(dtype, copy=False) if dtype != np.float32 else _normalize_global(a)
    elif mode == "percentile":
        out = _normalize_percentile(a, p_low, p_high).astype(dtype, copy=False) if dtype != np.float32 else _normalize_percentile(a, p_low, p_high)
    else:
        raise ValueError(f"Unknown normalize_mode: {normalize_mode!r}. Must be 'none', 'global', or 'percentile'.")

    return out


def write_tiff_stack(
    path: str,
    arr: np.ndarray,
    dtype: Literal["uint16", "float32", "float16"] = "float32",
    scale: Literal["auto", None] = "auto",
    bigtiff: bool = True,
    compress: bool = False,
) -> None:
    """
    Write (T, Y, X) array to TIFF with sensible defaults.

    Parameters
    ----------
    path : str
        Output path (.tif/.tiff).
    arr : np.ndarray
        Data of shape (T, Y, X). If 2D, will be promoted to (1, Y, X).
    dtype : {'uint16','float32','float16'}
        Output storage type. 'uint16' is compact and widely supported.
    scale : {'auto', None}
        If 'auto' and dtype is integer, scale [0,1] floats to full range.
        If None, data is cast directly (values will be clipped if out of range).
    bigtiff : bool
        Enable BigTIFF container (recommended for large files).
    compress : bool
        Use DEFLATE compression to reduce size (requires `imagecodecs`).

    Notes
    -----
    - Adds metadata axes='TYX' to make the stack self-descriptive.
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3:
        raise ValueError(f"Expected 3D stack (T,Y,X). Got shape {a.shape}.")

    # Prepare dtype & scaling
    if dtype == "uint16":
        if a.dtype.kind == "f" and scale == "auto":
            out = np.clip(a, 0.0, 1.0) * 65535.0
            out = out.astype(np.uint16, copy=False)
        else:
            # Cast/clamp into uint16 range
            out = np.clip(a, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
    elif dtype == "float32":
        out = a.astype(np.float32, copy=False)
    elif dtype == "float16":
        out = a.astype(np.float16, copy=False)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Choose compression safely
    compression = None
    if compress:
        if _HAVE_IMAGECODECS:
            compression = "deflate"
        else:
            warnings.warn(
                "imagecodecs is not available/compatible; writing uncompressed TIFF.",
                RuntimeWarning,
                stacklevel=2,
            )

    imwrite(
        path,
        out,
        bigtiff=bigtiff,
        metadata={"axes": "TYX"},
        photometric="minisblack",
        compression=compression,
    )
