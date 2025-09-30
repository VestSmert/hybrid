"""
TIFF I/O helpers (TYX stacks)

Functions
---------
read_tiff_stack(
    path,
    normalize=False,
    p_low=1.0,
    p_high=99.5,
    dtype=None,
    *,
    normalize_mode: {"none","global","percentile"} = "none",
    method: {"auto","imread","memmap"} = "auto",
    ram_budget_bytes: Optional[int] = 2_147_483_648,
)
    Read a TIFF stack into shape (T, Y, X) with **OOM-safe defaults**.

    Key behavior changes (long-term fix):
      - Default `dtype=None` (keep source dtype) to avoid implicit upcasts.
      - Default `method="auto"`: prefers memory-mapped reading for large files.
      - `ram_budget_bytes` protects from accidental full-RAM allocations. If an
        operation would exceed the budget, a clear, actionable error is raised.

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
from typing import Literal, Optional

import warnings
import numpy as np
from tifffile import imread, imwrite, memmap as tiff_memmap, TiffWriter, TiffFile

__all__ = [
    "read_tiff_stack",
    "write_tiff_stack",
    # new streaming-friendly APIs:
    "read_tiff_memmap",
    "TiffStreamWriter",
    "iter_tiff_blocks",
]

# --- optional codec backend (for compression) ---------------------------------
try:
    import imagecodecs  # noqa: F401
    _HAVE_IMAGECODECS = True
except Exception:
    _HAVE_IMAGECODECS = False


# --- small helpers ------------------------------------------------------------
def _nbytes(shape, dtype) -> int:
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize


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


# --- core reader (OOM-safe) ---------------------------------------------------
def read_tiff_stack(
    path: str,
    normalize: bool = False,
    p_low: float = 1.0,
    p_high: float = 99.5,
    dtype: Optional[type] = None,
    *,
    normalize_mode: Literal["none", "global", "percentile"] = "none",
    method: Literal["auto", "imread", "memmap"] = "auto",
    ram_budget_bytes: Optional[int] = 2_147_483_648,
) -> np.ndarray:
    """
    Read a TIFF stack as (T, Y, X) with OOM-safe defaults.

    Parameters
    ----------
    path : str
        Path to a multi-page TIFF.
    normalize : bool
        Legacy switch kept for backward compatibility:
          - True  -> behaves like normalize_mode="percentile".
          - False -> behaves like normalize_mode="none" (unless `normalize_mode` is set).
        Prefer using `normalize_mode` directly.
    p_low, p_high : float
        Percentiles for robust normalization (used only if normalize_mode="percentile").
    dtype : numpy dtype or None
        Output dtype. If None, keep original dtype (no cast).
    normalize_mode : {"none","global","percentile"}, optional (keyword-only)
        Select normalization strategy:
          - "none":     no scaling, keep original intensity scale.
          - "global":   global min/max → [0,1].
          - "percentile": robust p_low/p_high → [0,1].
    method : {"auto","imread","memmap"}
        - "auto" (default): prefer memmap for large stacks; fall back to imread
          when within RAM budget.
        - "memmap": force memory-mapped read (will raise if the file is not
          memmappable due to compression/tiling).
        - "imread": force materialization into RAM (only do this for small files).
    ram_budget_bytes : int or None
        Soft limit to prevent inadvertent large allocations. If None, no limit.

    Returns
    -------
    np.ndarray
        Array of shape (T, Y, X). May be a numpy.memmap if method uses memmap and
        no normalization/casting was requested.

    Notes
    -----
    - If both `normalize` and `normalize_mode` are provided, `normalize_mode`
      takes precedence unless `normalize=True` and `normalize_mode="none"`,
      in which case percentile normalization is applied (legacy behavior).
    - For very large, **compressed** TIFFs that are not memory-mappable, prefer
      iterating via `iter_tiff_blocks(...)` and write out with `TiffStreamWriter`.
    """
    # discover shape & dtype cheaply, and detect storage order
    with TiffFile(path) as tif:
        series = tif.series[0]
        shp = series.shape  # could be (T,Y,X) or (Y,X,T)
        if len(shp) == 2:
            shp = (1, shp[0], shp[1])
        elif len(shp) != 3:
            raise ValueError(f"Expected 3D stack (T,Y,X). Got shape {shp}.")
        src_dtype = series.dtype
        # heuristics: if last axis is time → (Y,X,T)
        stored_yxt = (shp[0] < 64 and shp[-1] > 64)
        T_est = shp[-1] if stored_yxt else shp[0]

    # decide default mode and safety budget
    mode = normalize_mode
    if normalize and (normalize_mode == "none"):
        mode = "percentile"  # legacy

    # prefer memmap for big sources or when explicitly asked
    src_nbytes = _nbytes((T_est,) + shp[-2:], src_dtype)
    prefer_memmap = (method == "memmap") or (
        method == "auto" and (ram_budget_bytes is not None and src_nbytes > ram_budget_bytes)
    )

    a: np.ndarray
    if prefer_memmap:
        # Try to open as memmap; if not possible and forced → raise; else fallback
        try:
            a = read_tiff_memmap(path, as_TYX=True)
        except ValueError:
            if method == "memmap":
                raise  # caller explicitly requested memmap
            # compressed/tilled and too big to imread safely → guard
            if ram_budget_bytes is not None and src_nbytes > ram_budget_bytes:
                raise MemoryError(
                    "TIFF is not memory-mappable and exceeds ram_budget_bytes. "
                    "Use iter_tiff_blocks(...) + TiffStreamWriter for streaming, "
                    "or lower the budget explicitly."
                )
            # else, small enough → fall through to imread
            a = imread(path)
            if a.ndim == 2:
                a = a[None, ...]
            if a.ndim != 3:
                raise ValueError(f"Expected 3D stack (T,Y,X). Got shape {a.shape}.")
    else:
        # imread path (within budget)
        if ram_budget_bytes is not None and src_nbytes > ram_budget_bytes:
            # prevent accidental OOM
            raise MemoryError(
                "Stack exceeds ram_budget_bytes for imread. Choose method='memmap' "
                "or increase budget."
            )
        a = imread(path)
        if a.ndim == 2:
            a = a[None, ...]
        if a.ndim != 3:
            raise ValueError(f"Expected 3D stack (T,Y,X). Got shape {a.shape}.")

    # Early return: no normalization and no cast requested → cheap
    if mode == "none" and dtype is None:
        return a

    # Estimate output bytes to avoid surprise allocations
    out_dtype = (
        (np.float32 if mode in ("global", "percentile") else a.dtype)
        if dtype is None else np.dtype(dtype)
    )
    out_nbytes = _nbytes(a.shape, out_dtype)
    if isinstance(a, np.memmap) and (ram_budget_bytes is not None) and (out_nbytes > ram_budget_bytes):
        raise MemoryError(
            "Requested normalization/cast would allocate a full in-RAM array "
            f"(~{out_nbytes/2**30:.2f} GiB) exceeding ram_budget_bytes. "
            "Tip: keep dtype=None & normalize_mode='none' to work lazily, or "
            "stream via iter_tiff_blocks(...) into a disk memmap using TiffStreamWriter."
        )

    # Perform normalization/cast in-RAM (safe given the budget checks above)
    if mode == "none":
        out = a.astype(out_dtype, copy=False)
    elif mode == "global":
        scaled = _normalize_global(a)
        out = scaled.astype(out_dtype, copy=False)
    elif mode == "percentile":
        scaled = _normalize_percentile(a, p_low, p_high)
        out = scaled.astype(out_dtype, copy=False)
    else:
        raise ValueError(
            f"Unknown normalize_mode: {normalize_mode!r}. Must be 'none', 'global', or 'percentile'."
        )

    return out

# --- make read_tiff_memmap robust: fallback to raising a clearer hint ----------
def read_tiff_memmap(path: str, *, as_TYX: bool = True) -> np.ndarray:
    """
    Memory-mapped read; if the file layout is not memmappable (compressed/tilled),
    this will raise ValueError. In that case, use `iter_tiff_blocks(...)` instead.
    """
    try:
        a = tiff_memmap(path)
    except ValueError as e:
        raise ValueError(
            "TIFF is not memory-mappable (likely compressed/tilled). "
            "Use iter_tiff_blocks(...) for streaming reads."
        ) from e

    if a.ndim == 2:
        a = a[None, ...]
    elif a.ndim == 3 and as_TYX:
        if a.shape[-1] < 64 and a.shape[0] > 64:
            a = np.moveaxis(a, -1, 0)
    if a.ndim != 3:
        raise ValueError(f"Expected 3D stack (T,Y,X). Got shape {a.shape}.")
    return a

# --- streaming-friendly iterator over time blocks -----------------------------
def iter_tiff_blocks(
    path: str,
    block: int = 64,
    *,
    halo: int = 0,
    start: int = 0,
    stop: Optional[int] = None,
    as_TYX: bool = True,
):
    """
    Iterate TIFF stack in temporal blocks [start:stop) with an optional halo.

    Yields
    ------
    (s, e, arr) where:
        s, e : int
            absolute [s:e) indices in time (0-based) of the CENTER region (no halo).
        arr : np.ndarray, shape (e-s + 2*halo_effective, Y, X)
            loaded block including halo; center is arr[halo : halo+(e-s)].

    Notes
    -----
    - Works for compressed/tilled OME-TIFF (not memory-mappable).
    - Moves axes to (T, Y, X) when `as_TYX=True`.
    """
    with TiffFile(path) as tif:
        series = tif.series[0]
        full_shape = series.shape  # could be (T,Y,X) or (Y,X,T)
        if len(full_shape) != 3:
            raise ValueError(f"Expected 3D stack, got {full_shape}")

        # normalize to (T,Y,X) if needed
        tyx_order = None
        if as_TYX:
            if full_shape[0] < 64 and full_shape[-1] > 64:
                # likely (Y,X,T) -> (T,Y,X)
                tyx_order = (2, 0, 1)

        T_total = full_shape[0] if tyx_order is None else full_shape[-1]
        if stop is None:
            stop = T_total
        start = max(0, start); stop = min(stop, T_total)

        cur = start
        while cur < stop:
            end = min(stop, cur + block)
            s = max(0, cur - halo)
            e = min(T_total, end + halo)

            # read only the needed pages; use slice on the time axis
            if tyx_order is None:
                # data already (T, Y, X)
                arr = series.asarray(key=slice(s, e))
            else:
                # data stored as (Y, X, T): read all and slice last axis
                # (this reads YX first, but tifffile handles paging internally)
                arr_yxt = series.asarray(key=None)
                arr = np.moveaxis(arr_yxt[..., s:e], -1, 0)

            yield (cur, end, arr)  # arr includes halo; center = arr[cur-s : cur-s + (end-cur)]
            cur = end



# --- writer (single-shot) -----------------------------------------------------
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


# --- streaming writer (per-frame) --------------------------------------------
class TiffStreamWriter:
    """
    Stream-writer for large stacks (writes frames incrementally).

    Usage
    -----
    >>> with TiffStreamWriter("out.tiff", dtype="float32", bigtiff=True, compress=True) as tw:
    ...     for frame in frames:   # frame is (Y, X)
    ...         tw.write(frame)

    Parameters
    ----------
    path : str
        Output TIFF path.
    dtype : {'uint16','float32','float16'}
        Storage dtype for each written frame.
    scale : {'auto', None}
        If 'auto' and dtype is integer, scale [0,1] floats to full range.
    bigtiff : bool
        Use BigTIFF container (recommended for large stacks).
    compress : bool
        Use DEFLATE compression when available.

    Notes
    -----
    - Adds axes='TYX' metadata. Frames must be 2D arrays (Y, X).
    - No full-stack buffering — suitable for very large time series.
    """

    def __init__(
        self,
        path: str,
        dtype: Literal["uint16", "float32", "float16"] = "float32",
        scale: Literal["auto", None] = "auto",
        bigtiff: bool = True,
        compress: bool = False,
    ) -> None:
        self.path = path
        self.dtype = dtype
        self.scale = scale
        self.bigtiff = bigtiff
        self.compress = compress

        compression = None
        if compress and _HAVE_IMAGECODECS:
            compression = "deflate"
        elif compress and not _HAVE_IMAGECODECS:
            warnings.warn(
                "imagecodecs is not available/compatible; writing uncompressed TIFF.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._tw = TiffWriter(path, bigtiff=bigtiff)
        self._compression = compression
        self._wrote_any = False

    def write(self, frame: np.ndarray) -> None:
        """Append a single 2D (Y, X) frame to the TIFF."""
        a = np.asarray(frame)
        if a.ndim != 2:
            raise ValueError(f"Each frame must be 2D (Y,X). Got shape {a.shape}.")

        # dtype/scale handling (per-frame)
        if self.dtype == "uint16":
            if a.dtype.kind == "f" and self.scale == "auto":
                out = np.clip(a, 0.0, 1.0) * 65535.0
                out = out.astype(np.uint16, copy=False)
            else:
                out = np.clip(a, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
        elif self.dtype == "float32":
            out = a.astype(np.float32, copy=False)
        elif self.dtype == "float16":
            out = a.astype(np.float16, copy=False)
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        self._tw.write(
            out,
            photometric="minisblack",
            compression=self._compression,
            metadata={"axes": "TYX"} if not self._wrote_any else None,
        )
        self._wrote_any = True

    def close(self) -> None:
        """Finalize the file."""
        if self._tw is not None:
            self._tw.close()
            self._tw = None  # type: ignore[assignment]

    def __enter__(self) -> "TiffStreamWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
