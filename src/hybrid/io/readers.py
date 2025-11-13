# -*- coding: utf-8 -*-
"""
readers.py — robust TIFF readers: memmap, full-RAM imread, block streaming, and staging.
Stacks are returned in (T, Y, X) orientation.
"""
from __future__ import annotations
import os
import warnings
from typing import Iterator, Optional, Tuple

import numpy as np
import tifffile as tiff
from tifffile import TiffFile

from .formats import (
    _to_tyx,
    _guess_tyx_from_shape,
    auto_ram_budget_bytes,
)

__all__ = ["read_tiff_memmap", "iter_tiff_blocks", "read_tiff_stack"]

# ---- low-level readers ------------------------------------------------------

def read_tiff_memmap(path: str, as_TYX: bool = True) -> np.ndarray:
    """
    Memmap the whole TIFF stack. Raises ValueError if the file is not a single
    contiguous stack series (e.g., many separate pages, compressed, tiled).
    Returns np.ndarray/np.memmap with shape (T, Y, X).
    """
    with TiffFile(path) as tif:
        total_pages = len(tif.pages)
        ser0 = tif.series[0]
        axes = getattr(ser0, "axes", "")
        is_stack_series = (len(ser0.shape) == 3) or ("T" in axes)
        if total_pages > 1 and not is_stack_series:
            raise ValueError("TIFF has multiple pages but no contiguous stack series.")
    try:
        a = tiff.memmap(path)
    except ValueError as e:
        raise ValueError("TIFF is not memory-mappable (compressed/tiling/non-contiguous).") from e
    return _to_tyx(a) if as_TYX else a

def iter_tiff_blocks(
    path: str, block: int = 64, halo: int = 0, as_TYX: bool = True
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """
    Stream frames in temporal blocks with optional halo.

    Yields
    ------
    (start, end, arr): arr has shape (end-start + 2*halo', Y, X).
    """
    if block <= 0:
        raise ValueError("block must be > 0")
    if halo < 0:
        raise ValueError("halo must be >= 0")

    with TiffFile(path) as tif:
        total_pages = len(tif.pages)
        ser0 = tif.series[0]
        axes = getattr(ser0, "axes", "")
        is_stack_series = (len(ser0.shape) == 3) or ("T" in axes)
        T = _guess_tyx_from_shape(ser0.shape)[0] if is_stack_series else total_pages

    cur = 0
    while cur < T:
        end = min(T, cur + block)
        s = max(0, cur - halo)
        e = min(T, end + halo)

        if is_stack_series:
            with TiffFile(path) as tif:
                arr = tif.series[0].asarray(key=range(s, e))
        else:
            arr = tiff.imread(path, key=range(s, e))

        arr = np.asarray(arr)
        yield (cur, end, _to_tyx(arr) if as_TYX else arr)
        cur = end

# ---- high-level API ---------------------------------------------------------

def read_tiff_stack(
    path: str,
    *,
    normalize: bool | None = None,
    p_low: float = 1.0,
    p_high: float = 99.9,
    dtype: Optional[np.dtype | str] = None,
    normalize_mode: str = "none",     # "none" | "percentile"
    method: str = "auto",             # "auto" | "memmap" | "imread" | "auto_stage"
    ram_budget_bytes: Optional[int] = 2_147_483_648,
    auto_ram: bool = True,
    auto_f_total: float = 0.90,
    auto_f_avail: float = 0.90,
    hard_cap_gib: Optional[float] = None,
    stage_block: int = 64,
    stage_dir: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Read a TIFF stack as (T, Y, X) using a resilient, memory-aware strategy.
    """
    path = os.path.abspath(path)

    # 1) fast path — memmap
    if method in ("auto", "memmap", "auto_stage"):
        try:
            a = read_tiff_memmap(path, as_TYX=True)
            if verbose:
                print("[I/O] Using memmap:", path)
            return _postprocess_stack(a, normalize, p_low, p_high, dtype, normalize_mode)
        except ValueError:
            if method == "memmap":
                raise

    # 2) inspect series info and estimate memory footprint
    with TiffFile(path) as tif:
        ser = tif.series[0]
        shp = ser.shape
        src_dtype = ser.dtype if hasattr(ser, "dtype") else np.dtype("uint16")
        T, Y, X, _ = _guess_tyx_from_shape(shp)
        if len(tif.pages) > 1 and len(shp) == 2:
            T = len(tif.pages)
            Y, X = shp[0], shp[1]
    src_nbytes = int(T) * int(Y) * int(X) * np.dtype(src_dtype).itemsize

    # 3) choose path: imread vs staging
    def _do_imread(force_no_guard: bool = False) -> np.ndarray:
        if not force_no_guard and ram_budget_bytes is not None and src_nbytes > ram_budget_bytes:
            raise MemoryError(
                "Stack is not memory-mappable and exceeds ram_budget_bytes. "
                "Use iter_tiff_blocks + staging, or set method='imread' with ram_budget_bytes=None."
            )
        if verbose:
            size_gib = src_nbytes / (1024**3)
            print(f"[I/O] Using full-RAM imread (~{size_gib:.2f} GiB):", path)
        a = tiff.imread(path, key=slice(None))  # read all pages explicitly
        return _to_tyx(a)

    def _do_stage() -> np.memmap:
        base = os.path.splitext(path)[0]
        dst_dir = stage_dir or os.path.dirname(path)
        os.makedirs(dst_dir, exist_ok=True)
        stage_path = os.path.join(dst_dir, os.path.basename(base) + f".staged.{src_dtype}.dat")
        if verbose:
            size_gib = src_nbytes / (1024**3)
            print(f"[I/O] Staging to disk memmap (~{size_gib:.2f} GiB):", stage_path)
        mm = np.memmap(stage_path, mode="w+", dtype=src_dtype, shape=(T, Y, X))
        filled = 0
        for s, e, arr in iter_tiff_blocks(path, block=stage_block, halo=0, as_TYX=True):
            mm[s:e] = arr
            filled += (e - s)
            if verbose and (filled % (stage_block * 4) == 0 or e == T):
                print(f"  staged {filled}/{T} frames")
        mm.flush()
        del mm
        return np.memmap(stage_path, mode="r+", dtype=src_dtype, shape=(T, Y, X))

    if method == "imread":
        a = _do_imread(force_no_guard=(ram_budget_bytes is None))
        return _postprocess_stack(a, normalize, p_low, p_high, dtype, normalize_mode)

    if method == "auto_stage":
        a = _do_stage()
        return _postprocess_stack(a, normalize, p_low, p_high, dtype, normalize_mode)

    if ram_budget_bytes is not None and src_nbytes <= ram_budget_bytes:
        a = _do_imread(force_no_guard=False)
        return _postprocess_stack(a, normalize, p_low, p_high, dtype, normalize_mode)

    if auto_ram:
        auto_budget = auto_ram_budget_bytes(
            f_total=auto_f_total, f_avail=auto_f_avail, hard_cap_gib=hard_cap_gib
        )
        if src_nbytes <= auto_budget:
            a = _do_imread(force_no_guard=True)
            return _postprocess_stack(a, normalize, p_low, p_high, dtype, normalize_mode)

    a = _do_stage()
    return _postprocess_stack(a, normalize, p_low, p_high, dtype, normalize_mode)

# ---- post-processing (normalization / dtype cast) ---------------------------

def _postprocess_stack(
    a: np.ndarray,
    normalize: Optional[bool],
    p_low: float,
    p_high: float,
    dtype: Optional[np.dtype | str],
    normalize_mode: str,
) -> np.ndarray:
    """Apply optional normalization and dtype cast."""
    a = _to_tyx(np.asarray(a))
    out = a

    if normalize_mode not in ("none", "percentile"):
        warnings.warn(f"Unknown normalize_mode={normalize_mode!r}, using 'none'.", RuntimeWarning)
        normalize_mode = "none"

    if normalize_mode == "percentile" or (normalize is True and normalize_mode == "none"):
        arr = a.astype(np.float32, copy=False)
        lo = np.percentile(arr, p_low)
        hi = np.percentile(arr, p_high)
        if hi <= lo:
            hi = lo + 1.0
        out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)
    else:
        out = a

    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out
