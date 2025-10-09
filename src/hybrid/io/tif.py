# -*- coding: utf-8 -*-
"""
tif.py — robust TIFF I/O helpers (memmap-first, RAM-aware, streaming fallback)

Key functions
-------------
- read_tiff_stack(path, ...): return (T, Y, X) stack; strategy:
    1) try memmap → return
    2) else compute auto RAM budget; if fits → full-RAM imread
    3) else stream + stage to disk-backed memmap → return memmap
- read_tiff_memmap(path, as_TYX=True): memmap-only reader (raises on compressed/tilled)
- iter_tiff_blocks(path, block=64, halo=0, as_TYX=True): stream (start, end, arr)
- TiffStreamWriter(path, ...): append frames to a TIFF stack
- write_tiff_stack(path, arr, ...): write whole stack (float32 by default)

Notes
-----
- Orientation guess: we normalize to (T, Y, X). If a stack looks like (Y, X, T)
  (i.e., first dim < 64 and last dim > 64), it is rotated to TYX.
- Auto RAM budget uses psutil if available; otherwise a conservative fallback.
- Staged memmap file is created next to the source: "<base>.staged.<dtype>.dat".
- Compression: uncompressed TIFFs can be memmapped; compressed ones cannot.
"""

from __future__ import annotations
import os
import sys
import warnings
from typing import Iterator, Tuple, Optional

import numpy as np
import tifffile as tiff
from tifffile import TiffFile, TiffWriter


# ------------------------------ misc helpers -------------------------------- #

def _needs_bigtiff(arr: np.ndarray, dtype=None) -> bool:
    dt = arr.dtype if dtype is None else np.dtype(dtype)
    # BigTIFF if file size would exceed 4 GiB
    return arr.size * dt.itemsize >= 2**32


def _guess_tyx_from_shape(shape: Tuple[int, ...]) -> Tuple[int, int, int, bool]:
    """Return (T, Y, X, stored_yxt_flag) from an arbitrary TIFF series shape."""
    if len(shape) == 2:
        T, Y, X = 1, shape[0], shape[1]
        return T, Y, X, False
    # Heuristic: if first dim is small (<64) and last is big (>64), assume YXT
    stored_yxt = (shape[0] < 64 and shape[-1] > 64 and len(shape) == 3)
    if stored_yxt:
        T, Y, X = shape[-1], shape[0], shape[1]
    else:
        # assume TYX
        T, Y, X = shape[0], shape[1], shape[2]
    return T, Y, X, stored_yxt


def _to_tyx(arr: np.ndarray) -> np.ndarray:
    """Normalize array to (T, Y, X)."""
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3 and arr.shape[0] < 64 and arr.shape[-1] > 64:
        # likely (Y, X, T) → (T, Y, X)
        return np.moveaxis(arr, -1, 0)
    return arr


def auto_ram_budget_bytes(
    f_total: float = 0.90,
    f_avail: float = 0.90,
    hard_cap_gib: Optional[float] = None,
    floor_mib: int = 512,
) -> int:
    """Compute a conservative RAM budget in bytes."""
    cand = None
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        cand = min(int(vm.total * f_total), int(vm.available * f_avail))
    except Exception:
        # Fallbacks if psutil isn't available
        try:
            if os.name == "posix":
                page = os.sysconf("SC_PAGE_SIZE")
                phys = os.sysconf("SC_PHYS_PAGES")
                cand = int(page * phys * f_total)
            else:
                # Windows or unknown: assume 16 GiB total
                cand = int(16 * (1024**3) * f_total)
        except Exception:
            cand = int(8 * (1024**3) * f_total)  # last resort: 8 GiB
    if hard_cap_gib is not None:
        cand = min(cand, int(hard_cap_gib * (1024**3)))
    floor = int(floor_mib * (1024**2))
    return max(cand, floor)


# ------------------------------- readers ------------------------------------ #

def read_tiff_memmap(path: str, as_TYX: bool = True) -> np.ndarray:
    """
    Memmap a TIFF stack. Raises ValueError if the image data are not mappable
    (e.g., compressed/tilled TIFF without contiguous data offsets).

    Returns
    -------
    np.ndarray or np.memmap of shape (T, Y, X)
    """
    try:
        a = tiff.memmap(path)  # may fail if series lacks a single dataoffset
    except ValueError as e:
        raise ValueError(
            "TIFF is not memory-mappable (likely compressed/tilled). "
            "Use iter_tiff_blocks(...) for streaming reads."
        ) from e

    if as_TYX:
        a = _to_tyx(a)
    return a


def iter_tiff_blocks(
    path: str,
    block: int = 64,
    halo: int = 0,
    as_TYX: bool = True,
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """
    Stream TIFF frames in time blocks with an optional temporal halo.

    Yields
    ------
    (start, end, arr) where:
        start : int (inclusive)
        end   : int (exclusive)
        arr   : np.ndarray with shape (end-start + 2*halo', Y, X)
                (halo at edges is smaller if near [0, T))
    """
    if block <= 0:
        raise ValueError("block must be a positive integer")
    if halo < 0:
        raise ValueError("halo must be >= 0")

    with TiffFile(path) as tif:
        ser = tif.series[0]
        shp = ser.shape
        T, Y, X, stored_yxt = _guess_tyx_from_shape(shp)

        cur = 0
        while cur < T:
            end = min(T, cur + block)
            s = max(0, cur - halo)
            e = min(T, end + halo)

            # tifffile supports slicing via 'key' argument
            arr = ser.asarray(key=range(s, e))  # read only the needed pages
            if as_TYX:
                if stored_yxt:
                    arr = np.moveaxis(arr, -1, 0)  # (Y,X,Tblk) → (Tblk,Y,X)
                else:
                    # expected (Tblk,Y,X)
                    pass
            yield cur, end, np.asarray(arr)
            cur = end


# ------------------------------- writers ------------------------------------ #

class TiffStreamWriter:
    """
    Append frames to a TIFF stack. Use as a context manager:

        with TiffStreamWriter("out.tiff", dtype="float32") as tw:
            for frame in frames:
                tw.write(frame)

    Notes
    -----
    - Writes plain TIFF pages with minimal tags (no OME/ImageJ at page level).
    - For very large stacks use bigtiff=True.
    - If compress=True, uses DEFLATE (cannot be memmapped later).
    """

    def __init__(
        self,
        path: str,
        dtype: str | np.dtype = "float32",
        bigtiff: bool = True,
        compress: bool = True,
    ):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.bigtiff = bool(bigtiff)
        self.compress = bool(compress)
        self._tw: Optional[TiffWriter] = None
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def __enter__(self):
        self._tw = TiffWriter(self.path, bigtiff=self.bigtiff)  # no ome/imagej here
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._tw is not None:
            self._tw.close()
        self._tw = None

    def write(self, frame: np.ndarray):
        if self._tw is None:
            raise RuntimeError("TiffStreamWriter not opened; use as a context manager.")
        f = np.asarray(frame, dtype=self.dtype, order="C")
        if f.ndim != 2:
            raise ValueError("Each written frame must be 2D (Y, X).")
        self._tw.write(
            f,
            compression=("deflate" if self.compress else None),
            photometric="minisblack",
        )

def write_tiff_stack(
    path: str,
    arr: np.ndarray,
    dtype: str | np.dtype = "float32",
    bigtiff: Optional[bool] = None,
    compress: bool = True,
    imagej_prefer: bool = False,
    axes: str = "TYX",
):
    """
    Write a whole stack to a single TIFF.

    Parameters
    ----------
    dtype : target dtype ('float32' default). 'uint16' is allowed for previews.
    bigtiff : auto if None (based on size).
    compress : DEFLATE if True; uncompressed if False (memmap-friendly).
    imagej_prefer : write ImageJ-compatible TIFF instead of OME when possible.
    """
    a = _to_tyx(np.asarray(arr))
    dt = np.dtype(dtype)
    a = a.astype(dt, copy=False)
    if bigtiff is None:
        bigtiff = _needs_bigtiff(a, dt)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    if imagej_prefer and not compress and not _needs_bigtiff(a, dt):
        # ImageJ classic — simpler metadata, sometimes nicer for quick previews
        tiff.imwrite(
            path,
            a,
            imagej=True,
            bigtiff=False,
            compression=None,
            metadata={"axes": axes},
        )
        return

    tiff.imwrite(
        path,
        a,
        ome=True,
        imagej=False,
        bigtiff=bigtiff,
        compression=("deflate" if compress else None),
        metadata={"axes": axes},
        photometric="minisblack",
    )


# ---------------------- main high-level read API ---------------------------- #

def read_tiff_stack(
    path: str,
    *,
    normalize: bool | None = None,   # legacy; kept for backward-compat
    p_low: float = 1.0,
    p_high: float = 99.9,
    dtype: Optional[np.dtype | str] = None,
    normalize_mode: str = "none",    # "none" | "percentile"
    method: str = "auto",            # "auto" | "memmap" | "imread" | "auto_stage"
    ram_budget_bytes: Optional[int] = 2_147_483_648,  # default ~2 GiB guard
    auto_ram: bool = True,
    auto_f_total: float = 0.90,
    auto_f_avail: float = 0.90,
    hard_cap_gib: Optional[float] = None,
    stage_block: int = 64,
    stage_dir: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Read a TIFF stack as (T, Y, X) with a robust, memory-aware strategy.

    Strategy
    --------
    - method="memmap": force memmap; raise if not possible.
    - method="imread": force full-RAM read (guarded by ram_budget_bytes unless None).
    - method="auto" (default):
        try memmap → if fail:
            estimate source size → if it fits RAM (using explicit budget OR auto),
            do full-RAM imread; else stage to disk-backed memmap and return it.
    - method="auto_stage": same as auto but always stage when memmap fails.

    Normalization
    -------------
    normalize_mode="none"  : return as-is
    normalize_mode="percentile": scale per-stack using [p_low, p_high] percentiles

    dtype
    -----
    dtype=None keeps source dtype; otherwise cast on return.
    """
    path = os.path.abspath(path)

    # -- 1) fast path: memmap --
    if method in ("auto", "memmap", "auto_stage"):
        try:
            a = read_tiff_memmap(path, as_TYX=True)
            if verbose:
                print("[I/O] Using memmap:", path)
            return _postprocess_stack(a, normalize, p_low, p_high, dtype, normalize_mode)
        except ValueError:
            # fall through to non-memmap paths
            if method == "memmap":
                raise

    # -- 2) prepare series info & size estimate --
    with TiffFile(path) as tif:
        ser = tif.series[0]
        shp = ser.shape
        src_dtype = ser.dtype
    T, Y, X, _ = _guess_tyx_from_shape(shp)
    src_nbytes = int(T) * int(Y) * int(X) * np.dtype(src_dtype).itemsize

    # -- 3) decide the path: imread vs staging --
    def _do_imread(force_no_guard: bool = False) -> np.ndarray:
        # Guard by budget unless explicitly disabled
        if not force_no_guard and ram_budget_bytes is not None and src_nbytes > ram_budget_bytes:
            raise MemoryError(
                "TIFF is not memory-mappable and exceeds ram_budget_bytes. "
                "Use iter_tiff_blocks(...) + TiffStreamWriter for streaming, "
                "or set method='imread' with ram_budget_bytes=None to force."
            )
        if verbose:
            size_gib = src_nbytes / (1024**3)
            print(f"[I/O] Using full-RAM imread (~{size_gib:.2f} GiB):", path)
        a = tiff.imread(path)
        a = _to_tyx(a)
        return a

    def _do_stage() -> np.memmap:
        base = os.path.splitext(path)[0]
        if stage_dir is None:
            dst_dir = os.path.dirname(path)
        else:
            dst_dir = stage_dir
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
        mm = np.memmap(stage_path, mode="r+", dtype=src_dtype, shape=(T, Y, X))
        return mm

    # method="imread": force full RAM path (respecting guard unless None)
    if method == "imread":
        a = _do_imread(force_no_guard=(ram_budget_bytes is None))
        return _postprocess_stack(a, normalize, p_low, p_high, dtype, normalize_mode)

    # method="auto_stage": always stage when memmap fails
    if method == "auto_stage":
        a = _do_stage()
        return _postprocess_stack(a, normalize, p_low, p_high, dtype, normalize_mode)

    # method="auto": choose between imread (if fits) or stage
    # First, check explicit user budget
    if ram_budget_bytes is not None and src_nbytes <= ram_budget_bytes:
        a = _do_imread(force_no_guard=False)
        return _postprocess_stack(a, normalize, p_low, p_high, dtype, normalize_mode)

    # Next, compute auto budget from system RAM (if allowed)
    if auto_ram:
        auto_budget = auto_ram_budget_bytes(
            f_total=auto_f_total, f_avail=auto_f_avail, hard_cap_gib=hard_cap_gib
        )
        if src_nbytes <= auto_budget:
            a = _do_imread(force_no_guard=True)  # override guard since we just checked
            return _postprocess_stack(a, normalize, p_low, p_high, dtype, normalize_mode)

    # Otherwise, stage to disk memmap
    a = _do_stage()
    return _postprocess_stack(a, normalize, p_low, p_high, dtype, normalize_mode)


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

    # normalization (legacy flag vs mode)
    if normalize_mode not in ("none", "percentile"):
        warnings.warn(f"Unknown normalize_mode={normalize_mode!r}, using 'none'.", RuntimeWarning)
        normalize_mode = "none"

    if normalize_mode == "percentile" or (normalize is True and normalize_mode == "none"):
        # scale per-stack using percentiles
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
