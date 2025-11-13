# -*- coding: utf-8 -*-
"""
formats.py — shared helpers for TIFF I/O: orientation, RAM budget, BigTIFF check.
No file I/O here — only utilities used by both readers and writers.
"""
from __future__ import annotations
import os
from typing import Optional, Tuple
import numpy as np

__all__ = [
    "_to_tyx",
    "_guess_tyx_from_shape",
    "auto_ram_budget_bytes",
    "_needs_bigtiff",
]

def _needs_bigtiff(arr: np.ndarray, dtype=None) -> bool:
    """Return True if serialized array size would exceed 4 GiB (BigTIFF required)."""
    dt = arr.dtype if dtype is None else np.dtype(dtype)
    return arr.size * dt.itemsize >= 2**32

def _guess_tyx_from_shape(shape: Tuple[int, ...]) -> Tuple[int, int, int, bool]:
    """
    Return (T, Y, X, stored_yxt_flag) from an arbitrary TIFF series shape.

    Heuristic:
      - 2D:          treat as single frame (1, Y, X).
      - 3D (Y,X,T):  if the LAST dim is 'small' (typical T < 64) -> stored_yxt=True.
      - 3D (T,Y,X):  otherwise assume already TYX.
    """
    if len(shape) == 2:
        T, Y, X = 1, shape[0], shape[1]
        return T, Y, X, False

    if len(shape) == 3:
        Y, X, Z = shape  # not the final interpretation, just names
        # if last dim looks like time (small), we assume (Y, X, T)
        if shape[-1] < 64 and shape[0] >= 64 and shape[1] >= 64:
            T, Y, X = shape[-1], shape[0], shape[1]
            return T, Y, X, True  # stored as YXT
        else:
            # assume already TYX
            T, Y, X = shape[0], shape[1], shape[2]
            return T, Y, X, False

    # Fallback — treat first as T
    T, Y, X = shape[0], shape[1], shape[2]
    return T, Y, X, False

def _to_tyx(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array to (T, Y, X).

    Handles the common cases:
      - (Y, X)          -> (1, Y, X)
      - (T, Y, X)       -> (T, Y, X) (already TYX)
      - (Y, X, T)       -> (T, Y, X)  [YXT]
    Heuristics for 3D case:
      * If first dim < 64 and last dim > 64 -> assume TYX already.
      * Else if last dim is strictly smaller than both first two (typical T < Y,X)
        and first two dims are reasonably large -> assume YXT and moveaxis(-1 -> 0).
      * Otherwise leave as-is (best-effort fallback).
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        return a[None, ...]  # (Y, X) -> (1, Y, X)

    if a.ndim != 3:
        return a  # do nothing for unexpected ranks

    T0, Y0, X0 = a.shape[0], a.shape[1], a.shape[2]

    # Case 1: old fast path — already (T,Y,X) like (small, big, big)
    if T0 < 64 and X0 > 64:
        return a

    # Case 2: typical YXT: (Y, X, T) with T smaller than both Y and X
    if (a.shape[-1] < min(a.shape[0], a.shape[1])) and (a.shape[0] > 64 and a.shape[1] > 64):
        return np.moveaxis(a, -1, 0)  # (Y, X, T) -> (T, Y, X)

    # Fallback: return as-is
    return a
    
def auto_ram_budget_bytes(
    f_total: float = 0.90,
    f_avail: float = 0.90,
    hard_cap_gib: Optional[float] = None,
    floor_mib: int = 512,
) -> int:
    """Compute a conservative RAM budget (bytes) for full in-RAM reads."""
    cand = None
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        cand = min(int(vm.total * f_total), int(vm.available * f_avail))
    except Exception:
        try:
            if os.name == "posix":
                page = os.sysconf("SC_PAGE_SIZE")
                phys = os.sysconf("SC_PHYS_PAGES")
                cand = int(page * phys * f_total)
            else:
                cand = int(16 * (1024**3) * f_total)
        except Exception:
            cand = int(8 * (1024**3) * f_total)
    if hard_cap_gib is not None:
        cand = min(cand, int(hard_cap_gib * (1024**3)))
    floor = int(floor_mib * (1024**2))
    return max(int(cand), floor)
