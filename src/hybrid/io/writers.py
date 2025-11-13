# -*- coding: utf-8 -*-
"""
writers.py — TIFF writing utilities: streaming writer and whole-stack writer.
Always writes stacks in (T, Y, X) orientation.
"""
from __future__ import annotations
import os
from typing import Optional
import numpy as np
import tifffile as tiff
from tifffile import TiffWriter

from .formats import _to_tyx, _needs_bigtiff

__all__ = ["TiffStreamWriter", "write_tiff_stack"]

class TiffStreamWriter:
    """Append frames to a TIFF stack (context manager)."""
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
        self._tw: TiffWriter | None = None
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    def __enter__(self):
        self._tw = TiffWriter(self.path, bigtiff=self.bigtiff)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._tw is not None:
            self._tw.close()
        self._tw = None

    def write(self, frame: np.ndarray):
        """Append a single 2D frame (Y, X)."""
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
    """Write the entire stack to a single TIFF file."""
    a = _to_tyx(np.asarray(arr))
    dt = np.dtype(dtype)
    a = a.astype(dt, copy=False)
    if bigtiff is None:
        bigtiff = _needs_bigtiff(a, dt)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    # Classic ImageJ-style (non-OME), if explicitly requested
    if imagej_prefer and not compress and not _needs_bigtiff(a, dt):
        tiff.imwrite(
            path, a, imagej=True, bigtiff=False, compression=None, metadata={"axes": axes}
        )
        return

    # Default: OME-compliant write
    tiff.imwrite(
        path,
        a,
        ome=True,
        imagej=False,
        bigtiff=bool(bigtiff),
        compression=("deflate" if compress else None),
        metadata={"axes": axes},
        photometric="minisblack",
    )
