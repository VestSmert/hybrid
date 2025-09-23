"""
hybrid.io
==============

I/O utilities for reading and writing imaging data.

Modules
-------
tif : Helpers for reading/writing multi-page TIFF stacks using `tifffile`.
      Exposes:
        - read_tiff_stack(path, normalize=..., normalize_mode=...)
        - write_tiff_stack(path, array, ...)
        - read_tiff_memmap(path, as_TYX=True)
        - TiffStreamWriter(path, ...)

Notes
-----
- Keep I/O functions small and dependency-light.
- Prefer lossless formats (TIFF/Zarr) for intermediate results.
- For very large stacks, use `read_tiff_memmap` + `TiffStreamWriter`
  to avoid allocating the whole movie in memory.
"""

from .tif import (
    read_tiff_stack,
    write_tiff_stack,
    read_tiff_memmap,
    TiffStreamWriter,
    iter_tiff_blocks,
)

import importlib as _importlib
tif = _importlib.import_module(".tif", __name__)

__all__ = [
    "read_tiff_stack",
    "write_tiff_stack",
    "read_tiff_memmap",
    "TiffStreamWriter",
    "iter_tiff_blocks",
    "tif",
]
