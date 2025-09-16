"""
hybrid.io
==============

I/O utilities for reading and writing imaging data.

Modules
-------
tif : Helpers for reading/writing multi-page TIFF stacks using `tifffile`.
      Exposes `read_tiff_stack(path, normalize=True, ...)` and
      `write_tiff_stack(path, array)`.

Notes
-----
- Keep I/O functions small and dependency-light.
- Prefer lossless formats (TIFF/Zarr) for intermediate results.
"""

from .tif import read_tiff_stack, write_tiff_stack

__all__ = ["read_tiff_stack", "write_tiff_stack"]
