# -*- coding: utf-8 -*-
"""
hybrid.masks.crop
=================

Simple, explicit cropping helpers for 2D images (Y,X) and 3D stacks (T,Y,X).

Design goals:
- Return NumPy *views* (slices) without copying data whenever possible.
- Validate bounds and sizes early with clear errors.
- Do not change axis order; only crop the last two axes (Y, X).
- Keep API small and predictable.

Provided:
- crop_sides(arr, top, bottom, left, right)
- crop_to_bbox(arr, y0, x0, y1, x1)
- crop_center(arr, height, width, cy=None, cx=None, align="round")
- crop_square(arr, size, cy=None, cx=None, align="round")
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

__all__ = [
    "crop_sides",
    "crop_to_bbox",
    "crop_center",
    "crop_square",
]


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _check_array_nd(arr: np.ndarray) -> None:
    """Ensure arr is 2D (Y,X) or 3D (T,Y,X)."""
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be a numpy.ndarray")
    if arr.ndim not in (2, 3):
        raise ValueError(f"arr.ndim must be 2 or 3, got {arr.ndim}")


def _yx_shape(arr: np.ndarray) -> Tuple[int, int]:
    """Return (H, W) from a 2D/3D array without reordering axes."""
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    return int(arr.shape[-2]), int(arr.shape[-1])


def _slice_2d(arr: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """
    Return a slice arr[..., y0:y1, x0:x1] (view) for 2D/3D arrays.
    Assumes indices are already validated.
    """
    if arr.ndim == 2:
        return arr[y0:y1, x0:x1]
    # 3D: (T, Y, X) — keep T intact
    return arr[..., y0:y1, x0:x1]


def _clamp_bounds(y0: int, x0: int, y1: int, x1: int, H: int, W: int) -> Tuple[int, int, int, int]:
    """Clamp proposed bounds into [0,H]x[0,W] while preserving ordering."""
    y0c = max(0, min(H, y0))
    y1c = max(0, min(H, y1))
    x0c = max(0, min(W, x0))
    x1c = max(0, min(W, x1))
    return y0c, x0c, y1c, x1c


def _require_valid_box(y0: int, x0: int, y1: int, x1: int, H: int, W: int) -> None:
    """Raise with a descriptive error if the box is invalid or empty."""
    if not (0 <= y0 < y1 <= H) or not (0 <= x0 < x1 <= W):
        raise ValueError(
            f"Invalid bbox: y0={y0}, y1={y1}, x0={x0}, x1={x1} for image size (H={H}, W={W})."
        )


def _round_like(x: float, mode: str) -> int:
    """Round strategy for center-based crops: 'round' | 'floor' | 'ceil'."""
    mode = str(mode).lower()
    if mode == "round":
        return int(round(x))
    if mode == "floor":
        return int(np.floor(x))
    if mode == "ceil":
        return int(np.ceil(x))
    raise ValueError("align must be one of {'round','floor','ceil'}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def crop_sides(
    arr: np.ndarray,
    *,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
) -> np.ndarray:
    """
    Crop by removing a given number of pixels from each side.

    Works for 2D (Y,X) and 3D (T,Y,X). Returns a view (no copy).

    Parameters
    ----------
    top, bottom, left, right : int
        Non-negative pixels to remove from corresponding sides.

    Raises
    ------
    ValueError
        If any side is negative or the resulting crop is empty/out of bounds.
    """
    _check_array_nd(arr)
    if min(top, bottom, left, right) < 0:
        raise ValueError("top/bottom/left/right must be >= 0")

    H, W = _yx_shape(arr)
    y0, y1 = top, H - bottom
    x0, x1 = left, W - right

    _require_valid_box(y0, x0, y1, x1, H, W)
    return _slice_2d(arr, y0, y1, x0, x1)


def crop_to_bbox(
    arr: np.ndarray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    *,
    clamp: bool = False,
) -> np.ndarray:
    """
    Crop to an explicit bounding box (y0:y1, x0:x1). Returns a view.

    Parameters
    ----------
    y0, x0, y1, x1 : int
        Pixel indices with the usual half-open convention: [y0, y1), [x0, x1).
    clamp : bool
        If True, clamp the box inside image bounds; otherwise validate strictly.

    Raises
    ------
    ValueError
        If the box is invalid/out of bounds and clamp=False.
    """
    _check_array_nd(arr)
    H, W = _yx_shape(arr)

    if clamp:
        y0, x0, y1, x1 = _clamp_bounds(y0, x0, y1, x1, H, W)
    _require_valid_box(y0, x0, y1, x1, H, W)

    return _slice_2d(arr, y0, y1, x0, x1)


def crop_center(
    arr: np.ndarray,
    height: int,
    width: int,
    *,
    cy: Optional[float] = None,
    cx: Optional[float] = None,
    align: str = "round",
) -> np.ndarray:
    """
    Centered crop of size (height, width) around (cy, cx).

    Works for 2D (Y,X) and 3D (T,Y,X). Returns a view.

    Parameters
    ----------
    height, width : int
        Target crop size (must be > 0 and <= image size).
    cy, cx : float or None
        Center in pixel coordinates. If None, use geometric image center:
        cy = (H-1)/2, cx = (W-1)/2 (i.e., pixel-center convention).
    align : {'round','floor','ceil'}
        How to convert fractional start to integer index.

    Notes
    -----
    - We compute the top-left as:
        y0 = align(cy - height/2),  x0 = align(cx - width/2)
      Then y1 = y0 + height, x1 = x0 + width,
      and validate/clamp to image bounds.
    """
    _check_array_nd(arr)
    H, W = _yx_shape(arr)

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if height > H or width > W:
        raise ValueError(f"Requested crop ({height}x{width}) exceeds image size ({H}x{W}).")

    if cy is None:
        cy = (H - 1) / 2.0
    if cx is None:
        cx = (W - 1) / 2.0

    y0 = _round_like(cy - height / 2.0, align)
    x0 = _round_like(cx - width / 2.0, align)
    y1 = y0 + height
    x1 = x0 + width

    # Keep crop fully inside the image by shifting if necessary.
    if y0 < 0:
        y1 -= y0  # move down
        y0 = 0
    if x0 < 0:
        x1 -= x0  # move right
        x0 = 0
    if y1 > H:
        shift = y1 - H
        y0 -= shift
        y1 = H
    if x1 > W:
        shift = x1 - W
        x0 -= shift
        x1 = W

    _require_valid_box(y0, x0, y1, x1, H, W)
    return _slice_2d(arr, y0, y1, x0, x1)


def crop_square(
    arr: np.ndarray,
    size: int,
    *,
    cy: Optional[float] = None,
    cx: Optional[float] = None,
    align: str = "round",
) -> np.ndarray:
    """
    Centered *square* crop of side `size` around (cy, cx). Returns a view.

    This is a convenience wrapper around `crop_center(...)` with height=width=size.
    """
    return crop_center(arr, size, size, cy=cy, cx=cx, align=align)
