# -*- coding: utf-8 -*-
"""
hybrid.masks.build
==================

High-level helpers to (optionally) crop, build a soft circular FOV mask,
and apply it to images/stacks. Pure in-memory; no disk I/O.

Goals:
- Keep it orchestration-only (use shapes/crop primitives; no plotting, no I/O).
- Work for 2D (Y,X) and 3D (T,Y,X) arrays without changing axis order.
- Return both the masked array and the mask/params needed by downstream code
  (e.g., ROI selection, QC, registration).

Public API
----------
- build_soft_disk_mask(H, W, cy=None, cx=None, r=None, r_frac=0.95, edge_px=12.0)
- apply_mask_to_array(arr, mask, out_dtype="float32")
- build_and_apply_fov(arr, crop=None, center=None, radius=None, r_frac=0.95, edge_px=12.0, out_dtype="float32")
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import numpy as np

# Prefer using shapes/crop from the same package
try:
    from .shapes import cosine_disk  # soft-edge circular mask
except Exception:
    cosine_disk = None  # we'll use a local fallback if not available

try:
    from .crop import (
        crop_sides,
        crop_to_bbox,
        crop_center,
        crop_square,
    )
except Exception:
    crop_sides = crop_to_bbox = crop_center = crop_square = None  # fallback is error


ArrayLike = np.ndarray
Center = Tuple[float, float]
BBox = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Internal util
# ---------------------------------------------------------------------------

def _check_arr(arr: ArrayLike) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be a numpy.ndarray")
    if arr.ndim not in (2, 3):
        raise ValueError(f"arr.ndim must be 2 or 3, got {arr.ndim}")


def _yx_shape(arr: ArrayLike) -> Tuple[int, int]:
    return (int(arr.shape[-2]), int(arr.shape[-1])) if arr.ndim == 3 else (int(arr.shape[0]), int(arr.shape[1]))


def _soft_disk_fallback(H: int, W: int, cy: float, cx: float, r: float, edge_px: float) -> np.ndarray:
    """
    Soft circular mask with cosine ramp at the edge (fallback if shapes.cosine_disk is unavailable).
    1.0 inside the disk core; cosine fall-off on [r-edge_px, r]; 0 outside.
    """
    yy, xx = np.ogrid[:H, :W]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)

    mask = np.zeros((H, W), dtype=np.float32)
    # inner (fully 1.0)
    inner = dist <= (r - max(edge_px, 0.0))
    mask[inner] = 1.0

    # ring (cosine ramp)
    if edge_px > 0:
        ring = (dist > (r - edge_px)) & (dist <= r)
        # map [r-edge, r] -> [0..1] and apply 0.5*(1+cos(pi*t))
        t = (r - dist[ring]) / float(edge_px)
        mask[ring] = 0.5 * (1.0 + np.cos(np.pi * (1.0 - t)))

    # outside stays 0
    return mask


def _build_soft_disk(H: int,
                     W: int,
                     cy: float,
                     cx: float,
                     r: float,
                     edge_px: float) -> np.ndarray:
    if cosine_disk is not None:
        return cosine_disk(H, W, cy=cy, cx=cx, r=r, edge=edge_px).astype(np.float32, copy=False)
    return _soft_disk_fallback(H, W, cy, cx, r, edge_px)


# ---------------------------------------------------------------------------
# Public: mask builder & applier
# ---------------------------------------------------------------------------

def build_soft_disk_mask(
    H: int,
    W: int,
    *,
    cy: Optional[float] = None,
    cx: Optional[float] = None,
    r: Optional[float] = None,
    r_frac: float = 0.95,
    edge_px: float = 12.0,
) -> Tuple[np.ndarray, Center, float]:
    """
    Build a soft circular FOV mask (float32 in [0,1]) with a cosine edge.

    Parameters
    ----------
    H, W : int
        Target height/width for the mask.
    cy, cx : float or None
        Mask center in pixel coordinates. Defaults to geometric center.
    r : float or None
        Disk radius in pixels. If None, use r_frac * min(H, W)/2.
    r_frac : float
        Fraction of half-min-dimension if r is not given (typ. 0.90..0.98).
    edge_px : float
        Width (in pixels) of the soft edge. 0 means hard edge.

    Returns
    -------
    mask : (H, W) float32
        Soft mask (1 in core, cosine fall-off at the edge, 0 outside).
    center : (cy, cx)
    radius : float
    """
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive.")
    if cy is None:
        cy = (H - 1) / 2.0
    if cx is None:
        cx = (W - 1) / 2.0
    if r is None:
        r = float(r_frac) * min(H, W) / 2.0
    if r <= 0:
        raise ValueError("r must be positive.")
    if edge_px < 0:
        raise ValueError("edge_px must be >= 0.")

    mask = _build_soft_disk(H, W, float(cy), float(cx), float(r), float(edge_px))
    return mask.astype(np.float32, copy=False), (float(cy), float(cx)), float(r)


def apply_mask_to_array(
    arr: ArrayLike,
    mask: np.ndarray,
    *,
    out_dtype: Union[str, np.dtype] = "float32",
) -> ArrayLike:
    """
    Multiply a 2D/3D array by a 2D mask, broadcasting across T if needed.

    Parameters
    ----------
    arr : (Y,X) or (T,Y,X) ndarray
        Input image or stack.
    mask : (H,W) float32/float64
        Soft mask (values in [0,1] are expected).
    out_dtype : numpy dtype or str
        Output dtype (default 'float32').

    Returns
    -------
    masked : ndarray
        Same dimensionality as arr; values multiplied by mask.
    """
    _check_arr(arr)
    if mask.ndim != 2:
        raise ValueError("mask must be 2D (H,W).")
    H, W = _yx_shape(arr)
    if mask.shape != (H, W):
        raise ValueError(f"mask shape {mask.shape} must match (H,W)=({H},{W}).")

    out = arr.astype(out_dtype, copy=False)
    if out.ndim == 2:
        out = out * mask
    else:
        out = out * mask[None, ...]
    return out


# ---------------------------------------------------------------------------
# Public: one-shot orchestration (crop -> mask -> apply)
# ---------------------------------------------------------------------------

def build_and_apply_fov(
    arr: ArrayLike,
    *,
    crop: Optional[Dict] = None,
    center: Optional[Center] = None,
    radius: Optional[float] = None,
    r_frac: float = 0.95,
    edge_px: float = 12.0,
    out_dtype: Union[str, np.dtype] = "float32",
) -> Dict[str, object]:
    """
    Optionally crop (using masks.crop helpers), build a soft disk mask on the
    resulting frame size, and apply it to the array.

    Parameters
    ----------
    arr : (Y,X) or (T,Y,X) ndarray
        Input image or stack (no axis reordering is performed).
    crop : dict or None
        If provided, a dict describing how to crop. Supported forms:
        - {'mode': 'sides', 'top':..., 'bottom':..., 'left':..., 'right':...}
        - {'mode': 'bbox',  'y0':..., 'x0':..., 'y1':..., 'x1':...}
        - {'mode': 'center','height':..., 'width':..., 'cy':None|float, 'cx':None|float, 'align':'round|floor|ceil'}
        - {'mode': 'square','size':..., 'cy':None|float, 'cx':None|float, 'align':'round|floor|ceil'}
        If crop helpers are not importable, passing crop will raise.
    center : (cy, cx) or None
        Mask center. If None, use geometric center of (possibly cropped) array.
    radius : float or None
        Mask radius. If None, use r_frac * min(H, W)/2.
    r_frac : float
        Fraction for radius fallback.
    edge_px : float
        Soft edge width in pixels (0 for hard edge).
    out_dtype : str or numpy dtype
        Output dtype after applying mask.

    Returns
    -------
    result : dict
        {
          'stack': masked_array,           # same ndim as input, dtype=out_dtype
          'mask_soft': (H,W) float32,      # the soft mask used
          'center': (cy, cx),
          'radius': r,
          'crop_box': (y0,x0,y1,x1),       # None if no crop
          'shape_in': original_shape,
          'shape_out': masked_array.shape,
          'params': {
              'r_frac': r_frac,
              'edge_px': edge_px,
              'out_dtype': str(np.dtype(out_dtype))
          }
        }
    """
    _check_arr(arr)
    original_shape = tuple(arr.shape)

    # ---- 1) optional crop
    crop_box: Optional[BBox] = None
    a = arr
    if crop is not None:
        if crop_sides is None:
            raise RuntimeError("masks.crop helpers are not available but 'crop' was provided.")
        mode = str(crop.get("mode", "")).lower()
        if mode == "sides":
            a = crop_sides(
                a,
                top=int(crop.get("top", 0)),
                bottom=int(crop.get("bottom", 0)),
                left=int(crop.get("left", 0)),
                right=int(crop.get("right", 0)),
            )
            # derive y0/x0 from sides for reporting
            H0, W0 = _yx_shape(arr)
            y0 = int(crop.get("top", 0))
            x0 = int(crop.get("left", 0))
            y1 = H0 - int(crop.get("bottom", 0))
            x1 = W0 - int(crop.get("right", 0))
            crop_box = (y0, x0, y1, x1)

        elif mode == "bbox":
            y0 = int(crop["y0"]); x0 = int(crop["x0"]); y1 = int(crop["y1"]); x1 = int(crop["x1"])
            a = crop_to_bbox(a, y0, x0, y1, x1, clamp=False)
            crop_box = (y0, x0, y1, x1)

        elif mode == "center":
            a = crop_center(
                a,
                height=int(crop["height"]),
                width=int(crop["width"]),
                cy=crop.get("cy", None),
                cx=crop.get("cx", None),
                align=str(crop.get("align", "round")),
            )
            # reconstruct exact crop_box from result vs input
            H_out, W_out = _yx_shape(a)
            H_in,  W_in  = _yx_shape(arr)
            # If center/align lead to boundary shifts, exact box is hard to recover here
            # without duplicating crop_center math; leave None to avoid false accuracy.
            crop_box = None

        elif mode == "square":
            a = crop_square(
                a,
                size=int(crop["size"]),
                cy=crop.get("cy", None),
                cx=crop.get("cx", None),
                align=str(crop.get("align", "round")),
            )
            crop_box = None
        else:
            raise ValueError("Unsupported crop['mode']. Expected: 'sides' | 'bbox' | 'center' | 'square'.")

    # ---- 2) mask on the (possibly cropped) array
    H, W = _yx_shape(a)
    if center is None:
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    else:
        cy, cx = float(center[0]), float(center[1])

    if radius is None:
        r = float(r_frac) * min(H, W) / 2.0
    else:
        r = float(radius)

    mask_soft, (cy, cx), r = build_soft_disk_mask(H, W, cy=cy, cx=cx, r=r, r_frac=r_frac, edge_px=float(edge_px))

    # ---- 3) apply
    masked = apply_mask_to_array(a, mask_soft, out_dtype=out_dtype)

    result = {
        "stack": masked,
        "mask_soft": mask_soft,
        "center": (cy, cx),
        "radius": r,
        "crop_box": crop_box,
        "shape_in": original_shape,
        "shape_out": tuple(masked.shape),
        "params": {
            "r_frac": float(r_frac),
            "edge_px": float(edge_px),
            "out_dtype": str(np.dtype(out_dtype)),
        },
    }
    return result
