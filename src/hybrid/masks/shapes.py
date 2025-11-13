# -*- coding: utf-8 -*-
"""
hybrid.masks.shapes
===================

Small collection of *shape* generators for masks and windows.

Design:
- Binary ("hard") masks return uint8 arrays with values {0, 1}.
- Soft masks/windows return float32 arrays with values in [0, 1].
- Geometry is expressed in pixel units unless stated otherwise.

Included:
- 2D Tukey window (rectangular apodization).
- Hard & soft circular disk (plus cosine_disk alias).
- Hard & soft ellipse.
- Hard & soft axis-aligned rectangle (rect: box coords y0:x0..y1:x1).

Notes:
- The "soft" variants use a cosine ramp near the boundary so the value
  decays smoothly from 1 (inside) to 0 (outside) over a given `edge` width.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

__all__ = [
    "tukey2d",
    "hard_disk", "soft_disk", "cosine_disk",
    "hard_ellipse", "soft_ellipse",
    "hard_rect", "soft_rect",
    "clip01",
]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clip01(a: np.ndarray) -> np.ndarray:
    """Return a float32 array clipped to [0, 1]."""
    return np.clip(a.astype(np.float32, copy=False), 0.0, 1.0)

def _yyxx(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Index grids (y, x) as float32."""
    yy, xx = np.ogrid[:h, :w]
    return yy.astype(np.float32, copy=False), xx.astype(np.float32, copy=False)

# ---------------------------------------------------------------------------
# 2D Tukey window (rectangular apodization)
# ---------------------------------------------------------------------------

def tukey2d(h: int, w: int, alpha: float = 0.5) -> np.ndarray:
    """
    2D Tukey window of size (h, w), values in [0, 1], float32.

    The Tukey window is 1 in the center and tapers to 0 at the borders
    using a cosine ramp. `alpha` (0..1] controls the fraction of the
    window that is tapered on each side (per axis). `alpha=0` degenerates
    to a rectangular window (not recommended); `alpha=1` becomes a Hann.

    Implementation detail: we compute the distance to the nearest image
    border (in pixels) and apply the standard raised-cosine ramp over the
    edge thickness implied by `alpha`.

    Parameters
    ----------
    h, w : int
        Window height and width.
    alpha : float
        Taper fraction per axis (0..1]. Typical 0.25..0.75.

    Returns
    -------
    win : np.ndarray (float32, shape (h, w))
        2D Tukey window.
    """
    if h <= 0 or w <= 0:
        raise ValueError("h and w must be positive")
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    # Effective taper widths in pixels along Y and X.
    # Each side gets (alpha/2)*size worth of cosine ramp.
    edge_y = 0.5 * alpha * max(1, h - 1)
    edge_x = 0.5 * alpha * max(1, w - 1)

    yy, xx = _yyxx(h, w)
    # distance to the nearest border along each axis
    dtop, dleft = yy, xx
    dbot, dright = (h - 1) - yy, (w - 1) - xx
    dy = np.minimum(dtop, dbot)
    dx = np.minimum(dleft, dright)

    wy = np.ones((h, 1), np.float32)
    wx = np.ones((1, w), np.float32)

    if edge_y > 0:
        mask = dy < edge_y
        t = np.zeros_like(dy, dtype=np.float32)
        t[mask] = 0.5 * (1.0 - np.cos(np.pi * (dy[mask] / edge_y)))
        t[~mask] = 1.0
        wy = t.reshape(h, 1).astype(np.float32, copy=False)

    if edge_x > 0:
        mask = dx < edge_x
        t = np.zeros_like(dx, dtype=np.float32)
        t[mask] = 0.5 * (1.0 - np.cos(np.pi * (dx[mask] / edge_x)))
        t[~mask] = 1.0
        wx = t.reshape(1, w).astype(np.float32, copy=False)

    win = (wy * wx).astype(np.float32, copy=False)
    return clip01(win)


# ---------------------------------------------------------------------------
# Disks (circular masks)
# ---------------------------------------------------------------------------

def hard_disk(h: int, w: int, cy: float, cx: float, r: float,
                dtype: Optional[np.dtype] = np.uint8) -> np.ndarray:
    """
    Binary circular mask: 1 inside radius r, 0 outside.

    Parameters
    ----------
    h, w : int
        Image size.
    cy, cx : float
        Circle center (pixel coordinates, can be fractional).
    r : float
        Radius in pixels.
    dtype : np.dtype
        Output dtype; default uint8.

    Returns
    -------
    mask : (h, w) uint8 with {0,1}
    """
    yy, xx = _yyxx(h, w)
    dist = np.hypot(yy - cy, xx - cx)
    out = (dist <= r).astype(dtype or np.uint8, copy=False)
    return out


def soft_disk(h: int, w: int, cy: float, cx: float, r: float, edge: float) -> np.ndarray:
    """
    Soft circular mask with a cosine edge of width `edge` pixels.

    Value is 1 for dist <= r - edge,
            cosine-ramped for r - edge < dist <= r,
            0 for dist > r.

    This is exactly the mask we used for FF/visualization (smooth disk).

    Returns
    -------
    mask : (h, w) float32 in [0,1]
    """
    if edge < 0:
        raise ValueError("edge must be >= 0")
    yy, xx = _yyxx(h, w)
    dist = np.hypot(yy - cy, xx - cx).astype(np.float32, copy=False)

    mask = np.zeros((h, w), np.float32)

    if edge <= 0:
        mask[dist <= r] = 1.0
        return mask

    core = dist <= (r - edge)
    ring = (dist > (r - edge)) & (dist <= r)

    mask[core] = 1.0
    # Cosine ramp from 1 at inner boundary to 0 at r
    mask[ring] = 0.5 * (1.0 + np.cos(np.pi * (dist[ring] - (r - edge)) / edge))
    # outside stays 0
    return clip01(mask)


def cosine_disk(h: int, w: int, cy: float, cx: float, r: float, edge: float) -> np.ndarray:
    """Alias for `soft_disk` (explicit cosine taper)."""
    return soft_disk(h, w, cy, cx, r, edge)


# ---------------------------------------------------------------------------
# Ellipses
# ---------------------------------------------------------------------------

def hard_ellipse(h: int, w: int, cy: float, cx: float, ry: float, rx: float,
                 dtype: Optional[np.dtype] = np.uint8) -> np.ndarray:
    """
    Binary ellipse: 1 inside, 0 outside.

    The ellipse is defined by center (cy,cx) and radii (ry, rx) in pixels.

    Returns
    -------
    mask : (h, w) uint8 with {0,1}
    """
    yy, xx = _yyxx(h, w)
    ny = (yy - cy) / max(1e-6, ry)
    nx = (xx - cx) / max(1e-6, rx)
    rnorm = np.hypot(ny, nx)  # <=1 inside the ellipse
    out = (rnorm <= 1.0).astype(dtype or np.uint8, copy=False)
    return out


def soft_ellipse(h: int, w: int, cy: float, cx: float, ry: float, rx: float,
                 edge_px: float) -> np.ndarray:
    """
    Soft ellipse with cosine taper.

    We define a normalized radius r̂ = sqrt(((y-cy)/ry)^2 + ((x-cx)/rx)^2).
    Inside the ellipse r̂<=1. We apply a cosine ramp on the band
    [1 - edge_norm, 1], where `edge_norm = edge_px / min(ry, rx)`.

    This interprets `edge_px` in *pixel units* consistent with the *minor*
    (smaller) ellipse radius, which keeps the taper visually similar in y/x.

    Returns
    -------
    mask : (h, w) float32 in [0,1]
    """
    if edge_px < 0:
        raise ValueError("edge_px must be >= 0")
    yy, xx = _yyxx(h, w)
    ny = (yy - cy) / max(1e-6, ry)
    nx = (xx - cx) / max(1e-6, rx)
    rhat = np.hypot(ny, nx).astype(np.float32, copy=False)

    mask = np.zeros((h, w), np.float32)
    if edge_px == 0:
        mask[rhat <= 1.0] = 1.0
        return mask

    edge_norm = float(edge_px) / max(1e-6, min(ry, rx))
    inner = 1.0 - edge_norm

    core = rhat <= inner
    ring = (rhat > inner) & (rhat <= 1.0)

    mask[core] = 1.0
    # linear parameter t from 0 at inner to 1 at outer boundary
    t = (rhat[ring] - inner) / max(1e-6, (1.0 - inner))
    mask[ring] = 0.5 * (1.0 + np.cos(np.pi * t))
    return clip01(mask)


# ---------------------------------------------------------------------------
# Axis-aligned rectangles
# ---------------------------------------------------------------------------

def hard_rect(h: int, w: int, cy: float, cx: float, hy: float, hx: float,
              dtype: Optional[np.dtype] = np.uint8) -> np.ndarray:
    """
    Binary axis-aligned rectangle centered at (cy, cx) with half-sizes (hy, hx).

    Pixels satisfying |y-cy|<=hy and |x-cx|<=hx are set to 1, others 0.
    """
    yy, xx = _yyxx(h, w)
    out = ((np.abs(yy - cy) <= hy) & (np.abs(xx - cx) <= hx)).astype(dtype or np.uint8, copy=False)
    return out


def soft_rect(h: int, w: int, cy: float, cx: float, hy: float, hx: float,
              edge_y: float, edge_x: Optional[float] = None) -> np.ndarray:
    """
    Soft axis-aligned rectangle with cosine edges.

    We define the *inside distance to the rectangle boundary* along the
    nearest side and ramp from 0 at the boundary to 1 at `edge_*` pixels
    deeper inside. Past that the value stays 1. Outside the rectangle the
    value is 0.

    Parameters
    ----------
    h, w : int
        Image size.
    cy, cx : float
        Center of the rectangle.
    hy, hx : float
        Half-sizes (vertical, horizontal) in pixels.
    edge_y : float
        Cosine taper width (pixels) along the vertical direction.
    edge_x : float or None
        Taper width along horizontal direction; if None, uses edge_y.

    Returns
    -------
    mask : (h, w) float32 in [0,1]
    """
    if edge_x is None:
        edge_x = edge_y
    if edge_y < 0 or edge_x < 0:
        raise ValueError("edge_y/edge_x must be >= 0")

    yy, xx = _yyxx(h, w)  # yy:(h,1), xx:(1,w)
    # make full (h, w) maps for safe boolean indexing
    ay = np.broadcast_to(np.abs(yy - cy), (h, w)).astype(np.float32, copy=False)
    ax = np.broadcast_to(np.abs(xx - cx), (h, w)).astype(np.float32, copy=False)

    inside = (ay <= hy) & (ax <= hx)
    mask = np.zeros((h, w), np.float32)
    if not np.any(inside):
        return mask

    mask[inside] = 1.0

    # distance to nearest boundary inside the rectangle
    dy = hy - ay          # (h, w)
    dx = hx - ax          # (h, w)

    if edge_y > 0:
        iy = inside & (dy < edge_y)
        ty = 0.5 * (1.0 - np.cos(np.pi * (dy[iy] / edge_y)))
        mask[iy] = np.minimum(mask[iy], ty.astype(np.float32, copy=False))

    if edge_x > 0:
        ix = inside & (dx < edge_x)
        tx = 0.5 * (1.0 - np.cos(np.pi * (dx[ix] / edge_x)))
        mask[ix] = np.minimum(mask[ix], tx.astype(np.float32, copy=False))

    return clip01(mask)