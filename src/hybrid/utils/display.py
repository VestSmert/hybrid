# -*- coding: utf-8 -*-
"""
Lightweight plotting helpers for quick QC.

Functions
---------
- robust_limits(img, p=(1,99), mask=None)
    Robust [vmin, vmax] percentile limits. If `mask` is given, compute
    percentiles only where mask > 0.

- show_pair(a, b, *, titles=("A","B"), mask=None, p=(1,99),
            link_scales=False, cmap="gray", show_colorbar=True, suptitle=None)
    Side-by-side display with independent (default) or linked scales.
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt


def robust_limits(img: np.ndarray,
                  p: Tuple[float, float] = (1.0, 99.0),
                  mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Compute robust display limits for a 2D image.

    Parameters
    ----------
    img : np.ndarray
        2D array.
    p : (low, high)
        Percentiles to use.
    mask : np.ndarray or None
        If provided, only pixels where mask > 0 are considered.

    Returns
    -------
    (vmin, vmax) : tuple[float, float]
    """
    a = np.asarray(img)
    if a.ndim != 2:
        raise ValueError(f"robust_limits expects 2D input, got shape {a.shape}.")
    if mask is not None:
        m = np.asarray(mask) > 0
        vals = a[m]
    else:
        vals = a.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        # fallback to full image min/max if mask removes everything
        return float(np.nanmin(a)), float(np.nanmax(a))
    lo, hi = np.percentile(vals, p)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals)) if float(np.max(vals)) > lo else lo + 1.0
    return float(lo), float(hi)


def show_pair(a: np.ndarray,
              b: np.ndarray,
              *,
              titles: Sequence[str] = ("A", "B"),
              mask: Optional[np.ndarray] = None,
              p: Tuple[float, float] = (1.0, 99.0),
              link_scales: bool = False,
              cmap: str = "gray",
              show_colorbar: bool = True,
              suptitle: Optional[str] = None):
    """
    Display two images side-by-side with robust scaling.

    Parameters
    ----------
    a, b : np.ndarray
        2D images to display.
    titles : (str, str)
        Panel titles.
    mask : np.ndarray or None
        If provided, scaling is computed inside mask (>0).
    p : (low, high)
        Percentiles for robust scaling.
    link_scales : bool
        If True, both panels share the same [vmin, vmax] computed across both
        images; otherwise each panel uses its own limits.
    cmap : str
        Matplotlib colormap name.
    show_colorbar : bool
        Whether to draw a colorbar for each panel.
    suptitle : str or None
        Optional figure title.

    Returns
    -------
    (fig, axes) : (matplotlib.figure.Figure, np.ndarray[Axes])
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("show_pair expects 2D images.")

    if link_scales:
        vmin_a, vmax_a = robust_limits(a, p=p, mask=mask)
        vmin_b, vmax_b = robust_limits(b, p=p, mask=mask)
        vmin = min(vmin_a, vmin_b)
        vmax = max(vmax_a, vmax_b)
        limits = ((vmin, vmax), (vmin, vmax))
    else:
        limits = (robust_limits(a, p=p, mask=mask),
                  robust_limits(b, p=p, mask=mask))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    im0 = ax[0].imshow(a, cmap=cmap, vmin=limits[0][0], vmax=limits[0][1])
    im1 = ax[1].imshow(b, cmap=cmap, vmin=limits[1][0], vmax=limits[1][1])

    ax[0].set_title(titles[0]); ax[0].set_axis_off()
    ax[1].set_title(titles[1]); ax[1].set_axis_off()

    if show_colorbar:
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    if suptitle is not None:
        fig.suptitle(suptitle, y=0.98)

    plt.tight_layout()
    plt.show()
    return fig, ax
