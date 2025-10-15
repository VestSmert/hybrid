# --- file: hybrid/summary/viz.py ---
"""
Reusable visualization helpers for QC and reports.
All functions return the Matplotlib figure handle for further customization/saving.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt


def grid_before_after(img_before: np.ndarray, img_after: np.ndarray, titles=("Before", "After")):
    """Show two images side by side (grayscale)."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(img_before, cmap="gray"); axs[0].set_title(titles[0]); axs[0].axis("off")
    axs[1].imshow(img_after,  cmap="gray"); axs[1].set_title(titles[1]);  axs[1].axis("off")
    plt.tight_layout()
    return fig


def show_qc_maps(mean_im: np.ndarray, corr_map: np.ndarray, pnr_map: np.ndarray):
    """Three-panel QC figure: mean, correlation, PNR."""
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].imshow(mean_im, cmap="gray"); axs[0].set_title("Mean image"); axs[0].axis("off")
    im1 = axs[1].imshow(corr_map, vmin=-0.2, vmax=1.0, cmap="magma")
    axs[1].set_title("Correlation image"); axs[1].axis("off"); fig.colorbar(im1, ax=axs[1], fraction=0.046)
    im2 = axs[2].imshow(pnr_map, cmap="magma")
    axs[2].set_title("PNR image"); axs[2].axis("off"); fig.colorbar(im2, ax=axs[2], fraction=0.046)
    plt.tight_layout()
    return fig


def montage(
    files: list[str],
    loader,
    panel: str = "mean",          # {"mean","corr","pnr"}
    ncols: int = 3,
    time_axis: int = 0,           # pass through to correlation_image/pnr_image
    hp_window: int = 9,           # PNR window (will be made odd inside pnr_image)
):
    """
    Create a montage from a list of files using a loader callable.

    Parameters
    ----------
    files : list[str]
        Paths to movies.
    loader : callable
        Function that returns a 3D array (T,H,W) from a path.
    panel : {"mean","corr","pnr"}
        Which panel to show for each file.
    ncols : int
        Number of columns in the grid.
    time_axis : int
        Time axis for correlation/PNR computations.
    hp_window : int
        Temporal HP window for PNR.
    """
    ims = []
    names = []
    for p in files:
        m = loader(p)
        if panel == "mean":
            ims.append(m.mean(axis=time_axis if m.ndim == 3 else 0))
        elif panel == "corr":
            from .qc_maps import correlation_image
            ims.append(correlation_image(m, time_axis=time_axis))
        elif panel == "pnr":
            from .qc_maps import pnr_image
            ims.append(pnr_image(m, time_axis=time_axis, hp_window=hp_window))
        else:
            raise ValueError("panel must be one of {'mean','corr','pnr'}")
        names.append(os.path.basename(p))

    n = len(ims)
    ncols = max(1, int(ncols))
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axs = np.atleast_1d(axs).ravel()

    for ax, im, name in zip(axs, ims, names + [""] * (len(axs) - n)):
        cmap = "gray" if panel == "mean" else "magma"
        ax.imshow(im, cmap=cmap)
        ax.set_title(f"{panel}\n{name}")
        ax.axis("off")

    # Hide extra axes if grid bigger than images
    for ax in axs[n:]:
        ax.axis("off")

    plt.tight_layout()
    return fig
