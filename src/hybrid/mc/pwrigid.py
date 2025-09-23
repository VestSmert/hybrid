# --- file: hybrid/mc/pwrigid.py ---
"""
Piecewise-rigid motion correction (pw-rigid) with optional GPU acceleration.

Goal
----
Estimate local subpixel shifts on a grid of tiles, interpolate a dense (H,W) flow
field, and warp each frame accordingly. Backend can be CPU (NumPy/SciPy) or
GPU (CuPy) with automatic selection.

Typical usage
-------------
>>> from hybrid.mc import pwrigid_movie
>>> res = pwrigid_movie(movie, tiles=(4,4), overlap=32, upsample=10, device="auto")
>>> corrected = res.corrected
>>> shifts = res.grid_shifts
>>> print(res.mean_corr_after)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.transform import warp

# --- Optional GPU stack (CuPy) ---
HAVE_CUPY = False
try:
    import cupy as cp
    from cupyx.scipy import fft as cufft
    from cupyx.scipy import ndimage as cnd  # <-- явный импорт подпакета ndimage
    HAVE_CUPY = True
except Exception:
    cp = None       # type: ignore
    cufft = None    # type: ignore
    cnd = None      # type: ignore


# ======================================================================
# Result container
# ======================================================================

@dataclass
class PWMotionResult:
    """Result of piecewise-rigid per-frame motion correction.

    Attributes
    ----------
    corrected : np.ndarray
        Motion-corrected movie, shape (T, H, W), dtype float32 (CPU side).
    grid_shifts : np.ndarray
        Tile-wise shifts (dy, dx) in pixels, shape (T, gy, gx, 2), float32.
    mean_corr_after : float
        Crude QC metric: mean correlation with a median template after correction.
    template : np.ndarray
        Registration template (float32) used to estimate shifts.
    """
    corrected: np.ndarray
    grid_shifts: np.ndarray
    mean_corr_after: float
    template: np.ndarray


# ======================================================================
# Helpers
# ======================================================================

def _make_tiles(
    shape_hw: Tuple[int, int],
    tiles: Tuple[int, int] = (4, 4),
    overlap: int = 32,
):
    """Create overlapping tile windows and their centers.

    Parameters
    ----------
    shape_hw : (int, int)
        Image shape as (H, W).
    tiles : (int, int)
        Number of tiles along (y, x) → (gy, gx).
    overlap : int
        Full overlap in pixels added around each core tile.

    Returns
    -------
    boxes : list[tuple[int,int,int,int]]
        (y0, y1, x0, x1) inclusive-exclusive tile windows.
    centers : np.ndarray
        Centers (y, x) of core tiles, shape (gy*gx, 2), float32.
    grid_y : np.ndarray
        1D y-centers of tiles, length gy, float32.
    grid_x : np.ndarray
        1D x-centers of tiles, length gx, float32.
    """
    H, W = shape_hw
    gy, gx = tiles

    y_edges = np.linspace(0, H, gy + 1, dtype=int)
    x_edges = np.linspace(0, W, gx + 1, dtype=int)

    half = overlap // 2
    boxes, centers, grid_y, grid_x = [], [], [], []

    for i in range(gy):
        a_y, b_y = y_edges[i], y_edges[i + 1]
        cy = (a_y + b_y - 1) / 2.0
        grid_y.append(cy)
    for j in range(gx):
        a_x, b_x = x_edges[j], x_edges[j + 1]
        cx = (a_x + b_x - 1) / 2.0
        grid_x.append(cx)

    for i in range(gy):
        a_y, b_y = y_edges[i], y_edges[i + 1]
        y0 = max(0, a_y - half)
        y1 = min(H, b_y + half)
        for j in range(gx):
            a_x, b_x = x_edges[j], x_edges[j + 1]
            x0 = max(0, a_x - half)
            x1 = min(W, b_x + half)
            boxes.append((y0, y1, x0, x1))
            centers.append(((a_y + b_y - 1) / 2.0, (a_x + b_x - 1) / 2.0))

    return (
        boxes,
        np.array(centers, dtype=np.float32),
        np.array(grid_y, dtype=np.float32),
        np.array(grid_x, dtype=np.float32),
    )


def _interpolate_flow(
    dy_grid: np.ndarray,
    dx_grid: np.ndarray,
    H: int,
    W: int,
    grid_y: np.ndarray,
    grid_x: np.ndarray,
    xp=np,
):
    """Interpolate a (gy, gx) shift grid into dense (H, W) flow fields.

    CPU: `RegularGridInterpolator` (linear), with extrapolation at borders.
    GPU: bilinear sampling via `cupyx.scipy.ndimage.map_coordinates`.

    Returns
    -------
    flow_y, flow_x : arrays on the same backend (NumPy or CuPy), float32
    """
    if xp is np:
        f_dy = RegularGridInterpolator((grid_y, grid_x), dy_grid, bounds_error=False, fill_value=None)
        f_dx = RegularGridInterpolator((grid_y, grid_x), dx_grid, bounds_error=False, fill_value=None)
        yy = np.arange(H, dtype=np.float32)
        xx = np.arange(W, dtype=np.float32)
        Y, X = np.meshgrid(yy, xx, indexing="ij")
        pts = np.stack([Y, X], axis=-1)
        flow_y = f_dy(pts).astype(np.float32)
        flow_x = f_dx(pts).astype(np.float32)
        np.nan_to_num(flow_y, copy=False)
        np.nan_to_num(flow_x, copy=False)
        return flow_y, flow_x

    # ---- GPU path (CuPy) ----
    gy, gx = dy_grid.shape
    dy_g = cp.asarray(dy_grid, dtype=cp.float32)
    dx_g = cp.asarray(dx_grid, dtype=cp.float32)

    yy = cp.arange(H, dtype=cp.float32)
    xx = cp.arange(W, dtype=cp.float32)
    Y, X = cp.meshgrid(yy, xx, indexing="ij")

    # map image pixels to tile-grid index space [0..gy-1], [0..gx-1]
    Yg = (Y / cp.float32(max(H - 1, 1))) * cp.float32(gy - 1)
    Xg = (X / cp.float32(max(W - 1, 1))) * cp.float32(gx - 1)

    # >>> key fix: stack coordinates into a single array of shape (ndim, H, W)
    coords = cp.stack([Yg, Xg], axis=0)  # shape = (2, H, W)

    flow_y = cnd.map_coordinates(dy_g, coords, order=1, mode="nearest")
    flow_x = cnd.map_coordinates(dx_g, coords, order=1, mode="nearest")
    return flow_y.astype(cp.float32), flow_x.astype(cp.float32)


def _warp_with_flow(
    frame: np.ndarray,
    flow_y,
    flow_x,
    mode: str = "edge",
    xp=np,
):
    """Warp a single 2D frame using dense flow (inverse map: y+dy, x+dx).

    CPU: delegates to `skimage.transform.warp` with a fast inverse_map closure.
    GPU: bilinear sampling via `cupyx.scipy.ndimage.map_coordinates`.

    Returns
    -------
    out : array on the same backend (NumPy or CuPy), float32
    """
    assert frame.ndim == 2
    H, W = frame.shape

    if xp is np:
        base_y = np.arange(H, dtype=np.float32)[:, None]
        base_x = np.arange(W, dtype=np.float32)[None, :]
        map_y = base_y + flow_y
        map_x = base_x + flow_x

        def inv_map(yx: np.ndarray) -> np.ndarray:
            y = np.clip(np.rint(yx[:, 0]).astype(np.int64), 0, H - 1)
            x = np.clip(np.rint(yx[:, 1]).astype(np.int64), 0, W - 1)
            return np.stack([map_y[y, x], map_x[y, x]], axis=1)

        out = warp(frame, inverse_map=inv_map, order=1, mode=mode, preserve_range=True, clip=True)
        return out.astype(np.float32)

    # --- GPU path ---
    fr = cp.asarray(frame, dtype=cp.float32)
    yy = cp.arange(H, dtype=cp.float32)[:, None]
    xx = cp.arange(W, dtype=cp.float32)[None, :]
    src_y = yy + flow_y
    src_x = xx + flow_x
    coords = cp.stack([src_y, src_x])  # shape (2, H, W)
    out = cnd.map_coordinates(fr, coords, order=1, mode="nearest")
    return out.astype(cp.float32)


# ---------- GPU phase-correlation (coarse FFT + quadratic subpixel) ----------

def _quad_subpixel_offset(vm1, v0, vp1):
    """1D quadratic peak refinement: offset in [-1,1] from center sample."""
    denom = (vm1 - 2 * v0 + vp1)
    return 0.0 if float(denom) == 0.0 else 0.5 * float(vm1 - vp1) / float(denom)


def _phase_cross_correlation_gpu(img_ref, img_mov, upsample: int = 10):
    """Phase cross correlation on GPU via FFT (CuPy).

    Notes
    -----
    - Returns (dy, dx) in pixels. Subpixel refinement is done by a
      simple quadratic fit around the correlation peak (one-step).
      Это быстро и достаточно точно для MC; при желании можно заменить
      на DFT upsampling Guizar-Sicairos.
    """
    ref = cp.asarray(img_ref, dtype=cp.float32)
    mov = cp.asarray(img_mov, dtype=cp.float32)

    F1 = cufft.rfftn(ref)
    F2 = cufft.rfftn(mov)
    R = F1 * cp.conj(F2)
    R /= cp.abs(R) + cp.float32(1e-8)
    corr = cufft.irfftn(R)

    # coarse integer peak
    max_idx = cp.unravel_index(cp.argmax(corr), corr.shape)
    peak_y = int(max_idx[0])
    peak_x = int(max_idx[1])

    # wrap to negative for peaks > N/2
    H, W = corr.shape
    dy = peak_y - (H if peak_y > H // 2 else 0)
    dx = peak_x - (W if peak_x > W // 2 else 0)

    if upsample and upsample >= 2:
        # 1D quadratic fit along each axis around the peak
        # clamp neighbors to valid indices
        ym1 = max(0, peak_y - 1); yp1 = min(H - 1, peak_y + 1)
        xm1 = max(0, peak_x - 1); xp1 = min(W - 1, peak_x + 1)
        cy   = float(corr[peak_y, peak_x])
        cy_m = float(corr[ym1, peak_x])
        cy_p = float(corr[yp1, peak_x])
        cx_m = float(corr[peak_y, xm1])
        cx_p = float(corr[peak_y, xp1])
        dy += _quad_subpixel_offset(cy_m, cy, cy_p)
        dx += _quad_subpixel_offset(cx_m, cy, cx_p)

    return np.array([dy, dx], dtype=np.float32), None, None


# ======================================================================
# Main API
# ======================================================================

def pwrigid_movie(
    movie: np.ndarray,
    template: Optional[np.ndarray] = None,
    tiles: Tuple[int, int] = (4, 4),
    overlap: int = 32,
    upsample: int = 10,
    dog_for_registration: Optional[np.ndarray] = None,
    progress: bool = True,
    device: Literal["auto", "cpu", "gpu"] = "auto",
) -> PWMotionResult:
    """Piecewise-rigid motion correction with CPU/GPU backends.

    Parameters
    ----------
    movie : np.ndarray
        Input stack (T, H, W). Any numeric dtype; casted to float32 internally.
    template : np.ndarray, optional
        (H, W) template for registration. If None, median of first min(200, T) frames.
    tiles : (int, int)
        Number of tiles (gy, gx).
    overlap : int
        Full tile overlap (pixels).
    upsample : int
        Subpixel refinement factor. On GPU используется квадратичная аппроксимация
        вокруг пика (быстро и близко к skimage). На CPU — `skimage.registration.phase_cross_correlation`.
    dog_for_registration : np.ndarray, optional
        DoG-filtered copy for estimating shifts. Warp всё равно применяется к `movie`.
    progress : bool
        Show tqdm progress bar.
    device : {"auto","cpu","gpu"}
        Backend selection.

    Returns
    -------
    PWMotionResult
        Corrected movie (CPU side), grid shifts, QC metric, template.
    """
    import sys
    from tqdm import tqdm

    # --- choose backend ---
    xp = np
    use_gpu = False
    if device == "gpu":
        if not (HAVE_CUPY and cp.cuda.runtime.getDeviceCount() > 0):
            raise RuntimeError("GPU requested but CuPy/CUDA device is not available.")
        xp = cp; use_gpu = True
        print("[GPU] Using CuPy backend for pw-rigid")
    elif device == "auto":
        if HAVE_CUPY and cp.cuda.runtime.getDeviceCount() > 0:
            xp = cp; use_gpu = True
            print("[GPU-auto] CuPy available → using GPU backend for pw-rigid")
        else:
            print("[CPU-auto] Using NumPy backend for pw-rigid")

    # --- inputs ---
    assert movie.ndim == 3, "movie must have shape (T, H, W)"
    T, H, W = movie.shape
    ref_stack = (dog_for_registration.astype(np.float32, copy=False)
                 if dog_for_registration is not None else movie.astype(np.float32, copy=False))

    boxes, _, grid_y, grid_x = _make_tiles((H, W), tiles=tiles, overlap=overlap)
    gy, gx = tiles

    if template is None:
        k = int(min(200, T))
        template = np.median(ref_stack[:k], axis=0).astype(np.float32)
    else:
        template = template.astype(np.float32, copy=False)

    # --- outputs ---
    grid_shifts = np.zeros((T, gy, gx, 2), dtype=np.float32)
    corrected = np.empty_like(movie, dtype=np.float32)

    rng = range(T)
    if progress:
        title = "pw-rigid MC (GPU)" if use_gpu else "pw-rigid MC (CPU)"
        rng = tqdm(rng, desc=title, file=sys.stdout)

    # --- per-frame loop ---
    for t in rng:
        frame_ref = ref_stack[t]
        dy_grid = np.zeros((gy, gx), dtype=np.float32)
        dx_grid = np.zeros((gy, gx), dtype=np.float32)

        idx = 0
        for i in range(gy):
            for j in range(gx):
                y0, y1, x0, x1 = boxes[idx]; idx += 1
                sub_t = template[y0:y1, x0:x1]
                sub_f = frame_ref[y0:y1, x0:x1]

                if use_gpu:
                    shift, _, _ = _phase_cross_correlation_gpu(sub_t, sub_f, upsample=upsample)
                else:
                    from skimage.registration import phase_cross_correlation
                    shift, _, _ = phase_cross_correlation(sub_t, sub_f, upsample_factor=upsample)

                dy_grid[i, j], dx_grid[i, j] = shift

        grid_shifts[t, :, :, 0] = dy_grid
        grid_shifts[t, :, :, 1] = dx_grid

        flow_y, flow_x = _interpolate_flow(dy_grid, dx_grid, H, W, grid_y, grid_x, xp=(cp if use_gpu else np))
        out = _warp_with_flow(movie[t].astype(np.float32, copy=False), -flow_y, -flow_x, mode="edge", xp=(cp if use_gpu else np))
        corrected[t] = cp.asnumpy(out) if use_gpu else out

    # --- simple QC: mean correlation with median template (subsampled) ---
    tmpl = np.median(corrected[: int(min(T, 200))], axis=0).astype(np.float32)
    a = tmpl.ravel()[::16]
    a0 = (a - a.mean()) / (a.std() + 1e-8)
    cors = []
    for t in range(T):
        b = corrected[t].ravel()[::16]
        b0 = (b - b.mean()) / (b.std() + 1e-8)
        cors.append(float((a0 * b0).mean()))
    mean_corr_after = float(np.mean(cors)) if cors else float("nan")

    return PWMotionResult(
        corrected=corrected.astype(np.float32, copy=False),
        grid_shifts=grid_shifts,
        mean_corr_after=mean_corr_after,
        template=template,
    )
