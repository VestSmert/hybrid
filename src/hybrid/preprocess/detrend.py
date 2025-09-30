# --- file: hybrid/preprocess/detrend.py ---
"""
Temporal baseline estimation & detrending (CPU/GPU, OOM-aware, with progress).

- Baseline F0: temporal Gaussian smoothing (sigma_t, frames) along time axis.
- Detrended F = X - F0.
- device={"auto","cpu","gpu"} picks CuPy when available (like pwrigid).
- chunk controls time-slab size; gpu_halo can extend slabs to reduce boundary effects.

Typical
-------
>>> F, F0 = detrend(movie, sigma_t=100.0, chunk=64, device="auto", progress=True)

Notes
-----
* Output dtypes are float32.
* Chunked filtering without halo reproduces прежнюю «быструю» схему.
  Для более точной аппроксимации границ используйте gpu_halo≈3*sigma_t (дороже по VRAM).
"""

from __future__ import annotations
from typing import Optional, Tuple, Literal
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import sys

# --- Optional GPU stack (CuPy) ---
HAVE_CUPY = False
try:
    import cupy as cp  # type: ignore
    from cupyx.scipy.ndimage import gaussian_filter1d as c_gaussian_filter1d  # type: ignore
    HAVE_CUPY = True
except Exception:  # pragma: no cover
    cp = None  # type: ignore
    c_gaussian_filter1d = None  # type: ignore
    
# =============================== helpers ======================================

def _ensure_float32(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32, copy=False)

def _move_time_first(a: np.ndarray, time_axis: int) -> np.ndarray:
    if time_axis < 0:
        time_axis += a.ndim
    return a if time_axis == 0 else np.moveaxis(a, time_axis, 0)


# ============================ CPU implementations =============================

def _gaussian_baseline_cpu(
    movie: np.ndarray,
    sigma_t: float,
    time_axis: int,
    mode: str,
    chunk: Optional[int],
    progress: bool,
    desc: Optional[str],
) -> np.ndarray:
    x = _ensure_float32(movie)
    if chunk is None or x.ndim == 1:
        F0 = gaussian_filter1d(x, sigma=sigma_t, axis=time_axis, mode=mode)
        return F0.astype(np.float32, copy=False)

    x_t = _move_time_first(x, time_axis)
    T = x_t.shape[0]
    F0_t = np.empty_like(x_t, dtype=np.float32)
    pbar = tqdm(total=T, desc=(desc or "baseline"), unit="fr", file=sys.stdout) if progress else None

    # Быстрая chunk-схема без гало (как была раньше)
    for start in range(0, T, chunk):
        stop = min(T, start + chunk)
        out = gaussian_filter1d(x_t[start:stop], sigma=sigma_t, axis=0, mode=mode)
        F0_t[start:stop] = out.astype(np.float32, copy=False)
        if pbar: pbar.update(stop - start)
    if pbar: pbar.close()

    return F0_t if time_axis == 0 else np.moveaxis(F0_t, 0, time_axis)


# ============================ GPU implementations =============================

def _auto_gpu_chunk(T: int, H: int, W: int, halo: int, safety: float = 0.6) -> int:
    """Pick a time-chunk size from free VRAM (very rough heuristic)."""
    free, total = cp.cuda.Device().mem_info  # bytes
    bytes_per_frame = H * W * 4  # float32
    max_frames = int((free * safety) // bytes_per_frame) - 2 * halo
    return max(1, min(T, max_frames))

def _gaussian_baseline_gpu(
    movie: np.ndarray,
    sigma_t: float,
    time_axis: int,
    mode: str,
    chunk: Optional[int],
    gpu_halo: int,
    progress: bool,
    desc: Optional[str],
) -> np.ndarray:
    x = _ensure_float32(movie)
    x_t = _move_time_first(x, time_axis)
    T, H, W = x_t.shape
    F0_t = np.empty_like(x_t, dtype=np.float32)

    halo = int(max(0, gpu_halo))
    if chunk is None:
        chunk = _auto_gpu_chunk(T, H, W, halo)
    pbar = tqdm(total=T, desc=(desc or "baseline"), unit="fr", file=sys.stdout) if progress else None

    t = 0
    while t < T:
        stop = min(T, t + chunk)
        s = max(0, t - halo)
        e = min(T, stop + halo)

        slab = cp.asarray(x_t[s:e])                          # (e-s, H, W) on GPU
        out_slab = c_gaussian_filter1d(slab, sigma=sigma_t, axis=0, mode=mode)
        cs = t - s
        ce = cs + (stop - t)
        F0_center = out_slab[cs:ce]                          # center (no halo)
        F0_t[t:stop] = cp.asnumpy(F0_center)

        if pbar: pbar.update(stop - t)
        t = stop

    if pbar: pbar.close()
    return F0_t if time_axis == 0 else np.moveaxis(F0_t, 0, time_axis)


# ================================ Public API ==================================

def gaussian_baseline(
    movie: np.ndarray,
    sigma_t: float = 100.0,
    time_axis: int = 0,
    mode: str = "reflect",
    chunk: Optional[int] = None,
    *,
    device: Literal["auto", "cpu", "gpu"] = "auto",
    gpu_halo: int = 0,
    progress: bool = False,
    desc: Optional[str] = None,
) -> np.ndarray:
    """Estimate per-pixel baseline by temporal Gaussian smoothing."""
    use_gpu = (device == "gpu") or (device == "auto" and HAVE_CUPY)
    if use_gpu and not HAVE_CUPY:
        use_gpu = False  # safety

    if use_gpu:
        return _gaussian_baseline_gpu(
            movie, sigma_t, time_axis, mode, chunk, gpu_halo, progress, desc
        )
    else:
        return _gaussian_baseline_cpu(
            movie, sigma_t, time_axis, mode, chunk, progress, desc
        )


def detrend(
    movie: np.ndarray,
    sigma_t: float = 100.0,
    time_axis: int = 0,
    mode: str = "reflect",
    chunk: Optional[int] = None,
    *,
    device: Literal["auto", "cpu", "gpu"] = "auto",
    gpu_halo: int = 0,
    progress: bool = False,
    desc: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract a smooth per-pixel baseline from the movie.

    Returns
    -------
    F, F0 : (float32, float32)
        Detrended stack and baseline, same shape as input.
    """
    # 1) Baseline (CPU/GPU)
    F0 = gaussian_baseline(
        movie,
        sigma_t=sigma_t,
        time_axis=time_axis,
        mode=mode,
        chunk=chunk,
        device=device,
        gpu_halo=gpu_halo,
        progress=progress,
        desc=(f"{desc}·baseline" if desc else "baseline"),
    )

    x = _ensure_float32(movie)

    # 2) Subtract baseline (CPU fast path)
    if chunk is None or not (device in ("gpu", "auto") and HAVE_CUPY):
        F = (x - F0).astype(np.float32, copy=False) if chunk is None else _subtract_cpu_chunked(
            x, F0, time_axis, chunk, progress, (f"{desc}·subtract" if desc else "subtract")
        )
        return F, F0

    # 2) Subtract baseline on GPU, chunked похожим образом
    x_t = _move_time_first(x, time_axis)
    F0_t = _move_time_first(F0, time_axis)
    T = x_t.shape[0]
    F_t = np.empty_like(x_t, dtype=np.float32)

    # Выбираем тот же chunk, что и для baseline (без halo для вычитания)
    pbar = tqdm(total=T, desc=(f"{desc}·subtract" if desc else "subtract"), unit="fr", file=sys.stdout) if progress else None
    for start in range(0, T, (chunk or T)):
        stop = min(T, start + (chunk or T))
        xb = cp.asarray(x_t[start:stop])
        f0b = cp.asarray(F0_t[start:stop])
        Fb = xb - f0b
        F_t[start:stop] = cp.asnumpy(Fb)
        if pbar: pbar.update(stop - start)
    if pbar: pbar.close()

    return (F_t if time_axis == 0 else np.moveaxis(F_t, 0, time_axis),
            F0_t if time_axis == 0 else np.moveaxis(F0_t, 0, time_axis))


def _subtract_cpu_chunked(
    x: np.ndarray,
    F0: np.ndarray,
    time_axis: int,
    chunk: int,
    progress: bool,
    desc: str,
) -> np.ndarray:
    x_t = _move_time_first(x, time_axis)
    F0_t = _move_time_first(F0, time_axis)
    T = x_t.shape[0]
    F_t = np.empty_like(x_t, dtype=np.float32)

    pbar = tqdm(total=T, desc=desc, unit="fr", file=sys.stdout) if progress else None
    for start in range(0, T, chunk):
        stop = min(T, start + chunk)
        np.subtract(x_t[start:stop], F0_t[start:stop], out=F_t[start:stop], dtype=np.float32)
        if pbar: pbar.update(stop - start)
    if pbar: pbar.close()

    return F_t if time_axis == 0 else np.moveaxis(F_t, 0, time_axis)
