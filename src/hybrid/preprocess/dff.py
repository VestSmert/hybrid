# --- file: hybrid/preprocess/dff.py ---
"""
DeltaF/F (dF/F) computation.

Goal
----
Provide utilities to compute ΔF/F either from a precomputed baseline `F0`
(or background) or directly from a rolling percentile baseline along time.

Typical usage
-------------
>>> from hybrid.preprocess.dff import dff_from_baseline, dff_percentile
>>> dff = dff_from_baseline(movie, F0)
>>> dff = dff_percentile(movie, p=10.0, win=301)                     # auto device
>>> dff = dff_percentile(movie, p=10.0, win=301, device="cpu")       # force CPU
>>> dff = dff_percentile(movie, p=10.0, win=301, device="gpu")       # force GPU

Notes
-----
- Input is assumed to be (T, H, W) by default. Use `time_axis` to adapt.
- For `dff_percentile`, the rolling baseline is computed **along the time axis only**
  by using `percentile_filter` with `size=(win, 1, 1)`.
- The function supports CPU (NumPy/SciPy) and GPU (CuPy) backends.
  With `device="auto"` it picks GPU if available & memory allows; otherwise CPU.
- `win` should be odd for a centered window; if even, it is incremented by 1.
- Outputs are returned as float32.
"""

from __future__ import annotations

from typing import Optional, Tuple, Literal
import warnings
import numpy as np
from scipy.ndimage import percentile_filter as _np_percentile_filter

# Try importing CuPy lazily (optional dependency)
try:
    import cupy as cp  # type: ignore
    from cupyx.scipy.ndimage import percentile_filter as _cp_percentile_filter  # type: ignore
    _HAVE_CUPY = True
except Exception:  # pragma: no cover - environment without GPU/CuPy
    cp = None  # type: ignore
    _cp_percentile_filter = None  # type: ignore
    _HAVE_CUPY = False

__all__ = ["dff_from_baseline", "dff_percentile"]


def _as_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=False)


def dff_from_baseline(movie: np.ndarray, F0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute ΔF/F given a baseline `F0`."""
    x = _as_float32(movie)
    f0 = _as_float32(F0)
    return ((x - f0) / (f0 + eps)).astype(np.float32, copy=False)


# ---------- GPU helpers ----------

def _gpu_available() -> bool:
    """Return True if CuPy+CUDA are available and at least one device is present."""
    if not _HAVE_CUPY:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0  # type: ignore[attr-defined]
    except Exception:
        return False


def _gpu_free_mem() -> Tuple[int, int]:
    """Return (free_bytes, total_bytes) for the current GPU. If unavailable, (0, 0)."""
    if not _gpu_available():
        return 0, 0
    try:
        free, total = cp.cuda.runtime.memGetInfo()  # type: ignore[attr-defined]
        return int(free), int(total)
    except Exception:
        return 0, 0


def _block_bytes_elems(block_len: int, H: int, W: int, dtype=np.float32) -> Tuple[int, int]:
    """Estimate bytes and elements for a (block_len, H, W) float32 block."""
    elems = int(block_len) * int(H) * int(W)
    return elems * np.dtype(dtype).itemsize, elems


def _pick_device_for_block(
    block_len: int,
    H: int,
    W: int,
    elem_bytes: int = 4,
    gpu_mem_factor: float = 3.0,
    gpu_mem_safety: float = 0.7,
    min_gpu_elems: int = 30_000_000,
) -> Literal["gpu", "cpu"]:
    """
    Decide whether GPU is worthwhile for a given block.
    Heuristics:
      - GPU must be available.
      - 3x block bytes fit into 70% of free GPU memory.
      - Block has at least ~30M elements to amortize PCIe overhead.
    """
    if not _gpu_available():
        return "cpu"
    free, _total = _gpu_free_mem()
    if free <= 0:
        return "cpu"
    bytes_need, elems = _block_bytes_elems(block_len, H, W, np.float32)
    bytes_need_gpu = int(gpu_mem_factor * bytes_need)
    if elems < min_gpu_elems:
        return "cpu"
    if bytes_need_gpu < int(gpu_mem_safety * free):
        return "gpu"
    return "cpu"


def _shrink_chunk_to_fit_gpu(
    base_chunk: int,
    half: int,
    H: int,
    W: int,
    min_chunk: int = 16,
    **pick_kwargs,
) -> Optional[int]:
    """
    Reduce chunk (by /2) until the expanded block (chunk+2*half) fits GPU heuristics.
    Return adjusted chunk or None if nothing fits.
    """
    chunk = max(base_chunk, min_chunk)
    while chunk >= min_chunk:
        block_len = chunk + 2 * half
        if _pick_device_for_block(block_len, H, W, **pick_kwargs) == "gpu":
            return chunk
        chunk //= 2
    return None


# ---------- main function ----------

def dff_percentile(
    movie: np.ndarray,
    p: float = 10.0,
    win: int = 201,
    time_axis: int = 0,
    mode: str = "nearest",
    eps: float = 1e-6,
    chunk: Optional[int] = None,
    progress: bool = True,
    device: Literal["auto", "cpu", "gpu"] = "auto",
    gpu_chunk: Optional[int] = None,
) -> np.ndarray:
    """
    Compute ΔF/F using a rolling percentile baseline along the time axis.

    If the stack is long, processing automatically switches to chunked mode.
    With `device="auto"` the function tries GPU (CuPy) when available and the
    expanded block fits into GPU memory; otherwise it uses CPU (NumPy/SciPy).

    Parameters
    ----------
    movie : np.ndarray
        Input stack (T, H, W) by default; override with `time_axis`.
    p : float
        Percentile for the rolling baseline, e.g. 10 for 10th percentile.
    win : int
        Window size in **frames**; should be odd for centered window. If even,
        it is incremented by 1 internally.
    time_axis : int
        Axis index for time (default 0).
    mode : str
        Border handling for the percentile filter.
    eps : float
        Stabilizer in the denominator.
    chunk : Optional[int]
        Time chunk length for CPU path (and as a base for GPU auto-chunk). If None,
        auto-selects (64 when T>=512, else vectorized).
    progress : bool
        Show tqdm progress bar for chunked mode.
    device : {"auto","cpu","gpu"}
        Backend selection. "auto" tries GPU if feasible; falls back to CPU.
    gpu_chunk : Optional[int]
        Override chunk length when using GPU (after memory check).

    Returns
    -------
    dff : np.ndarray
        ΔF/F, float32, same shape as input.
    """
    if movie.ndim < 1:
        raise ValueError("movie must be an array")

    # Ensure odd window for centered behavior
    if win % 2 == 0:
        win += 1
    half = win // 2

    x = _as_float32(movie)

    # Move time to axis 0 for easier slicing
    x_tfirst = np.moveaxis(x, time_axis, 0)
    T, H, W = x_tfirst.shape[0], x_tfirst.shape[1], x_tfirst.shape[2]

    # Default chunking policy
    if chunk is None and T >= 512:
        chunk = 64  # conservative default to see progress sooner on large stacks

    size = (win,) + (1,) * (x_tfirst.ndim - 1)

    # Decide device
    chosen_device: Literal["cpu", "gpu"] = "cpu"
    chosen_chunk = chunk

    if device == "gpu":
        # Force GPU, try to adjust chunk if needed
        if not _gpu_available():
            warnings.warn("dff_percentile: GPU requested but CuPy/CUDA not available. Falling back to CPU.")
            chosen_device = "cpu"
        else:
            base_chunk = gpu_chunk if gpu_chunk is not None else (chunk or 64)
            adj = _shrink_chunk_to_fit_gpu(base_chunk, half, H, W)
            if adj is None:
                warnings.warn("dff_percentile: GPU requested but block does not fit GPU memory. Falling back to CPU.")
                chosen_device = "cpu"
            else:
                chosen_device = "gpu"
                chosen_chunk = adj
    elif device == "auto":
        if _gpu_available():
            base_chunk = gpu_chunk if gpu_chunk is not None else (chunk or 64)
            adj = _shrink_chunk_to_fit_gpu(base_chunk, half, H, W)
            if adj is not None:
                chosen_device = "gpu"
                chosen_chunk = adj
            else:
                chosen_device = "cpu"
        else:
            chosen_device = "cpu"
    else:
        chosen_device = "cpu"

    # --- Execution paths ---

    if chosen_chunk is None:
        # Vectorized path (fast on moderate T, no progress bar)
        if chosen_device == "gpu":
            # Entire stack to GPU (only if it fits heuristics; usually not for large T)
            if not _gpu_available() or _cp_percentile_filter is None:
                warnings.warn("dff_percentile: GPU path unavailable; using CPU.")
                F0_tfirst = _np_percentile_filter(x_tfirst, percentile=p, size=size, mode=mode).astype(
                    np.float32, copy=False
                )
            else:
                # NOTE: This allocates full T*H*W on GPU — use with caution.
                g = cp.asarray(x_tfirst, dtype=cp.float32)  # type: ignore
                F0g = _cp_percentile_filter(g, percentile=p, size=size, mode=mode)  # type: ignore
                F0_tfirst = cp.asnumpy(F0g).astype(np.float32, copy=False)  # type: ignore
                del g, F0g  # free GPU mem
        else:
            F0_tfirst = _np_percentile_filter(x_tfirst, percentile=p, size=size, mode=mode).astype(
                np.float32, copy=False
            )

    else:
        # Chunked path with halo and progress bar
        import sys
        from tqdm import tqdm

        F0_tfirst = np.empty_like(x_tfirst, dtype=np.float32)
        rng = range(0, T, chosen_chunk)
        desc_prefix = "[GPU] " if chosen_device == "gpu" else "[CPU] "
        bar_desc = f"{desc_prefix}dFF (percentile)  win={win}  chunk={chosen_chunk}  block={chosen_chunk + 2*half}"
        iters = (T + chosen_chunk - 1) // chosen_chunk
        if progress:
            rng = tqdm(rng, total=iters, desc=bar_desc, file=sys.stdout)

        for start in rng:
            stop = min(T, start + chosen_chunk)
            s = max(0, start - half)
            e = min(T, stop + half)
            block = x_tfirst[s:e]  # (e-s, H, W)

            if chosen_device == "gpu" and _gpu_available() and _cp_percentile_filter is not None:
                # To GPU
                gb = cp.asarray(block, dtype=cp.float32)  # type: ignore
                F0g = _cp_percentile_filter(gb, percentile=p, size=size, mode=mode)  # type: ignore
                F0_block = cp.asnumpy(F0g).astype(np.float32, copy=False)  # type: ignore
                del gb, F0g  # free GPU mem early
            else:
                # CPU
                if chosen_device == "gpu":
                    # We intended GPU but fell back locally — warn once per run
                    warnings.warn(
                        "dff_percentile: GPU chosen but CuPy path unavailable for this block. Using CPU instead."
                    )
                F0_block = _np_percentile_filter(block, percentile=p, size=size, mode=mode).astype(
                    np.float32, copy=False
                )

            # Copy central region back
            b0 = start - s
            b1 = b0 + (stop - start)
            F0_tfirst[start:stop] = F0_block[b0:b1]

    F0 = np.moveaxis(F0_tfirst, 0, time_axis)
    
    # Memory-friendly formation of dF/F
    T = x.shape[time_axis]
    if chunk is None:
        # выберем безопасный дефолт для больших стеков
        chunk = 64

    out = np.empty_like(x, dtype=np.float32)

    # организуем срез вдоль оси времени
    sl_all = [slice(None)] * x.ndim
    for start in range(0, T, chunk):
        stop = min(T, start + chunk)
        sl = sl_all.copy(); sl[time_axis] = slice(start, stop)

        # dFF = (x - F0) / (F0 + eps)  — без лишних временных массивов
        np.subtract(x[tuple(sl)], F0[tuple(sl)], out=out[tuple(sl)], dtype=np.float32, casting="unsafe")
        # делитель положим в буфер tmp, чтобы не плодить еще один 4.5ГиБ массив
        denom = F0[tuple(sl)].astype(np.float32, copy=False)
        denom += np.float32(eps)
        np.divide(out[tuple(sl)], denom, out=out[tuple(sl)], casting="unsafe")

    return out
