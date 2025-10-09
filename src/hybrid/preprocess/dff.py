# hybrid/preprocess/dff.py
"""
DeltaF/F (dF/F) computation with single progress bar and auto chunking.

- One tqdm bar per call (like detrend). If you pass an external `pbar`,
  no internal bar is created and we update your pbar by frames processed.
- Device "auto": prefer GPU when available & fits VRAM, otherwise CPU.
- Memory-aware: vectorized full-frame when RAM/VRAM allows; otherwise chunked
  processing with a temporal halo (keeps correctness at chunk boundaries).

Typical use
-----------
>>> from hybrid.preprocess import dff_percentile, dff_from_baseline
>>> dff = dff_percentile(movie, p=10.0, win=301)          # auto device, single tqdm
>>> dff = dff_percentile(movie, p=10.0, win=301, device="cpu")
>>> dff = dff_percentile(movie, p=10.0, win=301, device="gpu")

Notes
-----
- Input is (T, H, W) by default; override via `time_axis`.
- Percentile baseline is computed along time only: size=(win, 1, 1).
- Outputs are float32.
"""

from __future__ import annotations

from typing import Optional, Tuple, Literal, Any
import sys
import warnings
import numpy as np
from scipy.ndimage import percentile_filter as _np_percentile_filter

# Optional: system RAM probe
try:
    import psutil  # type: ignore
    _HAVE_PSUTIL = True
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
    _HAVE_PSUTIL = False

# Optional GPU stack
try:
    import cupy as cp  # type: ignore
    from cupyx.scipy.ndimage import percentile_filter as _cp_percentile_filter  # type: ignore
    _HAVE_CUPY = True
except Exception:  # pragma: no cover
    cp = None  # type: ignore
    _cp_percentile_filter = None  # type: ignore
    _HAVE_CUPY = False

__all__ = ["dff_from_baseline", "dff_percentile"]


# ------------------------------ helpers ---------------------------------------

def _as_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=False)


def dff_from_baseline(movie: np.ndarray, F0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute ΔF/F given a baseline `F0`."""
    x = _as_float32(movie)
    f0 = _as_float32(F0)
    return ((x - f0) / (f0 + eps)).astype(np.float32, copy=False)


def _gpu_available() -> bool:
    if not _HAVE_CUPY:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0  # type: ignore[attr-defined]
    except Exception:
        return False


def _gpu_mem_free_total() -> Tuple[int, int]:
    if not _gpu_available():
        return 0, 0
    try:
        free, total = cp.cuda.runtime.memGetInfo()  # type: ignore[attr-defined]
        return int(free), int(total)
    except Exception:
        return 0, 0


def _cpu_mem_available() -> int:
    if _HAVE_PSUTIL:
        try:
            return int(psutil.virtual_memory().available)  # type: ignore[attr-defined]
        except Exception:
            return 0
    return 0


def _elems_bytes(T: int, H: int, W: int, dtype=np.float32) -> Tuple[int, int]:
    elems = int(T) * int(H) * int(W)
    return elems, elems * np.dtype(dtype).itemsize


def _could_fit_gpu_full(T: int, H: int, W: int, win: int) -> bool:
    # Rough estimate: need baseline + input buffers in VRAM for the filter.
    # Keep it conservative: ~3x array bytes must be within 70% of free VRAM.
    free, _tot = _gpu_mem_free_total()
    if free <= 0:
        return False
    _, bytes_all = _elems_bytes(T, H, W, np.float32)
    need = int(3.0 * bytes_all)
    return need < int(0.7 * free)


def _pick_gpu_chunk(base: int, half: int, H: int, W: int) -> Optional[int]:
    """Shrink chunk (÷2) until (chunk+2*half) fits VRAM; None if nothing fits."""
    if base < 1:
        base = 1
    chunk = base
    while chunk >= 16:
        block_len = chunk + 2 * half
        free, _ = _gpu_mem_free_total()
        if free <= 0:
            return None
        _, block_bytes = _elems_bytes(block_len, H, W, np.float32)
        if int(3.0 * block_bytes) < int(0.7 * free):
            return chunk
        chunk //= 2
    return None


def _should_vectorize_cpu(T: int, H: int, W: int, safety: float = 0.35) -> bool:
    """
    Decide if we can process full-frame on CPU.

    We already hold the input as float32 (x). During baseline we allocate ~1x
    more float32 array (F0). During formation dFF we allocate output (1x).
    Internal buffers exist, so budget ~3x array bytes. `safety` is the fraction
    of available RAM we are allowed to consume (default 35%).
    """
    avail = _cpu_mem_available()
    if avail <= 0:
        return False
    _, bytes_all = _elems_bytes(T, H, W, np.float32)
    need = int(3.0 * bytes_all)
    return need < int(safety * avail)


# ------------------------------ main API --------------------------------------

def dff_percentile(
    movie: np.ndarray,
    p: float = 10.0,
    win: int = 201,
    time_axis: int = 0,
    mode: str = "nearest",
    eps: float = 1e-6,
    chunk: Optional[int] = None,
    *,
    # progress semantics:
    # - progress=True and pbar is None → one classic tqdm bar (frames) with [GPU]/[CPU] prefix
    # - pbar is provided               → no internal bar; we update external pbar by frames processed
    # - progress=False                 → silent
    progress: bool = True,
    pbar: Optional[Any] = None,
    device: Literal["auto", "cpu", "gpu"] = "auto",
    gpu_chunk: Optional[int] = None,
) -> np.ndarray:
    """
    Compute ΔF/F using a rolling percentile baseline along the time axis.

    Memory/Device policy
    --------------------
    - GPU: try full-frame if VRAM allows; else auto-chunk (with halo).
    - CPU: if enough RAM → vectorized full-frame; else chunked.
    """
    if movie.ndim < 1:
        raise ValueError("movie must be an array")

    # Ensure odd window (centered)
    if win % 2 == 0:
        win += 1
    half = win // 2
    size = (win,) + (1,) * (movie.ndim - 1)

    # Time axis first
    x = _as_float32(movie)
    x_tfirst = np.moveaxis(x, time_axis, 0)
    T, H, W = int(x_tfirst.shape[0]), int(x_tfirst.shape[1]), int(x_tfirst.shape[2])

    # Decide device
    if device == "gpu":
        use_gpu = _gpu_available()
        if not use_gpu:
            warnings.warn("dff_percentile: GPU requested but CuPy/CUDA not available; falling back to CPU.")
    elif device == "cpu":
        use_gpu = False
    else:  # auto
        use_gpu = _gpu_available()

    # Decide vectorized vs chunked
    chosen_chunk: Optional[int] = chunk
    vectorized = False

    if use_gpu:
        # full-frame if VRAM allows
        if _could_fit_gpu_full(T, H, W, win):
            vectorized, chosen_chunk = True, None
        else:
            base = gpu_chunk if gpu_chunk is not None else (chunk or 64)
            adj = _pick_gpu_chunk(base=max(base, 16), half=half, H=H, W=W)
            if adj is None:  # give up on GPU; fall back to CPU policy
                use_gpu = False
            else:
                chosen_chunk = adj
    if not use_gpu:
        if chunk is None and _should_vectorize_cpu(T, H, W):
            vectorized, chosen_chunk = True, None
        else:
            chosen_chunk = chunk if chunk is not None else 64

    # Short preamble for users
    if progress and pbar is None:
        if vectorized:
            print(f"[{'GPU' if use_gpu else 'CPU'}] dFF (percentile): full-frame", file=sys.stdout, flush=True)
        else:
            nblocks = (T + chosen_chunk - 1) // chosen_chunk  # type: ignore[arg-type]
            print(
                f"[{'GPU' if use_gpu else 'CPU'}] dFF (percentile): chunked {nblocks}×{chosen_chunk} "
                f"(win={win}, halo={half})",
                file=sys.stdout,
                flush=True,
            )

    # ----------------------- baseline F0 --------------------------------------
    if vectorized:
        if use_gpu and _HAVE_CUPY and _cp_percentile_filter is not None:
            g = cp.asarray(x_tfirst, dtype=cp.float32)  # type: ignore
            F0g = _cp_percentile_filter(g, percentile=p, size=size, mode=mode)  # type: ignore
            F0_tfirst = cp.asnumpy(F0g).astype(np.float32, copy=False)  # type: ignore
            del g, F0g
        else:
            F0_tfirst = _np_percentile_filter(
                x_tfirst, percentile=p, size=size, mode=mode
            ).astype(np.float32, copy=False)
    else:
        # Single frame-based bar (internal or external)
        from tqdm import tqdm

        desc = f"[{'GPU' if use_gpu else 'CPU'}] dFF (percentile)  win={win}  chunk={chosen_chunk}  block={chosen_chunk + 2*half}"  # type: ignore[operator]
        own_bar = (pbar is None and progress)
        bar = tqdm(total=T, desc=desc, unit="fr", file=sys.stdout) if own_bar else pbar

        F0_tfirst = np.empty_like(x_tfirst, dtype=np.float32)

        for start in range(0, T, chosen_chunk):  # type: ignore[arg-type]
            stop = min(T, start + chosen_chunk)  # type: ignore[operator]
            s = max(0, start - half)
            e = min(T, stop + half)
            block = x_tfirst[s:e]

            if use_gpu and _HAVE_CUPY and _cp_percentile_filter is not None:
                gb = cp.asarray(block, dtype=cp.float32)  # type: ignore
                F0g = _cp_percentile_filter(gb, percentile=p, size=size, mode=mode)  # type: ignore
                F0_block = cp.asnumpy(F0g).astype(np.float32, copy=False)  # type: ignore
                del gb, F0g
            else:
                F0_block = _np_percentile_filter(block, percentile=p, size=size, mode=mode).astype(
                    np.float32, copy=False
                )

            # strip halo
            b0 = start - s
            b1 = b0 + (stop - start)
            F0_tfirst[start:stop] = F0_block[b0:b1]

            if bar is not None:
                bar.update(stop - start)

        if own_bar and bar is not None and hasattr(bar, "close"):
            bar.close()

    # Move time back
    F0 = np.moveaxis(F0_tfirst, 0, time_axis)

    # ----------------------- form dF/F with limited peak RAM ------------------
    out = np.empty_like(x, dtype=np.float32)

    # If we were vectorized, choose a safe slab for the ratio stage too
    ratio_chunk = chosen_chunk if not vectorized else max(64, (T // 32) or 64)

    sl_all = [slice(None)] * x.ndim
    for start in range(0, T, ratio_chunk):
        stop = min(T, start + ratio_chunk)
        sl = sl_all.copy()
        sl[time_axis] = slice(start, stop)

        # out = (x - F0) / (F0 + eps) (in-place style to cap temporaries)
        np.subtract(x[tuple(sl)], F0[tuple(sl)], out=out[tuple(sl)], dtype=np.float32, casting="unsafe")
        denom = F0[tuple(sl)].astype(np.float32, copy=False)
        denom += np.float32(eps)
        np.divide(out[tuple(sl)], denom, out=out[tuple(sl)], casting="unsafe")

    return out
