# --- file: hybrid/cli/prep_cli.py ---
"""
Preprocessing pipeline runner: flat-field → detrend → dF/F

Features
--------
- OOM-safe TIFF input (memmap when possible, streaming "staging" fallback for
  compressed/tilled TIFFs that cannot be memmapped).
- Float32 outputs only (BigTIFF when needed, optional DEFLATE compression).
- GPU auto-selection for detrend and dF/F (uses CuPy if available).
- Consistent console progress bars (provided inside preprocess functions).
- Modular: each step can be enabled/disabled via flags.

Examples
--------
python -m hybrid.cli.prep_cli ^
  --input "D:/data/*pwrigid*.tif" ^
  --outdir "D:/data/_prep" ^
  --flat-sigma 51 --save-ff ^
  --detrend-sigma 100 --save-detrended ^
  --dff-pct 10 --dff-win 301 --save-dff ^
  --chunk 64 --stride 1 --max-frames 0

Outputs (all float32)
---------------------
<name>_flat_f32.tiff          - estimated flat-field image (if --save-flat)
<name>_ff_f32.tiff            - flat-field corrected stack (if --save-ff)
<name>_detrended_f32.tiff     - detrended stack (if --save-detrended)
<name>_dff_p{P}_w{W}_f32.tiff - dF/F stack (if --save-dff)
"""

from __future__ import annotations
import os
import sys
import glob
import argparse
import numpy as np
from typing import Iterable, Tuple

from tifffile import TiffFile

# I/O (safe readers/writers)
from hybrid.io import read_tiff_stack, write_tiff_stack, iter_tiff_blocks

# Preprocessing
from hybrid.preprocess import detrend as detrend_fn
from hybrid.preprocess import dff_from_baseline, dff_percentile

# Spatial Gaussian (for flat-field)
from scipy.ndimage import gaussian_filter


# ------------------------------- helpers ------------------------------------ #

def _basename_noext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _ensure_odd(n: int) -> int:
    n = int(n)
    return n if (n % 2 == 1) else (n + 1)


def _stage_tiff_to_memmap(path: str, block: int = 64) -> Tuple[np.memmap, Tuple[int, int, int], np.dtype, str]:
    """Stream a compressed/tilled TIFF into a disk-backed memmap (TYX, source dtype)."""
    with TiffFile(path) as tif:
        ser = tif.series[0]
        shp = ser.shape
        src_dtype = ser.dtype
        if len(shp) == 2:
            T, Y, X = 1, shp[0], shp[1]
        else:
            stored_yxt = (shp[0] < 64 and shp[-1] > 64)
            T, Y, X = (shp[-1], shp[0], shp[1]) if stored_yxt else (shp[0], shp[1], shp[2])

    stage_path = os.path.join(os.path.dirname(path), _basename_noext(path) + f".staged.{src_dtype}.dat")
    print(f"[stage] {path} -> {stage_path} | (T,Y,X)=({T},{Y},{X}) block={block}")

    mm = np.memmap(stage_path, mode="w+", dtype=src_dtype, shape=(T, Y, X))
    filled = 0
    for s, e, arr in iter_tiff_blocks(path, block=block, halo=0, as_TYX=True):
        mm[s:e] = arr
        filled += (e - s)
        if filled % (block * 4) == 0 or e == T:
            print(f"  staged {filled}/{T}")
    mm.flush()
    del mm
    mm = np.memmap(stage_path, mode="r+", dtype=src_dtype, shape=(T, Y, X))
    return mm, (T, Y, X), np.dtype(src_dtype), stage_path


def _flatfield(movie: np.ndarray, sigma: int, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate & apply multiplicative flat-field using a blurred mean frame."""
    mean_im = movie.mean(axis=0).astype(np.float32, copy=False)
    flat = gaussian_filter(mean_im, sigma=sigma).astype(np.float32, copy=False)
    scale = float(mean_im.mean())
    flat = flat * (scale / (flat.mean() + 1e-12))
    corrected = (movie.astype(np.float32, copy=False) / (flat + eps)) * scale
    return corrected.astype(np.float32, copy=False), flat.astype(np.float32, copy=False)


# ------------------------------- core --------------------------------------- #

def process_one(
    path: str,
    outdir: str,
    *,
    # flat-field
    flat_sigma: int | None = None,
    save_flat: bool = False,
    save_ff: bool = False,
    # detrend
    detrend_sigma: float | None = None,
    save_detrended: bool = False,
    # dF/F
    dff_pct: float | None = None,
    dff_win: int | None = None,
    save_dff: bool = False,
    # perf/limits
    stride: int = 1,
    max_frames: int = 0,           # 0 = keep all after stride
    chunk: int = 64,               # time-chunk for detrend/dFF
    gpu_halo: int = 0,             # temporal halo for GPU baseline; 0 = fastest
) -> None:
    os.makedirs(outdir, exist_ok=True)
    base = _basename_noext(path)

    # 1) Read with guards (memmap preferred; stage if needed)
    try:
        movie = read_tiff_stack(path, normalize_mode="none", dtype=None)
    except (MemoryError, ValueError) as e:
        print(f"[read] {e} → staging to disk-memmap…")
        movie, (T, H, W), src_dtype, stage_path = _stage_tiff_to_memmap(path, block=max(32, chunk))
        print(f"[read] staged memmap ready: {stage_path}")

    # Temporal stride / cap
    movie = movie[:: max(1, int(stride))]
    if max_frames and movie.shape[0] > max_frames:
        movie = movie[:max_frames]

    x = movie.astype(np.float32, copy=False)

    # 2) Flat-field (optional)
    if flat_sigma and flat_sigma > 0:
        print(f"[flat] sigma={flat_sigma}")
        x, flat = _flatfield(x, sigma=int(flat_sigma))
        if save_flat:
            out_flat = os.path.join(outdir, f"{base}_flat_f32.tiff")
            write_tiff_stack(out_flat, flat[None, ...], dtype="float32", bigtiff=True, compress=True)
            print(f"  -> saved {out_flat}")
        if save_ff:
            out_ff = os.path.join(outdir, f"{base}_ff_f32.tiff")
            write_tiff_stack(out_ff, x, dtype="float32", bigtiff=True, compress=True)
            print(f"  -> saved {out_ff}")

    # 3) Detrend (optional; GPU auto)
    F = None
    F0 = None
    if detrend_sigma and detrend_sigma > 0:
        print(f"[detrend] sigma_t={detrend_sigma} | chunk={chunk}")
        F, F0 = detrend_fn(
            x, sigma_t=float(detrend_sigma), time_axis=0, mode="reflect",
            chunk=int(chunk), device="auto", gpu_halo=int(gpu_halo),
            progress=True, desc="detrend"
        )
        if save_detrended:
            out_det = os.path.join(outdir, f"{base}_detrended_f32.tiff")
            write_tiff_stack(out_det, F, dtype="float32", bigtiff=True, compress=True)
            print(f"  -> saved {out_det}")
        # Use detrended stack as input for dF/F by default
        x = F

    # 4) dF/F (optional; GPU auto)
    if dff_pct and dff_win and dff_pct > 0 and dff_win > 0:
        win_eff = _ensure_odd(int(dff_win))
        print(f"[dF/F] pct={dff_pct} win={win_eff} | chunk={chunk}")
        dff = dff_percentile(
            x, p=float(dff_pct), win=win_eff, time_axis=0, mode="nearest",
            eps=1e-6, chunk=int(chunk), device="auto", progress=True
        )
        if save_dff:
            out_dff = os.path.join(outdir, f"{base}_dff_p{int(dff_pct)}_w{win_eff}_f32.tiff")
            write_tiff_stack(out_dff, dff, dtype="float32", bigtiff=True, compress=True)
            print(f"  -> saved {out_dff}")

    print(f"[OK] {path}")


# ------------------------------- CLI ---------------------------------------- #

def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Preprocessing pipeline: flat-field → detrend → dF/F (float32, GPU-auto).")
    ap.add_argument("--input", required=True, help="Glob for input TIFFs, e.g. D:/data/*pwrigid*.tif")
    ap.add_argument("--outdir", required=True, help="Output directory")

    # flat-field
    ap.add_argument("--flat-sigma", type=int, default=0, help="Spatial Gaussian sigma (pixels). 0=off")
    ap.add_argument("--save-flat", action="store_true", help="Save the estimated flat-field image")
    ap.add_argument("--save-ff", action="store_true", help="Save flat-field corrected stack")

    # detrend
    ap.add_argument("--detrend-sigma", type=float, default=0.0, help="Temporal Gaussian sigma (frames). 0=off")
    ap.add_argument("--save-detrended", action="store_true", help="Save detrended stack")

    # dF/F
    ap.add_argument("--dff-pct", type=float, default=0.0, help="Rolling baseline percentile (0=off)")
    ap.add_argument("--dff-win", type=int, default=0, help="Window (frames), will be made odd")
    ap.add_argument("--save-dff", action="store_true", help="Save dF/F stack")

    # performance
    ap.add_argument("--stride", type=int, default=1, help="Take every Nth frame before processing")
    ap.add_argument("--max-frames", type=int, default=0, help="0=keep all after stride")
    ap.add_argument("--chunk", type=int, default=64, help="Time chunk for detrend/dF/F")
    ap.add_argument("--gpu-halo", type=int, default=0, help="Temporal halo for GPU baseline (0=fastest)")

    args = ap.parse_args(list(argv) if argv is not None else None)

    files = sorted(glob.glob(args.input))
    if not files:
        raise SystemExit(f"No files match: {args.input}")

    print(f"[prep] files={len(files)} outdir={args.outdir}")
    for f in files:
        process_one(
            f, args.outdir,
            flat_sigma=(args.flat_sigma if args.flat_sigma > 0 else None),
            save_flat=args.save_flat, save_ff=args.save_ff,
            detrend_sigma=(args.detrend_sigma if args.detrend_sigma > 0 else None),
            save_detrended=args.save_detrended,
            dff_pct=(args.dff_pct if args.dff_pct > 0 else None),
            dff_win=(args.dff_win if args.dff_win > 0 else None),
            save_dff=args.save_dff,
            stride=args.stride,
            max_frames=args.max_frames,
            chunk=args.chunk,
            gpu_halo=args.gpu_halo,
        )


if __name__ == "__main__":
    main()
