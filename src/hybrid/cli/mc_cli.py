# --- file: hybrid/cli/mc_cli.py ---
"""
Batch piecewise-rigid motion correction (float32 outputs, memmap-friendly).

Features
--------
- OOM-safe input: prefer memmap; for compressed/tilled TIFFs fall back to a
  streaming "staging" into a disk-backed memmap.
- PW-rigid MC with GPU auto-selection (delegated to pwrigid_movie).
- Optional DoG bandpass for robust shift estimation (applied only to the
  registration pathway; shifts are applied to the original frames).
- Float32 outputs, BigTIFF when needed, no compression (memmap-friendly).

Usage
-----
python -m hybrid.cli.mc_cli ^
  --input "D:/data/*.tif" ^
  --outdir "D:/data/_mc" ^
  --tiles 4 4 --overlap 32 --upsample 10 ^
  --device auto --dog-for-registration ^
  --stride 1 --max-frames 0
"""

from __future__ import annotations
import os
import glob
import argparse
import numpy as np
from typing import Iterable, Tuple

from tifffile import TiffFile

# Public IO & MC APIs
from hybrid.io import read_tiff_stack, write_tiff_stack, iter_tiff_blocks
from hybrid.mc import pwrigid_movie

# Optional DoG helper (if available in your tree)
try:
    from hybrid.filters.dog import dog_bandpass
except Exception:
    dog_bandpass = None  # type: ignore


# ------------------------- helpers (staging & paths) ------------------------- #

def _basename_noext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


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


# ------------------------------- core --------------------------------------- #

def run_mc(
    path: str,
    outdir: str,
    *,
    tiles: tuple[int, int],
    overlap: int,
    upsample: int,
    device: str = "auto",            # auto/cpu/gpu — passed to pwrigid_movie
    use_dog: bool = False,
    stride: int = 1,
    max_frames: int = 0,             # 0 = keep all
) -> None:
    os.makedirs(outdir, exist_ok=True)
    base = _basename_noext(path)

    # Safe read: prefer memmap; stage if the TIFF is compressed/tilled & large
    try:
        movie = read_tiff_stack(path, normalize_mode="none", dtype=None)
    except (MemoryError, ValueError) as e:
        print(f"[read] {e} → staging to disk-memmap…")
        movie, (T, H, W), src_dtype, stage_path = _stage_tiff_to_memmap(path, block=max(32, 64))
        print(f"[read] staged memmap ready: {stage_path}")

    # Downsample & cap frames (if requested)
    movie = movie[:: max(1, int(stride))]
    if max_frames and movie.shape[0] > max_frames:
        movie = movie[:max_frames]

    # Optional DoG for registration only (robust shifts)
    dog_reg = None
    if use_dog:
        if dog_bandpass is None:
            print("[warn] dog_bandpass is not available; continuing without DoG.")
        else:
            # Note: this allocates a float32 copy; consider stride/cap if memory is tight.
            dog_reg = dog_bandpass(movie.astype(np.float32, copy=False), sigma_low=1.2, sigma_high=14.0)

    # Run MC (MC always works in float32)
    res = pwrigid_movie(
        movie=movie.astype(np.float32, copy=False),
        template=None,
        tiles=tiles,
        overlap=overlap,
        upsample=upsample,
        dog_for_registration=dog_reg,
        progress=True,
        device=device,
    )

    # Save float32, uncompressed (memmap-friendly)
    out_name = f"{base}_pwrigid_f32.tiff"
    out_path = os.path.join(outdir, out_name)
    write_tiff_stack(out_path, res.corrected, dtype="float32", bigtiff=True, compress=False)
    print(f"[OK] {path} -> {out_path} | mean_corr_after={res.mean_corr_after:.4f}")


# ------------------------------- CLI ---------------------------------------- #

def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Batch piecewise-rigid motion correction (float32, memmap-friendly).")
    ap.add_argument("--input", required=True, help="Glob for input TIFFs, e.g. D:/data/*.tif")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--tiles", nargs=2, type=int, default=[4, 4], help="Tile grid: gy gx")
    ap.add_argument("--overlap", type=int, default=32, help="Full overlap (pixels)")
    ap.add_argument("--upsample", type=int, default=10, help="Subpixel factor")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="Computation device for MC")
    ap.add_argument("--dog-for-registration", dest="use_dog", action="store_true", help="Use DoG-filtered frames for shift estimation")
    ap.add_argument("--stride", type=int, default=1, help="Take every Nth frame before MC")
    ap.add_argument("--max-frames", type=int, default=0, help="0=keep all frames after stride")
    args = ap.parse_args(list(argv) if argv is not None else None)

    files = sorted(glob.glob(args.input))
    if not files:
        raise SystemExit(f"No files match: {args.input}")

    print(f"[mc] files={len(files)} outdir={args.outdir}")
    for f in files:
        run_mc(
            f,
            args.outdir,
            tiles=(args.tiles[0], args.tiles[1]),
            overlap=args.overlap,
            upsample=args.upsample,
            device=args.device,
            use_dog=args.use_dog,
            stride=args.stride,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
