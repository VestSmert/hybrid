# --- file: hybrid/cli/prep_cli.py ---

"""
Preprocessing pipeline runner.

Steps (configurable)
--------------------
1) Flat-field estimation & application
2) (Optional) DoG bandpass for registration helper only
3) Detrend (temporal Gaussian baseline)
4) dF/F (rolling percentile)

Usage
-----
python -m hybrid.cli.prep_cli \
  --input "D:/data/*.tif" --outdir "D:/data/_prep" \
  --flat_sigma 60 \
  --save_flat --save_flat_corrected \
  --detrend_sigma 100 \
  --dff_pct 10 --dff_win 301 \
  --save_detrended --save_dff

Outputs
-------
- *_flat.tiff (illumination field)
- *_ff_f32.tiff (flat-field corrected)
- *_detrended_f32.tiff
- *_dff_pct_p10_w301_f32.tiff (example naming)
"""

import os
import glob
import argparse
import numpy as np

from hybrid.io.tif import read_tiff_stack, write_tiff_stack
from hybrid.filters.flatfield import estimate_flatfield, apply_flatfield
from hybrid.preprocess.detrend import detrend
from hybrid.preprocess.dff import dff_percentile


def process_one(path, outdir, flat_sigma, save_flat, save_ff, detrend_sigma,
                save_detrended, dff_pct, dff_win, save_dff, stride, max_frames):
    movie = read_tiff_stack(path, normalize=False, dtype=np.float32)
    if stride > 1:
        movie = movie[::stride]
    if max_frames is not None:
        movie = movie[:max_frames]

    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(outdir, exist_ok=True)

    # 1) Flat-field
    flat = None
    if flat_sigma is not None:
        flat = estimate_flatfield(movie, sigma_px=float(flat_sigma))
        if save_flat:
            write_tiff_stack(os.path.join(outdir, base + "_flat.tiff"), flat[None, ...], dtype="float32")
        movie_ff = apply_flatfield(movie, flat, renormalize="none")
        if save_ff:
            write_tiff_stack(os.path.join(outdir, base + "_ff_f32.tiff"), movie_ff, dtype="float32")
    else:
        movie_ff = movie

    # 2) Detrend
    F_detr, F0 = detrend(movie_ff, sigma_t=float(detrend_sigma)) if detrend_sigma is not None else (movie_ff, None)
    if save_detrended and detrend_sigma is not None:
        write_tiff_stack(os.path.join(outdir, base + "_detrended_f32.tiff"), F_detr, dtype="float32")

    # 3) dF/F (rolling percentile)
    dff = None
    if dff_pct is not None and dff_win is not None:
        dff = dff_percentile(F_detr, p=float(dff_pct), win=int(dff_win), progress=True, device="auto")
        if save_dff:
            suffix = f"_dff_pct_p{int(dff_pct)}_w{int(dff_win)}_f32.tiff"
            write_tiff_stack(os.path.join(outdir, base + suffix), dff, dtype="float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Glob pattern, e.g. D:/data/*.tif")
    ap.add_argument("--outdir", required=True)

    # Flat-field
    ap.add_argument("--flat_sigma", type=float, default=60.0, help="Gaussian sigma in px (None to skip)", nargs="?")
    ap.add_argument("--save_flat", action="store_true")
    ap.add_argument("--save_flat_corrected", action="store_true", dest="save_ff")

    # Detrend
    ap.add_argument("--detrend_sigma", type=float, default=100.0, nargs="?")
    ap.add_argument("--save_detrended", action="store_true")

    # dF/F
    ap.add_argument("--dff_pct", type=float, default=10.0, nargs="?")
    ap.add_argument("--dff_win", type=int, default=301, nargs="?")
    ap.add_argument("--save_dff", action="store_true")

    # Common
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        raise SystemExit(f"No files match: {args.input}")

    for f in files:
        process_one(
            f, args.outdir,
            flat_sigma=args.flat_sigma, save_flat=args.save_flat, save_ff=args.save_ff,
            detrend_sigma=args.detrend_sigma, save_detrended=args.save_detrended,
            dff_pct=args.dff_pct, dff_win=args.dff_win, save_dff=args.save_dff,
            stride=args.stride, max_frames=args.max_frames
        )

if __name__ == "__main__":
    main()
