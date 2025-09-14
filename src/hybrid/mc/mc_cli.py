# --- file: hybrid/cli/mc_cli.py ---

"""
Batch piecewise-rigid motion correction.

Usage
-----
python -m hybrid.cli.mc_cli \
  --input "D:/data/*.tif" \
  --outdir "D:/data/_mc" \
  --tiles 4 4 --overlap 32 --upsample 10 \
  --device auto --dog_for_registration

Notes
-----
- Outputs are float32 TIFFs with suffix "_pwrigid_f32".
- If --dog_for_registration is set, shifts are estimated on DoG-filtered frames
  but applied to the original movie.
"""

import os
import glob
import argparse
import numpy as np

from hybrid.io.tif import read_tiff_stack, write_tiff_stack
from hybrid.filters.dog import dog_bandpass
from hybrid.mc.pwrigid import pwrigid_movie


def run_one(path, outdir, tiles, overlap, upsample, device, use_dog, stride, max_frames):
    movie = read_tiff_stack(path, normalize=False, dtype=np.float32)
    if stride > 1:
        movie = movie[::stride]
    if max_frames is not None:
        movie = movie[:max_frames]

    dog_reg = None
    if use_dog:
        dog_reg = dog_bandpass(movie, sigma_low=1.2, sigma_high=14.0)

    res = pwrigid_movie(
        movie,
        tiles=tiles,
        overlap=overlap,
        upsample=upsample,
        dog_for_registration=dog_reg,
        progress=True,
        device=device,
    )
    base = os.path.splitext(os.path.basename(path))[0] + "_pwrigid_f32.tiff"
    out_path = os.path.join(outdir, base)
    os.makedirs(outdir, exist_ok=True)
    write_tiff_stack(out_path, res.corrected, dtype="float32", bigtiff=True, compress=False)
    print(f"[OK] {path} -> {out_path} | mean_corr_after={res.mean_corr_after:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Glob pattern for input TIFFs, e.g. D:/data/*.tif")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--tiles", nargs=2, type=int, default=[4, 4], help="Tile grid gy gx")
    ap.add_argument("--overlap", type=int, default=32, help="Full overlap (pixels)")
    ap.add_argument("--upsample", type=int, default=10, help="Subpixel factor")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    ap.add_argument("--dog_for_registration", action="store_true", dest="use_dog")
    ap.add_argument("--stride", type=int, default=1, help="Take every Nth frame")
    ap.add_argument("--max_frames", type=int, default=None, help="Cap frames per movie")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        raise SystemExit(f"No files match: {args.input}")

    for f in files:
        run_one(
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
