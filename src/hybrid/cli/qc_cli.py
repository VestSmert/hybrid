# --- file: hybrid/cli/qc_cli.py ---

"""
QC map generator: Correlation + PNR.

Usage
-----
python -m hybrid.cli.qc_cli \
  --input "D:/data/*pwrigid*.tif" \
  --outdir "D:/data/_qc" \
  --stride 2 --max_frames 800 --crop 512 \
  --hp_window 15 \
  --norm none            # or: global | percentile
  --p_low 1.0 --p_high 99.5

Outputs
-------
- *_corr.npy, *_pnr.npy
- *_qc.png (mean, corr, pnr panels)
- summary.csv (filename, T,H,W, corr_mean, corr_p95, pnr_median, pnr_p95)
"""

import os
import csv
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

from hybrid.io.tif import read_tiff_stack
from hybrid.summary.qc_maps import correlation_image, pnr_image


def center_crop(arr, size=None):
    if size is None:
        return arr
    t, h, w = arr.shape
    hh, ww = min(size, h), min(size, w)
    y0 = max(0, h // 2 - hh // 2); y1 = y0 + hh
    x0 = max(0, w // 2 - ww // 2); x1 = x0 + ww
    return arr[:, y0:y1, x0:x1]


def save_qc_png(base_out, mean_im, corr_map, pnr_map):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].imshow(mean_im, cmap="gray"); axs[0].set_title("Mean image"); axs[0].axis("off")
    im1 = axs[1].imshow(corr_map, vmin=-0.2, vmax=1.0, cmap="magma")
    axs[1].set_title("Correlation image"); axs[1].axis("off"); fig.colorbar(im1, ax=axs[1], fraction=0.046)
    im2 = axs[2].imshow(pnr_map, cmap="magma")
    axs[2].set_title("PNR image"); axs[2].axis("off"); fig.colorbar(im2, ax=axs[2], fraction=0.046)
    plt.tight_layout()
    png_path = base_out + "_qc.png"
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def process_one(
    path: str,
    outdir: str,
    stride: int,
    max_frames: int | None,
    crop: int | None,
    hp_window: int,
    norm_mode: str,
    p_low: float,
    p_high: float,
):
    # read with chosen normalization mode (none/global/percentile)
    movie = read_tiff_stack(
        path,
        normalize=False,               # legacy flag off (we use normalize_mode explicitly)
        p_low=p_low,
        p_high=p_high,
        dtype=np.float32,
        normalize_mode=norm_mode,
    )

    # temporal downsample and crop for speed
    if stride > 1:
        movie = movie[::stride]
    if max_frames is not None:
        movie = movie[:max_frames]
    movie = center_crop(movie, crop)

    # QC maps
    corr_map = correlation_image(movie, time_axis=0)
    pnr_map = pnr_image(movie, time_axis=0, hp_window=hp_window)

    # save outputs
    base = os.path.join(outdir, os.path.splitext(os.path.basename(path))[0])
    os.makedirs(outdir, exist_ok=True)
    np.save(base + "_corr.npy", corr_map.astype(np.float32, copy=False))
    np.save(base + "_pnr.npy",  pnr_map.astype(np.float32, copy=False))
    png_path = save_qc_png(base, movie.mean(axis=0), corr_map, pnr_map)

    # summary stats
    inner = corr_map[1:-1, 1:-1]
    stats = {
        "file": os.path.basename(path),
        "T": int(movie.shape[0]),
        "H": int(movie.shape[1]),
        "W": int(movie.shape[2]),
        "corr_mean": float(np.nanmean(inner)),
        "corr_p95": float(np.nanpercentile(inner, 95)),
        "pnr_median": float(np.median(pnr_map)),
        "pnr_p95": float(np.percentile(pnr_map, 95)),
        "png": os.path.basename(png_path),
        "norm_mode": norm_mode,
        "p_low": float(p_low),
        "p_high": float(p_high),
        "stride": int(stride),
        "max_frames": int(max_frames) if max_frames is not None else None,
        "crop": int(crop) if crop is not None else None,
        "hp_window": int(hp_window),
    }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Glob, e.g. D:/data/*.tif")
    ap.add_argument("--outdir", required=True)

    # speed/size controls
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max_frames", type=int, default=800)
    ap.add_argument("--crop", type=int, default=512, help="Center crop size (None -> full)", nargs="?")

    # QC params
    ap.add_argument("--hp_window", type=int, default=15, help="Temporal HP window for PNR")

    # NEW: normalization controls
    ap.add_argument("--norm", dest="norm_mode",
                    choices=["none", "global", "percentile"],
                    default="none",
                    help="Normalization mode for reading TIFF (default: none)")
    ap.add_argument("--p_low", type=float, default=1.0, help="Lower percentile for 'percentile' mode")
    ap.add_argument("--p_high", type=float, default=99.5, help="Upper percentile for 'percentile' mode")

    args = ap.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        raise SystemExit(f"No files match: {args.input}")

    rows = []
    for f in files:
        print(f"[QC] {f}")
        rows.append(
            process_one(
                f,
                args.outdir,
                args.stride,
                args.max_frames,
                args.crop,
                args.hp_window,
                args.norm_mode,
                args.p_low,
                args.p_high,
            )
        )

    # write CSV summary
    csv_path = os.path.join(args.outdir, "summary.csv")
    with open(csv_path, "w", newline="") as fw:
        w = csv.DictWriter(fw, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] summary -> {csv_path}")


if __name__ == "__main__":
    main()
