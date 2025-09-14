# --- file: hybrid/cli/cp_prep_cli.py ---
"""
Cellpose-ready input generator.

Builds a single 2D image per movie that is well-suited for Cellpose segmentation.
Each step is optional (pass None to skip) and all parameters are recorded to
summary.csv for easy grid-searching.

Pipeline
--------
RAW (TYX)
  └─ [optional] motion correction via mc_cli.run_one
       └─ read_tiff_stack(normalize_mode=...)
            └─ [optional] flat-field (estimate+apply)
                 └─ [optional] detrend
                      └─ [optional] dF/F (rolling percentile)
                           └─ [optional] DoG per frame
                                └─ PNR-like projection (max z-score)
                                     └─ [optional] correlation weighting
                                          └─ stretch + [optional] CLAHE (u8)

Main output (default): *_cp_input_f32.tiff  (float32, 2D)
Optional outputs     : *_cp_input_u8.tiff   (uint8, 2D, only if --save_u8)
                       *_cp_prep.png        (QC panel)
                       summary.csv          (one row per file)

"""

from __future__ import annotations

import os
import csv
import glob
import argparse
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt

# I/O
from hybrid.io.tif import read_tiff_stack, write_tiff_stack

# your filters / preprocess
from hybrid.filters.flatfield import estimate_flatfield, apply_flatfield
from hybrid.filters.dog import dog_bandpass
from hybrid.preprocess.detrend import detrend
from hybrid.preprocess.dff import dff_percentile
from hybrid.summary.qc_maps import pnr_image, correlation_image
from hybrid.filters.contrast import clahe_u8

# optional MC via your tested CLI runner
try:
    from hybrid.cli.mc_cli import run_one as mc_run_one
except Exception:
    mc_run_one = None


# ------------------------------- helpers ------------------------------------ #

def _to_uint8_for_vis(im: np.ndarray, clip_low: float, clip_high: float,
                      use_clahe: bool, clahe_clip: float, clahe_tile: int) -> np.ndarray:
    """Convert float image to uint8 for visualization (and optional u8 output)."""
    imf = im.astype(np.float32, copy=False)
    lo, hi = np.percentile(imf, [clip_low, clip_high])
    if not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(imf))
        hi = float(np.max(imf)) if float(np.max(imf)) > lo else lo + 1.0
    imf = np.clip((imf - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    im8 = (imf * 255.0).astype(np.uint8, copy=False)
    if use_clahe:
        im8 = clahe_u8(im8, clip_limit=float(clahe_clip), tile=int(clahe_tile))
    return im8


def _qc_png(base_out: str, raw_mean: np.ndarray, pnr: np.ndarray, cp_vis: np.ndarray) -> str:
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].imshow(raw_mean, cmap="gray"); axs[0].set_title("Raw mean"); axs[0].axis("off")
    im1 = axs[1].imshow(pnr, cmap="magma"); axs[1].set_title("PNR map"); axs[1].axis("off")
    fig.colorbar(im1, ax=axs[1], fraction=0.046)
    axs[2].imshow(cp_vis, cmap="gray"); axs[2].set_title("Cellpose input (u8 view)"); axs[2].axis("off")
    plt.tight_layout()
    png = base_out + "_cp_prep.png"
    plt.savefig(png, dpi=150)
    plt.close(fig)
    return png


# ------------------------------- core --------------------------------------- #

def process_one(
    path: str,
    outdir: str,
    *,
    # read/normalize
    normalize_mode: str = "none",   # {"none","global","percentile"}
    p_low: float = 1.0,
    p_high: float = 99.5,
    stride: int = 1,
    max_frames: Optional[int] = None,

    # motion correction (ON by default)
    run_mc: bool = True,
    mc_tiles: Optional[List[int]] = None,   # e.g., [4,4]
    mc_overlap: int = 32,
    mc_upsample: int = 0,
    mc_device: str = "auto",
    mc_use_dog: bool = True,
    mc_stride: int = 1,

    # flat-field
    flat_sigma: Optional[float] = 60.0,     # None → skip
    save_flat: bool = False,

    # detrend + dFF
    detrend_sigma: Optional[float] = 100.0, # None → skip
    dff_pct: Optional[float] = 10.0,        # None → skip
    dff_win: Optional[int] = 301,           # used when dff_pct not None

    # DoG + PNR
    dog_sigma_low: Optional[float] = 1.6,   # None → skip DoG
    dog_sigma_high: Optional[float] = 3.2,
    hp_window: int = 51,

    # correlation weighting
    corr_weight: Optional[float] = 0.5,     # None or 0 → skip

    # visualization/u8 export
    save_u8: bool = False,
    clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_tile: int = 16,
    clip_low: float = 1.0,
    clip_high: float = 99.0,

    # f32 output control
    no_save_f32: bool = False,
) -> dict:
    """
    Prepare a single Cellpose-ready 2D image from a (T,Y,X) movie.

    Returns 'stats' dict with paths and parameters for CSV.
    """
    in_path = path
    mc_out_path = ""

    # (0) motion correction (default ON)
    if run_mc:
        if mc_run_one is None:
            raise RuntimeError("run_mc=True but hybrid.cli.mc_cli.run_one is not available.")
        os.makedirs(outdir, exist_ok=True)
        print("[CP-PREP] Motion correction (pw-rigid)…")
        mc_run_one(
            path=in_path,
            outdir=outdir,
            tiles=tuple(mc_tiles) if mc_tiles else (4,4),
            overlap=int(mc_overlap),
            upsample=int(mc_upsample),
            device=str(mc_device),
            use_dog=bool(mc_use_dog),
            stride=int(mc_stride),
            max_frames=max_frames,
        )
        produced = [f for f in os.listdir(outdir) if f.endswith("_pwrigid_f32.tiff")]
        if not produced:
            raise RuntimeError("MC produced no *_pwrigid_f32.tiff.")
        mc_out_path = os.path.join(outdir, produced[0])
        in_path = mc_out_path
        print(f"[CP-PREP] MC output: {os.path.basename(in_path)}")

    # (1) read stack with unified normalization
    movie = read_tiff_stack(
        in_path,
        normalize=False,  # legacy flag off
        p_low=p_low, p_high=p_high,
        dtype=np.float32,
        normalize_mode=normalize_mode,
    )
    if stride > 1:
        movie = movie[::stride]
    if max_frames is not None:
        movie = movie[:max_frames]

    base = os.path.splitext(os.path.basename(in_path))[0]
    os.makedirs(outdir, exist_ok=True)
    base_out = os.path.join(outdir, base)

    # (2) flat-field (optional)
    flat_sigma_used = flat_sigma
    if flat_sigma is not None:
        flat = estimate_flatfield(movie, sigma_px=float(flat_sigma))
        if save_flat:
            write_tiff_stack(base_out + "_flat.tiff", flat[None, ...], dtype="float32")
        movie_ff = apply_flatfield(movie, flat, renormalize="none")
    else:
        movie_ff = movie
        flat_sigma_used = None

    # (3) detrend (optional)
    detrend_sigma_used = detrend_sigma
    if detrend_sigma is not None:
        F_detr, _ = detrend(movie_ff, sigma_t=float(detrend_sigma))
    else:
        F_detr = movie_ff
        detrend_sigma_used = None

    # (4) dF/F (optional)
    dff_pct_used, dff_win_used = dff_pct, dff_win
    if dff_pct is not None and dff_win is not None:
        dff = dff_percentile(F_detr, p=float(dff_pct), win=int(dff_win),
                             progress=False, device="auto")
    else:
        dff = F_detr
        dff_pct_used, dff_win_used = None, None

    # (5) DoG per frame (optional; uses your filters.dog.dog_bandpass)
    dog_low_used, dog_high_used = dog_sigma_low, dog_sigma_high
    work = dff
    if dog_sigma_low is not None and dog_sigma_high is not None:
        dog = np.empty_like(dff, dtype=np.float32)
        for t in range(dff.shape[0]):
            dog[t] = dog_bandpass(dff[t],
                                  sigma_low=float(dog_sigma_low),
                                  sigma_high=float(dog_sigma_high))
        work = dog
    else:
        dog_low_used = dog_high_used = None

    # (6) PNR-like projection
    pnr = pnr_image(work, time_axis=0, hp_window=int(hp_window))

    # (7) optional correlation weighting
    corr_weight_used = corr_weight
    if corr_weight is not None and float(corr_weight) != 0.0:
        C = correlation_image(movie_ff, time_axis=0)
        w = np.clip(float(corr_weight), 0.0, 1.0)
        cp_f32 = pnr * (1.0 - w) + pnr * w * C
    else:
        cp_f32 = pnr
        corr_weight_used = None

    # (8) save outputs
    if not no_save_f32:
        write_tiff_stack(base_out + "_cp_input_f32.tiff", cp_f32[None, ...], dtype="float32")

    # build a visualization/u8 version (stretch+optional CLAHE)
    cp_u8 = _to_uint8_for_vis(cp_f32, clip_low=clip_low, clip_high=clip_high,
                              use_clahe=bool(clahe), clahe_clip=float(clahe_clip), clahe_tile=int(clahe_tile))

    if save_u8:
        # write u8; keep as uint16 container to stay consistent with other TIFFs
        write_tiff_stack(base_out + "_cp_input_u8.tiff", cp_u8[None, ...], dtype="uint16", scale=None)

    png_path = _qc_png(base_out, movie.mean(axis=0), pnr, cp_u8)

    # (9) quick stats + protocol
    inner = pnr[1:-1, 1:-1]
    stats = dict(
        file=os.path.basename(path),
        used_mc=bool(run_mc),
        mc_out=os.path.basename(mc_out_path) if mc_out_path else "",
        normalize_mode=str(normalize_mode),
        p_low=float(p_low), p_high=float(p_high),
        stride=int(stride),
        max_frames=int(max_frames) if max_frames is not None else "",
        T=int(movie.shape[0]), H=int(movie.shape[1]), W=int(movie.shape[2]),
        flat_sigma=flat_sigma_used if flat_sigma_used is not None else "",
        detrend_sigma=detrend_sigma_used if detrend_sigma_used is not None else "",
        dff_pct=dff_pct_used if dff_pct_used is not None else "",
        dff_win=dff_win_used if dff_win_used is not None else "",
        dog_sigma_low=dog_low_used if dog_low_used is not None else "",
        dog_sigma_high=dog_high_used if dog_high_used is not None else "",
        hp_window=int(hp_window),
        corr_weight=corr_weight_used if corr_weight_used is not None else "",
        clahe=bool(clahe),
        clahe_clip=float(clahe_clip) if clahe else "",
        clahe_tile=int(clahe_tile) if clahe else "",
        stretch_low=float(clip_low),
        stretch_high=float(clip_high),
        pnr_mean=float(np.nanmean(inner)),
        pnr_p95=float(np.nanpercentile(inner, 95)),
        cp_png=os.path.basename(png_path),
        cp_u8=os.path.basename(base_out + "_cp_input_u8.tiff") if save_u8 else "",
        cp_f32=os.path.basename(base_out + "_cp_input_f32.tiff") if not no_save_f32 else "",
    )
    return stats


def _int_pair(s: str) -> List[int]:
    parts = s.replace(",", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected two integers, e.g. '4 4'")
    return [int(parts[0]), int(parts[1])]


def main():
    ap = argparse.ArgumentParser(description="Prepare Cellpose-ready 2D inputs from TIFF movies.")

    ap.add_argument("--input", required=True, help="Glob, e.g. D:/data/*.tif")
    ap.add_argument("--outdir", required=True)

    # normalization & slicing
    ap.add_argument("--normalize_mode", default="none", choices=["none", "global", "percentile"])
    ap.add_argument("--p_low", type=float, default=1.0)
    ap.add_argument("--p_high", type=float, default=99.5)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=None)

    # motion correction (ON by default)
    ap.add_argument("--no_mc", action="store_true", help="Disable motion correction")
    ap.add_argument("--mc_tiles", type=_int_pair, default=None, help="e.g. '4 4'")
    ap.add_argument("--mc_overlap", type=int, default=32)
    ap.add_argument("--mc_upsample", type=int, default=0)
    ap.add_argument("--mc_device", type=str, default="auto")
    ap.add_argument("--mc_use_dog", action="store_true")
    ap.add_argument("--mc_stride", type=int, default=1)

    # flat-field / detrend / dFF
    ap.add_argument("--flat_sigma", type=float, default=60.0, nargs="?")
    ap.add_argument("--save_flat", action="store_true")
    ap.add_argument("--detrend_sigma", type=float, default=100.0, nargs="?")
    ap.add_argument("--dff_pct", type=float, default=10.0, nargs="?")
    ap.add_argument("--dff_win", type=int, default=301, nargs="?")

    # DoG + PNR
    ap.add_argument("--dog_sigma_low", type=float, default=1.6, nargs="?")
    ap.add_argument("--dog_sigma_high", type=float, default=3.2, nargs="?")
    ap.add_argument("--hp_window", type=int, default=51)

    # correlation weighting
    ap.add_argument("--corr_weight", type=float, default=0.5, nargs="?")

    # visualization / u8 export
    ap.add_argument("--save_u8", action="store_true")
    ap.add_argument("--no_clahe", action="store_true")
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_tile", type=int, default=16)
    ap.add_argument("--clip_low", type=float, default=1.0)
    ap.add_argument("--clip_high", type=float, default=99.0)

    # f32 output control
    ap.add_argument("--no_save_f32", action="store_true")

    args = ap.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        raise SystemExit(f"No files match: {args.input}")
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for f in files:
        print(f"[CP-PREP] {f}")
        rows.append(process_one(
            f, args.outdir,
            normalize_mode=args.normalize_mode, p_low=args.p_low, p_high=args.p_high,
            stride=args.stride, max_frames=args.max_frames,
            run_mc=(not args.no_mc), mc_tiles=args.mc_tiles, mc_overlap=args.mc_overlap,
            mc_upsample=args.mc_upsample, mc_device=args.mc_device,
            mc_use_dog=args.mc_use_dog, mc_stride=args.mc_stride,
            flat_sigma=args.flat_sigma, save_flat=args.save_flat,
            detrend_sigma=args.detrend_sigma, dff_pct=args.dff_pct, dff_win=args.dff_win,
            dog_sigma_low=args.dog_sigma_low, dog_sigma_high=args.dog_sigma_high, hp_window=args.hp_window,
            corr_weight=args.corr_weight,
            save_u8=args.save_u8,
            clahe=(not args.no_clahe), clahe_clip=args.clahe_clip, clahe_tile=args.clahe_tile,
            clip_low=args.clip_low, clip_high=args.clip_high,
            no_save_f32=args.no_save_f32,
        ))

    csv_path = os.path.join(args.outdir, "summary.csv")
    with open(csv_path, "w", newline="") as fw:
        w = csv.DictWriter(fw, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] summary -> {csv_path}")


if __name__ == "__main__":
    main()
    