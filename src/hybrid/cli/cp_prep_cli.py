# --- file: hybrid/cli/cp_prep_cli.py ---
"""
Copy & prep workflow:
- (Optional) PW-rigid motion correction via hybrid.cli.mc_cli.run_mc
- Optional flat-field, detrend, and dF/F
- Optional DoG/PNR + simple contrast previews (u8/PNG)
- Saves float32 outputs; memmap-friendly when feasible.

This tool is intentionally opinionated and geared for quick, reproducible
preprocessing runs on single files (from notebooks or CLI).
"""

from __future__ import annotations
import os
from typing import Iterable, Dict, Any, Optional

import numpy as np

# Public IO API
from hybrid.io import read_tiff_stack, write_tiff_stack

# Preprocess building blocks
from hybrid.preprocess import (
    gaussian_baseline,
    detrend,
    dff_percentile,
    dff_from_baseline,
)

# QC / viz bits (only if present in your tree; used for previews/stats)
try:
    from hybrid.summary.qc_maps import correlation_image, pnr_image
except Exception:  # pragma: no cover
    correlation_image = None  # type: ignore
    pnr_image = None  # type: ignore

# Bring motion-correction runner (new name!)
try:
    # NOTE: run_one was renamed to run_mc
    from hybrid.cli.mc_cli import run_mc  # <- correct entry point now
except Exception:  # pragma: no cover
    run_mc = None  # type: ignore


def _basename_noext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _maybe_mc(
    path: str,
    outdir: str,
    *,
    run_mc_flag: bool,
    tiles: tuple[int, int],
    overlap: int,
    upsample: int,
    device: str,
    use_dog: bool,
    stride: int,
    max_frames: int,
) -> str:
    """Optionally run MC and return the path to the movie to continue with."""
    if not run_mc_flag:
        return path
    if run_mc is None:
        raise RuntimeError("run_mc=True but hybrid.cli.mc_cli.run_mc is not available.")  # :contentReference[oaicite:0]{index=0}

    os.makedirs(outdir, exist_ok=True)
    print("[CP-PREP] Motion correction (pw-rigid)…")
    run_mc(
        path=path,
        outdir=outdir,
        tiles=tiles,
        overlap=overlap,
        upsample=upsample,
        device=device,
        use_dog=use_dog,
        stride=stride,
        max_frames=max_frames,
    )
    base = _basename_noext(path)
    return os.path.join(outdir, f"{base}_pwrigid_f32.tiff")


def run_cp_prep(
    path: str,
    outdir: str,
    *,
    # read/normalize
    normalize_mode: str = "none",
    p_low: float = 1.0,
    p_high: float = 99.5,
    stride: int = 1,
    max_frames: Optional[int] = None,
    # motion correction block
    run_mc: bool = False,
    mc_tiles: tuple[int, int] = (4, 4),
    mc_overlap: int = 32,
    mc_upsample: int = 10,
    mc_device: str = "auto",
    mc_use_dog: bool = False,
    mc_stride: int = 1,
    # flat/detrend/dff
    flat_sigma: Optional[float] = None,
    save_flat: bool = False,
    detrend_sigma: Optional[float] = None,
    dff_pct: Optional[float] = None,
    dff_win: int = 301,
    # DoG/PNR preview & correlation weighting
    dog_sigma_low: float = 1.6,
    dog_sigma_high: float = 3.2,
    hp_window: int = 51,
    corr_weight: float = 0.5,
    # previews / outputs
    save_u8: bool = True,
    clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_tile: int = 16,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    no_save_f32: bool = False,
) -> Dict[str, Any]:
    """
    High-level, single-file “copy & prep” pipeline.
    Returns a dict with file paths and basic QC stats (when available).
    """
    os.makedirs(outdir, exist_ok=True)
    base = _basename_noext(path)
    stats: Dict[str, Any] = {"file": path}

    # 0) Optional motion correction
    in_path = _maybe_mc(
        path,
        outdir,
        run_mc_flag=run_mc,
        tiles=mc_tiles,
        overlap=mc_overlap,
        upsample=mc_upsample,
        device=mc_device,
        use_dog=mc_use_dog,
        stride=mc_stride,
        max_frames=(0 if max_frames is None else max_frames),
    )

    # 1) Read, normalization, temporal trimming
    movie = read_tiff_stack(
        in_path,
        normalize_mode=normalize_mode,
        p_low=p_low,
        p_high=p_high,
        dtype=None,
    )
    movie = movie[:: max(1, int(stride))]
    if max_frames is not None and movie.shape[0] > max_frames:
        movie = movie[:max_frames]
    T, H, W = movie.shape
    stats["shape_in"] = (T, H, W)
    stats["dtype_in"] = str(movie.dtype)

    # 2) Flat-field (optional, simple Gaussian divider)
    if flat_sigma is not None and flat_sigma > 0:
        from scipy.ndimage import gaussian_filter
        print(f"[flat] sigma={flat_sigma}")
        # estimate per-frame illumination field and divide
        flat = gaussian_filter(movie.astype(np.float32, copy=False), sigma=(0.0, flat_sigma, flat_sigma))
        flat = np.maximum(flat, 1e-6)
        ff = movie.astype(np.float32, copy=False) / flat
        movie = ff
        if save_flat:
            out_flat = os.path.join(outdir, f"{base}_flat_f32.tiff")
            write_tiff_stack(out_flat, flat.astype(np.float32, copy=False), dtype="float32", bigtiff=True, compress=True)
            stats["flat_f32"] = out_flat

    # 3) Detrend (optional)
    if detrend_sigma is not None and detrend_sigma > 0:
        print(f"[detrend] sigma_t={detrend_sigma}")
        F, F0 = detrend(
            movie.astype(np.float32, copy=False),
            sigma_t=float(detrend_sigma),
            time_axis=0,
            mode="reflect",
            chunk=64,
            device="auto",
            gpu_halo=0,
            progress=True,
            desc="detrend",
        )
        movie = F
        if not no_save_f32:
            out_f0 = os.path.join(outdir, f"{base}_baseline_f32.tiff")
            write_tiff_stack(out_f0, F0.astype(np.float32, copy=False), dtype="float32", bigtiff=True, compress=True)
            stats["baseline_f32"] = out_f0

    # 4) dF/F (optional)
    if dff_pct is not None and dff_pct > 0:
        print(f"[dF/F] pct={dff_pct} win={dff_win}")
        F_dff = dff_percentile(
            movie.astype(np.float32, copy=False),
            p=float(dff_pct),
            win=int(dff_win),
            time_axis=0,
            mode="nearest",
            eps=1e-6,
            chunk=64,
            device="auto",
            progress=True,
        )
        movie = F_dff
        if not no_save_f32:
            out_dff = os.path.join(outdir, f"{base}_dff_p{int(dff_pct)}_w{int(dff_win if dff_win % 2 else dff_win + 1)}_f32.tiff")
            write_tiff_stack(out_dff, movie.astype(np.float32, copy=False), dtype="float32", bigtiff=True, compress=True)
            stats["dff_f32"] = out_dff

    # 5) QC maps & previews (optional; requires summary.qc_maps)
    if correlation_image is not None and pnr_image is not None:
        try:
            corr = correlation_image(movie, time_axis=0)
            pnr = pnr_image(movie, time_axis=0, hp_window=hp_window)
            stats["corr_mean"] = float(np.nanmean(corr))
            stats["pnr_p95"] = float(np.percentile(pnr, 95))
        except Exception as e:  # pragma: no cover
            print(f"[warn] QC maps failed: {e!r}")

    # 6) Optional u8/PNG previews (contrast tools are intentionally minimal)
    if save_u8:
        def _to_u8(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
            lo_v, hi_v = np.percentile(x, [lo, hi])
            x = np.clip((x - lo_v) / max(1e-6, (hi_v - lo_v)), 0, 1)
            return (x * 255.0 + 0.5).astype(np.uint8)

        mean_im = movie.mean(axis=0).astype(np.float32, copy=False)
        u8 = _to_u8(mean_im, clip_low, clip_high)

        if clahe:
            try:
                import cv2  # OpenCV CLAHE (optional)
                clahe_op = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(int(clahe_tile), int(clahe_tile)))
                u8 = clahe_op.apply(u8)
            except Exception:  # pragma: no cover
                pass

        # Save u8 TIFF + PNG preview
        out_u8 = os.path.join(outdir, f"{base}_cp_u8.tiff")
        write_tiff_stack(out_u8, u8[None, ...], dtype="uint8", bigtiff=False, compress=True)
        stats["cp_u8"] = out_u8

        try:
            from imageio.v2 import imwrite as imsave_png  # lightweight, optional
            out_png = os.path.join(outdir, f"{base}_cp_preview.png")
            imsave_png(out_png, u8)
            stats["cp_png"] = out_png
        except Exception:  # pragma: no cover
            pass

    # 7) Save float32 copy of the prepped signal (unless suppressed)
    if not no_save_f32:
        out_f32 = os.path.join(outdir, f"{base}_cp_f32.tiff")
        write_tiff_stack(out_f32, movie.astype(np.float32, copy=False), dtype="float32", bigtiff=True, compress=True)
        stats["cp_f32"] = out_f32

    return stats


# Optional CLI wrapper (kept minimal on purpose)
def main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Copy & prep a single TIFF with optional MC and previews.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--run-mc", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    stats = run_cp_prep(args.input, args.outdir, run_mc=args.run_mc)
    print("Done:", {k: stats[k] for k in sorted(stats) if isinstance(stats[k], (str, float, int))})


if __name__ == "__main__":  # pragma: no cover
    main()
