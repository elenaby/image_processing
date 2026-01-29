import os
import re
import math
import numpy as np
import cv2
from PIL import Image

try:
    import tifffile
except Exception:
    tifffile = None

import openslide

# -----------------------------
# User input (WSL path)
# -----------------------------
NDPI_PATH = r""
TILE_SIZE = 512
TISSUE_COVERAGE_MIN = 0.20
OUT_DIR = ""


def infer_channels_from_axes_shape(axes: str, shape: tuple) -> int:
    if axes and "C" in axes:
        c_idx = axes.index("C")
        return int(shape[c_idx])
    if len(shape) >= 3 and shape[-1] in (3, 4):
        return int(shape[-1])
    return 1


def print_slide_stats(ndpi_path: str) -> None:
    print("\n=== Slide stats ===")
    print(f"Path: {ndpi_path}")
    print(f"Exists: {os.path.exists(ndpi_path)}")
    if not os.path.exists(ndpi_path):
        raise FileNotFoundError(f"File does not exist: {ndpi_path}")

    # Try tifffile first (nice for series/channel introspection)
    if tifffile is not None:
        try:
            print("\n--- NDPI TIFF Series / Channel Stats (tifffile) ---")
            with tifffile.TiffFile(ndpi_path) as tf:
                series_list = list(tf.series)
                print(f"Number of series: {len(series_list)}")
                for i, s in enumerate(series_list):
                    axes = getattr(s, "axes", None)
                    shape = getattr(s, "shape", None)
                    dtype = getattr(s, "dtype", None)
                    ch = infer_channels_from_axes_shape(axes, shape) if shape is not None else None
                    print(f"  Series[{i}]: axes={axes}, shape={shape}, dtype={dtype}, inferred_channels={ch}")
        except Exception as e:
            print(f"(tifffile could not read series info; continuing with OpenSlide only)\nReason: {e}")

    # Always print OpenSlide info
    print("\n--- OpenSlide Info ---")
    slide = openslide.OpenSlide(ndpi_path)
    print(f"OpenSlide vendor: {slide.properties.get('openslide.vendor', 'unknown')}")
    print(f"Level count: {slide.level_count}")
    for lvl in range(slide.level_count):
        w, h = slide.level_dimensions[lvl]
        ds = slide.level_downsamples[lvl]
        print(f"  Level {lvl}: {w}x{h}, downsample={ds}")
    slide.close()
    print("===============================================\n")


def build_tissue_mask(slide: openslide.OpenSlide, thumb_max_dim: int = 2048):
    w0, h0 = slide.level_dimensions[0]
    scale = max(w0, h0) / float(thumb_max_dim)
    thumb_w = max(1, int(round(w0 / scale)))
    thumb_h = max(1, int(round(h0 / scale)))

    thumb = slide.get_thumbnail((thumb_w, thumb_h))  # PIL RGB
    thumb_np = np.array(thumb)

    gray = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thr = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue = (thr == 0).astype(np.uint8)  # darker regions -> tissue

    # Morphology
    k = max(3, int(round(min(thumb_w, thumb_h) * 0.005)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, kernel)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, kernel)

    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tissue, connectivity=8)
    min_area = int(0.001 * thumb_w * thumb_h)
    tissue2 = np.zeros_like(tissue)
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_area:
            tissue2[labels == lab] = 1

    ratio = w0 / float(thumb_w)  # level0 pixels per thumb pixel
    return tissue2, ratio, (thumb_w, thumb_h)


def save_tiles(slide_path: str, out_dir: str, tile_size: int = 512, coverage_min: float = 0.2):
    os.makedirs(out_dir, exist_ok=True)

    slide = openslide.OpenSlide(slide_path)
    w0, h0 = slide.level_dimensions[0]

    mask_thumb, ratio, (tw, th) = build_tissue_mask(slide, thumb_max_dim=2048)

    def tile_tissue_coverage(x0: int, y0: int) -> float:
        x1 = min(w0, x0 + tile_size)
        y1 = min(h0, y0 + tile_size)

        tx0 = int(math.floor(x0 / ratio))
        ty0 = int(math.floor(y0 / ratio))
        tx1 = int(math.ceil(x1 / ratio))
        ty1 = int(math.ceil(y1 / ratio))

        tx0 = max(0, min(tw - 1, tx0))
        ty0 = max(0, min(th - 1, ty0))
        tx1 = max(0, min(tw, tx1))
        ty1 = max(0, min(th, ty1))

        region = mask_thumb[ty0:ty1, tx0:tx1]
        return float(region.mean()) if region.size else 0.0

    base = os.path.basename(out_dir.rstrip(os.sep))
    saved, skipped = 0, 0

    for y in range(0, h0, tile_size):
        for x in range(0, w0, tile_size):
            if tile_tissue_coverage(x, y) < coverage_min:
                skipped += 1
                continue

            tile = slide.read_region((x, y), 0, (tile_size, tile_size))  # RGBA
            tile_np = np.array(tile)
            if tile_np.ndim == 3 and tile_np.shape[-1] == 4:
                tile_np = tile_np[:, :, :3]

            fn = os.path.join(out_dir, f"{base}_x{x}_y{y}.tif")
            Image.fromarray(tile_np).save(fn, format="TIFF")
            saved += 1

    slide.close()
    print(f"Done. Saved tiles: {saved} | Skipped (low tissue): {skipped} | Output: {out_dir}")


if __name__ == "__main__":
    print_slide_stats(NDPI_PATH)
    save_tiles(NDPI_PATH, OUT_DIR, TILE_SIZE, TISSUE_COVERAGE_MIN)