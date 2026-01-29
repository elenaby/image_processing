"""
Seeded watershed membrane segmentation from point centers (TSV),
with NO MERGING between instances ensured by:
  - one unique marker label per point (no CC labeling on dilated masks)
  - watershed_line=True

Also handles RGBA images by converting them to RGB before rgb2hed.

Dependencies:
  pip install numpy scipy scikit-image imageio matplotlib
"""

import os
import csv
import numpy as np
import warnings
from skimage import io, color, filters, morphology, segmentation, exposure, util
import matplotlib.pyplot as plt


# -----------------------------
# Image utilities
# -----------------------------
def ensure_rgb(image):
    """
    Ensure image is RGB (H,W,3).
    Handles:
      - RGBA (H,W,4): drops alpha
      - grayscale (H,W): converts to RGB
      - c==1: expands to RGB
    """
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)

    if image.ndim == 3:
        c = image.shape[2]
        if c == 3:
            return image
        if c == 4:
            return image[..., :3]
        if c == 1:
            return np.concatenate([image, image, image], axis=-1)
        if c > 4:
            return image[..., :3]

    raise ValueError(f"Unsupported image shape: {image.shape}")


# -----------------------------
# Energy map
# -----------------------------
def brownness_energy_map(rgb, method="hed_dab", smooth_sigma=1.0):
    """
    Build an 'energy' (elevation) image where membranes/boundaries have HIGH values.
    """
    rgb = ensure_rgb(rgb)
    rgb = util.img_as_float(rgb)

    if method == "hed_dab":
        hed = color.rgb2hed(rgb)  # requires 3 channels
        dab = hed[..., 2]
        dab = exposure.rescale_intensity(dab, in_range="image", out_range=(0.0, 1.0))
        brown = dab
    elif method == "r_over_sum":
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        denom = (r + g + b + 1e-8)
        brown = r / denom
        brown = exposure.rescale_intensity(brown, in_range="image", out_range=(0.0, 1.0))
    else:
        raise ValueError(f"Unknown method: {method}")

    if smooth_sigma and smooth_sigma > 0:
        brown = filters.gaussian(brown, sigma=smooth_sigma, preserve_range=True)

    brown = exposure.equalize_adapthist(brown, clip_limit=0.02)
    return brown.astype(np.float32)


# -----------------------------
# Points I/O
# -----------------------------
def read_points_tsv(points_path):
    """
    Reads TSV with columns x, y (header optional). Returns Nx2 float32 array.
    """
    pts = []
    with open(points_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)

    # detect header
    start_i = 0
    if rows and len(rows[0]) >= 2:
        try:
            float(rows[0][0])
            float(rows[0][1])
        except Exception:
            start_i = 1

    for r in rows[start_i:]:
        if len(r) < 2:
            continue
        try:
            pts.append((float(r[0]), float(r[1])))
        except Exception:
            continue

    return np.array(pts, dtype=np.float32)


def dedup_points_by_pixel(points_xy, shape_hw):
    """
    If multiple points round to the same pixel, keep only one.
    Avoids marker overwrite at identical pixel locations.
    """
    if points_xy.size == 0:
        return points_xy

    H, W = shape_hw
    xs = np.rint(points_xy[:, 0]).astype(int)
    ys = np.rint(points_xy[:, 1]).astype(int)
    xs = np.clip(xs, 0, W - 1)
    ys = np.clip(ys, 0, H - 1)

    seen = set()
    keep = []
    for x, y, p in zip(xs, ys, points_xy):
        key = (int(x), int(y))
        if key in seen:
            continue
        seen.add(key)
        keep.append(p)
    return np.asarray(keep, dtype=np.float32)


def suppress_close_points(points_xy, min_dist_px):
    """
    Optional: remove points closer than min_dist_px (greedy).
    """
    if points_xy.size == 0:
        return points_xy
    keep = []
    for p in points_xy:
        if not keep:
            keep.append(p)
            continue
        d2 = np.sum((np.asarray(keep) - p) ** 2, axis=1)
        if np.min(d2) >= (min_dist_px**2):
            keep.append(p)
    return np.asarray(keep, dtype=np.float32)


# -----------------------------
# Markers (NO MERGE)
# -----------------------------
def markers_from_points(points_xy, shape_hw, seed_radius=1, min_separation_px=None, dedup_same_pixel=True):
    """
    Create an INT marker image with ONE UNIQUE LABEL PER POINT.
    Guarantees watershed instances don't merge due to marker collapse.
    """
    H, W = shape_hw
    markers = np.zeros((H, W), dtype=np.int32)

    if points_xy.size == 0:
        return markers

    if dedup_same_pixel:
        points_xy = dedup_points_by_pixel(points_xy, shape_hw)

    if min_separation_px is not None and min_separation_px > 0:
        points_xy = suppress_close_points(points_xy, min_separation_px)

    xs = np.rint(points_xy[:, 0]).astype(int)
    ys = np.rint(points_xy[:, 1]).astype(int)
    xs = np.clip(xs, 0, W - 1)
    ys = np.clip(ys, 0, H - 1)

    if seed_radius and seed_radius > 0:
        rr, cc = morphology.disk(seed_radius).nonzero()
        rr = rr - seed_radius
        cc = cc - seed_radius

        label = 1
        for x, y in zip(xs, ys):
            r = y + rr
            c = x + cc
            ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            markers[r[ok], c[ok]] = label
            label += 1
    else:
        for label, (x, y) in enumerate(zip(xs, ys), start=1):
            markers[y, x] = label

    return markers


# -----------------------------
# Watershed
# -----------------------------
def seeded_watershed_membranes(
    rgb,
    markers,
    brown_method="hed_dab",
    smooth_sigma=1.0,
    compactness=0.0,
    watershed_line=True,
    mask=None,
):
    energy = brownness_energy_map(rgb, method=brown_method, smooth_sigma=smooth_sigma)

    if mask is not None:
        mask = mask.astype(bool)

    labels = segmentation.watershed(
        image=energy,
        markers=markers,
        mask=mask,
        compactness=compactness,
        watershed_line=watershed_line,
    ).astype(np.int32)

    membrane = (labels == 0) if watershed_line else segmentation.find_boundaries(labels, mode="thick")
    return labels, membrane, energy


# -----------------------------
# Saving helpers
# -----------------------------
def make_out_dir(base_out_dir, image_path, out_subdir=None):
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    if out_subdir is None:
        out_subdir = image_stem
    out_dir = os.path.join(base_out_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_and_print(path, saver_fn, *args, **kwargs):
    saver_fn(path, *args, **kwargs)
    print(f"[saved] {os.path.abspath(path)}")


def save_summary_figure(
    out_dir,
    out_prefix,
    rgb,
    markers,
    energy,
    labels_rgb,
    membrane,
    dpi=200,
):
    """
    Saves the same 2x2 summary figure that is visualized, into out_dir,
    with filename: summary_<out_prefix>.png
    """
    summary_path = os.path.join(out_dir, f"summary_{out_prefix}.png")

    fig = plt.figure(figsize=(12, 10))

    ax = fig.add_subplot(2, 2, 1)
    ax.set_title("RGB (alpha dropped if RGBA)")
    ax.imshow(rgb)
    ax.axis("off")

    ax = fig.add_subplot(2, 2, 2)
    ax.set_title(f"Markers (n={int(markers.max())})")
    ax.imshow(markers, cmap="nipy_spectral")
    ax.axis("off")

    ax = fig.add_subplot(2, 2, 3)
    ax.set_title("Energy (brownness)")
    ax.imshow(energy, cmap="gray")
    ax.axis("off")

    ax = fig.add_subplot(2, 2, 4)
    ax.set_title("Watershed + membrane")
    ax.axis("off")
    mem_overlay = labels_rgb.copy()
    mem_overlay[membrane] = [1, 0, 0]
    ax.imshow(mem_overlay)

    fig.tight_layout()
    fig.savefig(summary_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {os.path.abspath(summary_path)}")
    return summary_path


# -----------------------------
# Runner
# -----------------------------
def run_on_tile(
    image_path,
    points_path,
    out_prefix="out",
    out_dir=None,                 # if None -> ./watershed_outputs/<image_stem>/
    base_out_dir="./watershed_outputs",
    brown_method="hed_dab",
    smooth_sigma=1.0,
    seed_radius=1,
    min_separation_px=None,
    compactness=0.001,
    watershed_line=True,
    tissue_mask=None,
    show_plots=True,
    save_summary=True,            # ✅ NEW: save the summary visualization
    suppress_low_contrast_warnings=True,
):
    # Warnings are harmless but noisy for binary/near-binary images
    if suppress_low_contrast_warnings:
        warnings.filterwarnings("ignore", message=".*is a low contrast image.*")

    img = io.imread(image_path)
    rgb = ensure_rgb(img)

    pts = read_points_tsv(points_path)

    markers = markers_from_points(
        pts,
        shape_hw=rgb.shape[:2],
        seed_radius=seed_radius,
        min_separation_px=min_separation_px,
        dedup_same_pixel=True,
    )

    labels, membrane, energy = seeded_watershed_membranes(
        rgb=rgb,
        markers=markers,
        brown_method=brown_method,
        smooth_sigma=smooth_sigma,
        compactness=compactness,
        watershed_line=watershed_line,
        mask=tissue_mask,
    )

    # Decide output folder
    if out_dir is None:
        out_dir = make_out_dir(base_out_dir, image_path)

    # Build full paths
    def p(name):
        return os.path.join(out_dir, f"{out_prefix}_{name}")

    # -----------------------------
    # SAVE ENERGY / BROWNNESS MAPS
    # -----------------------------
    save_and_print(p("energy_raw.npy"), np.save, energy)

    energy_vis = exposure.rescale_intensity(energy, in_range="image", out_range=(0, 1))
    save_and_print(p("energy_vis.png"), io.imsave, util.img_as_ubyte(energy_vis))

    energy_16 = exposure.rescale_intensity(energy, in_range="image", out_range=(0, 65535)).astype(np.uint16)
    save_and_print(p("energy_16bit.tif"), io.imsave, energy_16)

    # -----------------------------
    # SAVE SEGMENTATION OUTPUTS
    # -----------------------------
    labels_rgb = color.label2rgb(labels, image=rgb, bg_label=0, alpha=0.35)

    save_and_print(p("labels.png"), io.imsave, util.img_as_ubyte(labels_rgb))
    save_and_print(p("membrane.png"), io.imsave, util.img_as_ubyte(membrane))

    markers_vis = exposure.rescale_intensity(markers.astype(np.float32), in_range="image", out_range=(0, 1))
    save_and_print(p("markers.png"), io.imsave, util.img_as_ubyte(markers_vis))

    # ✅ NEW: save the same visualized summary figure in the SAME output directory
    if save_summary:
        save_summary_figure(
            out_dir=out_dir,
            out_prefix=out_prefix,
            rgb=rgb,
            markers=markers,
            energy=energy,
            labels_rgb=labels_rgb,
            membrane=membrane,
            dpi=200,
        )

    print(f"\nOutput directory: {os.path.abspath(out_dir)}\n")

    # -----------------------------
    # PLOT (and show)
    # -----------------------------
    if show_plots:
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        plt.title("RGB (alpha dropped if RGBA)")
        plt.imshow(rgb)
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.title(f"Markers (n={markers.max()})")
        plt.imshow(markers, cmap="nipy_spectral")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.title("Energy (brownness)")
        plt.imshow(energy, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.title("Watershed + membrane")
        plt.axis("off")
        mem_overlay = labels_rgb.copy()
        mem_overlay[membrane] = [1, 0, 0]
        plt.imshow(mem_overlay)

        plt.tight_layout()
        plt.show()

    return labels, membrane, energy, markers


if __name__ == "__main__":
    image_path = ""
    points_path = ""

    # Prefix used inside the output directory for all files
    out_prefix = "ihctest_ws_nomerge"

    run_on_tile(
        image_path=image_path,
        points_path=points_path,
        out_prefix=out_prefix,
        out_dir=None,                 # None -> ./watershed_outputs/<image_stem>/
        base_out_dir="IHC/watershed_outputs",
        brown_method="hed_dab",        # or "r_over_sum"
        smooth_sigma=1.0,
        seed_radius=1,
        min_separation_px=None,        # set e.g. 3-5 if you have near-duplicate points
        compactness=0.001,
        watershed_line=True,
        tissue_mask=None,
        show_plots=True,
        save_summary=True,            # ✅ will save: summary_<out_prefix>.png
        suppress_low_contrast_warnings=True,
    )

