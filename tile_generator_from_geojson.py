"""
Select EXACTLY 20 tiles (512x512) using regions from a GeoJSON file,
then save each tile (nuclei, cytoplasm, 2-channel, preview) like your original script.

Key change vs your script:
- Instead of random sampling + Otsu, we read tile locations from the GeoJSON.
- We then pick 20 locations (spread out) and crop them from the QPTIFF.

Assumptions about the GeoJSON:
- It contains polygon/rectangle annotations in pixel coordinates at "resolution #1".
- If the GeoJSON is in microns / slide coordinates, you must convert first (not handled here).
"""

import json
import os
import numpy as np
import tifffile
import cv2

# --------------------------
# Configuration
# --------------------------


tile_size = 512
n_tiles_to_keep = 20

os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Helpers
# --------------------------
def enhance_contrast(channel: np.ndarray, low_percentile=2, high_percentile=98) -> np.ndarray:
    """Percentile stretch -> uint8 for preview."""
    p_low = np.percentile(channel, low_percentile)
    p_high = np.percentile(channel, high_percentile)
    if p_high > p_low:
        enhanced = np.clip((channel - p_low) * 255.0 / (p_high - p_low), 0, 255).astype(np.uint8)
    else:
        enhanced = np.clip(channel, 0, 255).astype(np.uint8)
    return enhanced


def _iter_geojson_geometries(obj):
    """Yield geometry dicts from typical GeoJSON structures."""
    if isinstance(obj, dict):
        if obj.get("type") == "FeatureCollection":
            for feat in obj.get("features", []):
                geom = feat.get("geometry")
                if geom:
                    yield geom
        elif obj.get("type") == "Feature":
            geom = obj.get("geometry")
            if geom:
                yield geom
        elif "geometry" in obj:
            geom = obj["geometry"]
            if geom:
                yield geom
        elif obj.get("type") in ("Polygon", "MultiPolygon", "Point", "MultiPoint", "LineString", "MultiLineString"):
            yield obj
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_geojson_geometries(item)


def geom_to_centroid_xy(geom):
    """
    Convert GeoJSON geometry -> (cx, cy) centroid in pixel coordinates.
    Supports Polygon/MultiPolygon/Point.
    """
    gtype = geom.get("type")
    coords = geom.get("coordinates")

    if gtype == "Point":
        x, y = coords
        return float(x), float(y)

    if gtype == "Polygon":
        # coords[0] is exterior ring: [[x,y], [x,y], ...]
        ring = coords[0]
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return float(np.mean(xs)), float(np.mean(ys))

    if gtype == "MultiPolygon":
        # take centroid of all exterior points across polygons
        xs, ys = [], []
        for poly in coords:
            ring = poly[0]
            xs.extend([p[0] for p in ring])
            ys.extend([p[1] for p in ring])
        return float(np.mean(xs)), float(np.mean(ys))

    # If you have other geometry types, add handling here
    raise ValueError(f"Unsupported geometry type: {gtype}")


def pick_spread_out_points(points_xy, k=20, seed=42):
    """
    Pick k points that are spatially spread out (farthest-point sampling).
    points_xy: list[(x,y)]
    """
    pts = np.array(points_xy, dtype=np.float32)
    if len(pts) <= k:
        return points_xy

    rng = np.random.default_rng(seed)
    # start from a random point
    idx0 = int(rng.integers(0, len(pts)))
    chosen = [idx0]

    # distances to chosen set
    d2 = np.sum((pts - pts[idx0]) ** 2, axis=1)

    for _ in range(k - 1):
        idx = int(np.argmax(d2))
        chosen.append(idx)
        # update min distance to chosen set
        new_d2 = np.sum((pts - pts[idx]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)

    return [points_xy[i] for i in chosen]


def crop_top_left_from_center(cx, cy, width, height, tile_size):
    """Return integer top-left (x,y) for a tile centered at (cx,cy), clipped to image bounds."""
    x = int(round(cx - tile_size / 2))
    y = int(round(cy - tile_size / 2))
    x = max(0, min(x, width - tile_size))
    y = max(0, min(y, height - tile_size))
    return x, y


# --------------------------
# Read GeoJSON tile centers
# --------------------------
print("Reading GeoJSON...")
with open(geojson_path, "r", encoding="utf-8") as f:
    geo = json.load(f)

centroids = []
for geom in _iter_geojson_geometries(geo):
    try:
        cx, cy = geom_to_centroid_xy(geom)
        centroids.append((cx, cy))
    except Exception:
        continue

if len(centroids) == 0:
    raise RuntimeError(
        "No usable geometries found in GeoJSON. "
        "Check that it is a FeatureCollection with Polygon/Point geometries in pixel coordinates."
    )

print(f"Found {len(centroids)} geometries in GeoJSON.")

# Pick 20 spread-out centroids
picked_centroids = pick_spread_out_points(centroids, k=n_tiles_to_keep, seed=42)

# --------------------------
# Read QPTIFF and save tiles
# --------------------------
print("Reading QPTIFF...")
with tifffile.TiffFile(input_path) as tif:
    image = tif.asarray(series=0)  # (C, H, W)
    print(f"Image shape: {image.shape}")

    nuclei = image[0]
    membrane = image[1]
    height, width = nuclei.shape

    nuclei_enhanced = enhance_contrast(nuclei)
    membrane_enhanced = enhance_contrast(membrane)

    # Convert centroids -> tile top-left positions
    selected_positions = []
    for (cx, cy) in picked_centroids:
        x, y = crop_top_left_from_center(cx, cy, width, height, tile_size)
        selected_positions.append((x, y))

    # Deduplicate (in case multiple centroids map to same tile)
    selected_positions = list(dict.fromkeys(selected_positions))

    # If dedup reduced count, just take first N (or repeat sampling fallback)
    selected_positions = selected_positions[:n_tiles_to_keep]

    print(f"Saving {len(selected_positions)} tiles to: {output_dir}")

    for i, (x, y) in enumerate(selected_positions):
        nuclei_tile_original = nuclei[y:y + tile_size, x:x + tile_size]
        membrane_tile_original = membrane[y:y + tile_size, x:x + tile_size]

        nuclei_tile_enh = nuclei_enhanced[y:y + tile_size, x:x + tile_size]
        membrane_tile_enh = membrane_enhanced[y:y + tile_size, x:x + tile_size]

        base_name = f"tile_{i:03d}_x{x}_y{y}"

        # 1) nuclei
        tifffile.imwrite(os.path.join(output_dir, f"{base_name}_nuclei.tiff"), nuclei_tile_original)

        # 2) cytoplasm/membrane
        tifffile.imwrite(os.path.join(output_dir, f"{base_name}_cytoplasm.tiff"), membrane_tile_original)

        # 3) 2-channel composite
        two_channel_data = np.stack([nuclei_tile_original, membrane_tile_original], axis=0)
        tifffile.imwrite(
            os.path.join(output_dir, f"{base_name}_2channel.tiff"),
            two_channel_data,
            imagej=True,
            metadata={'axes': 'CYX', 'channels': 2, 'Channel0': 'nuclei', 'Channel1': 'cytoplasm'}
        )

        # 4) preview PNG: red=nuclei, green=cytoplasm
        preview_rgb = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        if nuclei_tile_enh.max() > 0:
            preview_rgb[..., 0] = (nuclei_tile_enh / nuclei_tile_enh.max() * 255).astype(np.uint8)
        if membrane_tile_enh.max() > 0:
            preview_rgb[..., 1] = (membrane_tile_enh / membrane_tile_enh.max() * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(output_dir, f"{base_name}_preview.png"), preview_rgb)

print("Done.")

# Write summary
summary_path = os.path.join(output_dir, "tile_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Tile Summary (GeoJSON-driven)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Source QPTIFF: {input_path}\n")
    f.write(f"Source GeoJSON: {geojson_path}\n")
    f.write(f"Tile size: {tile_size}x{tile_size}\n")
    f.write(f"Requested tiles: {n_tiles_to_keep}\n")
    f.write(f"Saved tiles: {len(selected_positions)}\n\n")
    for i, (x, y) in enumerate(selected_positions):
        f.write(f"Tile {i:03d}: top-left=({x},{y}) center≈({x+tile_size//2},{y+tile_size//2})\n")
print(f"Summary saved to: {summary_path}")
