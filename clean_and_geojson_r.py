from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import json
import re
from tqdm import tqdm


# -----------------------
# PATHS
# -----------------------
INPUT_DIR = Path("")

OUT_MASK_DIR = INPUT_DIR.parent / "clean_predictions"
OUT_GEOJSON_DIR = INPUT_DIR.parent / "clean_predictions_geojson"

OUT_MASK_DIR.mkdir(exist_ok=True)
OUT_GEOJSON_DIR.mkdir(exist_ok=True)


# -----------------------
# 🔥 IMPORTANT: GLOBAL OFFSET FROM OTSU CROP
# -----------------------
OTSUX = 0   # ❗ replace with real value
OTSUY = 0   # ❗ replace with real value


# -----------------------
# PARAMETERS
# -----------------------
THRESH = 127
DILATE_RADIUS = 1
ERODE_RADIUS = 1
N_MORPH = 2

MIN_COMPONENT_PIXELS = 20
MIN_CIRCULARITY = 0.10

CONTOUR_THICKNESS = 2


# -----------------------
# PARSE TILE OFFSET
# -----------------------
def parse_xy(filename):
    m = re.search(r"x(\d+)_y(\d+)", filename)
    if not m:
        raise ValueError(f"Cannot parse x/y from {filename}")
    return int(m.group(1)), int(m.group(2))


# -----------------------
# HELPERS
# -----------------------
def read_mask_u8(path):
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)

def binarize(u8):
    return (u8 > THRESH).astype(np.uint8)

def morph(bin01):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    out = bin01.copy()
    for _ in range(N_MORPH):
        out = cv2.dilate(out.astype(np.uint8), kernel)
    for _ in range(N_MORPH):
        out = cv2.erode(out.astype(np.uint8), kernel)
    return (out > 0).astype(np.uint8)

def filter_small(bin01):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    out = np.zeros_like(bin01)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_PIXELS:
            out[labels == i] = 1
    return out

def circularity(c):
    a = cv2.contourArea(c)
    p = cv2.arcLength(c, True)
    if a == 0 or p == 0:
        return 0
    return 4*np.pi*a/(p*p)


# -----------------------
# GEOJSON WITH GLOBAL COORDS ⭐
# -----------------------
def contours_to_geojson(contours, filename, x_tile, y_tile):
    features = []

    for c in contours:
        if len(c) < 3:
            continue

        coords = c.squeeze()

        # convert to global coords
        coords_global = []
        for pt in coords:
            xg = int(pt[0] + x_tile + OTSUX)
            yg = int(pt[1] + y_tile + OTSUY)
            coords_global.append([xg, yg])

        if coords_global[0] != coords_global[-1]:
            coords_global.append(coords_global[0])

        features.append({
            "type": "Feature",
            "properties": {
                "tile": filename
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords_global]
            }
        })

    return {"type": "FeatureCollection", "features": features}


# -----------------------
# MAIN
# -----------------------
def main():

    files = sorted(INPUT_DIR.glob("*.png"))

    for path in tqdm(files):

        x_tile, y_tile = parse_xy(path.name)

        mask = binarize(read_mask_u8(path))
        mask = morph(mask)
        mask = filter_small(mask)

        img255 = (mask * 255).astype(np.uint8)

        contours, _ = cv2.findContours(img255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        kept = [c for c in contours if circularity(c) >= MIN_CIRCULARITY]

        # save contour mask
        out_mask = np.zeros_like(mask)
        if kept:
            cv2.drawContours(out_mask, kept, -1, 1, thickness=CONTOUR_THICKNESS)

        Image.fromarray(out_mask*255).save(OUT_MASK_DIR / path.name)

        # save geojson
        geojson = contours_to_geojson(kept, path.name, x_tile, y_tile)

        with open(OUT_GEOJSON_DIR / (path.stem + ".geojson"), "w") as f:
            json.dump(geojson, f)

    print("✅ DONE")


if __name__ == "__main__":
    main()