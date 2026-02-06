import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

# -----------------------------
# Path to NDPI
# -----------------------------
ndpi_path = r""

# -----------------------------
# Load NDPI and get thumbnail
# -----------------------------
with tifffile.TiffFile(ndpi_path) as tif:
    print(f"Number of series: {len(tif.series)}\n")

    for i, series in enumerate(tif.series):
        print(f"Series {i}:")
        print(f"  shape : {series.shape}")
        print(f"  dtype : {series.dtype}")
        print(f"  axes  : {series.axes}")
        print(f"  levels: {len(series.levels)}\n")

    series = tif.series[0]
    thumb = series.levels[-1].asarray()

print("Thumbnail raw shape:", thumb.shape)

# -----------------------------
# Normalize to (C, H, W)
# -----------------------------
if thumb.ndim == 3:
    if thumb.shape[-1] <= 4:       # (H, W, C)
        img = np.moveaxis(thumb, -1, 0)
    else:                           # (C, H, W)
        img = thumb
elif thumb.ndim == 4:
    img = thumb[0]                 # drop T/Z if present
else:
    raise ValueError("Unexpected NDPI layout")

num_channels = img.shape[0]
print(f"Number of channels: {num_channels}")

# -----------------------------
# Channel assignment
# -----------------------------
H_channel   = img[0].astype(np.float32)  # Hematoxylin
DAB_channel = img[1].astype(np.float32)  # DAB

# -----------------------------
# Function to estimate I0
# -----------------------------
def estimate_I0_otsu(channel, percentile=99, min_bg_frac=0.01):
    """
    Estimate I0 using Otsu-based background detection.
    """
    t = threshold_otsu(channel)
    bg_mask = channel > t  # background assumed bright

    if bg_mask.mean() < min_bg_frac:
        I0 = np.percentile(channel, percentile)
        print("  [fallback] insufficient background, using global percentile")
    else:
        I0 = np.percentile(channel[bg_mask], percentile)

    return I0, bg_mask, t

# -----------------------------
# Compute I0 per channel
# -----------------------------
I0_H, bg_H, tH = estimate_I0_otsu(H_channel)
I0_D, bg_D, tD = estimate_I0_otsu(DAB_channel)

print("\nEstimated I0 values:")
print(f"  Hematoxylin I0 = {I0_H:.2f} (Otsu={tH:.2f})")
print(f"  DAB         I0 = {I0_D:.2f} (Otsu={tD:.2f})")

# -----------------------------
# Visualize channels + background masks
# -----------------------------
fig, ax = plt.subplots(2, 3, figsize=(12, 8))

ax[0,0].imshow(H_channel, cmap="gray")
ax[0,0].set_title("Hematoxylin")

ax[0,1].imshow(bg_H, cmap="gray")
ax[0,1].set_title("H background mask")

ax[0,2].hist(H_channel.ravel(), bins=256)
ax[0,2].axvline(tH, color="r")
ax[0,2].set_title("H histogram")

ax[1,0].imshow(DAB_channel, cmap="gray")
ax[1,0].set_title("DAB")

ax[1,1].imshow(bg_D, cmap="gray")
ax[1,1].set_title("DAB background mask")

ax[1,2].hist(DAB_channel.ravel(), bins=256)
ax[1,2].axvline(tD, color="r")
ax[1,2].set_title("DAB histogram")

for a in ax.ravel():
    a.axis("off")

plt.tight_layout()
plt.show()

