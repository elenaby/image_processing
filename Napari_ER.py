import napari
import imageio.v2 as imageio
import numpy as np

pred_path = r""

rgb_path = r""

pred = imageio.imread(pred_path)
rgb = imageio.imread(rgb_path)

# Convert prediction to editable labels
if pred.ndim == 3:
    pred_gray = pred[..., 0]
else:
    pred_gray = pred

labels = (pred_gray > 0).astype(np.int32)

viewer = napari.Viewer()

# LEFT: original RGB image
viewer.add_image(
    rgb,
    name="Original RGB"
)

# RIGHT: RGB image again, shifted horizontally
shift_x = rgb.shape[1] + 50

viewer.add_image(
    rgb,
    name="Overlay RGB",
    translate=(0, shift_x)
)

# RIGHT: editable prediction overlay
viewer.add_labels(
    labels,
    name="Prediction",
    opacity=0.5,
    translate=(0, shift_x)
)

napari.run()

