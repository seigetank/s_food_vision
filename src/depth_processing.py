from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "virtual_depth"


def generate_depth_map(image):
    if image is None:
        raise ValueError("Input image is missing.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    return depth_map


def generate_point_cloud(image):
    if image is None:
        raise ValueError("Input image is missing.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    x, y = np.meshgrid(
        np.arange(width),
        np.arange(height)
    )
    z = gray.astype(np.float32)

    points_3d = np.dstack((x, y, z))

    return points_3d


if __name__ == "__main__":
    image = cv2.imread(str(DATA_DIR / "sample.jpg"))

    if image is None:
        raise FileNotFoundError("data/sample.jpg was not found.")

    depth_map = generate_depth_map(image)
    points_3d = generate_point_cloud(image)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / "depth_map.jpg"), depth_map)

    print(f"points_3d shape: {points_3d.shape}")

    cv2.imshow("Original Image", image)
    cv2.imshow("Depth Map", depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
