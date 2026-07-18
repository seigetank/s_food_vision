from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_PATH = BASE_DIR / "data" / "sample.png"


def run_basic_processing(image_path=DEFAULT_IMAGE_PATH):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"{image_path} was not found.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    result = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow("Original", image)
    cv2.imshow("Red Filtered", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
