from pathlib import Path

import cv2
import numpy as np
from datasets import load_dataset


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "preprocessing"


image = cv2.imread(str(DATA_DIR / "sample.png"))
if image is None:
    raise FileNotFoundError("data/sample.png was not found.")

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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_food_image():
    dataset = load_dataset(
        "ethz/food101",
        split="train",
        streaming=True
    )

    for item in dataset:
        image = np.array(item["image"].convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (224, 224))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if gray.mean() < 40:
            continue

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        largest_area = max(cv2.contourArea(contour) for contour in contours)

        if largest_area < 224 * 224 * 0.05:
            continue

        return image

    raise RuntimeError("No suitable Food-101 image was found.")


food_image = load_food_image()

gray = cv2.cvtColor(food_image, cv2.COLOR_BGR2GRAY)
normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
flipped = cv2.flip(food_image, 1)

matrix = cv2.getRotationMatrix2D((112, 112), 15, 1)
rotated = cv2.warpAffine(
    food_image,
    matrix,
    (224, 224),
    borderMode=cv2.BORDER_REFLECT
)

hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV).astype(np.float32)
hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)

rotated_color = cv2.cvtColor(
    hsv.astype(np.uint8),
    cv2.COLOR_HSV2BGR
)

cv2.imwrite(str(OUTPUT_DIR / "01_resized.jpg"), food_image)
cv2.imwrite(str(OUTPUT_DIR / "02_gray_normalized.jpg"), normalized)
cv2.imwrite(str(OUTPUT_DIR / "03_blurred.jpg"), blurred)
cv2.imwrite(str(OUTPUT_DIR / "04_flipped.jpg"), flipped)
cv2.imwrite(str(OUTPUT_DIR / "05_rotated_color.jpg"), rotated_color)
