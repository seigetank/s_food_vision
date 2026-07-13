import cv2
import numpy as np
from datasets import load_dataset


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

        largest_area = max(
            cv2.contourArea(contour)
            for contour in contours
        )

        if largest_area < 224 * 224 * 0.05:
            continue

        return image