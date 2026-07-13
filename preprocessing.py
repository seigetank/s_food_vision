from pathlib import Path

import cv2
import numpy as np


def save_preprocessed_images(
    image,
    output_path="preprocessed_samples"
):
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)

    flipped = cv2.flip(image, 1)

    matrix = cv2.getRotationMatrix2D(
        (112, 112),
        15,
        1
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (224, 224),
        borderMode=cv2.BORDER_REFLECT
    )

    hsv = cv2.cvtColor(
        rotated,
        cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1] * 1.5,
        0,
        255
    )

    hsv[:, :, 2] = np.clip(
        hsv[:, :, 2] * 1.1,
        0,
        255
    )

    rotated_color = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR
    )

    cv2.imwrite(
        str(output_dir / "01_resized.jpg"),
        image
    )
    cv2.imwrite(
        str(output_dir / "02_gray_normalized.jpg"),
        normalized
    )
    cv2.imwrite(
        str(output_dir / "03_blurred.jpg"),
        blurred
    )
    cv2.imwrite(
        str(output_dir / "04_flipped.jpg"),
        flipped
    )
    cv2.imwrite(
        str(output_dir / "05_rotated_color.jpg"),
        rotated_color
    )