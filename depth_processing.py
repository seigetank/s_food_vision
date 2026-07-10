import cv2
import numpy as np


def generate_depth_map(image):
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    return depth_map
