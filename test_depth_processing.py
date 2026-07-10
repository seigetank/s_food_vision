import cv2
import numpy as np
import pytest

from depth_processing import generate_depth_map


def test_generate_depth_map():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    depth_map = generate_depth_map(image)

    assert isinstance(depth_map, np.ndarray)
    assert depth_map.shape == image.shape


def test_generate_depth_map_none():
    with pytest.raises(ValueError):
        generate_depth_map(None)