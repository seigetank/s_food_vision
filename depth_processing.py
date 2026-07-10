import cv2
import numpy as np


def generate_depth_map(image):
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    return depth_map


def generate_point_cloud(image):
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")

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
    image = cv2.imread("sample.jpg")

    if image is None:
        raise FileNotFoundError("sample.png를 불러올 수 없습니다.")

    depth_map = generate_depth_map(image)
    points_3d = generate_point_cloud(image)

    cv2.imwrite("depth_map.jpg", depth_map)

    cv2.imshow("Original Image", image)
    cv2.imshow("Depth Map", depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()