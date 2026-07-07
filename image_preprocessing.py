from pathlib import Path
from datasets import load_dataset
import cv2
import numpy as np

# 이미지 로드
image = cv2.imread('sample.png')

# BGR에서 HSV 색상 공간으로 변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 빨간색 범위 지정 (두 개의 범위를 설정해야 함)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# 마스크 생성
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2

# 원본 이미지에서 빨간색 부분만 추출
result = cv2.bitwise_and(image, image, mask=mask)

# 결과 이미지 출력
cv2.imshow('Original', image)
cv2.imshow('Red Filtered', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

output_dir = Path("preprocessed_samples")
output_dir.mkdir(exist_ok=True)


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

        # 너무 어두운 이미지 제외
        if gray.mean() < 40:
            continue

        # 객체 크기가 너무 작은 이미지 제외
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


food_image = load_food_image()

# Grayscale 및 Normalize
gray = cv2.cvtColor(food_image, cv2.COLOR_BGR2GRAY)
normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

# Blur
blurred = cv2.GaussianBlur(normalized, (5, 5), 0)

# 좌우 반전
flipped = cv2.flip(food_image, 1)

# 회전
matrix = cv2.getRotationMatrix2D((112, 112), 15, 1)
rotated = cv2.warpAffine(
    food_image,
    matrix,
    (224, 224),
    borderMode=cv2.BORDER_REFLECT
)

# 색상 변화
hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV).astype(np.float32)
hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)

rotated_color = cv2.cvtColor(
    hsv.astype(np.uint8),
    cv2.COLOR_HSV2BGR
)



# 처리 이미지 5장 저장
cv2.imwrite(str(output_dir / "01_resized.jpg"), food_image)
cv2.imwrite(str(output_dir / "02_gray_normalized.jpg"), normalized)
cv2.imwrite(str(output_dir / "03_blurred.jpg"), blurred)
cv2.imwrite(str(output_dir / "04_flipped.jpg"), flipped)
cv2.imwrite(
    str(output_dir / "05_rotated_color.jpg"),
    rotated_color
)