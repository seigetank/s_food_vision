# 스마트 냉장고 식품 이미지 처리 및 2D-3D 변환 프로젝트

## 프로젝트 개요

이 저장소는 스마트 냉장고 카메라로 촬영한 식품 이미지를 처리하는 Computer Vision 실습 프로젝트입니다.

- 1주차에는 이미지 전처리와 색상 영역 검출을 수행했습니다.
- 2주차에는 Unit Test, Depth Map 시각화, 3D 좌표 배열 변환을 수행했습니다.
- 기본 Depth Map과 `sample_point_cloud.png`는 실제 깊이 센서값이 아니라 Grayscale 밝기값을 깊이처럼 사용한 실습 결과입니다.
- RealSense D435 카메라를 이용해 실제 depth frame 기반 Grayscale Depth Map과 3D 점 그래프도 생성했습니다.
- 코드, 입력 데이터, 결과물을 기능별 디렉토리로 분리했습니다.

## 주요 기능

### 이미지 전처리

구현 파일: `src/image_preprocessing.py`

- `data/sample.png`를 OpenCV로 읽고 HSV 색상 공간에서 빨간색 영역을 감지합니다.
- 빨간색 HSV 범위를 두 구간으로 나누어 mask를 만들고, `cv2.bitwise_and`로 빨간색 영역만 필터링합니다.
- Hugging Face `ethz/food101` 데이터셋을 streaming 방식으로 로드합니다.
- Food-101 이미지 중 평균 밝기가 너무 낮은 이미지와 객체 영역이 너무 작은 이미지를 제외합니다.
- 선택된 식품 이미지를 `224x224` 크기로 조정합니다.
- Grayscale 변환 후 `cv2.normalize`로 밝기 범위를 정규화합니다.
- Gaussian Blur, 좌우 반전, 15도 회전, HSV 기반 색상 변화를 적용합니다.
- 처리 결과 이미지를 `outputs/preprocessing/` 폴더에 저장합니다.

### Grayscale 기반 가상 Depth Map

구현 파일: `src/depth_processing.py`

- `generate_depth_map(image)`
  - 입력 이미지가 `None`이면 `ValueError`를 발생시킵니다.
  - 입력 이미지를 Grayscale로 변환합니다.
  - `cv2.applyColorMap(gray, cv2.COLORMAP_JET)`으로 Depth Map처럼 보이는 컬러 시각화 이미지를 만듭니다.

- `generate_point_cloud(image)`
  - 입력 이미지가 `None`이면 `ValueError`를 발생시킵니다.
  - 입력 이미지를 Grayscale로 변환합니다.
  - 각 픽셀의 X, Y 좌표와 Grayscale 밝기값을 Z값으로 사용해 `(height, width, 3)` 형태의 3D 좌표 배열을 만듭니다.

`src/depth_processing.py`를 실행하면 `data/sample.jpg`를 읽고 `outputs/virtual_depth/depth_map.jpg`를 저장합니다.

`outputs/virtual_depth/sample_point_cloud.png`는 `data/sample.jpg`에서 생성한 `(540, 960, 3)` 형태의 3D 좌표 배열을 점 그래프로 시각화한 결과입니다. 이 그래프의 Z축은 실제 거리값이 아니라 Grayscale 밝기값입니다.

### RealSense 실제 Depth Map

구현 파일: `src/realsense_depth_processing.py`

- `pyrealsense2`로 RealSense D435의 depth stream과 color stream을 엽니다.
- depth frame의 `z16` 값을 읽고 depth scale을 적용해 meter 단위 depth 배열을 만듭니다.
- 유효 거리 범위는 `0.1m` 초과, `2.0m` 이하로 제한합니다.
- 가까운 지점이 더 밝게 보이도록 Grayscale Depth Map을 생성합니다.
- `cv2.applyColorMap`을 이용해 컬러 Depth Map도 저장합니다.
- depth intrinsics와 meter 단위 depth 값을 이용해 실제 3D 좌표 점 그래프를 생성합니다.
- meter 단위 depth 배열은 `outputs/realsense_depth/realsense_depth_m.npy`로 저장합니다.

## 프로젝트 구조

```text
.
├── README.md
├── data/
│   ├── sample.jpg
│   └── sample.png
├── outputs/
│   ├── preprocessing/
│   │   ├── 01_resized.jpg
│   │   ├── 02_gray_normalized.jpg
│   │   ├── 03_blurred.jpg
│   │   ├── 04_flipped.jpg
│   │   └── 05_rotated_color.jpg
│   ├── realsense_depth/
│   │   ├── realsense_color.jpg
│   │   ├── realsense_depth_colormap.jpg
│   │   ├── realsense_depth_gray.jpg
│   │   ├── realsense_depth_m.npy
│   │   └── realsense_point_cloud.png
│   └── virtual_depth/
│       ├── depth_map.jpg
│       └── sample_point_cloud.png
├── src/
│   ├── __init__.py
│   ├── depth_processing.py
│   ├── image_preprocessing.py
│   └── realsense_depth_processing.py
└── tests/
    └── test_depth_processing.py
```

## 개발 환경

- Python 3
- OpenCV
- NumPy
- Hugging Face datasets
- Pillow
- pytest
- matplotlib
- pyrealsense2

현재 저장소에는 `requirements.txt`가 없습니다.

## 설치 방법

Windows Git Bash 기준 설치 흐름입니다.

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install opencv-python numpy datasets pillow pytest matplotlib pyrealsense2
```

## 실행 방법

### 이미지 전처리 실행

```bash
python src/image_preprocessing.py
```

실행하면 `data/sample.png`의 빨간색 영역 필터링 결과를 OpenCV 창으로 표시하고, Food-101 데이터셋에서 선택한 식품 이미지의 전처리 결과를 `outputs/preprocessing/` 폴더에 저장합니다.

### Grayscale 기반 가상 Depth Map 실행

```bash
python src/depth_processing.py
```

실행하면 `data/sample.jpg`를 읽어 Grayscale 밝기 기반 Depth Map 시각화 이미지를 만들고 `outputs/virtual_depth/depth_map.jpg`로 저장합니다. OpenCV 창으로 원본 이미지와 Depth Map을 표시합니다.

### RealSense 실제 Depth Map 실행

```bash
python src/realsense_depth_processing.py
```

실행하면 RealSense D435 카메라에서 실제 depth frame을 캡처하고 `outputs/realsense_depth/` 폴더에 결과를 저장합니다.

저장 결과는 다음과 같습니다.

- `outputs/realsense_depth/realsense_color.jpg`: RGB 카메라 이미지
- `outputs/realsense_depth/realsense_depth_gray.jpg`: 실제 depth 값 기반 Grayscale Depth Map
- `outputs/realsense_depth/realsense_depth_colormap.jpg`: Grayscale Depth Map에 컬러맵을 적용한 이미지
- `outputs/realsense_depth/realsense_point_cloud.png`: 실제 depth 좌표 기반 3D 점 그래프
- `outputs/realsense_depth/realsense_depth_m.npy`: meter 단위 depth 배열

### 테스트 실행

```bash
pytest tests/test_depth_processing.py -v
```

`src/depth_processing.py`의 기본 Depth Map 함수와 3D 좌표 배열 생성 함수를 검증합니다.

## Unit Test

테스트 파일은 `tests/test_depth_processing.py`입니다.

- `test_generate_depth_map`
  - `100x100x3` 크기의 NumPy 이미지를 입력합니다.
  - 반환값이 `np.ndarray`인지 확인합니다.
  - 반환 이미지 크기가 입력 이미지와 같은지 확인합니다.

- `test_generate_depth_map_none`
  - `generate_depth_map(None)` 호출 시 `ValueError`가 발생하는지 확인합니다.

- `test_generate_point_cloud`
  - `100x100x3` 크기의 NumPy 이미지를 입력합니다.
  - 반환값이 `np.ndarray`인지 확인합니다.
  - 반환 배열 크기가 `(100, 100, 3)`인지 확인합니다.

현재 확인된 테스트 결과는 다음과 같습니다.

- `test_generate_depth_map`: PASSED
- `test_generate_depth_map_none`: PASSED
- `test_generate_point_cloud`: PASSED
- 총 3개 테스트 통과

## 시각적 결과

### 이미지 전처리 결과

![Resize result](outputs/preprocessing/01_resized.jpg)

![Gray normalized result](outputs/preprocessing/02_gray_normalized.jpg)

![Rotated color result](outputs/preprocessing/05_rotated_color.jpg)

### Grayscale 기반 가상 Depth Map

![Before](data/sample.jpg)

![Depth Map](outputs/virtual_depth/depth_map.jpg)

### Grayscale 기반 3D 점 그래프

![Sample point cloud](outputs/virtual_depth/sample_point_cloud.png)

### RealSense 실제 Depth 결과

![RealSense color](outputs/realsense_depth/realsense_color.jpg)

![RealSense gray depth map](outputs/realsense_depth/realsense_depth_gray.jpg)

![RealSense point cloud](outputs/realsense_depth/realsense_point_cloud.png)

## 구현상의 한계

- `src/depth_processing.py`의 기본 Depth Map은 실제 깊이 추정 결과가 아니라 Grayscale 밝기 기반 시각화입니다.
- `outputs/virtual_depth/sample_point_cloud.png`의 Z축은 실제 거리값이 아니라 Grayscale 밝기값입니다.
- 기본 포인트 클라우드는 X, Y, 밝기 기반 Z값으로 구성된 기초 실습 형태입니다.
- RealSense 결과는 실제 depth sensor 값을 사용하지만, 촬영 환경, 반사 재질, 거리 범위, 카메라 노이즈에 영향을 받습니다.
- 실제 식품 영역별 depth 분석이나 3D 모델 복원은 아직 구현하지 않았습니다.

## 향후 개선 방향

- 실제 Depth Estimation 모델 적용
- RealSense depth와 RGB 이미지 정렬 후 식품 영역 기준 depth 분석
- Open3D를 이용한 포인트 클라우드 시각화
- 다양한 식품 이미지와 실제 냉장고 환경에서 테스트 확장
- 테스트 케이스 보강
