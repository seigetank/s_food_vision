# 3주차 AI 모델링 및 OpenCV 결과 시각화 보고서

## 1. 목표와 범위

3주차 과제의 목표는 YOLOv8 객체 탐지 모델의 학습·평가·추론 흐름을 구현하고, OpenCV로 예측 결과를 시각화하는 것입니다.

이번 구현은 Ultralytics COCO8 예제 데이터셋을 사용합니다. COCO8은 학습 이미지 4장과 검증 이미지 4장으로 구성된 워크플로 검증용 데이터이므로, 스마트 냉장고 식품 인식 성능을 대표하지 않습니다.

## 2. 구현 내용

- `src/train_yolo.py`
  - COCO8 다운로드 및 runtime YAML 생성
  - `yolov8n.pt`를 시작 모델로 10 epoch 미세조정
  - augmentation 활성화, seed 42, deterministic 설정
  - `best.pt` 및 학습 그래프 생성
- `src/evaluate_yolo.py`
  - 사전학습 모델과 미세조정 모델을 동일한 검증셋에서 비교
  - Precision, Recall, mAP50, mAP50-95 저장
  - 지표 비교 그래프, PR curve, confusion matrix 저장
- `src/object_detection.py`
  - 학습된 모델로 식품 샘플 이미지 추론
  - OpenCV로 bounding box와 신뢰도 표시
  - 결과 이미지와 JSON 저장

## 3. 평가 결과

| 지표 | 사전학습 모델 | COCO8 미세조정 | 변화 |
|---|---:|---:|---:|
| Precision | 0.6210 | 0.7600 | +0.1391 |
| Recall | 0.8333 | 0.8333 | +0.0000 |
| mAP50 | 0.8875 | 0.8890 | +0.0015 |
| mAP50-95 | 0.6291 | 0.6244 | -0.0046 |

미세조정 후 Precision은 증가했지만 mAP50-95는 소폭 감소했습니다. 이는 학습 이미지가 4장뿐인 데이터셋에서는 개별 지표의 변동이 크고, 미세조정이 일반화 성능 향상을 보장하지 않는다는 점을 보여줍니다.

평가 산출물:

- `outputs/week3_evaluation/metrics.json`
- `outputs/week3_evaluation/metrics_comparison.png`
- `outputs/week3_evaluation/BoxPR_curve.png`
- `outputs/week3_evaluation/confusion_matrix_normalized.png`

## 4. 객체 탐지 결과 분석

학습된 모델은 샘플 식품 이미지에서 여러 객체를 탐지하고 bounding box를 표시했습니다. 그러나 COCO 분류 체계와 작은 학습 데이터의 한계로 인해 두부나 반찬을 `sandwich`, `bowl`, `dining table` 등으로 잘못 분류하거나 넓은 영역에 중복 box를 생성했습니다.

따라서 현재 결과는 YOLO 학습·평가·추론 파이프라인의 동작을 확인한 POC이며, 식품 전용 인식 모델로 해석하면 안 됩니다.

## 5. 검증

```text
pytest -q
5 passed
```

테스트 항목:

- 가상 Depth Map 반환 타입과 크기
- `None` 입력 예외 처리
- 3D 좌표 배열 크기
- YOLO 평가 지표 변환
- baseline 대비 지표 변화 계산

## 6. 결론과 다음 단계

3주차에서는 모델 학습, 정량 평가, OpenCV 시각화, 결과 저장까지 재현 가능한 흐름을 완성했습니다. 동시에 소규모 범용 데이터만으로는 스마트 냉장고 식품 인식 성능을 주장할 수 없다는 한계를 확인했습니다.

4주차에서는 탐지 결과를 사용자가 수정하고 유통기한을 입력할 수 있는 로컬 MVP를 구현합니다. 식품 전용 데이터셋 구축과 재학습은 후속 개선 범위로 둡니다.
