import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_PATH = BASE_DIR / "data" / "sample.jpg"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "object_detection"
DEFAULT_MODEL = (
    BASE_DIR
    / "runs"
    / "detect"
    / "week3_yolo_train"
    / "weights"
    / "best.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pretrained YOLOv8 object detection on an image."
    )
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE_PATH),
        help="Input image path."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="YOLO model path. Run src/train_yolo.py first for the default model."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for annotated image and JSON results."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold."
    )
    return parser.parse_args()


def draw_detections(image, detections):
    annotated = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        label = detection["label"]
        confidence = detection["confidence"]

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        text = f"{label} {confidence:.2f}"
        text_origin = (x1, max(y1 - 10, 20))
        cv2.putText(
            annotated,
            text,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return annotated


def run_object_detection(
    image_path=DEFAULT_IMAGE_PATH,
    model_name=DEFAULT_MODEL,
    output_dir=DEFAULT_OUTPUT_DIR,
    conf=0.25
):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"{image_path} was not found.")

    model_path = Path(model_name)
    if model_path.suffix == ".pt" and not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} was not found. Run python src/train_yolo.py first."
        )

    model = YOLO(str(model_path) if model_path.exists() else model_name)
    results = model.predict(
        source=str(image_path),
        conf=conf,
        verbose=False
    )

    result = results[0]
    names = result.names
    detections = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [
            int(round(value))
            for value in box.xyxy[0].tolist()
        ]

        detections.append(
            {
                "label": names[cls_id],
                "class_id": cls_id,
                "confidence": confidence,
                "bbox_xyxy": [x1, y1, x2, y2]
            }
        )

    annotated = draw_detections(image, detections)

    annotated_path = output_dir / "sample_detected.jpg"
    json_path = output_dir / "detection_results.json"

    cv2.imwrite(str(annotated_path), annotated)

    output = {
        "model": str(model_path.relative_to(BASE_DIR))
        if model_path.exists() and model_path.is_relative_to(BASE_DIR)
        else model_name,
        "source_image": str(image_path.relative_to(BASE_DIR))
        if image_path.is_relative_to(BASE_DIR)
        else str(image_path),
        "confidence_threshold": conf,
        "image_shape": {
            "height": image.shape[0],
            "width": image.shape[1],
            "channels": image.shape[2]
        },
        "detection_count": len(detections),
        "detections": detections
    }

    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"detection_count: {len(detections)}")
    print(f"saved: {annotated_path}")
    print(f"saved: {json_path}")

    return output


if __name__ == "__main__":
    args = parse_args()
    run_object_detection(
        image_path=args.image,
        model_name=args.model,
        output_dir=args.output_dir,
        conf=args.conf
    )
