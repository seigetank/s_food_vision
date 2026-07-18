import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import SETTINGS


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "datasets" / "week3_yolo_sample"
DATA_YAML = DATASET_DIR / "data.yaml"
RUNTIME_DATA_YAML = DATASET_DIR / "data_runtime.yaml"
COCO8_DIR = DATASET_DIR / "coco8"
COCO8_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
RUNS_DIR = BASE_DIR / "runs" / "detect"
TRAIN_NAME = "week3_yolo_train"
DEFAULT_MODEL = "yolov8n.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on the COCO8 sample dataset."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Base YOLO model name or path."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size."
    )
    return parser.parse_args()


def ensure_coco8_dataset():
    train_images = COCO8_DIR / "images" / "train"
    val_images = COCO8_DIR / "images" / "val"

    if train_images.exists() and val_images.exists():
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATASET_DIR / "coco8.zip"

    print(f"downloading: {COCO8_URL}")
    urllib.request.urlretrieve(COCO8_URL, zip_path)

    extract_dir = DATASET_DIR / "_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)

    extracted_coco8 = extract_dir / "coco8"
    if COCO8_DIR.exists():
        shutil.rmtree(COCO8_DIR)

    shutil.move(str(extracted_coco8), str(COCO8_DIR))
    shutil.rmtree(extract_dir)
    zip_path.unlink(missing_ok=True)

    print(f"dataset_ready: {COCO8_DIR}")


def write_runtime_data_yaml():
    source = DATA_YAML.read_text(encoding="utf-8")
    lines = []

    for line in source.splitlines():
        if line.startswith("path:"):
            lines.append(f"path: {COCO8_DIR.as_posix()}")
        else:
            lines.append(line)

    RUNTIME_DATA_YAML.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    return RUNTIME_DATA_YAML


def train_yolo(model_name=DEFAULT_MODEL, epochs=10, imgsz=640):
    ensure_coco8_dataset()
    SETTINGS["datasets_dir"] = str(DATASET_DIR)
    data_yaml = write_runtime_data_yaml()

    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        project=str(RUNS_DIR),
        name=TRAIN_NAME,
        exist_ok=True,
        device="cpu"
    )

    best_path = RUNS_DIR / TRAIN_NAME / "weights" / "best.pt"
    print(f"best_model: {best_path}")

    return results


if __name__ == "__main__":
    args = parse_args()
    train_yolo(
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz
    )
