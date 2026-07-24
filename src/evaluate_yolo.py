import argparse
import json
import shutil
from pathlib import Path

if __package__:
    from .train_yolo import (
        BASE_DIR,
        DEFAULT_MODEL,
        RUNS_DIR,
        TRAIN_NAME,
        ensure_coco8_dataset,
        write_runtime_data_yaml,
    )
else:
    from train_yolo import (
        BASE_DIR,
        DEFAULT_MODEL,
        RUNS_DIR,
        TRAIN_NAME,
        ensure_coco8_dataset,
        write_runtime_data_yaml,
    )


DEFAULT_TRAINED_MODEL = (
    RUNS_DIR / TRAIN_NAME / "weights" / "best.pt"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "week3_evaluation"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the pretrained YOLOv8n baseline with the COCO8 "
            "fine-tuned model on the same validation split."
        )
    )
    parser.add_argument(
        "--baseline-model",
        default=DEFAULT_MODEL,
        help="Baseline model name or path."
    )
    parser.add_argument(
        "--trained-model",
        default=str(DEFAULT_TRAINED_MODEL),
        help="Fine-tuned model path."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Validation image size."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Validation device, for example cpu, 0, or cuda."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for committed evaluation artifacts."
    )
    return parser.parse_args()


def metrics_to_dict(metrics):
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def compare_metrics(baseline, trained):
    deltas = {
        key: trained[key] - baseline[key]
        for key in baseline
    }
    improved = deltas["map50_95"] > 0

    return {
        "baseline": baseline,
        "fine_tuned": trained,
        "delta": deltas,
        "summary": (
            "Fine-tuning improved mAP50-95 on the COCO8 validation split."
            if improved
            else (
                "Fine-tuning did not improve mAP50-95 on the COCO8 "
                "validation split. The four-image training split is too "
                "small for a reliable performance claim."
            )
        ),
    }


def evaluate_model(model_name, data_yaml, run_name, imgsz, device):
    from ultralytics import YOLO

    model_path = Path(model_name)
    if model_path.suffix == ".pt" and not model_path.exists():
        if model_name != DEFAULT_MODEL:
            raise FileNotFoundError(
                f"{model_path} was not found. Run python src/train_yolo.py first."
            )

    model = YOLO(
        str(model_path)
        if model_path.exists()
        else model_name
    )
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        device=device,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        plots=True,
        workers=0,
    )

    return metrics_to_dict(metrics), Path(metrics.save_dir)


def plot_comparison(comparison, output_path):
    import matplotlib.pyplot as plt

    metric_names = ["precision", "recall", "map50", "map50_95"]
    labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
    positions = range(len(metric_names))
    width = 0.36

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        [position - width / 2 for position in positions],
        [comparison["baseline"][name] for name in metric_names],
        width,
        label="Pretrained baseline",
        color="#94A3B8",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        [comparison["fine_tuned"][name] for name in metric_names],
        width,
        label="COCO8 fine-tuned",
        color="#2563EB",
    )
    axis.set_xticks(list(positions), labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("YOLOv8n validation metrics on COCO8")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def copy_validation_artifacts(validation_dir, output_dir):
    artifact_names = [
        "BoxPR_curve.png",
        "confusion_matrix_normalized.png",
    ]

    for name in artifact_names:
        source = validation_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)


def run_evaluation(
    baseline_model=DEFAULT_MODEL,
    trained_model=DEFAULT_TRAINED_MODEL,
    imgsz=640,
    device="cpu",
    output_dir=DEFAULT_OUTPUT_DIR,
):
    trained_model = Path(trained_model)
    output_dir = Path(output_dir)

    ensure_coco8_dataset()
    data_yaml = write_runtime_data_yaml()

    baseline, _ = evaluate_model(
        baseline_model,
        data_yaml,
        "week3_yolo_val_baseline",
        imgsz,
        device,
    )
    fine_tuned, validation_dir = evaluate_model(
        str(trained_model),
        data_yaml,
        "week3_yolo_val_fine_tuned",
        imgsz,
        device,
    )

    comparison = compare_metrics(baseline, fine_tuned)
    comparison["experiment"] = {
        "dataset": "Ultralytics COCO8",
        "train_images": 4,
        "validation_images": 4,
        "baseline_model": str(baseline_model),
        "fine_tuned_model": (
            str(trained_model.relative_to(BASE_DIR))
            if trained_model.is_relative_to(BASE_DIR)
            else str(trained_model)
        ),
        "imgsz": imgsz,
        "device": device,
        "limitations": (
            "COCO8 is a workflow sample, not a food-specific benchmark. "
            "The metrics are not representative of smart-refrigerator "
            "recognition performance."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    plot_comparison(
        comparison,
        output_dir / "metrics_comparison.png",
    )
    copy_validation_artifacts(validation_dir, output_dir)

    print(json.dumps(comparison, indent=2))
    print(f"saved: {metrics_path}")
    return comparison


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        baseline_model=args.baseline_model,
        trained_model=args.trained_model,
        imgsz=args.imgsz,
        device=args.device,
        output_dir=args.output_dir,
    )
