from types import SimpleNamespace

import pytest

from src.evaluate_yolo import compare_metrics, metrics_to_dict


def test_metrics_to_dict():
    metrics = SimpleNamespace(
        box=SimpleNamespace(
            mp=0.81,
            mr=0.72,
            map50=0.77,
            map=0.55,
        )
    )

    assert metrics_to_dict(metrics) == {
        "precision": 0.81,
        "recall": 0.72,
        "map50": 0.77,
        "map50_95": 0.55,
    }


def test_compare_metrics_calculates_deltas():
    baseline = {
        "precision": 0.5,
        "recall": 0.4,
        "map50": 0.6,
        "map50_95": 0.3,
    }
    trained = {
        "precision": 0.7,
        "recall": 0.5,
        "map50": 0.8,
        "map50_95": 0.45,
    }

    comparison = compare_metrics(baseline, trained)

    assert comparison["delta"]["precision"] == pytest.approx(0.2)
    assert comparison["delta"]["map50_95"] == pytest.approx(0.15)
    assert "improved" in comparison["summary"]
