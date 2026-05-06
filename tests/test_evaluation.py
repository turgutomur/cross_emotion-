"""
Tests for evaluation metrics and bootstrap.

Run with:
    pytest tests/test_evaluation.py -v
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import compute_metrics, EvalResult
from src.evaluation.bootstrap import (
    paired_bootstrap_test,
    multi_seed_summary,
    BootstrapResult,
)


class TestMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 2, 3, 4, 5])
        result = compute_metrics(y, y)
        assert result.macro_f1 == 1.0
        assert result.accuracy == 1.0

    def test_restrict_to_present(self):
        # Only classes 0,1,2 present — should compute over 3 classes
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        result = compute_metrics(y_true, y_pred, restrict_to_present=True)
        assert len(result.labels_used) == 3
        assert result.macro_f1 == 1.0

    def test_confusion_shape(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        result = compute_metrics(y_true, y_pred)
        assert result.confusion.shape == (4, 4)

    def test_per_class_f1_keys(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        result = compute_metrics(y_true, y_pred)
        assert "anger" in result.per_class_f1  # 0 = anger
        assert "disgust" in result.per_class_f1  # 1 = disgust


class TestBootstrap:
    def test_identical_predictions(self):
        y_true = np.array([0, 1, 2, 3, 4, 5] * 50)
        y_pred = np.array([0, 1, 2, 3, 4, 5] * 50)
        result = paired_bootstrap_test(y_pred, y_pred, y_true, n_resamples=100)
        assert result.delta == 0.0
        assert result.p_value >= 0.4  # identical systems → p ≈ 1.0

    def test_clearly_better_system(self):
        rng = np.random.RandomState(42)
        n = 500
        y_true = rng.randint(0, 6, size=n)
        y_pred_a = rng.randint(0, 6, size=n)       # random
        y_pred_b = y_true.copy()                     # perfect
        result = paired_bootstrap_test(
            y_pred_a, y_pred_b, y_true, n_resamples=500
        )
        assert result.delta > 0           # B is better
        assert result.significant_005     # should be significant

    def test_result_fields(self):
        y = np.array([0, 1, 2, 3, 4, 5] * 10)
        result = paired_bootstrap_test(y, y, y, n_resamples=50)
        assert isinstance(result, BootstrapResult)
        assert result.metric_name == "macro_f1"
        assert isinstance(result.summary(), str)


class TestMultiSeedSummary:
    def test_basic(self):
        scores = [0.68, 0.70, 0.69]
        result = multi_seed_summary(scores, label="test")
        assert abs(result["mean"] - 0.69) < 0.01
        assert result["n"] == 3
        assert "±" in result["formatted"]
