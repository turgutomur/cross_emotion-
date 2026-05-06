"""
Regression tests for paired bootstrap significance testing.

These three tests pin the core statistical properties that the paper's
claims depend on.  They use small, deterministic inputs so they run in
< 5 s even on CPU.

Run with:
    pytest tests/test_bootstrap.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.bootstrap import (
    BootstrapResult,
    aggregate_bootstrap_across_seeds,
    paired_bootstrap_test,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(labels, preds, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPairedBootstrap:
    def test_identical_predictions_have_p_value_one(self) -> None:
        """If A == B, p_value should be ≈ 1.0 (no detectable difference).

        When observed_diff == 0 and both systems produce identical (perfect)
        predictions, every resampled delta is also 0.  The two-sided formula
        gives 2 * mean(deltas <= 0) = 2 * 1.0, capped at 1.0.
        """
        labels = np.array([0, 1, 2, 3, 4, 5] * 100)
        preds = np.array([0, 1, 2, 3, 4, 5] * 100)

        result = paired_bootstrap_test(
            predictions_a=preds,
            predictions_b=preds,
            labels=labels,
            n_resamples=1000,
            seed=42,
        )

        assert isinstance(result, BootstrapResult)
        assert result.mean_diff == 0.0, "Identical predictions must have zero delta"
        assert result.p_value >= 1.0 - 1e-2, (
            f"Expected p_value ≈ 1.0, got {result.p_value:.4f}"
        )

    def test_clearly_different_predictions_have_low_p(self) -> None:
        """All-correct B vs all-wrong A must yield p < 0.05.

        With 6 balanced classes, all-correct gives macro-F1 = 1.0 and
        all-wrong gives macro-F1 = 0.0.  Bootstrap should detect this
        obvious difference in every resample.
        """
        n_per_class = 50
        labels = np.repeat(np.arange(6), n_per_class)   # balanced, 300 total

        preds_a = (labels + 1) % 6   # all-wrong (systematic shift)
        preds_b = labels.copy()       # all-correct

        result = paired_bootstrap_test(
            predictions_a=preds_a,
            predictions_b=preds_b,
            labels=labels,
            n_resamples=1000,
            seed=42,
        )

        assert result.mean_b > result.mean_a, "B (all-correct) must outscore A (all-wrong)"
        assert result.mean_diff > 0, "mean_diff = mean_b - mean_a must be positive"
        assert result.p_value < 0.05, (
            f"Obvious difference should be highly significant, got p={result.p_value:.4f}"
        )

    def test_result_structure(self) -> None:
        """BootstrapResult fields are present and have correct types."""
        labels = np.array([0, 1, 2, 3] * 25)
        preds = labels.copy()

        result = paired_bootstrap_test(preds, preds, labels, n_resamples=100, seed=42)

        assert isinstance(result.mean_a, float)
        assert isinstance(result.mean_b, float)
        assert isinstance(result.mean_diff, float)
        assert isinstance(result.ci_low, float)
        assert isinstance(result.ci_high, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.n_resamples, int)
        assert result.n_resamples == 100
        assert isinstance(result.summary(), str)
        # Backward-compat aliases must work
        assert result.delta == result.mean_diff
        assert result.ci_lower == result.ci_low
        assert result.ci_upper == result.ci_high


class TestAggregateBootstrap:
    def test_aggregate_across_seeds_basic(self) -> None:
        """3 seeds where A consistently beats B → mean_diff > 0 and p < 0.05.

        Method A always predicts correctly; method B always predicts the next
        class (systematic error).  Across all three seeds the ground truth is
        the same test set — this models the real use-case where the test set
        is fixed and seeds only affect training.
        """
        rng = np.random.RandomState(0)
        n_per_class = 40
        labels = np.repeat(np.arange(6), n_per_class)   # 240 examples

        seed_results_a = []  # method A: all correct
        seed_results_b = []  # method B: systematic error

        for _ in range(3):
            preds_a = labels.copy()           # F1 = 1.0
            preds_b = (labels + 1) % 6       # F1 = 0.0
            seed_results_a.append((preds_a.tolist(), labels.tolist()))
            seed_results_b.append((preds_b.tolist(), labels.tolist()))

        result = aggregate_bootstrap_across_seeds(
            seed_results_a=seed_results_a,
            seed_results_b=seed_results_b,
            n_resamples=1000,
            seed=42,
        )

        assert isinstance(result, BootstrapResult)
        assert result.mean_a > result.mean_b, "Method A (correct) must outscore B (wrong)"
        # Note: mean_diff = mean_b - mean_a; A is better so this should be negative.
        # The test asks: does A beat B?  mean_a > mean_b → mean_diff < 0.
        # p_value tests sign mismatch; since observed_diff < 0 and nearly all
        # resampled deltas are also < 0, p_value ≈ 0 (B is significantly worse).
        assert result.mean_diff < 0, "mean_b - mean_a must be negative when A is better"
        assert result.p_value < 0.05, (
            f"A clearly dominates B across all 3 seeds; expected p < 0.05, "
            f"got p={result.p_value:.4f}"
        )

    def test_aggregate_requires_matching_seed_counts(self) -> None:
        """Mismatched number of seeds raises ValueError."""
        labels = [0, 1, 2, 3]
        preds = [0, 1, 2, 3]
        seed_a = [(preds, labels), (preds, labels)]
        seed_b = [(preds, labels)]

        with pytest.raises(ValueError, match="seeds"):
            aggregate_bootstrap_across_seeds(seed_a, seed_b, n_resamples=50, seed=42)

    def test_aggregate_requires_matching_labels(self) -> None:
        """Mismatched labels between A and B for the same seed raises ValueError."""
        preds = [0, 1, 2]
        seed_a = [(preds, [0, 1, 2])]
        seed_b = [(preds, [0, 1, 3])]   # different labels

        with pytest.raises(ValueError, match="labels mismatch"):
            aggregate_bootstrap_across_seeds(seed_a, seed_b, n_resamples=50, seed=42)
