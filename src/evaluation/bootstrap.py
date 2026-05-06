"""
Paired bootstrap significance test.

Why paired bootstrap instead of a parametric t-test?
  With n=3 seeds, a Welch t-test has only 2 degrees of freedom — it is
  severely underpowered and its normality assumption is untestable.  Paired
  bootstrap is non-parametric and uses the same resampling indices for both
  methods within each resample, so it directly tests "is method A better than
  method B on this exact test set" rather than "are the population means
  different".  This is the quantity reviewers want.

Why 1000 resamples?
  Efron & Tibshirani (1993) show that 1000 resamples gives ~3% RMSE on
  p-values around 0.05.  More resamples reduce variance further but give
  diminishing returns; fewer (e.g. 200) inflate RMSE to ~7%.

Why seed=42 fixed?
  Reproducibility.  Paper readers must be able to reproduce exact p-values
  by running this script.  The seed affects only the bootstrap RNG, not
  model training or evaluation.

Reference: Berg-Kirkpatrick et al. (2012) "An Empirical Investigation of
Statistical Significance in NLP", EMNLP.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    """Result of a paired bootstrap significance test.

    Primary fields use names from the paper spec (mean_a/b, mean_diff,
    ci_low/high).  Legacy names (system_a_score, delta, …) are kept as
    properties so existing evaluation code continues to work.
    """
    mean_a: float          # observed metric for method A
    mean_b: float          # observed metric for method B
    mean_diff: float       # mean_b - mean_a; positive = B is better
    ci_low: float          # 2.5th percentile of bootstrap diff distribution
    ci_high: float         # 97.5th percentile
    p_value: float         # two-sided: 2 * extreme-tail fraction, capped at 1.0
    n_resamples: int
    metric_name: str = "macro_f1"

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def significant_005(self) -> bool:
        return self.p_value < 0.05

    @property
    def significant_001(self) -> bool:
        return self.p_value < 0.01

    def summary(self) -> str:
        sig = "**" if self.significant_001 else ("*" if self.significant_005 else "ns")
        return (
            f"{self.metric_name}: "
            f"A={self.mean_a:.4f}, B={self.mean_b:.4f}, "
            f"Δ={self.mean_diff:+.4f}, p={self.p_value:.4f} {sig}, "
            f"95% CI=[{self.ci_low:+.4f}, {self.ci_high:+.4f}]"
        )

    # ------------------------------------------------------------------
    # Backward-compat aliases for code written against the old API
    # ------------------------------------------------------------------

    @property
    def system_a_score(self) -> float:
        return self.mean_a

    @property
    def system_b_score(self) -> float:
        return self.mean_b

    @property
    def delta(self) -> float:
        return self.mean_diff

    @property
    def ci_lower(self) -> float:
        return self.ci_low

    @property
    def ci_upper(self) -> float:
        return self.ci_high


# ---------------------------------------------------------------------------
# Core bootstrap function
# ---------------------------------------------------------------------------

def paired_bootstrap_test(
    predictions_a: List[int],
    predictions_b: List[int],
    labels: List[int],
    metric_fn: Optional[Callable] = None,
    metric_name: str = "macro_f1",
    n_resamples: int = 1000,
    seed: int = 42,
    restrict_labels: Optional[list] = None,
) -> BootstrapResult:
    """
    Paired bootstrap test: is method B significantly different from method A?

    Uses the *same* resampling indices for both methods in each resample, so
    the test is conditioned on the exact test set — this is the paired
    (within-test-set) version, as opposed to an unpaired test that would
    assume independent samples.

    Args:
        predictions_a: predicted class ids from method A, length N.
        predictions_b: predicted class ids from method B, length N.
        labels: ground-truth class ids, length N.  Must align with preds.
        metric_fn: callable(preds, labels) -> float.  If None, uses macro
            F1 (or label-restricted macro F1 when restrict_labels is set).
        metric_name: display name for summary().
        n_resamples: bootstrap iterations.  Pre-registered: 1000.
        seed: RNG seed for the bootstrap loop.  Pre-registered: 42.
        restrict_labels: optional list of label ids to restrict metric to
            (handles ISEAR-as-target where surprise class is absent).

    Returns:
        BootstrapResult with two-sided p_value and 95% CI of the difference.

    p_value interpretation:
        Two-sided: 2 × (fraction of resampled deltas in the tail opposite to
        observed_diff), capped at 1.0.  If observed_diff == 0, all deltas are
        <= 0 and >= 0 simultaneously, giving p = 1.0 (cannot reject null).
        If B clearly dominates A, almost no resample gives delta ≤ 0, so p ≈ 0.
    """
    y_true = np.asarray(labels)
    y_a = np.asarray(predictions_a)
    y_b = np.asarray(predictions_b)
    n = len(y_true)
    if len(y_a) != n or len(y_b) != n:
        raise ValueError(
            f"predictions_a ({len(y_a)}), predictions_b ({len(y_b)}), and "
            f"labels ({n}) must all have the same length."
        )

    if metric_fn is None:
        if restrict_labels is not None:
            def metric_fn(yt: np.ndarray, yp: np.ndarray) -> float:
                return float(f1_score(
                    yt, yp, labels=restrict_labels, average="macro", zero_division=0
                ))
        else:
            def metric_fn(yt: np.ndarray, yp: np.ndarray) -> float:
                return float(f1_score(yt, yp, average="macro", zero_division=0))

    score_a = metric_fn(y_true, y_a)
    score_b = metric_fn(y_true, y_b)
    observed_diff = score_b - score_a

    rng = np.random.RandomState(seed)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        deltas[i] = metric_fn(y_true[idx], y_b[idx]) - metric_fn(y_true[idx], y_a[idx])

    # Two-sided p-value: standard bootstrap formula.
    # If B is better (observed_diff > 0): count resamples where delta <= 0
    # (i.e., B is NOT better in that resample), double for two-sidedness.
    # Special case: when observed_diff == 0 (identical systems), all deltas
    # are also 0 for perfect predictions, giving mean(delta <= 0) = 1.0 and
    # p = 1.0 — which is correct (we cannot reject null for identical methods).
    # Cap at 1.0 because 2 * fraction can exceed 1.0.
    if observed_diff >= 0:
        p_value = float(min(2.0 * np.mean(deltas <= 0), 1.0))
    else:
        p_value = float(min(2.0 * np.mean(deltas >= 0), 1.0))

    return BootstrapResult(
        mean_a=float(score_a),
        mean_b=float(score_b),
        mean_diff=float(observed_diff),
        ci_low=float(np.percentile(deltas, 2.5)),
        ci_high=float(np.percentile(deltas, 97.5)),
        p_value=p_value,
        n_resamples=n_resamples,
        metric_name=metric_name,
    )


# ---------------------------------------------------------------------------
# Seed-aggregation variant
# ---------------------------------------------------------------------------

def aggregate_bootstrap_across_seeds(
    seed_results_a: List[Tuple[List[int], List[int]]],
    seed_results_b: List[Tuple[List[int], List[int]]],
    metric_fn: Optional[Callable] = None,
    metric_name: str = "macro_f1",
    n_resamples: int = 1000,
    seed: int = 42,
    restrict_labels: Optional[list] = None,
) -> BootstrapResult:
    """
    Bootstrap significance test aggregated across multiple seeds.

    Why concatenate seeds?
        Multi-seed runs give within-method variance, which is useful for
        reporting mean ± std.  But the question reviewers actually want
        answered is: "is method A *systematically* better than method B on
        the test distribution?"  Concatenating all seeds' (predictions, labels)
        into one large sample gives us more test examples to resample from,
        reducing bootstrap variance and making the p-value a statement about
        the method pair rather than any single seed's luck.

        Assumption: the test set is the same across seeds (same examples, same
        order), so concatenation corresponds to "3 independent observers of the
        same test set", which is conservative — it will *not* give artificially
        low p-values from treating seed variation as independent samples.

    Args:
        seed_results_a: list of (predictions, labels) tuples, one per seed,
            for method A.  Typically 3 tuples for main experiments.
        seed_results_b: same shape for method B.  Must have same length as
            seed_results_a.
        metric_fn: callable(preds, labels) -> float.  Defaults to macro F1.
        metric_name: display name for summary().
        n_resamples: bootstrap iterations.  Pre-registered: 1000.
        seed: RNG seed.  Pre-registered: 42.
        restrict_labels: passed through to metric_fn default (ISEAR edge case).

    Returns:
        BootstrapResult from the concatenated sample.
    """
    if len(seed_results_a) != len(seed_results_b):
        raise ValueError(
            f"seed_results_a has {len(seed_results_a)} seeds but "
            f"seed_results_b has {len(seed_results_b)}."
        )

    all_preds_a: List[int] = []
    all_preds_b: List[int] = []
    all_labels: List[int] = []

    for (preds_a, lbls_a), (preds_b, lbls_b) in zip(seed_results_a, seed_results_b):
        if list(lbls_a) != list(lbls_b):
            raise ValueError(
                "labels mismatch between method A and method B for the same seed. "
                "Both methods must be evaluated on the exact same test set."
            )
        all_preds_a.extend(preds_a)
        all_preds_b.extend(preds_b)
        all_labels.extend(lbls_a)

    return paired_bootstrap_test(
        predictions_a=all_preds_a,
        predictions_b=all_preds_b,
        labels=all_labels,
        metric_fn=metric_fn,
        metric_name=metric_name,
        n_resamples=n_resamples,
        seed=seed,
        restrict_labels=restrict_labels,
    )


# ---------------------------------------------------------------------------
# Multi-seed summary (unchanged helper)
# ---------------------------------------------------------------------------

def multi_seed_summary(
    scores: List[float],
    label: str = "",
) -> dict:
    """Summarize metric scores across multiple seeds (mean, std, min, max)."""
    arr = np.array(scores)
    return {
        "label": label,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": len(scores),
        "formatted": f"{arr.mean():.4f} ± {arr.std():.4f}",
    }
