"""
Paired bootstrap significance test.

Used to determine whether the difference between two model configurations
is statistically significant. This is REQUIRED for the paper — without it,
any claim that "Method A outperforms Method B" is unsupported.

Reference: Berg-Kirkpatrick et al. (2012) "An Empirical Investigation of
Statistical Significance in NLP"
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np
from sklearn.metrics import f1_score


@dataclass
class BootstrapResult:
    """Result of a paired bootstrap significance test."""
    metric_name: str
    system_a_score: float
    system_b_score: float
    delta: float                # B - A (positive = B is better)
    p_value: float              # one-sided: P(delta <= 0 under H0)
    ci_lower: float             # 95% CI lower bound of delta
    ci_upper: float             # 95% CI upper bound of delta
    n_resamples: int
    significant_005: bool       # p < 0.05
    significant_001: bool       # p < 0.01

    def summary(self) -> str:
        sig = "**" if self.significant_001 else ("*" if self.significant_005 else "ns")
        return (
            f"{self.metric_name}: "
            f"A={self.system_a_score:.4f}, B={self.system_b_score:.4f}, "
            f"Δ={self.delta:+.4f}, p={self.p_value:.4f} {sig}, "
            f"95% CI=[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]"
        )


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn: Optional[Callable] = None,
    metric_name: str = "macro_f1",
    n_resamples: int = 1000,
    seed: int = 42,
    restrict_labels: Optional[list] = None,
) -> BootstrapResult:
    """
    Paired bootstrap test: is system B significantly better than system A?

    Args:
        y_true: ground truth labels, shape (N,).
        y_pred_a: predictions from system A (baseline), shape (N,).
        y_pred_b: predictions from system B (proposed), shape (N,).
        metric_fn: callable(y_true, y_pred) -> float. If None, uses macro F1.
        metric_name: human-readable name for the metric.
        n_resamples: number of bootstrap iterations (1000 is standard).
        seed: RNG seed for reproducibility.
        restrict_labels: if set, compute metric only over these label ids
            (handles ISEAR-as-target where surprise is absent).

    Returns:
        BootstrapResult with p-value and confidence interval.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    n = len(y_true)
    assert len(y_pred_a) == n and len(y_pred_b) == n

    if metric_fn is None:
        if restrict_labels is not None:
            metric_fn = lambda yt, yp: f1_score(
                yt, yp, labels=restrict_labels, average="macro", zero_division=0
            )
        else:
            metric_fn = lambda yt, yp: f1_score(
                yt, yp, average="macro", zero_division=0
            )

    # Observed scores
    score_a = metric_fn(y_true, y_pred_a)
    score_b = metric_fn(y_true, y_pred_b)
    observed_delta = score_b - score_a

    # Bootstrap
    rng = np.random.RandomState(seed)
    deltas = np.zeros(n_resamples)

    for i in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        yt = y_true[indices]
        ya = y_pred_a[indices]
        yb = y_pred_b[indices]
        deltas[i] = metric_fn(yt, yb) - metric_fn(yt, ya)

    # One-sided p-value: fraction of resamples where delta <= 0
    # (i.e., B is NOT better than A)
    p_value = float(np.mean(deltas <= 0))

    # 95% confidence interval of the delta
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))

    return BootstrapResult(
        metric_name=metric_name,
        system_a_score=score_a,
        system_b_score=score_b,
        delta=observed_delta,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_resamples=n_resamples,
        significant_005=p_value < 0.05,
        significant_001=p_value < 0.01,
    )


def multi_seed_summary(
    scores: list,
    label: str = "",
) -> dict:
    """
    Summarize scores across multiple seeds.

    Args:
        scores: list of float metric values (one per seed).
        label: method name for display.

    Returns:
        dict with mean, std, min, max, n.
    """
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
