"""Evaluation module public API."""
from .metrics import EvalResult, compute_metrics
from .bootstrap import (
    BootstrapResult,
    paired_bootstrap_test,
    aggregate_bootstrap_across_seeds,
    multi_seed_summary,
)

__all__ = [
    "EvalResult", "compute_metrics",
    "BootstrapResult", "paired_bootstrap_test",
    "aggregate_bootstrap_across_seeds", "multi_seed_summary",
]
