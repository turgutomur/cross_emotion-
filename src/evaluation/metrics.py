"""
Evaluation metrics.

Macro-F1, weighted-F1, per-class F1, accuracy. Handles the ISEAR-as-target
edge case where the `surprise` class is absent from the target test set.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

from ..data.ekman_mapping import EKMAN_LABELS, NUM_LABELS, ID2LABEL


@dataclass
class EvalResult:
    """Container for a single evaluation run's metrics."""
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    accuracy: float
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    confusion: Optional[np.ndarray] = None
    support: Dict[str, int] = field(default_factory=dict)
    labels_used: List[str] = field(default_factory=list)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    restrict_to_present: bool = True,
) -> EvalResult:
    """
    Compute classification metrics.

    Args:
        y_true, y_pred: int arrays of shape (N,) with Ekman ids (0..5).
        restrict_to_present: if True, compute macro metrics only over
            classes that appear in y_true. This is critical for the
            ISEAR-as-target case (surprise absent) — we do NOT average in
            an F1 of 0 for a class that cannot possibly appear.

    Returns:
        EvalResult with macro/weighted F1, per-class F1, support, confusion.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if restrict_to_present:
        present_ids = sorted(np.unique(y_true).tolist())
    else:
        present_ids = list(range(NUM_LABELS))

    labels_used = [ID2LABEL[i] for i in present_ids]

    macro_f1 = f1_score(y_true, y_pred, labels=present_ids,
                        average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=present_ids,
                           average="weighted", zero_division=0)
    macro_p = precision_score(y_true, y_pred, labels=present_ids,
                              average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, labels=present_ids,
                           average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    per_class_f1_arr = f1_score(y_true, y_pred, labels=present_ids,
                                average=None, zero_division=0)
    per_class_f1 = {
        ID2LABEL[cls_id]: float(f1)
        for cls_id, f1 in zip(present_ids, per_class_f1_arr)
    }
    support = {ID2LABEL[cls_id]: int(np.sum(y_true == cls_id))
               for cls_id in present_ids}

    conf = confusion_matrix(y_true, y_pred, labels=present_ids)

    return EvalResult(
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        accuracy=float(acc),
        per_class_f1=per_class_f1,
        confusion=conf,
        support=support,
        labels_used=labels_used,
    )
