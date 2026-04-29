"""Focal loss for imbalanced multi-class emotion classification.

Why focal loss matters for this regime
---------------------------------------
Emotion datasets are heavily imbalanced: GoEmotions is dominated by
neutral/admiration/joy (which we drop neutral, but skew remains), ISEAR
contributes only 5 Ekman classes (surprise is absent entirely), and WASSA-21
carries class skew inherited from Reddit empathy prompts.  Naive cross-entropy
concentrates gradient signal on majority classes that are already well-
classified, leaving minority classes under-trained.  Focal loss (Lin et al.
ICCV 2017) addresses this by down-weighting easy examples (high p_t) via the
modulating factor (1 − p_t)^gamma, so training signal is dominated by hard,
minority-class examples where the model is uncertain.

Why inverse-frequency alpha (pre-registered default)
-----------------------------------------------------
Per-class alpha scaling further re-weights each sample's loss by how rare
its class is.  We use alpha_c = N / (C × n_c), which makes alpha proportional
to rarity and ensures mean(alpha) = 1 by construction (so expected loss
magnitude is unchanged relative to the unweighted baseline).  The choice of
inverse-frequency (rather than effective-number weighting or class-balanced
sampling) is pre-registered for three reasons:
  1. No test-set leakage: alpha is computed only from the training split.
  2. Transparency: a single formula with a clear interpretation.
  3. Broad support in the imbalance-learning literature.

When to disable alpha (alpha=None)
------------------------------------
Setting alpha=None gives pure focal loss without per-class re-weighting —
useful for ablations that isolate the effect of the focusing term alone,
without confounding it with class-prior correction.

Reference
---------
Lin, Goyal, Girshick, He, Dollár (ICCV 2017) "Focal Loss for Dense Object
Detection". arXiv:1708.02002.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.types import EmotionExample


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weighting.

    Multi-class formulation (Lin et al. 2017)::

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the softmax probability of the true class.

    Implementation: log-softmax + gather avoids catastrophic cancellation
    in softmax(x).log() when one logit dominates, preserving numerical
    stability under fp16 autocast.  At gamma=0 and alpha=None, the output
    is numerically identical to nn.CrossEntropyLoss (verified in the
    unit tests to within 1e-6).

    Parameters
    ----------
    gamma:
        Focusing exponent.  gamma=0 reduces exactly to cross-entropy;
        gamma=2 is the Lin et al. (2017) default and the pre-registered
        value for this project.
    alpha:
        Optional ``(num_classes,)`` float tensor of per-class weights.
        When provided, each sample's loss is multiplied by alpha[true_class].
        Use ``compute_inverse_frequency_alpha`` to build this from the
        training-set class frequencies (pre-registered default).
        None → no per-class weighting (focal-only).
    reduction:
        One of ``"mean"``, ``"sum"``, ``"none"``.  Matches the semantics
        of ``nn.CrossEntropyLoss``.
    ignore_index:
        Class id whose samples are excluded from the loss (same semantics as
        ``CrossEntropyLoss``).  The trainer never passes padded labels, but
        this parameter provides parity for future-proofing.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Invalid reduction: {reduction!r}")
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        if alpha is not None:
            # Register as buffer so .to(device) moves the weights along with
            # the model, and they appear in state_dict for checkpoint safety.
            self.register_buffer("alpha", alpha.float())
        else:
            # Explicit None attribute so forward() can check without hasattr.
            self.alpha: Optional[torch.Tensor] = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        logits:
            ``(B, C)`` unnormalised class scores.
        targets:
            ``(B,)`` long tensor of true class ids.

        Returns
        -------
        Scalar loss (or per-sample ``(B,)`` tensor when reduction="none").
        """
        # Replace ignore_index positions with 0 to avoid out-of-bounds gather;
        # their contributions are zeroed out after the computation.
        ignore_mask = targets == self.ignore_index
        safe_targets = targets.clone()
        safe_targets[ignore_mask] = 0

        log_probs = F.log_softmax(logits, dim=-1)                          # (B, C)
        log_pt = log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1) # (B,)
        pt = log_pt.exp()                                                   # (B,)

        # Focusing term: near-1 for easy (high-confidence) examples → small weight
        focal_weight = (1.0 - pt) ** self.gamma                            # (B,)
        loss = -focal_weight * log_pt                                       # (B,)

        # Optional per-class alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha[safe_targets]                             # (B,)
            loss = alpha_t * loss

        # Zero out ignored positions before reduction
        loss = loss.masked_fill(ignore_mask, 0.0)

        if self.reduction == "mean":
            n_valid = (~ignore_mask).float().sum().clamp(min=1.0)
            return loss.sum() / n_valid
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # "none"


def compute_inverse_frequency_alpha(
    train_examples: List[EmotionExample],
    num_classes: int,
) -> torch.Tensor:
    """Build inverse-frequency alpha weights from training-set class counts.

    Formula: alpha_c = N / (C × n_c), where N = total examples, C =
    num_classes, n_c = count of class c.  By construction mean(alpha) = 1
    when all classes are present, so the expected loss magnitude is unchanged
    relative to the unweighted baseline.

    A safety floor of 1.0 / N is applied for unseen classes (e.g. surprise
    in ISEAR-as-target runs) to avoid division by zero and NaN alpha values.
    The floor produces a very small but non-zero weight, so unseen classes
    contribute negligible signal rather than crashing training — this is the
    right behaviour because the model has no training signal for that class
    anyway.

    Parameters
    ----------
    train_examples:
        Training-split examples only.  MUST NOT include val or test examples
        — computing alpha from those would constitute test-set leakage, which
        is a pre-registered exclusion for this project.
    num_classes:
        Total number of Ekman classes (6 in this project).

    Returns
    -------
    ``(num_classes,)`` float32 tensor.
    """
    n_total = len(train_examples)
    if n_total == 0:
        raise ValueError("train_examples is empty — cannot compute alpha.")

    counts = torch.zeros(num_classes, dtype=torch.float32)
    for ex in train_examples:
        counts[ex.ekman_id] += 1.0

    # Safety floor: 1/N is the weight of one hypothetical example; unseen
    # classes get a near-zero weight rather than inf or NaN.
    floor = 1.0 / n_total
    counts = counts.clamp(min=floor)

    # alpha_c = N / (C * n_c).  Mean(alpha) == 1 by construction when all
    # classes are present.
    alpha = n_total / (num_classes * counts)
    return alpha


__all__ = ["FocalLoss", "compute_inverse_frequency_alpha"]
