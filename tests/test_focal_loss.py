"""Regression tests for FocalLoss and compute_inverse_frequency_alpha.

Three tests verify the pre-registered properties of the focal loss:
  (a) gamma=0, alpha=None must equal nn.CrossEntropyLoss to within 1e-6.
  (b) Easy examples (high p_t) are aggressively down-weighted vs hard ones.
  (c) compute_inverse_frequency_alpha correctly ranks rare > frequent classes
      and applies the safety floor for unseen classes (e.g. surprise in ISEAR).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path

# Ensure the project root is on sys.path when running from any directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.losses import FocalLoss, compute_inverse_frequency_alpha
from src.data.types import EmotionExample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_example(ekman_id: int) -> EmotionExample:
    """Minimal EmotionExample with only the fields used by alpha computation."""
    label_map = {0: "anger", 1: "disgust", 2: "fear", 3: "joy", 4: "sadness", 5: "surprise"}
    return EmotionExample(
        text="x",
        ekman_label=label_map[ekman_id],
        ekman_id=ekman_id,
        domain="goemotions",
        domain_id=0,
    )


def _make_examples(counts: List[int]) -> List[EmotionExample]:
    """Build a list of EmotionExample with the given per-class counts."""
    examples: List[EmotionExample] = []
    for class_id, n in enumerate(counts):
        examples.extend(_make_example(class_id) for _ in range(n))
    return examples


# ---------------------------------------------------------------------------
# Test (a): gamma=0, alpha=None reduces to CE
# ---------------------------------------------------------------------------

def test_focal_reduces_to_ce_at_gamma_zero():
    """FocalLoss(gamma=0, alpha=None) must equal nn.CrossEntropyLoss within 1e-6.

    At gamma=0 the modulating factor (1 - p_t)^0 = 1, so FL(p_t) = -log(p_t),
    which is the definition of cross-entropy.  This test guards the
    log-softmax + gather implementation path against arithmetic drift.
    """
    torch.manual_seed(0)
    B, C = 8, 6
    logits = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))

    focal = FocalLoss(gamma=0.0, alpha=None, reduction="mean")
    ce = nn.CrossEntropyLoss(reduction="mean")

    focal_loss = focal(logits, targets)
    ce_loss = ce(logits, targets)

    assert torch.allclose(focal_loss, ce_loss, atol=1e-6), (
        f"FocalLoss(gamma=0) = {focal_loss.item():.8f} "
        f"but CrossEntropyLoss = {ce_loss.item():.8f} "
        f"(diff = {abs(focal_loss.item() - ce_loss.item()):.2e})"
    )


# ---------------------------------------------------------------------------
# Test (b): easy examples are aggressively down-weighted
# ---------------------------------------------------------------------------

def test_focal_downweights_easy_examples():
    """Focal loss must aggressively down-weight high-confidence (easy) examples.

    We construct one 'easy' sample (correct-class logit = 10, others 0) and
    several 'hard' samples (uniform logits, model is uncertain).

    Expected behaviour under Lin et al. (2017):
      - Easy sample focal loss < CE loss / 100
        (i.e. at least 100× reduction — the focusing factor (1-p_t)^2 is
        essentially zero when p_t ≈ 1 from softmax(10, 0, 0, 0, 0, 0)).
      - Hard samples focal loss within 50 % of CE loss
        (modulating factor ≈ 1 when p_t ≈ 1/C for uniform logits).
    """
    C = 6
    B = 5  # 1 easy + 4 hard

    targets = torch.zeros(B, dtype=torch.long)  # all predict class 0

    logits = torch.zeros(B, C)
    # Easy example: huge logit on the true class → p_t ≈ 1
    logits[0, 0] = 10.0
    # Hard examples: uniform logits → p_t ≈ 1/C
    # logits[1:] stay at 0 (uniform)

    focal = FocalLoss(gamma=2.0, alpha=None, reduction="none")
    ce = nn.CrossEntropyLoss(reduction="none")

    focal_per = focal(logits, targets)
    ce_per = ce(logits, targets)

    # Easy example: focal << ce / 100
    assert focal_per[0].item() < ce_per[0].item() / 100.0, (
        f"Easy example not aggressively down-weighted: "
        f"focal={focal_per[0].item():.6f}, ce/100={ce_per[0].item() / 100:.6f}"
    )

    # Hard examples: focal loss stays within 50 % of CE
    for i in range(1, B):
        ratio = focal_per[i].item() / ce_per[i].item()
        assert 0.5 <= ratio <= 1.5, (
            f"Hard example {i} focal/ce ratio = {ratio:.4f} is outside [0.5, 1.5]. "
            f"focal={focal_per[i].item():.6f}, ce={ce_per[i].item():.6f}"
        )


# ---------------------------------------------------------------------------
# Test (c): inverse-frequency alpha ordering and safety floor
# ---------------------------------------------------------------------------

def test_inverse_frequency_alpha():
    """compute_inverse_frequency_alpha must rank rare classes higher and apply
    a safety floor (not zero or NaN) for unseen classes.

    Setup: 100 joys, 50 angers, 25 sadness, 0 of the other 3 Ekman classes.
    Ekman ids: anger=0, disgust=1, fear=2, joy=3, sadness=4, surprise=5.

    Expected:
      - alpha[joy] < alpha[anger] < alpha[sadness]   (rarer → higher weight)
      - alpha[disgust], alpha[fear], alpha[surprise] == floor == 1 / N
        where N = 100 + 50 + 25 = 175.
      - No NaN values anywhere in the output.
    """
    # Ekman id mapping: anger=0, disgust=1, fear=2, joy=3, sadness=4, surprise=5
    counts = [50, 0, 0, 100, 25, 0]  # indexed by ekman_id
    examples = _make_examples(counts)

    NUM_CLASSES = 6
    N = sum(counts)  # 175

    alpha = compute_inverse_frequency_alpha(examples, num_classes=NUM_CLASSES)

    assert alpha.shape == (NUM_CLASSES,), f"Expected shape ({NUM_CLASSES},), got {alpha.shape}"
    assert not torch.any(torch.isnan(alpha)), "alpha contains NaN values"
    assert not torch.any(torch.isinf(alpha)), "alpha contains inf values"

    # Rarity ordering: joy (100) < anger (50) < sadness (25)
    assert alpha[3] < alpha[0], (
        f"alpha[joy]={alpha[3]:.6f} should be < alpha[anger]={alpha[0]:.6f}"
    )
    assert alpha[0] < alpha[4], (
        f"alpha[anger]={alpha[0]:.6f} should be < alpha[sadness]={alpha[4]:.6f}"
    )

    # Safety floor for unseen classes (disgust=1, fear=2, surprise=5)
    floor = 1.0 / N
    for cls_id, cls_name in [(1, "disgust"), (2, "fear"), (5, "surprise")]:
        # alpha_c = N / (C * n_c) where n_c = clamp(0, min=floor) = floor = 1/N
        # → alpha_c = N / (C * (1/N)) = N^2 / C
        expected = float(N) / (NUM_CLASSES * floor)
        actual = alpha[cls_id].item()
        assert actual > 0, f"alpha[{cls_name}] must be > 0 (safety floor), got {actual}"
        assert not torch.isnan(alpha[cls_id]), f"alpha[{cls_name}] is NaN"
        # Verify it matches the floor-based formula within floating-point tolerance
        assert abs(actual - expected) < 1.0, (
            f"alpha[{cls_name}] = {actual:.4f}, expected floor-based value ~{expected:.4f}"
        )
