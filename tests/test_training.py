"""
Smoke test for the training loop.

Uses a DummyBackbone (random linear projection, no pre-trained weights) so
the test runs on CPU in a few seconds without downloading DeBERTa.
The goal is NOT to reach a good metric — it is to confirm that:
  1. The Trainer can complete one epoch without errors.
  2. Training loss at step > 0 is lower than the initial loss at step 0.
     (A loss that decreases even by a tiny amount shows gradients are
     flowing through all components correctly.)
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.types import EmotionExample
from src.evaluation.metrics import EvalResult
from src.models.backbone import BackboneOutput
from src.models.classifier import ClassificationHead, EmotionClassifier
from src.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HIDDEN = 64
NUM_LABELS = 6
VOCAB_SIZE = 512


class DummyBackbone(nn.Module):
    """Tiny embedding-mean backbone that never downloads anything.

    Mimics exactly the API contract of DebertaBackbone:
        forward(input_ids, attention_mask, ...) → BackboneOutput
        .hidden_size property
    """

    def __init__(self, hidden_size: int = HIDDEN):
        super().__init__()
        self._hidden_size = hidden_size
        self.embedding = nn.Embedding(VOCAB_SIZE, hidden_size)
        self.dropout = nn.Dropout(0.0)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> BackboneOutput:
        seq = self.embedding(input_ids)                          # (B, T, H)
        # Masked mean pooling (same logic as DebertaBackbone "mean" strategy)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (seq * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # (B, H)
        pooled = self.dropout(pooled)
        return BackboneOutput(pooled=pooled, sequence_output=seq, attention_mask=attention_mask)


class MockTokenizer:
    """Tokenizer that produces random integer ids — no vocabulary needed."""

    def __call__(
        self,
        texts: List[str],
        padding: Any = True,
        truncation: bool = True,
        max_length: int = 32,
        return_tensors: str = "pt",
        return_token_type_ids: bool = False,
        **_kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(texts)
        seq_len = 16  # fixed short length for speed
        return {
            "input_ids": torch.randint(0, VOCAB_SIZE, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }


def _make_examples(n: int) -> List[EmotionExample]:
    """Generate n fake EmotionExample instances with random valid labels."""
    rng = random.Random(0)
    examples = []
    for i in range(n):
        eid = rng.randint(0, NUM_LABELS - 1)
        did = rng.randint(0, 2)
        from src.data.ekman_mapping import ID2LABEL
        from src.data.types import ID2DATASET
        examples.append(
            EmotionExample(
                text=f"sample text number {i}",
                ekman_label=ID2LABEL[eid],
                ekman_id=eid,
                domain=ID2DATASET[did],
                domain_id=did,
            )
        )
    return examples


def _minimal_config(epochs: int = 1) -> Dict[str, Any]:
    """Minimal config that satisfies Trainer without loading a real YAML."""
    return {
        "training": {
            "epochs": epochs,
            "batch_size": 8,
            "gradient_accumulation": 1,
            "fp16": False,
            "early_stopping_patience": 3,
            "encoder_lr": 1e-3,
            "head_lr": 2e-3,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
        },
        "model": {"max_length": 16, "num_labels": NUM_LABELS},
        "evaluation": {"restrict_to_present": True},
        "logging": {"level": "WARNING"},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_smoke_loss_decreases(tmp_path):
    """One epoch on 50 examples; final loss must be below initial loss."""
    examples = _make_examples(50)
    train_ex, val_ex = examples[:40], examples[40:]

    backbone = DummyBackbone(hidden_size=HIDDEN)
    model = EmotionClassifier(
        backbone=backbone,
        num_labels=NUM_LABELS,
        head_dropout=0.0,
    )

    # Record initial loss before any weight update
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        dummy_ids = torch.randint(0, VOCAB_SIZE, (8, 16))
        dummy_mask = torch.ones(8, 16, dtype=torch.long)
        dummy_labels = torch.randint(0, NUM_LABELS, (8,))
        initial_out = model(
            input_ids=dummy_ids,
            attention_mask=dummy_mask,
            labels=dummy_labels,
        )
        initial_loss = initial_out.loss.item()

    cfg = _minimal_config(epochs=1)
    trainer = Trainer(
        model=model,
        train_examples=train_ex,
        val_examples=val_ex,
        tokenizer=MockTokenizer(),
        config=cfg,
        experiment_name="smoke_test",
        output_dir=tmp_path,
        seed=42,
    )
    result = trainer.train()

    # Measure loss after training
    model.eval()
    with torch.no_grad():
        post_out = model(
            input_ids=dummy_ids,
            attention_mask=dummy_mask,
            labels=dummy_labels,
        )
        final_loss = post_out.loss.item()

    assert final_loss < initial_loss, (
        f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}. "
        "Gradients may not be flowing correctly."
    )
    assert result["best_epoch"] == 1
    assert 0.0 <= result["best_val_f1"] <= 1.0


def test_evaluate_returns_aggregate_and_per_domain(tmp_path):
    """evaluate() must return an EvalResult for 'aggregate' and each domain present."""
    examples = _make_examples(30)

    backbone = DummyBackbone()
    model = EmotionClassifier(backbone=backbone, num_labels=NUM_LABELS, head_dropout=0.0)

    cfg = _minimal_config(epochs=1)
    trainer = Trainer(
        model=model,
        train_examples=examples,
        val_examples=examples,
        tokenizer=MockTokenizer(),
        config=cfg,
        experiment_name="eval_test",
        output_dir=tmp_path,
        seed=42,
    )

    metrics = trainer.evaluate(examples)

    assert "aggregate" in metrics, "Missing 'aggregate' key in evaluate() result."
    assert isinstance(metrics["aggregate"], EvalResult)
    assert 0.0 <= metrics["aggregate"].macro_f1 <= 1.0

    # At least one per-domain result besides aggregate and val_loss
    domain_keys = [k for k in metrics if k not in {"aggregate", "val_loss"}]
    assert len(domain_keys) >= 1, "No per-domain metrics returned."


def test_checkpoint_save_and_load(tmp_path):
    """Checkpoint saved during train() must be loadable without errors."""
    examples = _make_examples(20)

    backbone = DummyBackbone()
    model = EmotionClassifier(backbone=backbone, num_labels=NUM_LABELS, head_dropout=0.0)

    cfg = _minimal_config(epochs=1)
    trainer = Trainer(
        model=model,
        train_examples=examples,
        val_examples=examples,
        tokenizer=MockTokenizer(),
        config=cfg,
        experiment_name="ckpt_test",
        output_dir=tmp_path,
        seed=42,
    )
    trainer.train()

    checkpoint_path = tmp_path / "checkpoints" / "ckpt_test" / "seed_42" / "best.pt"
    assert checkpoint_path.exists(), f"Checkpoint not found at {checkpoint_path}"

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    assert "model_state" in ckpt
    assert "val_macro_f1" in ckpt
    assert "epoch" in ckpt
    assert ckpt["seed"] == 42

    # load_best_checkpoint() must not raise
    trainer.load_best_checkpoint()
