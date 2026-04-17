"""
Emotion classification head with cross-entropy loss.

This file implements the *first* of the six methods listed in README.md:

    Source-only / Mixed  =  DebertaBackbone + ClassificationHead + CE loss

Rationale for the split between ``ClassificationHead`` and ``EmotionClassifier``
-------------------------------------------------------------------------------
``ClassificationHead`` is a stateless ``nn.Linear`` (preceded by a dropout)
that maps the pooled representation to 6 Ekman logits. Keeping it as its own
module has two concrete payoffs:

    * In DANN / CDAN (Week 3) the *same* head will be trained jointly with a
      domain discriminator that lives in its own module. Pulling the head
      out of the end-to-end model means the discriminator code does not
      need to know about the CE head at all.

    * When we do CDAN in Week 3, the conditional multilinear map needs the
      softmax distribution *out of this exact head*. Having a clean
      ``forward(features) -> logits`` contract makes that wiring trivial.

``EmotionClassifier`` is the training-time composite: backbone + head + loss.
It is the module the trainer instantiates for Source-only and Mixed
baselines (Week 2 of the project plan).

The module is deliberately agnostic about whether the loss is vanilla CE or
focal — Week 4 introduces focal loss by injecting a different ``loss_fn``
into the composite, NOT by rewriting this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BackboneConfig, BackboneOutput, DebertaBackbone, build_backbone


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------
class ClassificationHead(nn.Module):
    """Linear projection from pooled features to Ekman-6 logits.

    Architecture
    ------------
    ``dropout → Linear(hidden_size, num_labels)``

    We intentionally keep the head minimal (a single linear layer). The
    DeBERTa fine-tuning literature shows that a deeper MLP head rarely
    helps for single-sentence classification and adds free parameters that
    make the comparison across our six methods less controlled. If an
    ablation is needed later, swap this module with a deeper one; no other
    code changes should be required.

    Note on dropout placement
    -------------------------
    The backbone already applies dropout to its pooled output. We apply a
    *second* dropout here so that the head-specific regularisation can be
    tuned independently (this is the pattern used by
    ``AutoModelForSequenceClassification`` in HuggingFace). If the user
    wants to disable it, pass ``dropout=0.0`` via the config.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_labels)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform init for the logits layer.

        Matches the initialisation used by ``transformers`` for newly added
        classification heads, so our head behaves identically to the one
        inside ``AutoModelForSequenceClassification`` at step 0.
        """
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, hidden_size)`` features to ``(batch, num_labels)`` logits."""
        return self.linear(self.dropout(features))


# ---------------------------------------------------------------------------
# Composite model output
# ---------------------------------------------------------------------------
@dataclass
class EmotionClassifierOutput:
    """Return type of ``EmotionClassifier.forward``.

    A dataclass (rather than a tuple) is used so that the trainer can write
    ``out.loss.backward()`` / ``out.logits.argmax(-1)`` without caring about
    positional order — important because Week 3 will add ``domain_logits``
    to the DANN composite and we want symmetry across modules.
    """

    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    features: torch.Tensor       # pooled backbone output; reused by DANN/CDAN


# ---------------------------------------------------------------------------
# End-to-end composite: backbone + head + loss
# ---------------------------------------------------------------------------
class EmotionClassifier(nn.Module):
    """Source-only / Mixed baseline model.

    Consumes a batch in the format produced by
    ``src/data/torch_dataset.EmotionCollator`` and returns logits plus
    (optionally) a cross-entropy loss.

    Why this class exists
    ---------------------
    The trainer should not need to know the shape contract between the
    backbone output and the head. ``EmotionClassifier`` glues them together
    so that the training loop can simply do::

        out = model(**batch)
        out.loss.backward()

    Parameters
    ----------
    backbone:
        A ``DebertaBackbone`` instance. Injected (not constructed here) so
        that DANN / CDAN models later can reuse the same encoder.
    num_labels:
        Number of Ekman classes. Defaults to 6 per the Ekman-6 label space
        defined in ``src/data/ekman_mapping.py``.
    head_dropout:
        Dropout applied *inside* the classification head, on top of the
        backbone's own dropout on the pooled representation.
    loss_fn:
        Callable ``(logits, targets) -> loss``. Defaults to
        ``nn.CrossEntropyLoss`` (mean reduction). Week 4 swaps this for a
        focal loss; the composite does not need any other change.
    class_weights:
        Optional ``(num_labels,)`` tensor passed to ``CrossEntropyLoss``.
        None by default — we do NOT reweight by class frequency for the
        main runs, to keep the Source-only / Mixed baselines faithful to
        the common practice in the domain-adaptation literature.
    """

    def __init__(
        self,
        backbone: DebertaBackbone,
        num_labels: int = 6,
        head_dropout: float = 0.1,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels

        self.head = ClassificationHead(
            hidden_size=backbone.hidden_size,
            num_labels=num_labels,
            dropout=head_dropout,
        )

        # Default loss: mean-reduced CE. We store it as a module attribute
        # so that ``model.to(device)`` will also move any class-weight
        # buffer attached to it.
        if loss_fn is None:
            self.loss_fn: nn.Module = nn.CrossEntropyLoss(
                weight=class_weights, reduction="mean"
            )
        else:
            # Custom loss callable (e.g. FocalLoss in Week 4). Wrap in an
            # nn.Module only if the caller hasn't already, so that it takes
            # part in ``.to(device)`` / ``.state_dict()``.
            self.loss_fn = (
                loss_fn if isinstance(loss_fn, nn.Module) else _CallableLossWrapper(loss_fn)
            )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        # Accept (and ignore) extra keys the collator produces — notably
        # ``domain_labels`` — so that a single batch dict can be passed to
        # both this model and the DANN model in Week 3 without filtering.
        **_unused_batch_fields,
    ) -> EmotionClassifierOutput:
        """Encode → classify → optionally compute loss.

        Parameters
        ----------
        input_ids, attention_mask, token_type_ids:
            Forwarded to the backbone. See ``DebertaBackbone.forward``.
        labels:
            ``(batch,)`` long tensor of Ekman ids. If provided, the
            returned output contains a CE loss; otherwise ``loss`` is None
            (inference path).

        Returns
        -------
        EmotionClassifierOutput
        """
        enc: BackboneOutput = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = self.head(enc.pooled)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return EmotionClassifierOutput(
            loss=loss,
            logits=logits,
            features=enc.pooled,
        )

    # ------------------------------------------------------------------ #
    # Inference helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return argmax predictions. Thin convenience wrapper for eval code."""
        was_training = self.training
        self.eval()
        try:
            out = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            return out.logits.argmax(dim=-1)
        finally:
            if was_training:
                self.train()


# ---------------------------------------------------------------------------
# Internal: wrap a bare callable into an nn.Module
# ---------------------------------------------------------------------------
class _CallableLossWrapper(nn.Module):
    """Adapter that turns an arbitrary ``(logits, targets) -> loss`` callable
    into an ``nn.Module`` so it participates in ``model.to(device)``.

    Private because callers should either pass an ``nn.Module`` loss
    (preferred) or let ``EmotionClassifier`` pick the default
    ``CrossEntropyLoss``; this wrapper exists only to keep the API flexible.
    """

    def __init__(self, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        super().__init__()
        self._fn = fn

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._fn(logits, targets)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_emotion_classifier(
    config: Union[BackboneConfig, Mapping[str, Any], str, Path],
    class_weights: Optional[torch.Tensor] = None,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> EmotionClassifier:
    """Convenience factory: YAML/dict → ready-to-train ``EmotionClassifier``.

    Uses the same config surface as ``build_backbone`` so that a script can
    do::

        model = build_emotion_classifier("configs/default.yaml")

    and get a Source-only / Mixed baseline wired up end-to-end.
    """
    if isinstance(config, BackboneConfig):
        cfg = config
        backbone = DebertaBackbone(cfg)
    else:
        # ``build_backbone`` handles Mapping / path / BackboneConfig already.
        backbone = build_backbone(config)
        cfg = backbone.config

    return EmotionClassifier(
        backbone=backbone,
        num_labels=cfg.num_labels,
        head_dropout=cfg.dropout,
        loss_fn=loss_fn,
        class_weights=class_weights,
    )


__all__ = [
    "ClassificationHead",
    "EmotionClassifier",
    "EmotionClassifierOutput",
    "build_emotion_classifier",
]
