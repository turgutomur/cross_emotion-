"""
Backbone encoder wrapper.

This module isolates *all* knowledge about which Transformer encoder we use
(DeBERTa-v3-base for main runs, -large for final validation) behind a single
``DebertaBackbone`` class. Downstream code — the classification head, the
DANN/CDAN domain discriminators, the trainer — interacts exclusively with:

    * ``DebertaBackbone.forward``  returning a ``BackboneOutput`` dataclass
    * ``DebertaBackbone.hidden_size``  property
    * ``DebertaBackbone.get_tokenizer``  class/instance method

Design rationale
----------------
1. **Why a thin wrapper and not ``AutoModelForSequenceClassification``?**
   The domain-adversarial methods in this project (DANN, CDAN) need direct
   access to the *pooled sentence representation* so that a gradient reversal
   layer + domain discriminator can be attached alongside the task head.
   ``AutoModelForSequenceClassification`` hides that representation inside a
   self-contained module that already includes a head; rolling our own
   wrapper keeps the pooled features explicit and reusable.

2. **Why first-token ([CLS]-like) pooling rather than the model's pooler?**
   DeBERTa-v3 does NOT ship with a pre-trained pooler layer — its
   ``pooler_output`` is either absent or randomly initialised depending on
   the ``transformers`` version. Using the first-token hidden state is the
   convention in the DeBERTa literature for classification and matches what
   ``AutoModelForSequenceClassification`` does internally.

3. **Why carry the config in a dataclass?**
   ``configs/default.yaml`` is the single source of truth for hyper-parameters.
   ``BackboneConfig.from_yaml`` reads the ``model:`` block directly so we do
   not scatter string literals (like the HF model name) across the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class BackboneConfig:
    """Typed mirror of the ``model:`` block in ``configs/default.yaml``.

    Attributes
    ----------
    name:
        HuggingFace model identifier (``microsoft/deberta-v3-base`` for the
        main experiments; ``-large`` is only used for the final validation
        runs on A100 per the compute-budget note in README.md).
    max_length:
        Tokenizer truncation length. Stored on the config (not hard-coded in
        the tokenizer factory) so that Protocol A and Protocol B can share
        one value and so that the sanity-check script stays consistent.
    dropout:
        Dropout rate applied to the pooled representation *before* the
        classification head. This matches the DeBERTa fine-tuning recipe in
        the original paper where a single dropout precedes the linear head.
    num_labels / num_domains:
        Kept on the backbone config for convenience (the heads that live in
        ``classifier.py`` / ``dann.py`` need these numbers at construction
        time and will usually receive the same ``BackboneConfig`` instance).
    pooling:
        ``"cls"`` → first-token hidden state (default, matches HF convention).
        ``"mean"`` → attention-masked mean of token embeddings. Reserved for
        future ablations; not used in the main paper.
    """

    name: str = "microsoft/deberta-v3-base"
    max_length: int = 256
    dropout: float = 0.1
    num_labels: int = 6
    num_domains: int = 3
    pooling: str = "cls"

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "BackboneConfig":
        """Build a ``BackboneConfig`` from a dict matching the YAML schema.

        The YAML uses ``backbone`` as the key for the HF model id; we rename
        it to ``name`` here so the Python attribute stays Pythonic. Any
        unknown keys are silently ignored so that adding new entries in
        ``default.yaml`` does not break existing checkpoints.
        """
        return cls(
            name=cfg.get("backbone", cls.name),
            max_length=int(cfg.get("max_length", cls.max_length)),
            dropout=float(cfg.get("dropout", cls.dropout)),
            num_labels=int(cfg.get("num_labels", cls.num_labels)),
            num_domains=int(cfg.get("num_domains", cls.num_domains)),
            pooling=str(cfg.get("pooling", cls.pooling)),
        )

    @classmethod
    def from_yaml(
        cls, yaml_path: Union[str, Path], section: str = "model"
    ) -> "BackboneConfig":
        """Load the ``model:`` section of a YAML file into a ``BackboneConfig``.

        Kept separate from ``from_dict`` so that callers that already parsed
        the YAML (e.g. the trainer) do not re-read the file.
        """
        import yaml  # local import: yaml is not needed in unit tests

        with open(yaml_path, "r", encoding="utf-8") as fh:
            full = yaml.safe_load(fh)
        if section not in full:
            raise KeyError(
                f"Section {section!r} not found in {yaml_path}; "
                f"available sections: {list(full)}"
            )
        return cls.from_dict(full[section])


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class BackboneOutput:
    """Structured return value for ``DebertaBackbone.forward``.

    Using a dataclass (instead of a bare tuple or dict) makes the downstream
    plumbing — CE classifier, DANN discriminator, CDAN multilinear map —
    safe to refactor: fields are accessed by name and ``torch.jit`` / type
    checkers can reason about the shape contract.

    Fields
    ------
    pooled:
        ``(batch, hidden_size)`` sentence-level representation *after*
        dropout. This is the feature that both the emotion classifier and
        the domain discriminator consume.
    sequence_output:
        ``(batch, seq_len, hidden_size)`` raw hidden states from the last
        transformer layer, BEFORE dropout. Exposed for ablations that pool
        differently (e.g. attention pooling) and for probing analyses.
    attention_mask:
        Passed through so heads that do mean-pooling or token-level
        supervision do not need to re-receive it from the batch.
    """

    pooled: torch.Tensor
    sequence_output: torch.Tensor
    attention_mask: torch.Tensor


# ---------------------------------------------------------------------------
# Backbone module
# ---------------------------------------------------------------------------
class DebertaBackbone(nn.Module):
    """Thin wrapper around a HuggingFace DeBERTa-v3 encoder.

    The wrapper does three things and nothing else:
        1. Load the pre-trained encoder via ``AutoModel.from_pretrained``.
        2. Extract a sentence-level ``pooled`` tensor with a configurable
           pooling strategy (first-token by default).
        3. Apply dropout on the pooled tensor so that every head that
           consumes it receives the same regularised representation.

    Keeping the backbone deliberately minimal is what lets the Source-only /
    Mixed / DANN / CDAN / +Focal variants share exactly one encoder
    implementation — only the head and loss change between methods.
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        # Deferred import: ``transformers`` pulls in a lot and we want
        # ``BackboneConfig`` / ``BackboneOutput`` to remain importable in
        # lightweight CI environments that only need the dataclasses.
        from transformers import AutoConfig, AutoModel

        self.config = config

        hf_config = AutoConfig.from_pretrained(config.name)
        # ``hidden_dropout_prob`` / ``attention_probs_dropout_prob`` are
        # intentionally NOT overridden here — we keep the pre-training
        # defaults inside the encoder and only add our own head-side
        # dropout on top, matching the DeBERTa fine-tuning recipe.
        self.encoder = AutoModel.from_pretrained(config.name, config=hf_config)
        self._hidden_size: int = int(hf_config.hidden_size)

        if config.pooling not in {"cls", "mean"}:
            raise ValueError(
                f"Unknown pooling strategy {config.pooling!r}; "
                "expected 'cls' or 'mean'."
            )
        self.pooling_strategy: str = config.pooling

        self.dropout = nn.Dropout(config.dropout)

    # ------------------------------------------------------------------ #
    # Public accessors
    # ------------------------------------------------------------------ #
    @property
    def hidden_size(self) -> int:
        """Return the encoder's hidden dimension (768 for -base, 1024 for -large).

        Exposed as a property so that heads constructed *after* the backbone
        (e.g. ``ClassificationHead(hidden_size=backbone.hidden_size, ...)``)
        do not need to know which concrete DeBERTa variant was loaded.
        """
        return self._hidden_size

    def get_tokenizer(self):
        """Return the matching ``AutoTokenizer``.

        DeBERTa-v3 uses a SentencePiece tokenizer; loading it via
        ``AutoTokenizer.from_pretrained(self.config.name)`` guarantees that
        the vocabulary and special tokens stay in sync with ``self.encoder``
        even if someone swaps ``-base`` for ``-large`` in the YAML.
        """
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.config.name)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> BackboneOutput:
        """Encode a batch of tokenised texts.

        Parameters
        ----------
        input_ids:
            ``(batch, seq_len)`` integer token ids produced by
            ``EmotionCollator`` in ``src/data/torch_dataset.py``.
        attention_mask:
            ``(batch, seq_len)`` 0/1 mask marking non-padding positions.
        token_type_ids:
            Optional; DeBERTa-v3 ignores segment ids in single-sentence
            classification, but we forward them when the collator provides
            them so the wrapper stays compatible with sentence-pair tasks.

        Returns
        -------
        BackboneOutput
            See the class docstring for the exact shape contract.
        """
        encoder_kwargs: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs, return_dict=True)
        sequence_output: torch.Tensor = outputs.last_hidden_state  # (B, T, H)

        if self.pooling_strategy == "cls":
            # First-token hidden state. DeBERTa-v3 prepends a [CLS]-like
            # token during tokenization, so position 0 is the sentence vec.
            pooled = sequence_output[:, 0, :]
        else:  # "mean"
            # Attention-masked mean pooling. Implemented here (rather than
            # in the head) so that all heads see an identical ``pooled``.
            mask = attention_mask.unsqueeze(-1).to(sequence_output.dtype)
            summed = (sequence_output * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts

        pooled = self.dropout(pooled)

        return BackboneOutput(
            pooled=pooled,
            sequence_output=sequence_output,
            attention_mask=attention_mask,
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def build_backbone(
    config: Union[BackboneConfig, Mapping[str, Any], str, Path],
) -> DebertaBackbone:
    """Build a ``DebertaBackbone`` from any of the accepted config forms.

    Accepts a ready-made ``BackboneConfig``, a YAML ``model:``-section dict,
    or a path to a YAML file. Provided so that trainer scripts can stay
    agnostic about how the config arrived (CLI arg vs. imported module vs.
    hard-coded dict in a unit test).
    """
    if isinstance(config, BackboneConfig):
        cfg = config
    elif isinstance(config, Mapping):
        cfg = BackboneConfig.from_dict(config)
    elif isinstance(config, (str, Path)):
        cfg = BackboneConfig.from_yaml(config)
    else:
        raise TypeError(
            f"Unsupported config type: {type(config).__name__}. "
            "Expected BackboneConfig, Mapping, str, or Path."
        )
    return DebertaBackbone(cfg)


__all__ = [
    "BackboneConfig",
    "BackboneOutput",
    "DebertaBackbone",
    "build_backbone",
]
