"""
Domain-Adversarial Neural Network (DANN) components.

Implements the method from Ganin & Lempitsky (2015) "Unsupervised Domain
Adaptation by Backpropagation" and Ganin et al. (2016) "Domain-Adversarial
Training of Neural Networks" for cross-dataset emotion classification.

Why DANN here
-------------
The Week 2 baseline shows a ~29-point gap between Mixed protocol F1 and LODO
mean F1, indicating that the encoder learns dataset-specific features that
hurt generalisation.  DANN counters this by attaching a domain discriminator
whose gradient is *reversed* before flowing into the encoder, forcing it to
learn domain-invariant representations without requiring any target-domain
labels.

Design notes
------------
The GRL uses ``torch.autograd.Function`` so the reversal is transparent to
higher-order gradients and avoids the extra tensor allocation of the
``x * -lambda + x.detach() * (1 + lambda)`` trick.

The three-class discriminator (one class per dataset) uses a single hidden
layer of 256 units — the reference architecture in Ganin et al. (2016).
Deeper discriminators can overfit the domain signal and collapse gradient
reversal into noise.

Sigmoid lambda annealing (gamma=10, p = step/total_steps) gives near-zero
lambda at the start of training (when the encoder hasn't learned useful
features yet) and rises to lambda_max as training converges, matching the
schedule in the original paper and its empirical motivation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch
import torch.nn as nn

from .backbone import BackboneConfig, DebertaBackbone, build_backbone
from .classifier import ClassificationHead


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DANNConfig:
    """Typed mirror of the ``dann:`` block in ``configs/default.yaml``.

    Attributes
    ----------
    lambda_max:
        Upper bound of the annealing schedule.  Values > 1 amplify the
        adversarial gradient; the Ganin paper uses 1.0 for most experiments.
    annealing:
        Schedule shape.  Only ``"sigmoid"`` is supported in Week 3; a
        ``"linear"`` option is reserved for ablations.
    domain_hidden_dim:
        Hidden-layer width for the domain discriminator.  256 is the default
        in Ganin et al. (2016) and is pre-registered for this project.
    gamma:
        Steepness of the sigmoid ramp.  Fixed at 10.0 per the Ganin paper.
        Not exposed in ``default.yaml`` because we do not ablate it.
    """

    lambda_max: float = 1.0
    annealing: str = "sigmoid"
    domain_hidden_dim: int = 256
    gamma: float = 10.0

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "DANNConfig":
        """Build a ``DANNConfig`` from the ``dann:`` YAML section dict."""
        return cls(
            lambda_max=float(cfg.get("lambda_max", cls.lambda_max)),
            annealing=str(cfg.get("annealing", cls.annealing)),
            domain_hidden_dim=int(cfg.get("domain_hidden_dim", cls.domain_hidden_dim)),
            gamma=float(cfg.get("gamma", cls.gamma)),
        )


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradientReversalFn(torch.autograd.Function):
    """Identity forward, negated-and-scaled backward.

    Why a custom ``autograd.Function`` rather than the multiply-by-(-lambda)
    trick: the trick (``x * -lambda + x.detach() * (1 + lambda)``) creates
    two leaf tensors and a sum node, polluting the gradient tape and
    allocating extra memory.  A custom Function is transparent to the tape
    and avoids the extra allocation.

    The ``lambda_`` scalar is stored via ``ctx.save_for_backward`` as a
    zero-dimensional tensor so that the backward pass can retrieve it
    without capturing a Python float in the closure — which could prevent
    garbage collection on large models.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        lambda_: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ):
        (lambda_t,) = ctx.saved_tensors
        # Return None for the lambda_ input (not a tensor in the graph).
        return -lambda_t.item() * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal with scale ``lambda_`` to tensor ``x``."""
    return GradientReversalFn.apply(x, lambda_)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Domain Discriminator
# ---------------------------------------------------------------------------

class DomainDiscriminator(nn.Module):
    """Three-class domain classifier attached after the GRL.

    Architecture
    ------------
    ``GRL(features) → Linear(hidden_size, 256) → ReLU → Dropout → Linear(256, 3)``

    The discriminator receives the *gradient-reversed* pooled representation
    from the backbone, so its gradient w.r.t. backbone parameters is negated
    by the GRL.  This forces the encoder to produce features that fool the
    discriminator — the adversarial pressure that drives domain invariance.

    Why one hidden layer (not two or zero)
    ---------------------------------------
    A linear (0-layer) discriminator is too weak to detect domain signal in
    DeBERTa features, reducing adversarial pressure.  Two hidden layers risk
    overfitting the domain signal and saturating gradient reversal.  One
    layer of 256 units is the Ganin et al. (2016) reference architecture and
    is pre-registered for this project.
    """

    def __init__(
        self,
        hidden_size: int,
        domain_hidden_dim: int = 256,
        num_domains: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.domain_hidden_dim = domain_hidden_dim
        self.num_domains = num_domains

        self.layers = nn.Sequential(
            nn.Linear(hidden_size, domain_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(domain_hidden_dim, num_domains),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor, lambda_: float) -> torch.Tensor:
        """Apply GRL then classify domain.

        Parameters
        ----------
        features:
            ``(batch, hidden_size)`` pooled backbone output (pre-reversal).
        lambda_:
            Current gradient reversal scale.  Passed here (not stored on the
            module) so it can change each step without requiring mutable
            module state.

        Returns
        -------
        ``(batch, num_domains)`` domain logits (pre-softmax).
        """
        reversed_features = grad_reverse(features, lambda_)
        return self.layers(reversed_features)


# ---------------------------------------------------------------------------
# DANN Output
# ---------------------------------------------------------------------------

@dataclass
class DANNOutput:
    """Return type of ``DANNModel.forward``.

    Attributes
    ----------
    task_loss:
        Cross-entropy loss on Ekman labels.  None in inference mode
        (``labels=None``).
    domain_loss:
        Cross-entropy loss on domain labels.  None when ``domain_labels`` is
        None or when the model is in eval mode (domain adaptation is a
        training-time signal only).
    total_loss:
        ``task_loss + lambda_ * domain_loss``.  None if task_loss is None.
        When domain_loss is None (eval mode), equals task_loss.  This is
        the scalar the trainer calls ``.backward()`` on.
    task_logits:
        ``(batch, num_labels)`` emotion class logits.
    domain_logits:
        ``(batch, num_domains)`` domain class logits.
    features:
        ``(batch, hidden_size)`` pooled backbone representation
        (post-dropout).  Retained for CDAN (Week 4) and probing analyses.
    """

    task_loss: Optional[torch.Tensor]
    domain_loss: Optional[torch.Tensor]
    total_loss: Optional[torch.Tensor]
    task_logits: torch.Tensor
    domain_logits: torch.Tensor
    features: torch.Tensor

    # -- Generic trainer contract aliases -----------------------------------

    @property
    def loss(self) -> Optional[torch.Tensor]:
        """Alias for ``total_loss``; satisfies the generic trainer contract."""
        return self.total_loss

    @property
    def logits(self) -> torch.Tensor:
        """Alias for ``task_logits``; satisfies the generic trainer contract."""
        return self.task_logits


# ---------------------------------------------------------------------------
# DANN Model
# ---------------------------------------------------------------------------

class DANNModel(nn.Module):
    """Domain-Adversarial Neural Network for cross-dataset emotion classification.

    Composite of:
        * ``DebertaBackbone`` — shared encoder (becomes domain-invariant after training)
        * ``ClassificationHead`` — Ekman-6 emotion task head
        * ``DomainDiscriminator`` — adversarial three-class domain classifier

    Why composition rather than inheritance
    ----------------------------------------
    Each sub-module needs its own learning-rate group in the optimizer
    (``_get_parameter_groups`` in ``trainer.py`` handles backbone / head /
    discriminator).  Inheritance would require inspecting parameter names
    to split lr groups.

    Forward compatibility with the generic trainer
    -----------------------------------------------
    The forward signature is backwards-compatible with ``EmotionClassifier``:
    the required args (``input_ids``, ``attention_mask``) are the same; DANN-
    specific args (``lambda_``, ``domain_labels``) are keyword-only with
    safe defaults.  Vanilla eval code that calls ``model(**batch)`` without
    passing ``lambda_`` gets ``lambda_=0.0`` — no gradient reversal, no
    domain adaptation.
    """

    def __init__(
        self,
        backbone: DebertaBackbone,
        num_labels: int = 6,
        num_domains: int = 3,
        head_dropout: float = 0.1,
        domain_hidden_dim: int = 256,
        task_loss_fn: Optional[nn.Module] = None,
    ):
        """Initialise DANN components.

        Parameters
        ----------
        task_loss_fn:
            Loss function for the Ekman emotion task.  Defaults to
            ``nn.CrossEntropyLoss`` (mean reduction), which preserves full
            backward compatibility with all Week 2/3 results.  Pass a
            ``FocalLoss`` instance to enable the dann_focal method without
            subclassing.  The domain discriminator always uses plain CE
            regardless of this argument: domain labels are balanced by
            construction in our LODO setup, so focal modulation there is
            both unnecessary and potentially destabilising.
        """
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels
        self.num_domains = num_domains

        self.head = ClassificationHead(
            hidden_size=backbone.hidden_size,
            num_labels=num_labels,
            dropout=head_dropout,
        )
        self.discriminator = DomainDiscriminator(
            hidden_size=backbone.hidden_size,
            domain_hidden_dim=domain_hidden_dim,
            num_domains=num_domains,
            dropout=head_dropout,
        )

        self._task_loss_fn: nn.Module = (
            task_loss_fn if task_loss_fn is not None
            else nn.CrossEntropyLoss(reduction="mean")
        )
        self._domain_loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lambda_: float = 0.0,
        labels: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **_unused_batch_fields: Any,
    ) -> DANNOutput:
        """Encode → classify emotions → classify domain (with reversal).

        Parameters
        ----------
        input_ids, attention_mask, token_type_ids:
            Forwarded to ``DebertaBackbone``.
        lambda_:
            Gradient reversal scale for this step.  Controlled by
            ``SigmoidLambdaScheduler`` in the trainer.  Zero at step 0
            (pure task learning), rising to ``lambda_max`` over training.
        labels:
            ``(batch,)`` Ekman label ids.  If None, ``task_loss`` is None.
        domain_labels:
            ``(batch,)`` dataset-origin ids (0=goemotions, 1=isear,
            2=wassa21).  Domain loss is only computed in training mode and
            when this is not None.

        Returns
        -------
        DANNOutput
        """
        enc = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        task_logits = self.head(enc.pooled)
        domain_logits = self.discriminator(enc.pooled, lambda_)

        task_loss: Optional[torch.Tensor] = None
        domain_loss: Optional[torch.Tensor] = None
        total_loss: Optional[torch.Tensor] = None

        if labels is not None:
            task_loss = self._task_loss_fn(task_logits, labels)

        # Domain loss is a training signal only; skip in eval to avoid
        # spurious GRL application during inference.
        if self.training and domain_labels is not None:
            domain_loss = self._domain_loss_fn(domain_logits, domain_labels)

        if task_loss is not None:
            if domain_loss is not None:
                total_loss = task_loss + lambda_ * domain_loss
            else:
                total_loss = task_loss

        return DANNOutput(
            task_loss=task_loss,
            domain_loss=domain_loss,
            total_loss=total_loss,
            task_logits=task_logits,
            domain_logits=domain_logits,
            features=enc.pooled,
        )


# ---------------------------------------------------------------------------
# Lambda Scheduler
# ---------------------------------------------------------------------------

class SigmoidLambdaScheduler:
    """Sigmoid annealing schedule for the gradient reversal lambda.

    ``lambda(p) = lambda_max * (2 / (1 + exp(-gamma * p)) - 1)``

    where ``p = current_step / total_steps ∈ [0, 1]``.

    Key values:
        * ``p = 0.0`` → lambda = 0   (no reversal; task-only warm-up)
        * ``p = 0.5`` → lambda ≈ 0.46 * lambda_max
        * ``p = 1.0`` → lambda ≈ lambda_max

    Why sigmoid not linear
    ----------------------
    A linear ramp applies full adversarial pressure too early, before the
    encoder has learned meaningful emotion features — which can destabilise
    training.  The sigmoid's slow start lets the task head converge first,
    then steadily increases domain-invariance pressure, matching the
    intuition and empirical results in Ganin et al. (2015, 2016).
    """

    def __init__(self, lambda_max: float = 1.0, gamma: float = 10.0):
        self.lambda_max = lambda_max
        self.gamma = gamma

    def __call__(self, p: float) -> float:
        """Return lambda for training progress fraction ``p ∈ [0, 1]``."""
        return self.lambda_max * (2.0 / (1.0 + math.exp(-self.gamma * p)) - 1.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dann_model(
    backbone_config: Union[BackboneConfig, Mapping[str, Any], str, Path],
    dann_config: Union[DANNConfig, Mapping[str, Any], None] = None,
    task_loss_fn: Optional[nn.Module] = None,
) -> DANNModel:
    """Build a ``DANNModel`` from backbone and DANN configs.

    Accepts the same flexible input types as ``build_backbone`` for the
    backbone config (``BackboneConfig`` instance, dict, or YAML path).
    The DANN config can be a ``DANNConfig`` instance, a dict matching the
    ``dann:`` section of the YAML, or None (all defaults).

    Typical usage::

        cfg = yaml.safe_load(open("configs/default.yaml"))
        model = build_dann_model(cfg["model"], cfg.get("dann", {}))
    """
    if isinstance(backbone_config, BackboneConfig):
        backbone = DebertaBackbone(backbone_config)
        bcfg = backbone_config
    else:
        backbone = build_backbone(backbone_config)
        bcfg = backbone.config

    if dann_config is None:
        dcfg = DANNConfig()
    elif isinstance(dann_config, DANNConfig):
        dcfg = dann_config
    else:
        dcfg = DANNConfig.from_dict(dann_config)

    return DANNModel(
        backbone=backbone,
        num_labels=bcfg.num_labels,
        num_domains=bcfg.num_domains,
        head_dropout=bcfg.dropout,
        domain_hidden_dim=dcfg.domain_hidden_dim,
        task_loss_fn=task_loss_fn,
    )


__all__ = [
    "DANNConfig",
    "DANNModel",
    "DANNOutput",
    "DomainDiscriminator",
    "GradientReversalFn",
    "SigmoidLambdaScheduler",
    "build_dann_model",
    "grad_reverse",
]
