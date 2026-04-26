"""
Conditional Domain-Adversarial Network (CDAN) components.

Implements Long et al. (2018) "Conditional Adversarial Domain Adaptation"
(NeurIPS 2018) for cross-dataset emotion classification.

Why CDAN over plain DANN
------------------------
DANN aligns marginal feature distributions p(z) across domains.  CDAN
aligns *conditional* distributions p(z | y) by conditioning the
discriminator on class predictions.  The conditioning is realised as a
multilinear map: the outer product of the backbone feature vector and the
task softmax distribution.  This forces the adversary to match not just
"what the feature looks like" but "what the feature looks like *given*
the predicted emotion class" — a strictly tighter alignment target that
empirically outperforms DANN when source and target class distributions
differ.  Our regime is exactly this: ISEAR lacks surprise, WASSA/GoEmotions
differ in domain bias, so class-conditional alignment is the natural fit.

Random projection (Section 3.2)
--------------------------------
The raw outer product h ⊗ softmax(f) has dimension H × C = 768 × 6 = 4 608,
manageable for one linear layer but wasteful when stacked.  Two fixed random
matrices R_f ∈ ℝ^{H×d}, R_g ∈ ℝ^{C×d} project feature and probability
vectors separately; the interaction is their element-wise product
(h @ R_f) ⊙ (g @ R_g) ∈ ℝ^d.  By Claim 1 of Long et al. (2018),
E[(h @ R_f) ⊙ (g @ R_g)] equals a linear projection of the full outer
product, so the random-projected discriminator is an unbiased estimator.
Matrices are initialised from N(0, 1/√d) so each projected coordinate has
unit variance.  They are stored as non-trainable Parameters rather than
buffers so they survive ``load_state_dict`` without shape mismatches.

Entropy weighting / CDAN+E (Section 3.3)
------------------------------------------
Uncertain predictions (high-entropy softmax) carry less information about
class-conditional structure.  CDAN+E down-weights each sample's domain loss
by w_i = 1 + exp(−H(softmax_i)), then normalises so ∑w_i = batch_size
(preserving the expected loss magnitude).  Low entropy → w_i ≈ 2 (upweighted);
high entropy → w_i ≈ 1 (near-unchanged).  Disabled by default; enable with
``entropy_weighting: true`` in the config.  In our regime the task head is
confident on source examples but noisy on LODO target examples, so weighting
may help concentrate adversarial signal on confident, class-informative points.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BackboneConfig, DebertaBackbone, build_backbone
from .classifier import ClassificationHead
from .dann import SigmoidLambdaScheduler, grad_reverse


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CDANConfig:
    """Typed mirror of the ``cdan:`` block in ``configs/default.yaml``.

    Attributes
    ----------
    lambda_max:
        Upper bound of the adversarial gradient scale.  Matches the DANN
        default (0.5) so comparisons between methods are not confounded by
        differing adversarial budgets.
    annealing:
        Schedule shape.  Only ``"sigmoid"`` is supported; a ``"linear"``
        option is reserved for ablations.
    domain_hidden_dim:
        Hidden-layer width of the conditional discriminator.  256 is the
        Ganin et al. (2016) reference architecture, kept identical to DANN
        so that any F1 difference is attributable to conditioning alone.
    use_random_projection:
        If True, project features and softmax probs separately via fixed
        random matrices and take their element-wise product (Section 3.2).
        Reduces discriminator input from H × C = 4608 to ``projection_dim``.
    projection_dim:
        Target dimension ``d`` for the random projection.  1024 matches
        the paper's default.  Ignored when ``use_random_projection=False``.
    entropy_weighting:
        If True, enable CDAN+E (Section 3.3): weight each sample's domain
        loss by 1 + exp(−H(softmax_i)), normalised to preserve loss scale.
    gamma:
        Sigmoid steepness for the lambda schedule.  Fixed at 10.0 per Ganin
        et al. (2015); not ablated.
    """

    lambda_max: float = 0.5
    annealing: str = "sigmoid"
    domain_hidden_dim: int = 256
    use_random_projection: bool = True
    projection_dim: int = 1024
    entropy_weighting: bool = False
    gamma: float = 10.0

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "CDANConfig":
        """Build a ``CDANConfig`` from the ``cdan:`` YAML section dict."""
        return cls(
            lambda_max=float(cfg.get("lambda_max", cls.lambda_max)),
            annealing=str(cfg.get("annealing", cls.annealing)),
            domain_hidden_dim=int(cfg.get("domain_hidden_dim", cls.domain_hidden_dim)),
            use_random_projection=bool(cfg.get("use_random_projection", cls.use_random_projection)),
            projection_dim=int(cfg.get("projection_dim", cls.projection_dim)),
            entropy_weighting=bool(cfg.get("entropy_weighting", cls.entropy_weighting)),
            gamma=float(cfg.get("gamma", cls.gamma)),
        )


# ---------------------------------------------------------------------------
# Conditional Domain Discriminator
# ---------------------------------------------------------------------------

class ConditionalDiscriminator(nn.Module):
    """Domain discriminator conditioned on task-head class predictions.

    Input is the multilinear map features ⊗ softmax(task_logits), which
    conditions the adversary on class-conditional feature structure.

    Why multilinear over simple concatenation
    ------------------------------------------
    Concatenation [h; g] gives the discriminator two independent views.
    The outer product h ⊗ g = {h_i · g_j} exposes all pairwise products,
    encoding which feature dimensions activate for which emotion class.
    Long et al. (2018, Proposition 1) prove that minimising the CDAN
    objective with this multilinear input is equivalent to matching the
    full conditional distribution p(z | y), which cannot be achieved by
    any linear function of [h; g] alone.

    Random projection mode
    ----------------------
    Two fixed random matrices R_f ∈ ℝ^{H×d}, R_g ∈ ℝ^{C×d} replace the
    explicit outer product with (h @ R_f) ⊙ (g @ R_g) ∈ ℝ^d (Claim 1 of
    Long et al. 2018).  Matrices are initialised from N(0, 1/√d) so each
    projected coordinate has unit variance, and are stored as non-trainable
    Parameters so they are included in state_dict without consuming
    optimizer state.

    Architecture
    ------------
    ``GRL(interaction) → Linear(input_dim, 256) → ReLU → Dropout → Linear(256, K)``

    where input_dim is ``d`` (projection mode) or H × C = 4608 (full mode).
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        num_domains: int,
        domain_hidden_dim: int = 256,
        dropout: float = 0.1,
        use_random_projection: bool = True,
        projection_dim: int = 1024,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.num_domains = num_domains
        self.use_random_projection = use_random_projection
        self.projection_dim = projection_dim

        if use_random_projection:
            # Stored as Parameters (requires_grad=False) rather than buffers
            # so state_dict round-trips preserve the matrices when reloading
            # a checkpoint with load_state_dict (strict=True).
            self.R_f = nn.Parameter(
                torch.randn(hidden_size, projection_dim) / math.sqrt(projection_dim),
                requires_grad=False,
            )
            self.R_g = nn.Parameter(
                torch.randn(num_labels, projection_dim) / math.sqrt(projection_dim),
                requires_grad=False,
            )
            input_dim = projection_dim
        else:
            input_dim = hidden_size * num_labels

        self.layers = nn.Sequential(
            nn.Linear(input_dim, domain_hidden_dim),
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

    def _multilinear_map(
        self,
        features: torch.Tensor,
        softmax_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the class-conditional interaction vector.

        Parameters
        ----------
        features:
            ``(batch, H)`` pooled backbone output.
        softmax_probs:
            ``(batch, C)`` softmax output from the task head.

        Returns
        -------
        ``(batch, input_dim)`` interaction tensor.
        """
        if self.use_random_projection:
            # (B, d) via separate projections then element-wise product.
            # Using the pre-transposed matmul form (B, H) @ (H, d) = (B, d).
            proj_f = features @ self.R_f        # (B, d)
            proj_g = softmax_probs @ self.R_g   # (B, d)
            return proj_f * proj_g              # element-wise, (B, d)
        else:
            # Full outer product: unsqueeze to (B, H, 1) × (B, 1, C) then flatten.
            interaction = features.unsqueeze(2) * softmax_probs.unsqueeze(1)  # (B, H, C)
            return interaction.view(features.size(0), -1)  # (B, H*C)

    def forward(
        self,
        features: torch.Tensor,
        softmax_probs: torch.Tensor,
        lambda_: float,
    ) -> torch.Tensor:
        """Compute class-conditional interaction, apply GRL, classify domain.

        Parameters
        ----------
        features:
            ``(batch, H)`` pooled backbone output (pre-reversal).
        softmax_probs:
            ``(batch, C)`` task-head softmax probabilities.  Passed from the
            parent model so softmax is computed once and shared with entropy
            weighting.
        lambda_:
            Current gradient reversal scale (from ``SigmoidLambdaScheduler``).

        Returns
        -------
        ``(batch, num_domains)`` domain logits (pre-softmax).
        """
        interaction = self._multilinear_map(features, softmax_probs)
        reversed_interaction = grad_reverse(interaction, lambda_)
        return self.layers(reversed_interaction)


# ---------------------------------------------------------------------------
# CDAN Output
# ---------------------------------------------------------------------------

@dataclass
class CDANOutput:
    """Return type of ``CDANModel.forward``.

    Mirrors ``DANNOutput`` field-for-field so the trainer is method-agnostic:
    any code that works with DANNOutput works identically with CDANOutput.

    Attributes
    ----------
    task_loss:
        Cross-entropy loss on Ekman labels.  None in inference mode.
    domain_loss:
        Weighted (CDAN+E) or unweighted CE on domain labels.  None in eval
        mode — domain adaptation is a training-time signal only.
    total_loss:
        ``task_loss + lambda_ * domain_loss``.  None if task_loss is None.
        When domain_loss is None (eval), equals task_loss.
    task_logits:
        ``(batch, num_labels)`` emotion class logits.
    domain_logits:
        ``(batch, num_domains)`` domain class logits.
    features:
        ``(batch, hidden_size)`` pooled backbone representation (post-dropout).
        Retained for probing analyses and future visualisation.
    """

    task_loss: Optional[torch.Tensor]
    domain_loss: Optional[torch.Tensor]
    total_loss: Optional[torch.Tensor]
    task_logits: torch.Tensor
    domain_logits: torch.Tensor
    features: torch.Tensor

    @property
    def loss(self) -> Optional[torch.Tensor]:
        """Alias for ``total_loss``; satisfies the generic trainer contract."""
        return self.total_loss

    @property
    def logits(self) -> torch.Tensor:
        """Alias for ``task_logits``; satisfies the generic trainer contract."""
        return self.task_logits


# ---------------------------------------------------------------------------
# CDAN Model
# ---------------------------------------------------------------------------

class CDANModel(nn.Module):
    """Conditional Domain-Adversarial Network for cross-dataset emotion classification.

    Composite of:
        * ``DebertaBackbone`` — shared encoder (forced domain-invariant by adversarial
          training, like DANN, but conditioned on predicted emotion class)
        * ``ClassificationHead`` — Ekman-6 emotion task head
        * ``ConditionalDiscriminator`` — adversarial discriminator whose input is
          the multilinear map of backbone features and task softmax probabilities

    Why composition rather than inheritance
    ----------------------------------------
    Identical rationale to ``DANNModel``: each sub-module needs its own learning-
    rate group in the optimizer.  Backbone gets ``encoder_lr``; head and
    discriminator get ``head_lr`` (routed via the ``other_*`` groups in
    ``_get_parameter_groups``).  Inheritance would complicate that split.

    Forward compatibility with the generic trainer
    -----------------------------------------------
    The forward signature is drop-in compatible with both ``EmotionClassifier``
    and ``DANNModel``.  Calling ``model(**batch)`` without ``lambda_`` yields
    ``lambda_=0.0`` — no gradient reversal, no adversarial pressure.  This
    lets the same trainer code handle source_only, DANN, and CDAN without
    any special-casing on the model side.

    Entropy weighting (CDAN+E)
    --------------------------
    Controlled by the ``entropy_weighting`` flag passed at construction.
    When enabled, each sample's domain-loss contribution is scaled by
    w_i = 1 + exp(−H(softmax_i)), normalised so ∑w_i = B.  The entropy is
    computed from the same softmax_probs tensor already needed for the
    multilinear map, so there is no extra forward-pass cost.
    """

    def __init__(
        self,
        backbone: DebertaBackbone,
        num_labels: int = 6,
        num_domains: int = 3,
        head_dropout: float = 0.1,
        domain_hidden_dim: int = 256,
        use_random_projection: bool = True,
        projection_dim: int = 1024,
        entropy_weighting: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels
        self.num_domains = num_domains
        self.entropy_weighting = entropy_weighting

        self.head = ClassificationHead(
            hidden_size=backbone.hidden_size,
            num_labels=num_labels,
            dropout=head_dropout,
        )
        self.discriminator = ConditionalDiscriminator(
            hidden_size=backbone.hidden_size,
            num_labels=num_labels,
            num_domains=num_domains,
            domain_hidden_dim=domain_hidden_dim,
            dropout=head_dropout,
            use_random_projection=use_random_projection,
            projection_dim=projection_dim,
        )

        self._task_loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lambda_: float = 0.0,
        labels: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **_unused_batch_fields: Any,
    ) -> CDANOutput:
        """Encode → classify emotions → classify domain with conditional GRL.

        Parameters
        ----------
        input_ids, attention_mask, token_type_ids:
            Forwarded to ``DebertaBackbone``.
        lambda_:
            Gradient reversal scale for this step.  Zero at step 0 (pure task
            learning), rising to ``lambda_max`` over training via
            ``SigmoidLambdaScheduler`` in the trainer.
        labels:
            ``(batch,)`` Ekman label ids.  If None, ``task_loss`` is None.
        domain_labels:
            ``(batch,)`` dataset-origin ids (0=goemotions, 1=isear,
            2=wassa21) remapped to local 0..K-1 by the trainer.  Domain loss
            is only computed in training mode.

        Returns
        -------
        CDANOutput
        """
        enc = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        task_logits = self.head(enc.pooled)

        # Softmax probs are computed once here and passed to both the
        # discriminator (multilinear map) and entropy weighting (if enabled)
        # to avoid duplicate softmax operations.
        softmax_probs = F.softmax(task_logits, dim=-1)  # (B, C), detach not needed — GRL handles the reversal
        domain_logits = self.discriminator(enc.pooled, softmax_probs, lambda_)

        task_loss: Optional[torch.Tensor] = None
        domain_loss: Optional[torch.Tensor] = None
        total_loss: Optional[torch.Tensor] = None

        if labels is not None:
            task_loss = self._task_loss_fn(task_logits, labels)

        # Domain loss is a training-time signal only; skip in eval to avoid
        # spurious GRL application during inference.
        if self.training and domain_labels is not None:
            per_sample_ce = F.cross_entropy(domain_logits, domain_labels, reduction="none")

            if self.entropy_weighting:
                # CDAN+E: samples with low-entropy (confident) predictions get
                # higher weight.  Normalisation preserves the magnitude of
                # domain_loss relative to the unweighted baseline.
                entropy = -(softmax_probs * torch.log(softmax_probs + 1e-8)).sum(dim=-1)
                w = 1.0 + torch.exp(-entropy)       # (B,), range (1, 2)
                w = w * w.size(0) / w.sum()          # normalise so ∑w = B
                domain_loss = (w * per_sample_ce).mean()
            else:
                domain_loss = per_sample_ce.mean()

        if task_loss is not None:
            if domain_loss is not None:
                total_loss = task_loss + lambda_ * domain_loss
            else:
                total_loss = task_loss

        return CDANOutput(
            task_loss=task_loss,
            domain_loss=domain_loss,
            total_loss=total_loss,
            task_logits=task_logits,
            domain_logits=domain_logits,
            features=enc.pooled,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_cdan_model(
    backbone_config: Union[BackboneConfig, Mapping[str, Any], str, Path],
    cdan_config: Union[CDANConfig, Mapping[str, Any], None] = None,
) -> CDANModel:
    """Build a ``CDANModel`` from backbone and CDAN configs.

    Accepts the same flexible input types as ``build_backbone`` for the
    backbone config (``BackboneConfig`` instance, dict, or YAML path).
    The CDAN config can be a ``CDANConfig`` instance, a dict matching the
    ``cdan:`` YAML section, or None (all defaults).

    Typical usage::

        cfg = yaml.safe_load(open("configs/default.yaml"))
        model = build_cdan_model(cfg["model"], cfg.get("cdan", {}))
    """
    if isinstance(backbone_config, BackboneConfig):
        backbone = DebertaBackbone(backbone_config)
        bcfg = backbone_config
    else:
        backbone = build_backbone(backbone_config)
        bcfg = backbone.config

    if cdan_config is None:
        ccfg = CDANConfig()
    elif isinstance(cdan_config, CDANConfig):
        ccfg = cdan_config
    else:
        ccfg = CDANConfig.from_dict(cdan_config)

    return CDANModel(
        backbone=backbone,
        num_labels=bcfg.num_labels,
        num_domains=bcfg.num_domains,
        head_dropout=bcfg.dropout,
        domain_hidden_dim=ccfg.domain_hidden_dim,
        use_random_projection=ccfg.use_random_projection,
        projection_dim=ccfg.projection_dim,
        entropy_weighting=ccfg.entropy_weighting,
    )


__all__ = [
    "CDANConfig",
    "CDANModel",
    "CDANOutput",
    "ConditionalDiscriminator",
    "SigmoidLambdaScheduler",
    "build_cdan_model",
]
