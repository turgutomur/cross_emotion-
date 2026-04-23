"""
Method-agnostic training and evaluation harness.

Designed so that Week 3's DANN/CDAN models plug in without touching this
file: any model whose forward(**batch) returns an object with .loss and
.logits attributes is trainable here.  All hyperparameters are read from
the config dict — nothing is hard-coded.

Key design choices
------------------
* Two optimizer parameter groups (encoder_lr / head_lr) with weight-decay
  exclusion for LayerNorm weights and biases, matching the standard
  DeBERTa fine-tuning recipe.
* Scheduler implemented via PyTorch LambdaLR (no transformers dependency
  in this module) — linear warmup then linear decay.
* GradScaler is instantiated unconditionally; it is a no-op when
  enabled=False, so the fp16 flag only controls the autocast context.
* Early stopping patience is on *aggregate* val macro-F1, not per-domain.
  Aggregate is the fairest single signal because domain sizes differ.
* DANN support: pass ``method="dann"`` to enable gradient reversal.
  Lambda is updated once per optimizer-step window (constant within a
  gradient accumulation block) using sigmoid annealing.  ``global_step``
  increments even when AMP skips the optimizer step so lambda grows
  monotonically regardless of gradient overflow events.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..data.torch_dataset import EmotionCollator, EmotionTorchDataset
from ..data.types import EmotionExample, ID2DATASET
from ..evaluation.metrics import EvalResult, compute_metrics
from ..utils.seed import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optimizer helper
# ---------------------------------------------------------------------------

def _get_parameter_groups(
    model: nn.Module,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """Split parameters into (encoder|head|other) × (decay|no-decay) groups.

    LayerNorm weights and all bias terms are excluded from weight decay.
    This prevents over-regularising normalisation layers, matching the
    standard BERT / DeBERTa fine-tuning recipe used in the literature.

    Falls back gracefully if the model does not expose backbone/head
    attributes (e.g. DANN composites might rename them).
    """
    no_decay_keywords = {
        "bias", "LayerNorm.weight", "layer_norm.weight",
        "layernorm.weight", "norm.weight",
    }

    backbone = getattr(model, "backbone", None)
    head = getattr(model, "head", None)
    backbone_ids = {id(p) for p in backbone.parameters()} if backbone else set()
    head_ids = {id(p) for p in head.parameters()} if head else set()

    enc_decay: List[torch.Tensor] = []
    enc_nodecay: List[torch.Tensor] = []
    head_decay: List[torch.Tensor] = []
    head_nodecay: List[torch.Tensor] = []
    other_decay: List[torch.Tensor] = []
    other_nodecay: List[torch.Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_wd = any(kw in name for kw in no_decay_keywords)
        pid = id(param)
        if pid in backbone_ids:
            (enc_nodecay if no_wd else enc_decay).append(param)
        elif pid in head_ids:
            (head_nodecay if no_wd else head_decay).append(param)
        else:
            # Extra params: domain discriminator head in DANN/CDAN, etc.
            (other_nodecay if no_wd else other_decay).append(param)

    groups = []
    if enc_decay:
        groups.append({"params": enc_decay, "lr": encoder_lr, "weight_decay": weight_decay})
    if enc_nodecay:
        groups.append({"params": enc_nodecay, "lr": encoder_lr, "weight_decay": 0.0})
    if head_decay:
        groups.append({"params": head_decay, "lr": head_lr, "weight_decay": weight_decay})
    if head_nodecay:
        groups.append({"params": head_nodecay, "lr": head_lr, "weight_decay": 0.0})
    if other_decay:
        groups.append({"params": other_decay, "lr": head_lr, "weight_decay": weight_decay})
    if other_nodecay:
        groups.append({"params": other_nodecay, "lr": head_lr, "weight_decay": 0.0})
    return groups


def _make_linear_warmup_decay(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Pure-PyTorch linear warmup + linear decay scheduler.

    Kept in this module rather than pulled from transformers so the trainer
    can be imported in environments that have PyTorch but not the full
    transformers stack (e.g. lightweight CI, unit tests).
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        remaining = float(current_step - warmup_steps)
        total_remaining = float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - remaining / total_remaining)

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Training and evaluation harness for cross-dataset emotion classification.

    Parameters
    ----------
    model:
        Any nn.Module whose forward(**batch) returns an object with .loss
        (scalar tensor) and .logits (B × num_labels tensor).
        EmotionClassifier for Week 2; DANN/CDAN composites in Week 3 will
        satisfy the same contract.
    train_examples, val_examples:
        Raw EmotionExample lists from the protocol builder.
    tokenizer:
        HuggingFace tokenizer matching the backbone's vocabulary.
    config:
        Full YAML parsed to a nested dict.  All hyperparameters are read
        from here; nothing is hard-coded in the trainer itself.
    experiment_name:
        Used for log and checkpoint sub-directory names.
    output_dir:
        Root directory; checkpoints and logs are written under it.
    seed:
        Passed to set_seed() at the start of train(); also embedded in
        the checkpoint path so multi-seed runs don't clobber each other.
    """

    def __init__(
        self,
        model: nn.Module,
        train_examples: List[EmotionExample],
        val_examples: List[EmotionExample],
        tokenizer: Any,
        config: Dict[str, Any],
        experiment_name: str,
        output_dir: str | Path = "outputs",
        seed: int = 42,
        method: str = "source_only",
    ) -> None:
        self.model = model
        self.train_examples = train_examples
        self.val_examples = val_examples
        self.tokenizer = tokenizer
        self.cfg = config
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.method = method

        train_cfg = config.get("training", {})
        self.epochs: int = int(train_cfg.get("epochs", 15))
        self.batch_size: int = int(train_cfg.get("batch_size", 16))
        self.grad_accum: int = int(train_cfg.get("gradient_accumulation", 2))
        self.patience: int = int(train_cfg.get("early_stopping_patience", 3))

        encoder_lr: float = float(train_cfg.get("encoder_lr", 1e-5))
        head_lr: float = float(train_cfg.get("head_lr", 2e-5))
        weight_decay: float = float(train_cfg.get("weight_decay", 0.01))
        warmup_ratio: float = float(train_cfg.get("warmup_ratio", 0.1))

        model_cfg = config.get("model", {})
        self.max_length: int = int(model_cfg.get("max_length", 256))

        eval_cfg = config.get("evaluation", {})
        self.restrict_to_present: bool = bool(eval_cfg.get("restrict_to_present", True))

        # 1. Move model to device FIRST so optimizer param groups are on device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # AMP is only meaningful on CUDA; disable silently on CPU even if
        # the config says fp16=true.  Stored as an instance variable so it
        # is computed once and shared by _train_epoch and evaluate.
        self.use_amp: bool = bool(train_cfg.get("fp16", False)) and torch.cuda.is_available()

        # 2. Build optimizer AFTER model is on device.
        param_groups = _get_parameter_groups(model, encoder_lr, head_lr, weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups)

        # 3. Build scheduler AFTER optimizer.
        # Steps per epoch is approximate — loader length isn't known until
        # we build it, so we use ceil(n / effective_batch) as an estimate.
        effective_batch = self.batch_size * self.grad_accum
        steps_per_epoch = max(1, (len(train_examples) + effective_batch - 1) // effective_batch)
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        self.scheduler = _make_linear_warmup_decay(self.optimizer, warmup_steps, total_steps)
        self.total_optimizer_steps: int = total_steps

        # 4. GradScaler is device-agnostic when enabled=False (no-op on CPU).
        #    Uses the new torch.amp namespace to avoid the deprecation warning.
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Guard: AMP requires fp32 parameters.  Any .half() / .to(fp16) call
        # on the model before this point would silently corrupt training.
        first_param_dtype = next(self.model.parameters()).dtype
        assert first_param_dtype == torch.float32, (
            f"Model params must stay fp32 for AMP; got {first_param_dtype}. "
            "Remove any .half() / .to(torch.float16) calls before constructing the Trainer."
        )

        # 5. DANN-specific state: lambda schedule and step counter.
        #    global_step increments on every optimizer-step attempt so lambda
        #    grows monotonically even when AMP skips an update.
        self.global_step: int = 0
        self.current_lambda: float = 0.0
        if method == "dann":
            from ..models.dann import DANNConfig, SigmoidLambdaScheduler
            dann_cfg = DANNConfig.from_dict(config.get("dann", {}))
            self.dann_scheduler: Optional["SigmoidLambdaScheduler"] = SigmoidLambdaScheduler(
                lambda_max=dann_cfg.lambda_max,
                gamma=dann_cfg.gamma,
            )
        else:
            self.dann_scheduler = None

        self.checkpoint_dir = (
            self.output_dir / "checkpoints" / experiment_name / f"seed_{seed}"
        )
        self.log_dir = self.output_dir / "logs" / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def train(self) -> Dict[str, Any]:
        """Run training with early stopping; return best-epoch summary.

        Returns a dict with keys:
            best_epoch      int
            best_val_f1     float  (aggregate macro-F1)
            best_metrics    dict   (domain → EvalResult, plus 'aggregate')
        """
        from ..utils.logging_utils import setup_logging

        log_file = self.log_dir / f"seed_{self.seed}.log"
        run_logger = setup_logging(
            level=self.cfg.get("logging", {}).get("level", "INFO"),
            log_file=log_file,
            name=f"{self.experiment_name}.seed{self.seed}",
        )

        set_seed(self.seed)
        train_loader = self._make_dataloader(self.train_examples, shuffle=True)

        best_val_f1 = -1.0
        best_epoch = -1
        patience_counter = 0
        best_metrics: Optional[Dict[str, Any]] = None

        for epoch in range(1, self.epochs + 1):
            train_info = self._train_epoch(train_loader)
            train_loss = train_info["loss"]
            val_metrics = self.evaluate(self.val_examples)
            agg_f1: float = val_metrics["aggregate"].macro_f1

            per_ds_parts = [
                f"{ds}={res.macro_f1:.4f}"
                for ds, res in val_metrics.items()
                if isinstance(res, EvalResult) and ds != "aggregate"
            ]
            if self.method == "dann":
                dann_suffix = (
                    f"  task_loss={train_info['task_loss']:.4f}"
                    f"  domain_loss={train_info['domain_loss']:.4f}"
                    f"  lambda={train_info['lambda_now']:.4f}"
                )
            else:
                dann_suffix = ""
            run_logger.info(
                f"epoch={epoch:02d}  train_loss={train_loss:.4f}  "
                f"val_macro_f1={agg_f1:.4f}  [{' | '.join(per_ds_parts)}]"
                f"{dann_suffix}"
            )

            if agg_f1 > best_val_f1:
                best_val_f1 = agg_f1
                best_epoch = epoch
                patience_counter = 0
                best_metrics = val_metrics
                self._save_checkpoint(epoch, best_val_f1)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    run_logger.info(
                        f"Early stopping: patience={self.patience} exhausted at epoch {epoch}. "
                        f"Best val macro-F1={best_val_f1:.4f} at epoch {best_epoch}."
                    )
                    break

        run_logger.info(
            f"Training complete. Best epoch={best_epoch}, "
            f"val macro-F1={best_val_f1:.4f}"
        )
        return {
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "best_metrics": best_metrics,
        }

    @torch.no_grad()
    def evaluate(
        self,
        examples: List[EmotionExample],
    ) -> Dict[str, Any]:
        """Evaluate on any list of EmotionExample.

        Groups predictions by domain and returns per-domain EvalResult
        objects plus an 'aggregate' key over all examples.  The
        restrict_to_present flag (from config) is forwarded to
        compute_metrics so that ISEAR-as-target runs (no surprise class)
        do not deflate macro-F1 with a spurious zero.

        Returns
        -------
        dict mapping:
            '<domain_name>' → EvalResult
            'aggregate'     → EvalResult
            'val_loss'      → float  (mean CE loss; nan if labels absent)
        """
        self.model.eval()
        loader = self._make_dataloader(examples, shuffle=False)

        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        all_domain_ids: List[np.ndarray] = []
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.model(**batch)

            all_preds.append(out.logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
            all_domain_ids.append(batch["domain_labels"].cpu().numpy())

            if out.loss is not None:
                total_loss += out.loss.item()
                n_batches += 1

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        domain_ids = np.concatenate(all_domain_ids)

        results: Dict[str, Any] = {}
        for did in sorted(np.unique(domain_ids).tolist()):
            mask = domain_ids == did
            domain_name = ID2DATASET.get(int(did), f"domain_{did}")
            results[domain_name] = compute_metrics(
                labels[mask],
                preds[mask],
                restrict_to_present=self.restrict_to_present,
            )

        results["aggregate"] = compute_metrics(
            labels, preds, restrict_to_present=self.restrict_to_present
        )
        results["val_loss"] = total_loss / max(n_batches, 1)

        self.model.train()
        return results

    def load_best_checkpoint(self) -> None:
        """Restore model weights from the best checkpoint saved during train()."""
        path = self.checkpoint_dir / "best.pt"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint at {path}. Run train() first.")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        logger.info(
            f"Loaded best checkpoint: epoch={ckpt['epoch']}, "
            f"val macro-F1={ckpt['val_macro_f1']:.4f}"
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _make_dataloader(
        self,
        examples: List[EmotionExample],
        shuffle: bool = False,
    ) -> DataLoader:
        dataset = EmotionTorchDataset(examples)
        collator = EmotionCollator(tokenizer=self.tokenizer, max_length=self.max_length)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def _train_epoch(self, loader: DataLoader) -> Dict[str, Any]:
        """One epoch of training with gradient accumulation and fp16.

        Returns a dict with at minimum ``{"loss": float}``.  For
        ``method="dann"`` it also contains ``task_loss``, ``domain_loss``,
        and ``lambda_now`` so the caller can log them separately.

        Loss values are means per gradient-update step (not per micro-step),
        so they are comparable regardless of the grad_accum setting.
        """
        self.model.train()
        total_loss = 0.0
        total_task_loss = 0.0
        total_domain_loss = 0.0
        update_steps = 0
        self.optimizer.zero_grad(set_to_none=True)

        for micro_step, batch in enumerate(loader):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # DANN: inject the current lambda_ at the start of each
            # accumulation window so it stays constant within the window.
            if self.method == "dann":
                if micro_step % self.grad_accum == 0:
                    p = self.global_step / max(1, self.total_optimizer_steps)
                    self.current_lambda = self.dann_scheduler(p)  # type: ignore[misc]
                batch["lambda_"] = self.current_lambda

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.model(**batch)
                # Divide before backward so accumulated gradients equal a
                # single full-batch gradient at the optimizer step.
                loss = out.loss / self.grad_accum

            self.scaler.scale(loss).backward()

            is_last_micro = (micro_step + 1) % self.grad_accum == 0
            is_last_batch = (micro_step + 1) == len(loader)

            if is_last_micro or is_last_batch:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Undo the /grad_accum so the logged value is comparable
                # regardless of the accumulation setting.
                total_loss += loss.item() * self.grad_accum
                update_steps += 1

                if self.method == "dann":
                    # global_step always increments, even when AMP skipped
                    # the optimizer step, so lambda grows monotonically.
                    self.global_step += 1
                    if out.task_loss is not None:
                        total_task_loss += out.task_loss.item()
                    if out.domain_loss is not None:
                        total_domain_loss += out.domain_loss.item()

        result: Dict[str, Any] = {"loss": total_loss / max(update_steps, 1)}
        if self.method == "dann":
            result["task_loss"] = total_task_loss / max(update_steps, 1)
            result["domain_loss"] = total_domain_loss / max(update_steps, 1)
            result["lambda_now"] = self.current_lambda
        return result

    def _save_checkpoint(self, epoch: int, val_macro_f1: float) -> None:
        checkpoint = {
            "model_state": self.model.state_dict(),
            "config": self.cfg,
            "val_macro_f1": val_macro_f1,
            "epoch": epoch,
            "seed": self.seed,
        }
        path = self.checkpoint_dir / "best.pt"
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved → {path} (epoch={epoch}, F1={val_macro_f1:.4f})")


__all__ = ["Trainer", "_get_parameter_groups"]
