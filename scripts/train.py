#!/usr/bin/env python3
"""
Training entry point for cross-dataset emotion classification.

Usage examples
--------------
# Week 2 baselines (CE only):
python scripts/train.py --config configs/default.yaml \\
    --protocol mixed --method mixed --seeds 42,123,456

python scripts/train.py --config configs/default.yaml \\
    --protocol lodo --target isear --method source_only --seeds 42,123,456

# Smoke run (1 epoch, 500 training examples, CPU-compatible):
python scripts/train.py --config configs/default.yaml \\
    --protocol mixed --method mixed --seeds 42 \\
    --max-train-examples 500 --epochs 1
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.builder import DataConfig, build_datasets
from src.data.protocols import build_mixed_protocol, build_lodo_protocol
from src.data.types import EmotionExample
from src.models.backbone import BackboneConfig, DebertaBackbone
from src.models.classifier import EmotionClassifier
from src.models.dann import build_dann_model
from src.training.trainer import Trainer
from src.utils.logging_utils import setup_logging


logger = setup_logging(level="INFO", name="train")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train cross-dataset emotion classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config (single source of truth for hyperparameters).",
    )
    p.add_argument(
        "--protocol", choices=["mixed", "lodo"], required=True,
        help="mixed = all domains pooled; lodo = leave-one-domain-out.",
    )
    p.add_argument(
        "--target", choices=["goemotions", "isear", "wassa21"], default=None,
        help="Target domain for LODO protocol.  Required when --protocol=lodo.",
    )
    p.add_argument(
        "--method", choices=["source_only", "mixed", "dann"], required=True,
        help=(
            "source_only: standard CE, no domain adaptation. "
            "mixed: same model trained on pooled mixed data. "
            "dann: domain-adversarial training with gradient reversal."
        ),
    )
    p.add_argument(
        "--seeds", type=str, default=None,
        help=(
            "Comma-separated seed list, e.g. '42,123,456'. "
            "Defaults to the seeds in the config."
        ),
    )
    p.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Root for checkpoints, logs, and results.",
    )
    # Debug / smoke-test overrides — these patch the config dict and are
    # not meant for production runs (configs are the canonical source).
    p.add_argument(
        "--max-train-examples", type=int, default=None,
        help="Subsample training set to N examples (smoke test only).",
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Override config epochs (smoke test only).",
    )
    p.add_argument(
        "--goemotions-only", action="store_true",
        help="Load only GoEmotions (no local files needed; for CI / quick tests).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_per_domain(cfg: Dict[str, Any], goemotions_only: bool = False):
    data_cfg_dict = cfg.get("data", {})
    data_config = DataConfig(
        isear_csv=Path(data_cfg_dict.get("isear_csv", "data/raw/isear.csv")),
        wassa_dir=Path(data_cfg_dict.get("wassa_dir", "data/raw/wassa21/")),
        goemotions_strict_single_ekman=bool(
            data_cfg_dict.get("goemotions_strict_single_ekman", True)
        ),
        isear_split_seed=int(data_cfg_dict.get("isear_split_seed", 42)),
        isear_train_frac=float(data_cfg_dict.get("isear_train_frac", 0.8)),
        isear_val_frac=float(data_cfg_dict.get("isear_val_frac", 0.1)),
        min_text_length=int(data_cfg_dict.get("min_text_length", 10)),
        goemotions_min_text_length=int(
            data_cfg_dict.get("goemotions_min_text_length", 3)
        ),
        include_goemotions=True,
        include_isear=not goemotions_only,
        include_wassa=not goemotions_only,
    )
    try:
        return build_datasets(data_config)
    except FileNotFoundError as exc:
        logger.warning(f"Dataset file missing ({exc}). Falling back to GoEmotions-only.")
        data_config.include_isear = False
        data_config.include_wassa = False
        return build_datasets(data_config)


# ---------------------------------------------------------------------------
# Model factories — rebuilt per seed to start fresh each run
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(
    cfg: Dict[str, Any],
    method: str = "source_only",
    num_train_domains: Optional[int] = None,
):
    """Return (model, tokenizer) for the requested method.

    For source_only / mixed, returns an ``EmotionClassifier``.
    For dann, returns a ``DANNModel`` that shares the same forward contract.

    ``num_train_domains`` must be passed for DANN: it is the number of
    domains actually present in the training split (2 for LODO, 3 for
    Mixed).  Using the config default (always 3) in LODO would leave one
    discriminator output class forever empty, causing the encoder to dump
    features there and domain_loss to explode.
    """
    backbone_cfg = BackboneConfig.from_dict(cfg.get("model", {}))
    backbone = DebertaBackbone(backbone_cfg)
    tokenizer = backbone.get_tokenizer()

    if method == "dann":
        from src.models.dann import DANNConfig, DANNModel
        dann_cfg = DANNConfig.from_dict(cfg.get("dann", {}))
        n_domains = num_train_domains if num_train_domains is not None else backbone_cfg.num_domains
        model = DANNModel(
            backbone=backbone,
            num_labels=backbone_cfg.num_labels,
            num_domains=n_domains,
            head_dropout=backbone_cfg.dropout,
            domain_hidden_dim=dann_cfg.domain_hidden_dim,
        )
    else:
        model = EmotionClassifier(
            backbone=backbone,
            num_labels=backbone_cfg.num_labels,
            head_dropout=backbone_cfg.dropout,
        )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Results CSV writer
# ---------------------------------------------------------------------------

def _flatten_metrics(
    metrics: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    """Flatten EvalResult values into a flat dict with prefixed keys."""
    from src.evaluation.metrics import EvalResult

    row: Dict[str, Any] = {}
    agg = metrics.get("aggregate")
    if isinstance(agg, EvalResult):
        row[f"{prefix}_macro_f1_aggregate"] = round(agg.macro_f1, 6)

    for key, val in metrics.items():
        if isinstance(val, EvalResult) and key != "aggregate":
            row[f"{prefix}_macro_f1_{key}"] = round(val.macro_f1, 6)
    return row


def write_results_csv(
    rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Append a per-seed result row to a CSV, creating the file if needed.

    Why append (not overwrite): each invocation of ``train.py`` runs one or
    more seeds for a single (method, protocol, target) cell of the result
    matrix. Production runs launch one CLI per seed so the previous seed's
    row must survive. Header is written only on first creation; subsequent
    appends reuse the existing header. If a future row introduces new keys
    (e.g. extra per-domain columns from a LODO run sharing a filename), we
    keep the original column order — extra keys are dropped from the row
    rather than corrupting the header. Same-cell runs always share schema,
    so this is safe in practice.
    """
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0

    if file_exists:
        # Reuse the existing header (column order is authoritative).
        with open(output_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            try:
                fieldnames = next(reader)
            except StopIteration:
                fieldnames = list(rows[0].keys())
                file_exists = False
    else:
        fieldnames = list(rows[0].keys())

    with open(output_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Results appended → {output_path} ({len(rows)} new row(s))")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.protocol == "lodo" and args.target is None:
        logger.error("--target is required when --protocol=lodo.")
        return 1

    # Load and optionally patch config
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)

    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
        logger.info(f"Config override: epochs={args.epochs} (smoke test)")

    # Seeds: CLI > config
    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = cfg.get("training", {}).get("seeds", [42, 123, 456])

    output_dir = Path(args.output_dir)

    # Load data once; reuse across seeds
    logger.info("Loading datasets...")
    per_domain = load_per_domain(cfg, goemotions_only=args.goemotions_only)

    if args.protocol == "mixed":
        protocol = build_mixed_protocol(per_domain)
    else:
        protocol = build_lodo_protocol(per_domain, args.target)

    logger.info(protocol.describe())

    train_examples: List[EmotionExample] = protocol.train
    val_examples: List[EmotionExample] = protocol.val
    test_examples: List[EmotionExample] = protocol.test

    if args.max_train_examples is not None and args.max_train_examples < len(train_examples):
        train_examples = train_examples[: args.max_train_examples]
        logger.info(
            f"Training set subsampled to {len(train_examples)} examples (smoke test)."
        )

    target_tag = args.target if args.protocol == "lodo" else "mixed"
    experiment_name = f"{args.method}_{args.protocol}_{target_tag}"
    csv_name = f"{args.method}_{args.protocol}_{target_tag}.csv"

    # For DANN: derive the local domain mapping from the actual train split.
    # LODO has 2 train domains; Mixed has 3.  The discriminator must be sized
    # to match so that every output class receives signal and domain_loss stays
    # near ln(K) instead of exploding toward 10+.
    if args.method == "dann":
        unique_domains: List[str] = sorted({e.domain for e in train_examples})
        num_train_domains: int = len(unique_domains)
        domain_to_idx: Dict[str, int] = {d: i for i, d in enumerate(unique_domains)}
        logger.info(
            f"DANN domain mapping ({num_train_domains} train domains): {domain_to_idx}"
        )
    else:
        num_train_domains = 0  # unused for non-DANN methods
        domain_to_idx = {}

    csv_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        logger.info(f"=== Seed {seed} ===")

        # Rebuild model from scratch for each seed (no weight carryover).
        model, tokenizer = build_model_and_tokenizer(
            cfg,
            method=args.method,
            num_train_domains=num_train_domains if args.method == "dann" else None,
        )

        trainer = Trainer(
            model=model,
            train_examples=train_examples,
            val_examples=val_examples,
            tokenizer=tokenizer,
            config=cfg,
            experiment_name=experiment_name,
            output_dir=output_dir,
            seed=seed,
            method=args.method,
            domain_to_idx=domain_to_idx if args.method == "dann" else None,
        )

        train_result = trainer.train()
        trainer.load_best_checkpoint()
        test_metrics = trainer.evaluate(test_examples)

        row: Dict[str, Any] = {
            "seed": seed,
            "epoch_best": train_result["best_epoch"],
        }
        row.update(_flatten_metrics(train_result["best_metrics"] or {}, prefix="val"))
        row.update(_flatten_metrics(test_metrics, prefix="test"))
        csv_rows.append(row)

        logger.info(
            f"Seed {seed} done — best epoch={train_result['best_epoch']}, "
            f"val_agg_F1={train_result['best_val_f1']:.4f}, "
            f"test_agg_F1={test_metrics['aggregate'].macro_f1:.4f}"
        )

    write_results_csv(csv_rows, output_dir / "results" / csv_name)
    logger.info("All seeds complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
