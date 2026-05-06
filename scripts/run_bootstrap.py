#!/usr/bin/env python3
"""
Pairwise bootstrap significance testing across all trained method pairs.

Usage (run on Colab after all training seeds are complete):
    python scripts/run_bootstrap.py \\
        --config configs/default.yaml \\
        --output-dir /content/drive/MyDrive/cross_emotion_data/outputs \\
        --metric macro_f1 \\
        --n-resamples 1000 \\
        --seed 42

What this does
--------------
1. Scans output_dir/results/*.csv to discover (method, protocol, target) cells.
2. For each cell that has >= 3 seeds, loads the trained checkpoints and runs
   inference on the corresponding test set to collect (predictions, labels).
3. Groups cells by (protocol, target) and compares every pair of methods
   that share the same protocol+target.
4. Writes output_dir/bootstrap/pairwise_pvalues.csv.

Cells with n_seeds < 3 are skipped with a warning.
Cells with n_seeds == 1 get "n_seeds_insufficient" in the p_value column
to prevent spurious single-seed bootstrap results from entering the paper.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.builder import DataConfig, build_datasets
from src.data.protocols import build_lodo_protocol, build_mixed_protocol
from src.data.types import EmotionExample
from src.evaluation.bootstrap import aggregate_bootstrap_across_seeds
from src.models.backbone import BackboneConfig, DebertaBackbone
from src.models.classifier import EmotionClassifier
from src.utils.logging_utils import setup_logging

logger = setup_logging(level="INFO", name="run_bootstrap")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pairwise bootstrap significance tests for trained methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument(
        "--metric", default="macro_f1",
        choices=["macro_f1"],
        help="Metric to compare.  Currently only macro_f1 is supported.",
    )
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------

def _parse_csv_filename(stem: str) -> Optional[Tuple[str, str, str]]:
    """Parse '{method}_{protocol}_{target}' from a results CSV filename stem.

    Returns (method, protocol, target) or None if the stem doesn't match.

    The protocol token is always 'lodo' or 'mixed'; we use the last two
    underscore-delimited tokens to extract it, then treat everything before
    as the method name (since method names may themselves contain underscores,
    e.g. 'source_only_focal').
    """
    parts = stem.split("_")
    # Target is always the last part; protocol is always second-to-last.
    # Works for: source_only_mixed_mixed, dann_focal_lodo_wassa21, etc.
    if len(parts) < 3:
        return None
    target = parts[-1]
    protocol = parts[-2]
    if protocol not in ("lodo", "mixed"):
        return None
    method = "_".join(parts[:-2])
    return method, protocol, target


def discover_cells(
    results_dir: Path,
) -> Dict[Tuple[str, str, str], List[int]]:
    """Return a mapping of (method, protocol, target) → list of seeds found."""
    cells: Dict[Tuple[str, str, str], List[int]] = {}
    if not results_dir.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return cells

    for csv_path in sorted(results_dir.glob("*.csv")):
        parsed = _parse_csv_filename(csv_path.stem)
        if parsed is None:
            logger.warning(f"Could not parse cell from filename: {csv_path.name} — skipping.")
            continue
        method, protocol, target = parsed
        seeds: List[int] = []
        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if "seed" in row:
                        seeds.append(int(row["seed"]))
        except Exception as exc:
            logger.warning(f"Failed to read {csv_path.name}: {exc} — skipping.")
            continue
        key = (method, protocol, target)
        cells[key] = seeds
        logger.info(f"  {method}/{protocol}/{target}: {len(seeds)} seeds {seeds}")

    return cells


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_test_set(
    cfg: Dict[str, Any],
    protocol: str,
    target: str,
) -> List[EmotionExample]:
    """Build and return the test split for the given protocol/target."""
    data_cfg = cfg.get("data", {})
    data_config = DataConfig(
        isear_csv=Path(data_cfg.get("isear_csv", "data/raw/isear.csv")),
        wassa_dir=Path(data_cfg.get("wassa_dir", "data/raw/wassa21/")),
        goemotions_strict_single_ekman=bool(
            data_cfg.get("goemotions_strict_single_ekman", True)
        ),
        isear_split_seed=int(data_cfg.get("isear_split_seed", 42)),
        isear_train_frac=float(data_cfg.get("isear_train_frac", 0.8)),
        isear_val_frac=float(data_cfg.get("isear_val_frac", 0.1)),
        min_text_length=int(data_cfg.get("min_text_length", 10)),
        goemotions_min_text_length=int(
            data_cfg.get("goemotions_min_text_length", 3)
        ),
    )
    per_domain = build_datasets(data_config)
    if protocol == "mixed":
        split = build_mixed_protocol(per_domain)
    else:
        split = build_lodo_protocol(per_domain, target)
    return split.test


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _build_bare_model(
    cfg: Dict[str, Any],
    method: str,
    num_domains: int,
    train_examples: Optional[List[EmotionExample]] = None,
) -> torch.nn.Module:
    """Reconstruct the model architecture (no weights loaded yet)."""
    backbone_cfg = BackboneConfig.from_dict(cfg.get("model", {}))
    backbone = DebertaBackbone(backbone_cfg)

    task_loss_fn = None
    if method.endswith("_focal") and train_examples is not None:
        from src.training.losses import FocalLoss, compute_inverse_frequency_alpha
        focal_cfg = cfg.get("focal", {})
        gamma = float(focal_cfg.get("gamma", 2.0))
        alpha_mode = str(focal_cfg.get("alpha", "inverse_frequency"))
        if alpha_mode == "inverse_frequency":
            alpha = compute_inverse_frequency_alpha(
                train_examples, num_classes=backbone_cfg.num_labels
            )
        elif alpha_mode == "uniform":
            alpha = torch.ones(backbone_cfg.num_labels)
        else:
            alpha = None
        task_loss_fn = FocalLoss(
            gamma=gamma, alpha=alpha, reduction=str(focal_cfg.get("reduction", "mean"))
        )

    if method in ("dann", "dann_focal"):
        from src.models.dann import DANNConfig, DANNModel
        dann_cfg = DANNConfig.from_dict(cfg.get("dann", {}))
        return DANNModel(
            backbone=backbone,
            num_labels=backbone_cfg.num_labels,
            num_domains=num_domains,
            head_dropout=backbone_cfg.dropout,
            domain_hidden_dim=dann_cfg.domain_hidden_dim,
            task_loss_fn=task_loss_fn,
        )
    if method in ("cdan", "cdan_focal"):
        from src.models.cdan import CDANConfig, CDANModel
        cdan_cfg = CDANConfig.from_dict(cfg.get("cdan", {}))
        return CDANModel(
            backbone=backbone,
            num_labels=backbone_cfg.num_labels,
            num_domains=num_domains,
            head_dropout=backbone_cfg.dropout,
            domain_hidden_dim=cdan_cfg.domain_hidden_dim,
            use_random_projection=cdan_cfg.use_random_projection,
            projection_dim=cdan_cfg.projection_dim,
            entropy_weighting=cdan_cfg.entropy_weighting,
            task_loss_fn=task_loss_fn,
        )
    return EmotionClassifier(
        backbone=backbone,
        num_labels=backbone_cfg.num_labels,
        head_dropout=backbone_cfg.dropout,
        loss_fn=task_loss_fn,
    )


def _run_inference(
    model: torch.nn.Module,
    test_examples: List[EmotionExample],
    tokenizer: Any,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    """Run forward pass on test_examples; return (predictions, labels)."""
    from torch.utils.data import DataLoader
    from src.data.types import EmotionDataset, EmotionCollator

    max_length = cfg.get("model", {}).get("max_length", 256)
    batch_size = cfg.get("training", {}).get("batch_size", 16)

    dataset = EmotionDataset(test_examples)
    collator = EmotionCollator(tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    all_preds: List[int] = []
    all_labels: List[int] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(**batch)
            preds = out.logits.argmax(dim=-1).cpu().numpy().tolist()
            labels = batch["labels"].cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

    return all_preds, all_labels


def collect_seed_results(
    cfg: Dict[str, Any],
    method: str,
    protocol: str,
    target: str,
    seeds: List[int],
    output_dir: Path,
) -> List[Tuple[List[int], List[int]]]:
    """For each seed, load checkpoint and return (predictions, labels)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = f"{method}_{protocol}_{target}"

    test_examples = _load_test_set(cfg, protocol, target)
    logger.info(f"  Test set: {len(test_examples)} examples")

    # Determine num_domains from training split for adversarial methods.
    _ADVERSARIAL = ("dann", "cdan", "dann_focal", "cdan_focal")
    num_domains = 3  # Mixed default; LODO overrides below
    if protocol == "lodo" and method in _ADVERSARIAL:
        num_domains = 2  # source domains only

    seed_results: List[Tuple[List[int], List[int]]] = []

    for seed in seeds:
        ckpt_path = (
            output_dir / "checkpoints" / experiment_name / f"seed_{seed}" / "best.pt"
        )
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                "Run training with --output-dir pointing to the same directory."
            )

        model = _build_bare_model(cfg, method, num_domains, train_examples=None)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)

        backbone_cfg = BackboneConfig.from_dict(cfg.get("model", {}))
        from src.models.backbone import DebertaBackbone as _BB
        tokenizer = _BB(backbone_cfg).get_tokenizer()

        preds, labels = _run_inference(model, test_examples, tokenizer, cfg, device)
        seed_results.append((preds, labels))
        logger.info(f"    seed={seed} → {len(preds)} predictions collected")

    return seed_results


# ---------------------------------------------------------------------------
# Bootstrap p-value table
# ---------------------------------------------------------------------------

def _metric_fn_for(metric: str):
    from sklearn.metrics import f1_score
    if metric == "macro_f1":
        def fn(preds: np.ndarray, labels: np.ndarray) -> float:
            return float(f1_score(labels, preds, average="macro", zero_division=0))
        return fn
    raise ValueError(f"Unsupported metric: {metric!r}")


def run_all_comparisons(
    cfg: Dict[str, Any],
    cells: Dict[Tuple[str, str, str], List[int]],
    output_dir: Path,
    metric: str,
    n_resamples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Run bootstrap for every method pair sharing the same (protocol, target)."""
    metric_fn = _metric_fn_for(metric)

    # Group by (protocol, target)
    groups: Dict[Tuple[str, str], List[str]] = {}
    for (method, protocol, target), seeds in cells.items():
        key = (protocol, target)
        groups.setdefault(key, []).append(method)

    rows: List[Dict[str, Any]] = []

    for (protocol, target), methods in sorted(groups.items()):
        logger.info(f"\n=== Protocol={protocol}, Target={target} ===")
        n_seeds_per_method = {
            m: len(cells[(m, protocol, target)]) for m in methods
        }

        # Collect inference results for methods that have >= 3 seeds.
        inference_cache: Dict[str, List[Tuple[List[int], List[int]]]] = {}
        for method in methods:
            n = n_seeds_per_method[method]
            if n < 3:
                logger.warning(
                    f"  {method}: only {n} seed(s) — bootstrap skipped "
                    f"(need >= 3). Will write n_seeds_insufficient."
                )
                continue
            logger.info(f"  Collecting {n} seed(s) for {method} ...")
            try:
                inference_cache[method] = collect_seed_results(
                    cfg,
                    method=method,
                    protocol=protocol,
                    target=target,
                    seeds=cells[(method, protocol, target)],
                    output_dir=output_dir,
                )
            except FileNotFoundError as exc:
                logger.warning(f"  {method}: {exc} — skipping.")

        # Compare every pair
        for method_a, method_b in combinations(sorted(methods), 2):
            row_base: Dict[str, Any] = {
                "protocol": protocol,
                "target": target,
                "method_a": method_a,
                "method_b": method_b,
            }

            n_a = n_seeds_per_method[method_a]
            n_b = n_seeds_per_method[method_b]

            if n_a == 1 or n_b == 1:
                row_base.update({
                    "mean_a": "",
                    "mean_b": "",
                    "mean_diff": "",
                    "ci_low": "",
                    "ci_high": "",
                    "p_value": "n_seeds_insufficient",
                    "n_resamples": n_resamples,
                })
                rows.append(row_base)
                continue

            if method_a not in inference_cache or method_b not in inference_cache:
                logger.warning(
                    f"  Skipping {method_a} vs {method_b}: missing inference data."
                )
                continue

            result = aggregate_bootstrap_across_seeds(
                seed_results_a=inference_cache[method_a],
                seed_results_b=inference_cache[method_b],
                metric_fn=metric_fn,
                metric_name=metric,
                n_resamples=n_resamples,
                seed=seed,
            )

            row_base.update({
                "mean_a": round(result.mean_a, 6),
                "mean_b": round(result.mean_b, 6),
                "mean_diff": round(result.mean_diff, 6),
                "ci_low": round(result.ci_low, 6),
                "ci_high": round(result.ci_high, 6),
                "p_value": round(result.p_value, 6),
                "n_resamples": result.n_resamples,
            })
            rows.append(row_base)
            logger.info(f"  {method_a} vs {method_b}: {result.summary()}")

    return rows


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_bootstrap_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        logger.warning("No bootstrap results to write.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "protocol", "target", "method_a", "method_b",
        "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high",
        "p_value", "n_resamples",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"\nBootstrap results written → {output_path} ({len(rows)} pairs)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    with open(args.config, encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)

    logger.info(f"Scanning {output_dir / 'results'} for experiment CSVs ...")
    cells = discover_cells(output_dir / "results")
    if not cells:
        logger.error("No result CSVs found. Run training first.")
        return 1

    rows = run_all_comparisons(
        cfg=cfg,
        cells=cells,
        output_dir=output_dir,
        metric=args.metric,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )

    write_bootstrap_csv(rows, output_dir / "bootstrap" / "pairwise_pvalues.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
