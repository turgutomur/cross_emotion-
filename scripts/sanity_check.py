#!/usr/bin/env python3
"""
Sanity check script — run this FIRST on Kaggle/Colab.

What it does:
  1. Loads GoEmotions (from HuggingFace — no manual download needed).
  2. Loads ISEAR (requires data/raw/isear.csv — see instructions below).
  3. Loads WASSA-21 (requires data/raw/wassa21/ — see instructions below).
  4. Prints per-dataset and per-class statistics.
  5. Builds Mixed and LODO protocol splits.
  6. Runs a tokenization dry-run with DeBERTa tokenizer.
  7. Prints a GO/NO-GO summary.

If ISEAR or WASSA-21 files are not yet available, the script will still
run for GoEmotions-only mode and tell you what's missing.

Usage:
  python scripts/sanity_check.py
  python scripts/sanity_check.py --goemotions-only
  python scripts/sanity_check.py --config configs/default.yaml

Data download instructions:
  GoEmotions: automatic (HuggingFace datasets library)
  ISEAR: download from https://www.unige.ch/cisa/research/materials-and-online-research/research-material/
         place as data/raw/isear.csv
  WASSA-21: obtain from shared task page or authors
         place files in data/raw/wassa21/{train.tsv, dev.tsv, test.tsv}
"""
import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ekman_mapping import EKMAN_LABELS, LABEL2ID, NUM_LABELS
from src.data.types import DatasetName, DATASET_NAMES, NUM_DOMAINS
from src.data.builder import DataConfig, build_datasets
from src.data.protocols import (
    build_mixed_protocol,
    build_all_lodo_protocols,
)
from src.utils.logging_utils import setup_logging


logger = setup_logging(level="INFO", name="sanity_check")


def parse_args():
    p = argparse.ArgumentParser(description="Data pipeline sanity check")
    p.add_argument("--goemotions-only", action="store_true",
                    help="Only load GoEmotions (no ISEAR/WASSA files needed)")
    p.add_argument("--isear-csv", type=str, default="/content/drive/MyDrive/cross_emotion_data/raw/isear.csv")
    p.add_argument("--wassa-dir", type=str, default="/content/drive/MyDrive/cross_emotion_data/raw/wassa21/")
    p.add_argument("--skip-tokenizer", action="store_true",
                    help="Skip tokenizer dry-run (saves time if no GPU)")
    return p.parse_args()


def print_section(title: str):
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)


def print_label_distribution(examples, dataset_name: str, split: str):
    """Print label distribution for a set of examples."""
    if not examples:
        logger.info(f"  {dataset_name}/{split}: (empty)")
        return

    counts = Counter(e.ekman_label for e in examples)
    total = len(examples)
    parts = []
    for label in EKMAN_LABELS:
        c = counts.get(label, 0)
        pct = c / total * 100 if total > 0 else 0
        parts.append(f"{label}={c} ({pct:.1f}%)")
    logger.info(f"  {dataset_name}/{split}: {total} total | {' | '.join(parts)}")

    # Check for missing classes
    missing = [l for l in EKMAN_LABELS if counts.get(l, 0) == 0]
    if missing:
        logger.warning(f"    ⚠ Missing classes: {missing}")


def print_text_stats(examples, dataset_name: str):
    """Print text length statistics."""
    if not examples:
        return
    lengths = [len(e.text.split()) for e in examples]
    import numpy as np
    arr = np.array(lengths)
    logger.info(
        f"  {dataset_name} text length (words): "
        f"mean={arr.mean():.1f}, median={np.median(arr):.1f}, "
        f"min={arr.min()}, max={arr.max()}, std={arr.std():.1f}"
    )


def run_tokenizer_check(examples, max_samples=100):
    """Dry-run tokenization to verify DeBERTa compatibility."""
    print_section("TOKENIZER DRY-RUN")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        sample = examples[:max_samples]
        texts = [e.text for e in sample]
        encoded = tokenizer(
            texts, padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        )
        logger.info(f"  Tokenized {len(sample)} examples successfully.")
        logger.info(f"  input_ids shape: {encoded['input_ids'].shape}")
        logger.info(f"  Max token length in batch: {encoded['attention_mask'].sum(dim=1).max().item()}")

        # Check truncation rate
        full = tokenizer(texts, truncation=False, padding=False)
        n_truncated = sum(1 for ids in full["input_ids"] if len(ids) > 256)
        logger.info(f"  Truncated at 256 tokens: {n_truncated}/{len(sample)} "
                     f"({n_truncated/len(sample)*100:.1f}%)")
        return True
    except Exception as e:
        logger.error(f"  Tokenizer check failed: {e}")
        return False


def main():
    args = parse_args()

    print_section("EKMAN-6 LABEL SPACE")
    logger.info(f"  Labels: {EKMAN_LABELS}")
    logger.info(f"  Label2ID: {LABEL2ID}")
    logger.info(f"  Num labels: {NUM_LABELS}")
    logger.info(f"  Num domains: {NUM_DOMAINS}")
    logger.info(f"  Domains: {DATASET_NAMES}")

    # Build config
    config = DataConfig(
        isear_csv=Path(args.isear_csv),
        wassa_dir=Path(args.wassa_dir),
        include_goemotions=True,
        include_isear=not args.goemotions_only,
        include_wassa=not args.goemotions_only,
    )

    # Load datasets
    print_section("LOADING DATASETS")
    try:
        per_domain = build_datasets(config)
    except FileNotFoundError as e:
        logger.warning(f"  Some datasets missing: {e}")
        logger.info("  Falling back to GoEmotions-only mode...")
        config.include_isear = False
        config.include_wassa = False
        per_domain = build_datasets(config)

    # Per-dataset statistics
    print_section("PER-DATASET STATISTICS")
    all_examples = []
    for domain_name, splits in per_domain.items():
        for split_name, examples in splits.items():
            print_label_distribution(examples, domain_name, split_name)
            all_examples.extend(examples)
        # Text stats (on train only)
        if "train" in splits:
            print_text_stats(splits["train"], domain_name)

    # Protocol A: Mixed
    if len(per_domain) >= 2:
        print_section("PROTOCOL A: MIXED")
        mixed = build_mixed_protocol(per_domain)
        logger.info(mixed.describe())

        # Protocol B: LODO
        print_section("PROTOCOL B: LEAVE-ONE-DATASET-OUT")
        lodo_splits = build_all_lodo_protocols(per_domain)
        for lodo in lodo_splits:
            logger.info(lodo.describe())
            logger.info("")
    else:
        logger.info("  (Skipping protocols — need ≥2 datasets)")

    # Tokenizer check
    if not args.skip_tokenizer and all_examples:
        tok_ok = run_tokenizer_check(all_examples)
    else:
        tok_ok = None

    # Summary
    print_section("SANITY CHECK SUMMARY")
    checks = {
        "Ekman-6 labels valid": NUM_LABELS == 6,
        "GoEmotions loaded": DatasetName.GOEMOTIONS.value in per_domain,
        "ISEAR loaded": DatasetName.ISEAR.value in per_domain,
        "WASSA-21 loaded": DatasetName.WASSA.value in per_domain,
        "Tokenizer OK": tok_ok if tok_ok is not None else "SKIPPED",
    }
    all_ok = True
    for check, status in checks.items():
        icon = "✅" if status is True else ("⏭" if status == "SKIPPED" else "❌")
        logger.info(f"  {icon} {check}: {status}")
        if status is False and "ISEAR" not in check and "WASSA" not in check:
            all_ok = False

    if all_ok:
        logger.info("")
        logger.info("  🟢 Pipeline is ready. Proceed to training.")
    else:
        logger.info("")
        logger.info("  🔴 Some checks failed. Fix before proceeding.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
