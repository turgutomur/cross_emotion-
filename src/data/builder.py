"""
Top-level dataset builder.

Single entry point: `build_datasets(config)` returns a dict of per-domain
splits that can be fed into either Protocol A or Protocol B builders.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import logging

from .goemotions_loader import load_goemotions
from .isear_loader import load_isear
from .wassa_loader import load_wassa21
from .types import DatasetName, EmotionExample


logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing."""
    # Paths (raw data roots)
    isear_csv: Path = Path("data/raw/isear.csv")
    wassa_dir: Path = Path("data/raw/wassa21/")

    # GoEmotions options
    goemotions_strict_single_ekman: bool = True

    # ISEAR split options
    isear_split_seed: int = 42
    isear_train_frac: float = 0.8
    isear_val_frac: float = 0.1

    # Text filters
    min_text_length: int = 10          # ISEAR, WASSA
    goemotions_min_text_length: int = 3  # GoEmotions tweets are shorter

    # Which datasets to load (for partial debugging)
    include_goemotions: bool = True
    include_isear: bool = True
    include_wassa: bool = True


def build_datasets(
    config: DataConfig,
) -> Dict[str, Dict[str, List[EmotionExample]]]:
    """
    Load and harmonize all three datasets.

    Returns a nested dict:
        {
            'goemotions': {'train': [...], 'val': [...], 'test': [...]},
            'isear':      {'train': [...], 'val': [...], 'test': [...]},
            'wassa21':    {'train': [...], 'val': [...], 'test': [...]},
        }

    Missing loaders (per config flags) are simply omitted.
    """
    per_domain: Dict[str, Dict[str, List[EmotionExample]]] = {}

    if config.include_goemotions:
        per_domain[DatasetName.GOEMOTIONS.value] = _load_goemotions_all(config)

    if config.include_isear:
        per_domain[DatasetName.ISEAR.value] = _load_isear_all(config)

    if config.include_wassa:
        per_domain[DatasetName.WASSA.value] = _load_wassa_all(config)

    _log_summary(per_domain)
    return per_domain


def _load_goemotions_all(config: DataConfig):
    return {
        "train": load_goemotions(
            "train",
            strict_single_ekman=config.goemotions_strict_single_ekman,
            min_text_length=config.goemotions_min_text_length,
        ),
        "val": load_goemotions(
            "validation",
            strict_single_ekman=config.goemotions_strict_single_ekman,
            min_text_length=config.goemotions_min_text_length,
        ),
        "test": load_goemotions(
            "test",
            strict_single_ekman=config.goemotions_strict_single_ekman,
            min_text_length=config.goemotions_min_text_length,
        ),
    }


def _load_isear_all(config: DataConfig):
    return {
        split: load_isear(
            config.isear_csv,
            split=split,
            split_seed=config.isear_split_seed,
            train_frac=config.isear_train_frac,
            val_frac=config.isear_val_frac,
            min_text_length=config.min_text_length,
        )
        for split in ("train", "val", "test")
    }


def _load_wassa_all(config: DataConfig):
    return {
        split: load_wassa21(
            config.wassa_dir,
            split=split,
            min_text_length=config.min_text_length,
        )
        for split in ("train", "val", "test")
    }


def _log_summary(per_domain):
    """Print a summary of dataset sizes and label distributions."""
    from collections import Counter
    logger.info("=" * 70)
    logger.info("DATASET BUILD SUMMARY")
    logger.info("=" * 70)
    for domain, splits in per_domain.items():
        logger.info(f"\n[{domain}]")
        for split_name, items in splits.items():
            label_counts = Counter(e.ekman_label for e in items)
            total = len(items)
            dist_str = ", ".join(
                f"{lbl}={cnt} ({cnt/total*100:.1f}%)"
                for lbl, cnt in sorted(label_counts.items())
            ) if total > 0 else "(empty)"
            logger.info(f"  {split_name}: {total} examples | {dist_str}")
    logger.info("=" * 70)
