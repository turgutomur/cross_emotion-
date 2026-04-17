"""
Common types for cross-dataset emotion classification.

All three datasets are normalized to a shared `EmotionExample` schema
before any downstream processing. This keeps loaders decoupled from
models, splitters, and training code.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class DatasetName(str, Enum):
    """Canonical dataset identifiers. Used as domain labels for DANN."""
    GOEMOTIONS = "goemotions"
    ISEAR = "isear"
    WASSA = "wassa21"


DATASET_NAMES: List[str] = [d.value for d in DatasetName]
DATASET2ID = {name: idx for idx, name in enumerate(DATASET_NAMES)}
ID2DATASET = {idx: name for name, idx in DATASET2ID.items()}
NUM_DOMAINS = len(DATASET_NAMES)


@dataclass
class EmotionExample:
    """Unified schema for a single emotion-labeled text instance."""
    text: str                       # the input text
    ekman_label: str                # one of EKMAN_LABELS (e.g. 'joy')
    ekman_id: int                   # index in LABEL2ID (0..5)
    domain: str                     # DatasetName value (source dataset)
    domain_id: int                  # DATASET2ID index
    orig_label: str = ""            # original dataset label (for debugging)
    split: str = "train"            # 'train' / 'val' / 'test'
    example_id: str = ""            # unique ID (dataset-prefixed)
    extra: dict = field(default_factory=dict)  # optional metadata


def example_from_record(
    text: str,
    ekman_label: str,
    domain: str,
    orig_label: str = "",
    split: str = "train",
    example_id: str = "",
    **extra,
) -> EmotionExample:
    """Helper constructor that fills derived fields (ids) from strings."""
    from .ekman_mapping import LABEL2ID

    if ekman_label not in LABEL2ID:
        raise ValueError(f"Unknown Ekman label: {ekman_label!r}")
    if domain not in DATASET2ID:
        raise ValueError(f"Unknown domain: {domain!r}")

    return EmotionExample(
        text=text,
        ekman_label=ekman_label,
        ekman_id=LABEL2ID[ekman_label],
        domain=domain,
        domain_id=DATASET2ID[domain],
        orig_label=orig_label,
        split=split,
        example_id=example_id,
        extra=extra,
    )
