"""
GoEmotions loader (Demszky et al., 2020).

Loads the simplified GoEmotions splits from HuggingFace (`go_emotions` config
`'simplified'`) and harmonizes labels to Ekman-6.

Preprocessing decisions (pre-registered, see ekman_mapping.py for full rationale):
  1. Drop examples with only the 'neutral' label.
  2. For multi-label examples: keep if all labels collapse to a single
     Ekman class; drop otherwise (strict mode).
  3. Lowercase and light whitespace cleaning; no further text modification.
"""
from typing import List, Optional
import logging

from .ekman_mapping import LABEL2ID, map_goemotions
from .types import DatasetName, EmotionExample, example_from_record


logger = logging.getLogger(__name__)


# Original GoEmotions label names in the order used by the HF dataset.
# This is the reference index-to-name map for the 'simplified' config.
# Source: https://huggingface.co/datasets/google-research-datasets/go_emotions
GOEMOTIONS_ORIGINAL_LABELS: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]


def load_goemotions(
    split: str = "train",
    strict_single_ekman: bool = True,
    min_text_length: int = 3,
) -> List[EmotionExample]:
    """
    Load GoEmotions and map to Ekman-6.

    Args:
        split: 'train', 'validation', or 'test'.
        strict_single_ekman: see map_goemotions().
        min_text_length: drop examples with fewer than this many characters.

    Returns:
        List of EmotionExample with harmonized labels.
    """
    from datasets import load_dataset

    logger.info(f"Loading GoEmotions split={split}...")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified",
                      split=split)

    examples: List[EmotionExample] = []
    n_dropped_neutral = 0
    n_dropped_multiekman = 0
    n_dropped_short = 0

    for idx, row in enumerate(ds):
        text = (row["text"] or "").strip()
        if len(text) < min_text_length:
            n_dropped_short += 1
            continue

        label_ids = row["labels"]  # list[int]
        label_names = [GOEMOTIONS_ORIGINAL_LABELS[i] for i in label_ids]

        ekman = map_goemotions(label_names, strict_single_ekman=strict_single_ekman)
        if ekman is None:
            if label_names == ["neutral"] or (
                len(label_names) == 1 and label_names[0] == "neutral"
            ):
                n_dropped_neutral += 1
            else:
                n_dropped_multiekman += 1
            continue

        ex = example_from_record(
            text=text,
            ekman_label=ekman,
            domain=DatasetName.GOEMOTIONS.value,
            orig_label="|".join(label_names),
            split=_normalize_split(split),
            example_id=f"goe-{split}-{idx}",
        )
        examples.append(ex)

    logger.info(
        f"GoEmotions[{split}]: kept {len(examples)}, "
        f"dropped neutral-only={n_dropped_neutral}, "
        f"dropped multi-Ekman={n_dropped_multiekman}, "
        f"dropped too-short={n_dropped_short}"
    )
    return examples


def _normalize_split(split: str) -> str:
    """HF uses 'validation'; we normalize to 'val' internally."""
    return "val" if split == "validation" else split
