"""
ISEAR loader (Scherer & Wallbott, 1994).

ISEAR does not come with official train/val/test splits, and is not directly
available on HuggingFace in a canonical form. We expect the user to place
the raw ISEAR CSV at `data/raw/isear.csv` (see README) and we perform a
stratified 80/10/10 split with fixed seed for reproducibility.

Preprocessing decisions:
  1. Exclude shame and guilt (contested mapping to sadness).
  2. Keep 5 Ekman classes: joy, fear, anger, sadness, disgust.
  3. Note: surprise is NOT present in ISEAR (inherent dataset limitation).
  4. Light whitespace cleaning; drop empty or placeholder texts
     (ISEAR has some "NO RESPONSE" entries).
"""
from pathlib import Path
from typing import List, Optional, Tuple
import csv
import logging
import re

from .ekman_mapping import map_isear, ISEAR_EXCLUDED_LABELS
from .types import DatasetName, EmotionExample, example_from_record


logger = logging.getLogger(__name__)


# Known placeholder / non-response patterns in ISEAR raw data
PLACEHOLDER_PATTERNS = [
    re.compile(r"^\s*NO RESPONSE\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*NO RESP\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\[?\s*NO\s+RESPONSE\s*\]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*N/?A\s*$", re.IGNORECASE),
    re.compile(r"^\s*-+\s*$"),
    re.compile(r"^\s*\.+\s*$"),
]


def _is_placeholder(text: str) -> bool:
    return any(p.match(text) for p in PLACEHOLDER_PATTERNS)


def load_isear(
    csv_path: Path,
    split: str = "train",
    split_seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    min_text_length: int = 10,
    text_column_candidates: Tuple[str, ...] = ("SIT", "situation", "text", "content"),
    label_column_candidates: Tuple[str, ...] = ("EMOT", "emotion", "Field1", "label"),
) -> List[EmotionExample]:
    """
    Load ISEAR and return a deterministic split.

    Args:
        csv_path: path to the raw ISEAR CSV.
        split: 'train', 'val', or 'test'.
        split_seed: fixed seed for the stratified split (reproducibility).
        train_frac, val_frac: split proportions (test_frac = 1 - train - val).
        min_text_length: drop short texts (likely placeholders or truncated).
        text_column_candidates: candidate column names for the situation/text.
        label_column_candidates: candidate column names for the emotion.

    Returns:
        List of EmotionExample for the requested split.

    Raises:
        FileNotFoundError if csv_path does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ISEAR CSV not found at {csv_path}. "
            f"Download from https://www.unige.ch/cisa/research/materials-and-online-research/research-material/ "
            f"(or a mirror) and place it at this path."
        )

    logger.info(f"Loading ISEAR from {csv_path}...")

    # Read raw rows
    rows = _read_isear_csv(csv_path, text_column_candidates, label_column_candidates)
    logger.info(f"ISEAR raw rows: {len(rows)}")

    # Filter + harmonize
    harmonized: List[Tuple[str, str, str]] = []  # (text, ekman_label, orig_label)
    n_excluded_label = 0
    n_placeholder = 0
    n_short = 0

    for text, orig_label in rows:
        text = (text or "").strip()
        orig_label_lower = (orig_label or "").strip().lower()

        if orig_label_lower in ISEAR_EXCLUDED_LABELS:
            n_excluded_label += 1
            continue

        if not text or _is_placeholder(text):
            n_placeholder += 1
            continue

        if len(text) < min_text_length:
            n_short += 1
            continue

        ekman = map_isear(orig_label_lower)
        if ekman is None:
            # Unknown label, skip
            continue

        harmonized.append((text, ekman, orig_label_lower))

    logger.info(
        f"ISEAR harmonized: kept {len(harmonized)}, "
        f"dropped shame/guilt={n_excluded_label}, "
        f"placeholders={n_placeholder}, short={n_short}"
    )

    # Deterministic stratified split
    train_items, val_items, test_items = _stratified_split(
        harmonized, train_frac=train_frac, val_frac=val_frac, seed=split_seed
    )
    split_items = {"train": train_items, "val": val_items, "test": test_items}[split]

    examples = [
        example_from_record(
            text=text,
            ekman_label=ekman,
            domain=DatasetName.ISEAR.value,
            orig_label=orig,
            split=split,
            example_id=f"isear-{split}-{i}",
        )
        for i, (text, ekman, orig) in enumerate(split_items)
    ]
    logger.info(f"ISEAR[{split}]: {len(examples)} examples")
    return examples


def _read_isear_csv(
    path: Path,
    text_cols: Tuple[str, ...],
    label_cols: Tuple[str, ...],
) -> List[Tuple[str, str]]:
    """Read ISEAR CSV, auto-detecting the text and label columns."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        # ISEAR CSVs use various delimiters depending on source; try to sniff
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            reader = csv.DictReader(f, dialect=dialect)
        except csv.Error:
            reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"Could not read header from {path}")

        fieldnames = [fn.strip() for fn in reader.fieldnames]
        text_col = _first_match(fieldnames, text_cols)
        label_col = _first_match(fieldnames, label_cols)

        if text_col is None or label_col is None:
            raise ValueError(
                f"Could not locate text/label columns in ISEAR CSV. "
                f"Available: {fieldnames}. "
                f"Looked for text in {text_cols}, label in {label_cols}."
            )
        logger.info(f"ISEAR columns: text='{text_col}', label='{label_col}'")

        rows = []
        for row in reader:
            # Dict keys may have whitespace if sniffer picked odd delimiter
            clean_row = {k.strip() if k else k: v for k, v in row.items()}
            rows.append((clean_row.get(text_col, ""), clean_row.get(label_col, "")))
        return rows


def _first_match(candidates: List[str], options: Tuple[str, ...]) -> Optional[str]:
    lower_to_orig = {c.lower(): c for c in candidates}
    for opt in options:
        if opt.lower() in lower_to_orig:
            return lower_to_orig[opt.lower()]
    return None


def _stratified_split(
    items: List[Tuple[str, str, str]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[List, List, List]:
    """Stratified split by Ekman label."""
    import random
    from collections import defaultdict

    by_label = defaultdict(list)
    for item in items:
        by_label[item[1]].append(item)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for label, label_items in by_label.items():
        # Sort for determinism before shuffling with seed
        label_items = sorted(label_items, key=lambda x: x[0])
        rng.shuffle(label_items)
        n = len(label_items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(label_items[:n_train])
        val.extend(label_items[n_train:n_train + n_val])
        test.extend(label_items[n_train + n_val:])
    return train, val, test
