"""
ISEAR-style loader (Scherer & Wallbott, 1994; permissive for re-distributions).

ISEAR does not come with official train/val/test splits, and is not directly
available on HuggingFace in a canonical form. We expect the user to place the
raw CSV at ``data/raw/isear.csv`` (see README) and we perform a stratified
80/10/10 split with a fixed seed for reproducibility.

The loader is deliberately permissive about column names because several
community re-distributions rename the original ``SIT``/``EMOT`` columns.

Preprocessing decisions (pre-registered — see docs/research_notes.md):
  1. Exclude ``shame`` and ``guilt`` (contested mapping to sadness).
  2. Keep the intersection of {joy, fear, anger, sadness, disgust} that
     actually appears in the supplied file.
  3. Note: ``surprise`` is NOT present in ISEAR (inherent dataset limitation).
  4. Light whitespace cleaning; drop empty or placeholder texts
     (ISEAR has some "NO RESPONSE" entries).
"""
from pathlib import Path
from typing import List, Optional, Tuple
import csv
import logging
import re

from .ekman_mapping import (
    ISEAR_EXCLUDED_LABELS,
    canonicalize_isear_label,
    map_isear,
)
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
    text_column_candidates: Tuple[str, ...] = (
        # Canonical ISEAR names first; then common variants we saw across
        # Kaggle / GitHub re-distributions of "ISEAR-like" data.
        "SIT", "situation", "text", "content", "tweet", "sentence", "message",
    ),
    label_column_candidates: Tuple[str, ...] = (
        "EMOT", "emotion", "Field1", "label", "sentiment", "class", "emotions",
    ),
) -> List[EmotionExample]:
    """Load ISEAR and return a deterministic split.

    Rationale for the column-candidate tuples
    -----------------------------------------
    Re-distributions we have encountered use at least three header schemes:
        * Canonical ISEAR:          ``SIT`` / ``EMOT``
        * Kaggle & GitHub mirrors:  ``text`` / ``emotion``
        * Twitter-style variants:   ``content`` / ``sentiment``
    Trying all of them (case-insensitively) here keeps ``configs/default.yaml``
    untouched when the user swaps one raw file for another.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ISEAR CSV not found at {csv_path}. "
            f"Download from https://www.unige.ch/cisa/research/materials-and-online-research/research-material/ "
            f"(or a mirror) and place it at this path."
        )

    logger.info(f"Loading ISEAR from {csv_path}...")

    rows = _read_isear_csv(csv_path, text_column_candidates, label_column_candidates)
    logger.info(f"ISEAR raw rows: {len(rows)}")

    harmonized: List[Tuple[str, str, str]] = []
    n_excluded_label = 0
    n_placeholder = 0
    n_short = 0

    for text, orig_label in rows:
        text = (text or "").strip()
        # Canonicalize here (not in map_isear) so the shame/guilt exclusion
        # check below sees the same canonical name regardless of whether the
        # source file uses strings ("shame") or numeric codes ("6").
        orig_label_canon = canonicalize_isear_label(orig_label or "")

        if orig_label_canon in ISEAR_EXCLUDED_LABELS:
            n_excluded_label += 1
            continue

        if not text or _is_placeholder(text):
            n_placeholder += 1
            continue

        if len(text) < min_text_length:
            n_short += 1
            continue

        ekman = map_isear(orig_label_canon)
        if ekman is None:
            continue

        harmonized.append((text, ekman, orig_label_canon))

    logger.info(
        f"ISEAR harmonized: kept {len(harmonized)}, "
        f"dropped shame/guilt={n_excluded_label}, "
        f"placeholders={n_placeholder}, short={n_short}"
    )

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
    """Read the raw ISEAR CSV, auto-detecting delimiter and header columns.

    ``utf-8-sig`` transparently strips the BOM that some exporters (Excel,
    certain Kaggle re-distributions) prepend — otherwise the first header
    becomes ``\\ufeffID`` and column-name matching becomes fragile.
    """
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
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
            clean_row = {k.strip() if k else k: v for k, v in row.items()}
            rows.append((clean_row.get(text_col, ""), clean_row.get(label_col, "")))
        return rows


def _first_match(candidates: List[str], options: Tuple[str, ...]) -> Optional[str]:
    """Case-insensitive first-match of ``options`` within ``candidates``."""
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
    """Stratified split by Ekman label with a fixed seed for reproducibility."""
    import random
    from collections import defaultdict

    by_label = defaultdict(list)
    for item in items:
        by_label[item[1]].append(item)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for label, label_items in by_label.items():
        label_items = sorted(label_items, key=lambda x: x[0])
        rng.shuffle(label_items)
        n = len(label_items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(label_items[:n_train])
        val.extend(label_items[n_train:n_train + n_val])
        test.extend(label_items[n_train + n_val:])
    return train, val, test
