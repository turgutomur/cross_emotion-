"""
WASSA-21 loader (Tafreshi et al., 2021).

The WASSA-21 shared task provides essays written in response to news articles,
annotated with Ekman-6 emotions. The canonical release is distributed as
``train.tsv`` / ``dev.tsv`` / ``test.tsv``, but mirrors and Kaggle copies use
several variants (``train.csv.tsv``, ``train.csv``, renamed columns, ...).
This loader is deliberately permissive about file names and gracefully falls
back when a split is missing or unlabeled.

Key behaviours
--------------
1. **Flexible file-name matching.** For each logical split we try, in order,
   ``train.tsv``, ``train.csv.tsv``, ``train.csv`` and a final glob
   (``train.*``) so that a user dropping the files in any reasonable form
   is supported without asking them to rename.

2. **Graceful fallback to splitting train.**  In practice, the shared-task
   ``dev.tsv`` and ``test.tsv`` are sometimes released WITHOUT the emotion
   column (they were used as blind evaluation sets). When either file is
   missing *or* lacks the label column, we stratified-split the labeled
   ``train.tsv`` into 80/10/10 (seed 42) — this mirrors what we already do
   for ISEAR and keeps Protocols A/B consistent across datasets.

3. **Delimiter autodetection.** Some Kaggle re-distributions save the file
   with a comma delimiter despite the ``.tsv`` extension. We sniff the
   first line and fall back to tab if sniffing fails.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
import logging

from .ekman_mapping import map_wassa
from .types import DatasetName, EmotionExample, example_from_record


logger = logging.getLogger(__name__)


# Candidate column names, in priority order.
TEXT_COL_CANDIDATES = ("essay", "text", "content", "response", "message")
LABEL_COL_CANDIDATES = ("emotion", "Emotion", "emotion_label", "label", "gold_emotion")

# Logical split name → ordered list of file-name patterns to try.
# We attempt exact names first (fast + deterministic), then glob patterns.
_SPLIT_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "train": ("train.tsv", "train.csv.tsv", "train.csv", "train.txt"),
    "val":   ("dev.tsv",   "dev.csv.tsv",   "dev.csv",   "val.tsv", "valid.tsv"),
    "test":  ("test.tsv",  "test.csv.tsv",  "test.csv",  "test.txt"),
}
_SPLIT_GLOBS: Dict[str, Tuple[str, ...]] = {
    # Greedy globs: match any file that contains "train"/"dev"/"test" in its
    # name. This is how we pick up the WASSA-21 shared-task release filenames
    # like ``track-1-essay-empathy-train.tsv`` / ``goldstandard_dev_2022.tsv``
    # without asking the user to rename them. Correctness is preserved by
    # ``_has_label_column`` — a matched dev/test file that turns out to be
    # unlabeled is transparently skipped in favour of the fallback split.
    "train": ("train.*", "*train*.tsv", "*train*.csv"),
    "val":   ("dev.*", "*dev*.tsv", "*dev*.csv",
              "val*.*", "*val*.tsv", "valid*.*", "*valid*.tsv"),
    "test":  ("test.*", "*test*.tsv", "*test*.csv"),
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load_wassa21(
    data_dir: Path,
    split: str = "train",
    min_text_length: int = 10,
    fallback_split_seed: int = 42,
    fallback_train_frac: float = 0.8,
    fallback_val_frac: float = 0.1,
) -> List[EmotionExample]:
    """Load a WASSA-21 split, falling back to a stratified split of train if needed.

    Parameters
    ----------
    data_dir:
        Directory containing the WASSA-21 files. Any of the variants listed
        in ``_SPLIT_CANDIDATES`` / ``_SPLIT_GLOBS`` is accepted.
    split:
        ``"train"``, ``"val"`` (dev), or ``"test"``.
    min_text_length:
        Drop essays shorter than this many characters. Matches the filter
        applied to ISEAR for consistency across datasets.
    fallback_split_seed / _train_frac / _val_frac:
        Used *only* when the requested split file is missing or unlabeled.
        A deterministic seed guarantees that every caller sees the exact
        same train/val/test partitions.

    Returns
    -------
    List[EmotionExample]

    Raises
    ------
    FileNotFoundError
        If ``train`` cannot be resolved (we always need labeled data
        somewhere — without it there is nothing to load).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"WASSA-21 data directory not found at {data_dir}. "
            f"Place the shared-task files there (any of: train.tsv, "
            f"train.csv.tsv, train.csv)."
        )

    # Resolve the requested split. If the target file is absent OR present
    # but unlabeled, we will fall back to splitting train.
    target_path = _resolve_split_file(data_dir, split)
    target_has_labels = target_path is not None and _has_label_column(target_path)

    if target_has_labels:
        logger.info(f"WASSA-21[{split}] using file: {target_path}")
        rows = _read_wassa_table(target_path)
        return _rows_to_examples(rows, split, min_text_length)

    # ---------------- fallback: synthesize splits from train ----------------
    train_path = _resolve_split_file(data_dir, "train")
    if train_path is None or not _has_label_column(train_path):
        raise FileNotFoundError(
            f"WASSA-21 train file not found (or missing label column) "
            f"in {data_dir}. Expected one of: "
            f"{_SPLIT_CANDIDATES['train']}."
        )

    reason = "missing" if target_path is None else "unlabeled"
    logger.warning(
        f"WASSA-21[{split}] falling back to stratified split of "
        f"{train_path.name} ({reason} official {split} file). "
        f"Using seed={fallback_split_seed}, train/val/test="
        f"{fallback_train_frac:.2f}/{fallback_val_frac:.2f}/"
        f"{1 - fallback_train_frac - fallback_val_frac:.2f}."
    )
    all_rows = _read_wassa_table(train_path)

    # Stratified split (by Ekman label, not original label, so that
    # "neutral" — which maps to None — is balanced across splits too).
    harmonized = _harmonize(all_rows, min_text_length)
    train_items, val_items, test_items = _stratified_split(
        harmonized,
        train_frac=fallback_train_frac,
        val_frac=fallback_val_frac,
        seed=fallback_split_seed,
    )
    split_items = {"train": train_items, "val": val_items, "test": test_items}[split]

    examples = [
        example_from_record(
            text=text,
            ekman_label=ekman,
            domain=DatasetName.WASSA.value,
            orig_label=orig,
            split=split,
            example_id=f"wassa-{split}-{i}",
        )
        for i, (text, ekman, orig) in enumerate(split_items)
    ]
    logger.info(f"WASSA-21[{split}] (fallback): {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# File resolution helpers
# ---------------------------------------------------------------------------
def _resolve_split_file(data_dir: Path, split: str) -> Optional[Path]:
    """Find the first existing file for ``split`` using explicit + glob patterns."""
    for name in _SPLIT_CANDIDATES[split]:
        p = data_dir / name
        if p.exists():
            return p
    # Fall back to glob matching — useful for odd suffixes we did not enumerate.
    for pattern in _SPLIT_GLOBS[split]:
        matches = sorted(data_dir.glob(pattern))
        # Exclude placeholder .gitkeep and hidden files
        matches = [m for m in matches if not m.name.startswith(".")]
        if matches:
            return matches[0]
    return None


def _has_label_column(path: Path) -> bool:
    """Return True if the file's header contains any known emotion-label column."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
    if not first:
        return False
    # Try tab first (WASSA canonical), then comma.
    for delim in ("\t", ","):
        header = [h.strip() for h in first.rstrip("\r\n").split(delim)]
        if _first_match(header, LABEL_COL_CANDIDATES) is not None:
            return True
    return False


# ---------------------------------------------------------------------------
# Table reading + harmonization
# ---------------------------------------------------------------------------
def _read_wassa_table(path: Path) -> List[Tuple[str, str]]:
    """Read a WASSA-21 file, autodetecting delimiter and column names.

    Returns a list of ``(text, orig_label)`` tuples. Rows without a
    resolvable text column are silently skipped; rows without a label
    become ``(text, "")`` and are dropped downstream by ``map_wassa``.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,;|")
        except csv.Error:
            # Default to tab — the canonical WASSA-21 release is TSV.
            dialect = csv.excel_tab
        reader = csv.DictReader(f, dialect=dialect)

        if reader.fieldnames is None:
            raise ValueError(f"Empty or malformed file: {path}")

        fieldnames = [fn.strip() for fn in reader.fieldnames]
        text_col = _first_match(fieldnames, TEXT_COL_CANDIDATES)
        label_col = _first_match(fieldnames, LABEL_COL_CANDIDATES)  # may be None

        if text_col is None:
            raise ValueError(
                f"Could not locate a text column in {path}. "
                f"Available: {fieldnames}. "
                f"Looked for: {TEXT_COL_CANDIDATES}."
            )
        logger.info(
            f"WASSA columns: text='{text_col}', "
            f"label={label_col!r if label_col else 'MISSING'}"
        )

        rows: List[Tuple[str, str]] = []
        for row in reader:
            clean = {k.strip() if k else k: v for k, v in row.items()}
            text = clean.get(text_col, "") or ""
            orig_label = clean.get(label_col, "") if label_col else ""
            rows.append((text, orig_label or ""))
        return rows


def _harmonize(
    rows: List[Tuple[str, str]],
    min_text_length: int,
) -> List[Tuple[str, str, str]]:
    """Filter + Ekman-map rows into ``(text, ekman_label, orig_label)`` tuples."""
    harmonized: List[Tuple[str, str, str]] = []
    n_short = 0
    n_unknown = 0
    for text, orig_label in rows:
        text = (text or "").strip()
        if len(text) < min_text_length:
            n_short += 1
            continue
        ekman = map_wassa(orig_label)
        if ekman is None:
            # "neutral" and unknown labels fall through here.
            n_unknown += 1
            continue
        harmonized.append((text, ekman, orig_label.strip().lower()))
    logger.info(
        f"WASSA-21 harmonized: kept {len(harmonized)}, "
        f"dropped too-short={n_short}, unknown/neutral={n_unknown}"
    )
    return harmonized


def _rows_to_examples(
    rows: List[Tuple[str, str]],
    split: str,
    min_text_length: int,
) -> List[EmotionExample]:
    """Convert raw rows from a LABELED split file straight into examples."""
    harmonized = _harmonize(rows, min_text_length)
    examples = [
        example_from_record(
            text=text,
            ekman_label=ekman,
            domain=DatasetName.WASSA.value,
            orig_label=orig,
            split=split,
            example_id=f"wassa-{split}-{i}",
        )
        for i, (text, ekman, orig) in enumerate(harmonized)
    ]
    logger.info(f"WASSA-21[{split}]: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# Split / utility helpers
# ---------------------------------------------------------------------------
def _stratified_split(
    items: List[Tuple[str, str, str]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[List, List, List]:
    """Stratified split by Ekman label — same recipe as ``isear_loader``.

    Kept as a private helper (rather than imported from ``isear_loader``) so
    the two modules stay de-coupled; if one dataset's splitting strategy
    needs to change later, the other is untouched.
    """
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


def _first_match(candidates: List[str], options: Tuple[str, ...]) -> Optional[str]:
    """Case-insensitive first match of ``options`` against ``candidates``."""
    lower_to_orig = {c.lower(): c for c in candidates}
    for opt in options:
        if opt.lower() in lower_to_orig:
            return lower_to_orig[opt.lower()]
    return None
