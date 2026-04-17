"""
WASSA-21 loader (Tafreshi et al., 2021).

The WASSA-21 shared task provides essays written in response to news articles,
annotated with Ekman-6 emotions. We expect the user to place the provided TSV
files at `data/raw/wassa21/{train,dev,test}.tsv`.

Different WASSA-21 releases use slightly different column names; we try
several common variants. Typical columns include essay text and emotion label.
"""
from pathlib import Path
from typing import List, Optional, Tuple
import csv
import logging

from .ekman_mapping import map_wassa
from .types import DatasetName, EmotionExample, example_from_record


logger = logging.getLogger(__name__)


# Candidate column names, in priority order
TEXT_COL_CANDIDATES = ("essay", "text", "content", "response")
LABEL_COL_CANDIDATES = ("emotion", "Emotion", "emotion_label", "label")


def load_wassa21(
    data_dir: Path,
    split: str = "train",
    min_text_length: int = 10,
) -> List[EmotionExample]:
    """
    Load WASSA-21 split from a directory containing {train,dev,test}.tsv.

    Args:
        data_dir: directory with TSV files.
        split: 'train', 'val' (maps to 'dev.tsv'), or 'test'.
        min_text_length: drop short texts.

    Returns:
        List of EmotionExample.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"WASSA-21 data directory not found at {data_dir}. "
            f"Place train.tsv, dev.tsv, test.tsv there."
        )

    filename = {"train": "train.tsv", "val": "dev.tsv", "test": "test.tsv"}[split]
    file_path = data_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Missing WASSA-21 file: {file_path}")

    logger.info(f"Loading WASSA-21 from {file_path}...")

    rows = _read_wassa_tsv(file_path)
    examples: List[EmotionExample] = []
    n_dropped_label = 0
    n_dropped_short = 0

    for idx, (text, orig_label) in enumerate(rows):
        text = (text or "").strip()
        if len(text) < min_text_length:
            n_dropped_short += 1
            continue

        ekman = map_wassa(orig_label)
        if ekman is None:
            n_dropped_label += 1
            continue

        ex = example_from_record(
            text=text,
            ekman_label=ekman,
            domain=DatasetName.WASSA.value,
            orig_label=orig_label.lower(),
            split=split,
            example_id=f"wassa-{split}-{idx}",
        )
        examples.append(ex)

    logger.info(
        f"WASSA-21[{split}]: kept {len(examples)}, "
        f"dropped unknown-label={n_dropped_label}, "
        f"dropped too-short={n_dropped_short}"
    )
    return examples


def _read_wassa_tsv(path: Path) -> List[Tuple[str, str]]:
    """Read a WASSA-21 TSV file, auto-detecting text and label columns."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Empty or malformed TSV: {path}")

        fieldnames = [fn.strip() for fn in reader.fieldnames]
        text_col = _first_match(fieldnames, TEXT_COL_CANDIDATES)
        label_col = _first_match(fieldnames, LABEL_COL_CANDIDATES)

        if text_col is None or label_col is None:
            raise ValueError(
                f"Could not locate text/label columns in {path}. "
                f"Available: {fieldnames}. "
                f"Looked for text in {TEXT_COL_CANDIDATES}, "
                f"label in {LABEL_COL_CANDIDATES}."
            )
        logger.info(f"WASSA columns: text='{text_col}', label='{label_col}'")

        rows = []
        for row in reader:
            rows.append((row.get(text_col, ""), row.get(label_col, "")))
        return rows


def _first_match(candidates: List[str], options: Tuple[str, ...]) -> Optional[str]:
    lower_to_orig = {c.lower(): c for c in candidates}
    for opt in options:
        if opt.lower() in lower_to_orig:
            return lower_to_orig[opt.lower()]
    return None
