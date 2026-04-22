"""
Ekman-6 label harmonization across GoEmotions, ISEAR, and WASSA-21.

This module is the single source of truth for label mapping decisions.
All design choices are documented inline with rationale.

Ekman's six basic emotions (Ekman, 1992):
    joy, sadness, anger, fear, disgust, surprise

Label index convention (used throughout the codebase):
    0: anger
    1: disgust
    2: fear
    3: joy
    4: sadness
    5: surprise
"""
from typing import Dict, List, Optional, Set


# ============================================================================
# Canonical Ekman-6 label space
# ============================================================================
EKMAN_LABELS: List[str] = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
LABEL2ID: Dict[str, int] = {label: idx for idx, label in enumerate(EKMAN_LABELS)}
ID2LABEL: Dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS: int = len(EKMAN_LABELS)


# ============================================================================
# GoEmotions → Ekman-6 mapping
# ============================================================================
# Source: Google Research official mapping
# https://github.com/google-research/google-research/blob/master/goemotions/data/ekman_mapping.json
#
# - 27 fine-grained emotions + neutral → 6 Ekman emotions
# - Neutral is DROPPED entirely (no corresponding class in ISEAR/WASSA-21)
# - Multi-label handling: see map_goemotions() below
GOEMOTIONS_TO_EKMAN: Dict[str, str] = {
    # anger
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    # disgust
    "disgust": "disgust",
    # fear
    "fear": "fear",
    "nervousness": "fear",
    # joy (largest cluster)
    "joy": "joy",
    "amusement": "joy",
    "approval": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "love": "joy",
    "optimism": "joy",
    "relief": "joy",
    "pride": "joy",
    "admiration": "joy",
    "desire": "joy",
    "caring": "joy",
    # sadness
    "sadness": "sadness",
    "disappointment": "sadness",
    "embarrassment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    # surprise
    "surprise": "surprise",
    "realization": "surprise",
    "confusion": "surprise",
    "curiosity": "surprise",
    # DROPPED: "neutral" → no Ekman equivalent
}


# ============================================================================
# ISEAR → Ekman-6 mapping
# ============================================================================
# ISEAR has 7 emotions: joy, fear, anger, sadness, disgust, shame, guilt
#
# Design decision (pre-registered):
#   - shame and guilt are EXCLUDED (not mapped to sadness).
#   - Rationale: mapping shame/guilt → sadness is contested in the literature
#     and would introduce label noise reviewers can flag. Excluding is cleaner.
#   - Consequence: ISEAR contributes 5 classes (no surprise, no shame, no guilt).
#
# Known limitation: ISEAR has NO surprise class at all in the original dataset.
# In LODO with ISEAR as target, surprise F1 is undefined. We report this
# explicitly and compute macro-F1 over the 5 available classes when ISEAR
# is the target.
ISEAR_TO_EKMAN: Dict[str, str] = {
    "joy": "joy",
    "fear": "fear",
    "anger": "anger",
    "sadness": "sadness",
    "disgust": "disgust",
    # EXCLUDED: shame, guilt
}
ISEAR_EXCLUDED_LABELS: Set[str] = {"shame", "guilt"}
ISEAR_MISSING_EKMAN: Set[str] = {"surprise"}  # not present in ISEAR

# Canonical ISEAR numeric codes from the original SPSS distribution (Scherer
# & Wallbott, 1994). Some community mirrors export EMOT as integers 1-7 and
# others as string names; we accept both so the pre-registered exclusion of
# shame (6) and guilt (7) holds regardless of file format, and so that no
# file-specific special-casing leaks into the loaders.
ISEAR_NUMERIC_TO_NAME: Dict[str, str] = {
    "1": "joy",
    "2": "fear",
    "3": "anger",
    "4": "sadness",
    "5": "disgust",
    "6": "shame",   # still excluded downstream
    "7": "guilt",   # still excluded downstream
}


# ============================================================================
# WASSA-21 → Ekman-6 mapping
# ============================================================================
# WASSA-21 shared task uses Ekman-6 directly (per Tafreshi et al., 2021).
# Label names may vary in capitalization across distributions; we normalize
# to lowercase. Some distributions include "neutral" — it is dropped.
WASSA_TO_EKMAN: Dict[str, str] = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "sadness": "sadness",
    "surprise": "surprise",
    # DROPPED if present: "neutral"
}


# ============================================================================
# Mapping functions
# ============================================================================
def map_goemotions(
    label_names: List[str],
    strict_single_ekman: bool = True,
) -> Optional[str]:
    """
    Map a list of GoEmotions labels (1 or more) to a single Ekman-6 label.

    Args:
        label_names: list of original GoEmotions label strings for one example.
        strict_single_ekman: if True, require all mapped labels to collapse to
            a single Ekman class; return None otherwise. If False, return the
            majority Ekman class (ties broken by EKMAN_LABELS order).

    Returns:
        Ekman-6 label string, or None if the example should be dropped
        (neutral-only, or conflicting multi-label when strict=True).

    Design notes:
        - "neutral" is dropped: if the example has only "neutral", return None.
        - If neutral is mixed with other labels, drop neutral and map the rest.
        - Multi-label examples whose labels all map to the same Ekman class
          are KEPT (no information loss).
        - Multi-label examples whose labels map to DIFFERENT Ekman classes:
            * strict mode: dropped (cleanest for single-label classification).
            * non-strict: majority vote.
    """
    # Filter neutral and unknown labels
    filtered = [lbl for lbl in label_names if lbl != "neutral"]
    if not filtered:
        return None

    # Map each to Ekman
    ekman_mapped = []
    for lbl in filtered:
        if lbl in GOEMOTIONS_TO_EKMAN:
            ekman_mapped.append(GOEMOTIONS_TO_EKMAN[lbl])
        # Unknown labels silently dropped (shouldn't happen with clean data)

    if not ekman_mapped:
        return None

    unique_ekman = set(ekman_mapped)
    if len(unique_ekman) == 1:
        return ekman_mapped[0]

    # Multi-label example spanning multiple Ekman classes
    if strict_single_ekman:
        return None

    # Majority vote (non-strict mode)
    counts = {e: ekman_mapped.count(e) for e in unique_ekman}
    max_count = max(counts.values())
    winners = [e for e, c in counts.items() if c == max_count]
    # Tie-break by canonical order
    for lbl in EKMAN_LABELS:
        if lbl in winners:
            return lbl
    return None  # unreachable


def canonicalize_isear_label(label_name: str) -> str:
    """Return the canonical lowercase ISEAR label, translating numeric codes.

    Exposed so the loader can run its exclusion check against canonical names
    (shame/guilt) regardless of whether the source file stores labels as
    integers (``"6"``/``"7"``) or as strings (``"shame"``/``"guilt"``). Keeps
    the single-source-of-truth policy intact: no loader owns this knowledge.
    """
    s = str(label_name).strip().lower()
    return ISEAR_NUMERIC_TO_NAME.get(s, s)


def map_isear(label_name: str) -> Optional[str]:
    """Map an ISEAR label to Ekman-6; return None if excluded (shame/guilt).

    Accepts the canonical string names (joy/fear/anger/sadness/disgust) OR the
    numeric codes 1-7 from the original ISEAR SPSS distribution that several
    community mirrors preserve verbatim. Pre-registered exclusions (shame=6,
    guilt=7) are honored in both representations.
    """
    canon = canonicalize_isear_label(label_name)
    return ISEAR_TO_EKMAN.get(canon)


def map_wassa(label_name: str) -> Optional[str]:
    """Map a WASSA-21 label to Ekman-6; return None if unknown/neutral."""
    label_name = label_name.strip().lower()
    return WASSA_TO_EKMAN.get(label_name)


# ============================================================================
# Sanity checks (run on import in debug; skipped in production)
# ============================================================================
def _validate_mappings() -> None:
    """Assert that all mapping targets are valid Ekman labels."""
    for src_name, mapping in [
        ("GoEmotions", GOEMOTIONS_TO_EKMAN),
        ("ISEAR", ISEAR_TO_EKMAN),
        ("WASSA-21", WASSA_TO_EKMAN),
    ]:
        for orig, ekman in mapping.items():
            assert ekman in EKMAN_LABELS, (
                f"{src_name} mapping: '{orig}' → '{ekman}' "
                f"is not a valid Ekman label"
            )
    # GoEmotions should have exactly 27 mapped labels (neutral excluded)
    assert len(GOEMOTIONS_TO_EKMAN) == 27, (
        f"Expected 27 GoEmotions labels mapped, got {len(GOEMOTIONS_TO_EKMAN)}"
    )


_validate_mappings()
