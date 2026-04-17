"""
Unit tests for the data pipeline.

Run with:
    pytest tests/test_data.py -v
"""
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ekman_mapping import (
    EKMAN_LABELS, LABEL2ID, ID2LABEL, NUM_LABELS,
    GOEMOTIONS_TO_EKMAN, ISEAR_TO_EKMAN, WASSA_TO_EKMAN,
    ISEAR_EXCLUDED_LABELS, ISEAR_MISSING_EKMAN,
    map_goemotions, map_isear, map_wassa,
)
from src.data.types import (
    DatasetName, DATASET2ID, NUM_DOMAINS,
    EmotionExample, example_from_record,
)


# ── Ekman label space ──────────────────────────────────────────────────

class TestEkmanLabels:
    def test_six_labels(self):
        assert NUM_LABELS == 6

    def test_label_names(self):
        expected = {"anger", "disgust", "fear", "joy", "sadness", "surprise"}
        assert set(EKMAN_LABELS) == expected

    def test_label2id_roundtrip(self):
        for label, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == label

    def test_ids_are_contiguous(self):
        assert sorted(ID2LABEL.keys()) == list(range(NUM_LABELS))


# ── GoEmotions mapping ─────────────────────────────────────────────────

class TestGoEmotionsMapping:
    def test_27_labels_mapped(self):
        assert len(GOEMOTIONS_TO_EKMAN) == 27

    def test_neutral_not_mapped(self):
        assert "neutral" not in GOEMOTIONS_TO_EKMAN

    def test_all_targets_valid(self):
        for src, tgt in GOEMOTIONS_TO_EKMAN.items():
            assert tgt in EKMAN_LABELS, f"{src} → {tgt} invalid"

    def test_anger_cluster(self):
        anger_sources = {k for k, v in GOEMOTIONS_TO_EKMAN.items() if v == "anger"}
        assert anger_sources == {"anger", "annoyance", "disapproval"}

    def test_joy_cluster_size(self):
        joy_sources = {k for k, v in GOEMOTIONS_TO_EKMAN.items() if v == "joy"}
        assert len(joy_sources) == 12  # largest cluster

    def test_map_single_label(self):
        assert map_goemotions(["joy"]) == "joy"
        assert map_goemotions(["annoyance"]) == "anger"
        assert map_goemotions(["nervousness"]) == "fear"

    def test_map_neutral_only(self):
        assert map_goemotions(["neutral"]) is None

    def test_map_neutral_plus_real(self):
        # neutral should be ignored, anger kept
        assert map_goemotions(["neutral", "anger"]) == "anger"

    def test_map_multi_same_ekman(self):
        # admiration + joy both → joy
        assert map_goemotions(["admiration", "joy"]) == "joy"

    def test_map_multi_different_ekman_strict(self):
        # anger + joy → different Ekman → None in strict mode
        assert map_goemotions(["anger", "joy"], strict_single_ekman=True) is None

    def test_map_multi_different_ekman_nonstrict(self):
        # anger + joy → majority needed; tie → first in EKMAN_LABELS (anger)
        result = map_goemotions(["anger", "joy"], strict_single_ekman=False)
        assert result in EKMAN_LABELS  # should return one of them

    def test_map_empty(self):
        assert map_goemotions([]) is None


# ── ISEAR mapping ──────────────────────────────────────────────────────

class TestISEARMapping:
    def test_five_labels(self):
        assert len(ISEAR_TO_EKMAN) == 5

    def test_shame_excluded(self):
        assert "shame" in ISEAR_EXCLUDED_LABELS
        assert map_isear("shame") is None

    def test_guilt_excluded(self):
        assert "guilt" in ISEAR_EXCLUDED_LABELS
        assert map_isear("guilt") is None

    def test_surprise_missing(self):
        assert "surprise" in ISEAR_MISSING_EKMAN

    def test_valid_mappings(self):
        assert map_isear("joy") == "joy"
        assert map_isear("fear") == "fear"
        assert map_isear("anger") == "anger"
        assert map_isear("sadness") == "sadness"
        assert map_isear("disgust") == "disgust"

    def test_case_insensitive(self):
        assert map_isear("JOY") == "joy"
        assert map_isear("  Fear  ") == "fear"


# ── WASSA mapping ──────────────────────────────────────────────────────

class TestWASSAMapping:
    def test_six_labels(self):
        assert len(WASSA_TO_EKMAN) == 6

    def test_all_ekman_present(self):
        assert set(WASSA_TO_EKMAN.values()) == set(EKMAN_LABELS)

    def test_identity_mapping(self):
        for label in EKMAN_LABELS:
            assert map_wassa(label) == label

    def test_case_insensitive(self):
        assert map_wassa("Anger") == "anger"
        assert map_wassa("  SURPRISE  ") == "surprise"


# ── Types ──────────────────────────────────────────────────────────────

class TestTypes:
    def test_three_domains(self):
        assert NUM_DOMAINS == 3

    def test_domain_names(self):
        assert DatasetName.GOEMOTIONS.value == "goemotions"
        assert DatasetName.ISEAR.value == "isear"
        assert DatasetName.WASSA.value == "wassa21"

    def test_example_from_record(self):
        ex = example_from_record(
            text="I am happy",
            ekman_label="joy",
            domain="goemotions",
            orig_label="joy",
            split="train",
        )
        assert isinstance(ex, EmotionExample)
        assert ex.ekman_id == LABEL2ID["joy"]
        assert ex.domain_id == DATASET2ID["goemotions"]

    def test_invalid_ekman_raises(self):
        try:
            example_from_record(text="x", ekman_label="love", domain="goemotions")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_invalid_domain_raises(self):
        try:
            example_from_record(text="x", ekman_label="joy", domain="twitter")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
