"""Data module public API."""
from .ekman_mapping import (
    EKMAN_LABELS,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
    map_goemotions,
    map_isear,
    map_wassa,
)
from .types import (
    DatasetName,
    DATASET_NAMES,
    DATASET2ID,
    ID2DATASET,
    NUM_DOMAINS,
    EmotionExample,
    example_from_record,
)
from .builder import DataConfig, build_datasets
from .protocols import (
    ProtocolSplit,
    build_mixed_protocol,
    build_lodo_protocol,
    build_all_lodo_protocols,
)
__all__ = [
    # ekman
    "EKMAN_LABELS", "LABEL2ID", "ID2LABEL", "NUM_LABELS",
    "map_goemotions", "map_isear", "map_wassa",
    # types
    "DatasetName", "DATASET_NAMES", "DATASET2ID", "ID2DATASET", "NUM_DOMAINS",
    "EmotionExample", "example_from_record",
    # builder
    "DataConfig", "build_datasets",
    # protocols
    "ProtocolSplit",
    "build_mixed_protocol", "build_lodo_protocol", "build_all_lodo_protocols",
    # torch (lazy import — requires torch)
    "EmotionTorchDataset", "EmotionCollator",
]

# Lazy import for torch-dependent modules (avoids import errors in test/CI)
def __getattr__(name):
    if name in ("EmotionTorchDataset", "EmotionCollator"):
        from .torch_dataset import EmotionTorchDataset, EmotionCollator
        return {"EmotionTorchDataset": EmotionTorchDataset,
                "EmotionCollator": EmotionCollator}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
