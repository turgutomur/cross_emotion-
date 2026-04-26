"""
Models package.

Implemented (Week 2):
    backbone.py   -- DeBERTa-v3-base encoder wrapper
    classifier.py -- Simple classification head (CE)

Planned (Weeks 3-4):
    dann.py       -- Gradient Reversal Layer + domain discriminator
    cdan.py       -- Conditional DANN (class-conditional alignment)
    focal.py      -- Focal loss variants (used by DANN+Focal / CDAN+Focal)

The public API below uses a lazy ``__getattr__`` so that importing this
package in environments without ``torch`` / ``transformers`` (e.g. the
data-only unit tests in ``tests/test_data.py``) does not fail. Only when a
concrete model symbol is actually referenced do we touch the heavy deps.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .backbone import (
        BackboneConfig,
        BackboneOutput,
        DebertaBackbone,
        build_backbone,
    )
    from .classifier import (
        ClassificationHead,
        EmotionClassifier,
        EmotionClassifierOutput,
        build_emotion_classifier,
    )
    from .dann import (
        DANNConfig,
        DANNModel,
        DANNOutput,
        DomainDiscriminator,
        SigmoidLambdaScheduler,
        build_dann_model,
    )
    from .cdan import (
        CDANConfig,
        CDANModel,
        CDANOutput,
        ConditionalDiscriminator,
        build_cdan_model,
    )


__all__ = [
    # backbone
    "BackboneConfig",
    "BackboneOutput",
    "DebertaBackbone",
    "build_backbone",
    # classifier
    "ClassificationHead",
    "EmotionClassifier",
    "EmotionClassifierOutput",
    "build_emotion_classifier",
    # dann
    "DANNConfig",
    "DANNModel",
    "DANNOutput",
    "DomainDiscriminator",
    "SigmoidLambdaScheduler",
    "build_dann_model",
    # cdan
    "CDANConfig",
    "CDANModel",
    "CDANOutput",
    "ConditionalDiscriminator",
    "build_cdan_model",
]


_BACKBONE_EXPORTS = {
    "BackboneConfig",
    "BackboneOutput",
    "DebertaBackbone",
    "build_backbone",
}
_CLASSIFIER_EXPORTS = {
    "ClassificationHead",
    "EmotionClassifier",
    "EmotionClassifierOutput",
    "build_emotion_classifier",
}
_DANN_EXPORTS = {
    "DANNConfig",
    "DANNModel",
    "DANNOutput",
    "DomainDiscriminator",
    "SigmoidLambdaScheduler",
    "build_dann_model",
}
_CDAN_EXPORTS = {
    "CDANConfig",
    "CDANModel",
    "CDANOutput",
    "ConditionalDiscriminator",
    "build_cdan_model",
}


def __getattr__(name):
    if name in _BACKBONE_EXPORTS:
        from . import backbone as _bb
        return getattr(_bb, name)
    if name in _CLASSIFIER_EXPORTS:
        from . import classifier as _cl
        return getattr(_cl, name)
    if name in _DANN_EXPORTS:
        from . import dann as _dann
        return getattr(_dann, name)
    if name in _CDAN_EXPORTS:
        from . import cdan as _cdan
        return getattr(_cdan, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
