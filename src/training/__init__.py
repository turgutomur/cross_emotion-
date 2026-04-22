"""
Training package.

Modules:
    trainer.py     -- Method-agnostic training loop (CE / Focal / DANN / CDAN)
    losses.py      -- Focal loss, class-balanced focal loss  (Week 4)
    schedulers.py  -- DANN lambda sigmoid annealing           (Week 3)
"""
from .trainer import Trainer

__all__ = ["Trainer"]
