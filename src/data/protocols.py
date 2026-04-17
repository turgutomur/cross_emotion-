"""
Experimental protocol builders.

Two protocols are supported (see README):

  Protocol A (Mixed):
    Union of all three datasets' train splits for training; each dataset's
    val and test splits are used for validation and evaluation.
    Reports per-dataset test metrics AND aggregate test metrics.

  Protocol B (Leave-One-Dataset-Out, LODO):
    Train on the UNION of two source datasets' train splits.
    Validate on the UNION of the same two source datasets' val splits.
    Test on the TARGET dataset's test split (unseen during training).
    Target is truly unseen — none of its train/val examples are used.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

from .types import DatasetName, EmotionExample


logger = logging.getLogger(__name__)


@dataclass
class ProtocolSplit:
    """A (train, val, test) triple for a single experimental run."""
    name: str                          # human-readable identifier
    train: List[EmotionExample]
    val: List[EmotionExample]
    test: List[EmotionExample]
    target_domain: Optional[str] = None   # only set for LODO
    source_domains: List[str] = None      # domains included in training

    def describe(self) -> str:
        from collections import Counter
        def _counts(exs, key):
            return dict(Counter(getattr(e, key) for e in exs))
        return (
            f"Protocol '{self.name}':\n"
            f"  target_domain = {self.target_domain}\n"
            f"  source_domains = {self.source_domains}\n"
            f"  train: {len(self.train)} examples, "
            f"by domain={_counts(self.train, 'domain')}, "
            f"by label={_counts(self.train, 'ekman_label')}\n"
            f"  val:   {len(self.val)} examples, "
            f"by domain={_counts(self.val, 'domain')}\n"
            f"  test:  {len(self.test)} examples, "
            f"by domain={_counts(self.test, 'domain')}, "
            f"by label={_counts(self.test, 'ekman_label')}"
        )


def build_mixed_protocol(
    per_domain: Dict[str, Dict[str, List[EmotionExample]]],
) -> ProtocolSplit:
    """
    Build Protocol A (Mixed): union train / union val / union test.

    Args:
        per_domain: {domain_name: {'train': [...], 'val': [...], 'test': [...]}}

    Returns:
        Single ProtocolSplit with all three domains combined.
    """
    train, val, test = [], [], []
    for domain, splits in per_domain.items():
        train.extend(splits.get("train", []))
        val.extend(splits.get("val", []))
        test.extend(splits.get("test", []))

    return ProtocolSplit(
        name="mixed",
        train=train,
        val=val,
        test=test,
        target_domain=None,
        source_domains=sorted(per_domain.keys()),
    )


def build_lodo_protocol(
    per_domain: Dict[str, Dict[str, List[EmotionExample]]],
    target_domain: str,
) -> ProtocolSplit:
    """
    Build Protocol B (LODO): train on sources, test on target.

    Target's train and val are NOT used. Only target's test is touched,
    and only at evaluation time.

    Args:
        per_domain: as in build_mixed_protocol.
        target_domain: DatasetName value for the held-out target.

    Returns:
        ProtocolSplit where test = target's test set.
    """
    if target_domain not in per_domain:
        raise ValueError(
            f"Target domain {target_domain!r} not in available domains: "
            f"{list(per_domain.keys())}"
        )

    sources = [d for d in per_domain if d != target_domain]
    train, val = [], []
    for src in sources:
        train.extend(per_domain[src].get("train", []))
        val.extend(per_domain[src].get("val", []))

    test = per_domain[target_domain].get("test", [])

    return ProtocolSplit(
        name=f"lodo-target={target_domain}",
        train=train,
        val=val,
        test=test,
        target_domain=target_domain,
        source_domains=sorted(sources),
    )


def build_all_lodo_protocols(
    per_domain: Dict[str, Dict[str, List[EmotionExample]]],
) -> List[ProtocolSplit]:
    """Build all three LODO configurations (one per target domain)."""
    return [build_lodo_protocol(per_domain, target) for target in sorted(per_domain)]
