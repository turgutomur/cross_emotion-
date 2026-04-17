"""
PyTorch Dataset wrapper and collator.

Wraps a List[EmotionExample] into a torch Dataset and provides a collator
that tokenizes on-the-fly and returns tensors suitable for both plain
classification and domain-adversarial training (DANN needs domain labels).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset

from .types import EmotionExample


class EmotionTorchDataset(Dataset):
    """Thin wrapper — tokenization happens in the collator, not here."""

    def __init__(self, examples: List[EmotionExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> EmotionExample:
        return self.examples[idx]


@dataclass
class EmotionCollator:
    """
    Tokenizes a batch of EmotionExample on-the-fly.

    Returns a dict with:
        input_ids, attention_mask, labels (ekman_id), domain_labels (domain_id)
    Ready to be consumed by HF-style models.
    """
    tokenizer: "PreTrainedTokenizer"   # type: ignore  # noqa: F821
    max_length: int = 256
    padding: Union[str, bool] = "longest"
    return_token_type_ids: bool = False

    def __call__(self, batch: List[EmotionExample]) -> Dict[str, torch.Tensor]:
        texts = [ex.text for ex in batch]
        labels = torch.tensor([ex.ekman_id for ex in batch], dtype=torch.long)
        domain_labels = torch.tensor([ex.domain_id for ex in batch], dtype=torch.long)

        encoded = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=self.return_token_type_ids,
        )

        out = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "domain_labels": domain_labels,
        }
        if "token_type_ids" in encoded:
            out["token_type_ids"] = encoded["token_type_ids"]
        return out
