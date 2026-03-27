from __future__ import annotations

from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


class CXRMultimodalDataset(Dataset):
    """
    Dataset for paired chest X-ray images and report text.

    Each sample returns:
    - transformed image tensor
    - tokenized text input
    - metadata used later for retrieval analysis
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer,
        transform=None,
        max_length: int = 512,
        include_finding_in_text: bool = True,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.include_finding_in_text = include_finding_in_text

    def __len__(self) -> int:
        return len(self.samples)

    def _build_text(self, item: Dict[str, Any]) -> str:
        report_text = item["report_text"]

        if self.include_finding_in_text:
            finding = item["finding"]
            return f"Finding: {finding}. Report: {report_text}"

        return report_text

    def _load_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        image = self._load_image(item["image_path"])
        text = self._build_text(item)

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),

            # metadata
            "image_path": item["image_path"],
            "report_text": item["report_text"],
            "study_id": item["id"],
            "finding": item["finding"],
            "negation_text": item["negation_text"],
            "omitted_text": item["omitted_text"],
            "location": item["location"],
        }