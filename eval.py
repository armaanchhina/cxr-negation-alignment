from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from src.data.cxr_dataset import CXRMultimodalDataset
from src.models.model import MultimodalCXRModel
from src.training.train_utils import evaluate_retrieval


DATA_PATH = "cxr-align.json"
IMAGE_ROOT = "images"
CHECKPOINT_PATH = Path("outputs/checkpoints/best_model.pt")
RESULTS_PATH = Path("outputs/results/final_metrics.json")


def load_data(path: str):
    with open(path, "r") as f:
        return json.load(f)


def build_samples(data: dict, image_root: str):
    samples = []
    cases = data["mimic"]

    for report_id, case in cases.items():
        image_path = Path(image_root) / f"{report_id}.jpg"

        if image_path.exists():
            samples.append(
                {
                    "id": report_id,
                    "report_text": case["report"],
                    "image_path": str(image_path),
                    "finding": case["chosen"],
                    "negation_text": case["negation"],
                    "omitted_text": case["omitted"],
                    "location": case["location"],
                }
            )

    return samples


def save_metrics(metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    raw_data = load_data(DATA_PATH)
    samples = build_samples(raw_data, IMAGE_ROOT)

    _, val_samples = train_test_split(
        samples,
        test_size=0.2,
        random_state=42,
    )

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_dataset = CXRMultimodalDataset(
        samples=val_samples,
        tokenizer=tokenizer,
        transform=val_transform,
        max_length=128,
    )

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalCXRModel().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    results = evaluate_retrieval(model, val_loader, device, top_k=5)
    metrics = results["metrics"]

    print("Evaluation results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    save_metrics(metrics, RESULTS_PATH)
    print(f"\nSaved metrics to {RESULTS_PATH}")


if __name__ == "__main__":
    main()