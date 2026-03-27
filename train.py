from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.data.cxr_dataset import CXRMultimodalDataset
from src.models.model import MultimodalCXRModel
from src.training.train_utils import train_one_epoch, evaluate_retrieval


DATA_PATH = "cxr-align.json"
IMAGE_ROOT = "images"
OUTPUT_DIR = Path("outputs")
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pt"


def load_data(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def build_samples(data: Dict, image_root: str) -> List[Dict]:
    samples = []
    cases = data["mimic"]

    for report_id, case in cases.items():
        image_path = os.path.join(image_root, f"{report_id}.jpg")

        if os.path.exists(image_path):
            samples.append({
                "id": report_id,
                "report_text": case["report"],
                "image_path": image_path,
                "finding": case["chosen"],
                "negation_text": case["negation"],
                "omitted_text": case["omitted"],
                "location": case["location"],
            })

    return samples


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, val_transform


def save_metrics(metrics: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw_data = load_data(DATA_PATH)
    samples = build_samples(raw_data, IMAGE_ROOT)

    print(f"Total samples: {len(samples)}")

    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.2,
        random_state=42,
    )

    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    train_transform, val_transform = get_transforms()

    train_dataset = CXRMultimodalDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        transform=train_transform,
        max_length=128,
    )

    val_dataset = CXRMultimodalDataset(
        samples=val_samples,
        tokenizer=tokenizer,
        transform=val_transform,
        max_length=128,
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalCXRModel().to(device)

    optimizer = AdamW([
        {"params": model.text_projection.parameters(), "lr": 1e-3},
        {"params": model.image_projection.parameters(), "lr": 1e-3},
        {"params": model.text_encoder.parameters(), "lr": 5e-6},
        {"params": model.image_encoder.parameters(), "lr": 1e-5},
    ], weight_decay=0.01)

    epochs = 50
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_r1 = 0.0
    patience = 8
    no_improve = 0

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, scheduler
        )

        eval_results = evaluate_retrieval(model, val_loader, device)

        i2t_r1 = eval_results["metrics"]["i2t_r1_exact"]
        t2i_r1 = eval_results["metrics"]["t2i_r1_exact"]
        avg_r1 = (i2t_r1 + t2i_r1) / 2

        print(
            f"Epoch {epoch+1} | "
            f"Loss: {train_metrics['train_loss']:.4f} | "
            f"I2T R@1: {i2t_r1:.4f} | "
            f"T2I R@1: {t2i_r1:.4f}"
        )

        save_metrics(
            eval_results["metrics"],
            OUTPUT_DIR / f"metrics_epoch_{epoch+1}.json",
        )

        if avg_r1 > best_r1:
            best_r1 = avg_r1
            no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> saved new best model (avg R@1={best_r1:.4f})")
        else:
            no_improve += 1
            print(f"  -> no improvement ({no_improve}/{patience})")

        if no_improve >= patience:
            print("Early stopping triggered")
            break

    print(f"\nBest avg R@1: {best_r1:.4f}")

    # --- Final evaluation ---
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    final_results = evaluate_retrieval(model, val_loader, device, top_k=5)

    save_metrics(final_results["metrics"], OUTPUT_DIR / "final_metrics.json")

    print("\nFinal evaluation complete.")


if __name__ == "__main__":
    main()
