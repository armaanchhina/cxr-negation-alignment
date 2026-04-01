from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from src.data.cxr_dataset import CXRMultimodalDataset
from src.data.io import load_data, build_samples, save_metrics
from src.models.model import MultimodalCXRModel
from src.training.train_utils import evaluate_retrieval


DATA_PATH = "cxr-align.json"
IMAGE_ROOT = "images"
CHECKPOINT_PATH = Path("outputs/best_model.pt")
RESULTS_PATH = Path("outputs/results/final_metrics.json")

BATCH_SIZE = 64
MAX_LENGTH = 128
RANDOM_STATE = 42
VAL_SIZE = 0.2


def main():
    raw_data = load_data(DATA_PATH)
    samples = build_samples(raw_data, IMAGE_ROOT)

    _, val_samples = train_test_split(
        samples,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_dataset = CXRMultimodalDataset(
        samples=val_samples,
        tokenizer=tokenizer,
        transform=val_transform,
        max_length=MAX_LENGTH,
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalCXRModel().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    results = evaluate_retrieval(model, val_loader, device, top_k=5)
    metrics = results["metrics"]

    print("Evaluation results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    save_metrics(metrics, RESULTS_PATH)
    print(f"\nSaved metrics to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
