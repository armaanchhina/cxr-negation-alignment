import json
import logging
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.data.cxr_dataset import CXRMultimodalDataset
from src.data.io import load_data, build_samples, save_metrics
from src.models.model import MultimodalCXRModel
from src.training.train_utils import train_one_epoch, evaluate_retrieval


DATA_PATH = "cxr-align.json"
IMAGE_ROOT = "images"
OUTPUT_DIR = Path("outputs")
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pt"
LOG_PATH = OUTPUT_DIR / "train.log"
CONFIG_PATH = OUTPUT_DIR / "config.json"

BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 8
MAX_LENGTH = 128
EMBEDDING_DIM = 512
WARMUP_FRACTION = 0.05
WEIGHT_DECAY = 0.01
LR_PROJECTION = 1e-3
LR_TEXT_ENCODER = 5e-6
LR_IMAGE_ENCODER = 1e-5
RANDOM_STATE = 42
VAL_SIZE = 0.2
USE_FINDING_AWARE_LOSS = True


def setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )
    # makes log file more readble
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)


def save_config(path: Path) -> None:
    config = {
        "data_path": DATA_PATH,
        "image_root": IMAGE_ROOT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "max_length": MAX_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "warmup_fraction": WARMUP_FRACTION,
        "weight_decay": WEIGHT_DECAY,
        "lr_projection": LR_PROJECTION,
        "lr_text_encoder": LR_TEXT_ENCODER,
        "lr_image_encoder": LR_IMAGE_ENCODER,
        "random_state": RANDOM_STATE,
        "val_size": VAL_SIZE,
        "use_finding_aware_loss": USE_FINDING_AWARE_LOSS,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


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


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    setup_logging(LOG_PATH)
    save_config(CONFIG_PATH) # so user can see exactly what was used to train this model

    raw_data = load_data(DATA_PATH)
    samples = build_samples(raw_data, IMAGE_ROOT)

    logging.info(f"Total samples: {len(samples)}")

    train_samples, val_samples = train_test_split(
        samples,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    logging.info(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    train_transform, val_transform = get_transforms()

    train_dataset = CXRMultimodalDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        transform=train_transform,
        max_length=MAX_LENGTH,
    )

    val_dataset = CXRMultimodalDataset(
        samples=val_samples,
        tokenizer=tokenizer,
        transform=val_transform,
        max_length=MAX_LENGTH,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = MultimodalCXRModel(embedding_dim=EMBEDDING_DIM).to(device)

    optimizer = AdamW([
        {"params": model.text_projection.parameters(), "lr": LR_PROJECTION},
        {"params": model.image_projection.parameters(), "lr": LR_PROJECTION},
        {"params": model.text_encoder.parameters(), "lr": LR_TEXT_ENCODER},
        {"params": model.image_encoder.parameters(), "lr": LR_IMAGE_ENCODER},
    ], weight_decay=WEIGHT_DECAY)

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_FRACTION * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_r1 = 0.0
    no_improve = 0
    suffix = "_finding_aware" if USE_FINDING_AWARE_LOSS else "_exact_match"
    for epoch in range(EPOCHS):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, scheduler,
            use_finding_aware_loss=USE_FINDING_AWARE_LOSS,
        )

        eval_results = evaluate_retrieval(model, val_loader, device)

        i2t_r1 = eval_results["metrics"]["i2t_r1_exact"]
        t2i_r1 = eval_results["metrics"]["t2i_r1_exact"]
        avg_r1 = (i2t_r1 + t2i_r1) / 2

        logging.info(
            f"Epoch {epoch+1} | "
            f"Loss: {train_metrics['train_loss']:.4f} | "
            f"I2T R@1: {i2t_r1:.4f} | "
            f"T2I R@1: {t2i_r1:.4f}"
        )

        epoch_metrics = {**eval_results["metrics"], "train_loss": train_metrics["train_loss"]}
        analysis = eval_results["i2t_analysis"]
        save_metrics(analysis, OUTPUT_DIR / f"analysis_epoch_{epoch+1}{suffix}.json")
        save_metrics(epoch_metrics, OUTPUT_DIR / f"metrics_epoch_{epoch+1}{suffix}.json")

        if avg_r1 > best_r1:
            best_r1 = avg_r1
            no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            logging.info(f"  -> saved new best model (avg R@1={best_r1:.4f})")
        else:
            no_improve += 1
            logging.info(f"  -> no improvement ({no_improve}/{PATIENCE})")

        if no_improve >= PATIENCE:
            logging.info("Early stopping triggered")
            break

    logging.info(f"\nBest avg R@1: {best_r1:.4f}")

    # Final evaluation on best checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    final_results = evaluate_retrieval(model, val_loader, device, top_k=5)
    save_metrics(final_results["metrics"], OUTPUT_DIR / f"final_metrics{suffix}.json")

    logging.info("Final evaluation complete.")


if __name__ == "__main__":
    main()
