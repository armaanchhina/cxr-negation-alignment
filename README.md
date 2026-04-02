# Multimodal Chest X-ray Retrieval

This project trains a multimodal model to align chest X-ray images with radiology reports using finding-aware contrastive learning. The goal is to retrieve the correct report given an image (image-to-text) and vice versa (text-to-image), evaluated using Recall@K.

The model encodes images with DenseNet-121 and reports with Bio_ClinicalBERT, projecting both into a shared 512-dimensional L2-normalized embedding space. The contrastive loss groups positives by shared radiological finding, encouraging the model to learn clinically meaningful alignment.

## Setup

1. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

This project uses data from PhysioNet. You must have approved access before running anything.

Dataset link:  
https://physionet.org/content/cxr-align/1.0.0/

IMAGE_FILENAMES link:
https://physionet.org/content/mimic-cxr-jpg/2.1.0/

You may also need access to MIMIC-CXR-JPG for downloading images.

After you have access, export your PhysioNet username:

```bash
export PHYSIONET_USERNAME=your_username
```

You will be prompted for your password when downloading images.

## Download Images

The download script uses `wget`. On Linux/Mac it is typically pre-installed. On Windows, install it via [Chocolatey](https://chocolatey.org/) (`choco install wget`) or use WSL.

Before training, download the required chest X-ray images:

```bash
python scripts/download_images.py
```

This script will:
- Build the image ID to path mapping
- Generate required image URLs
- Download images into the `images/` folder

## Train the Model

```bash
python train.py
```

This will:
- Load paired image and report data
- Train the multimodal model using the configured contrastive loss
- Save the best checkpoint
- Log metrics per epoch

**Switching loss function:**
Open `train.py` and set the `USE_FINDING_AWARE_LOSS` constant near the top:

```python
USE_FINDING_AWARE_LOSS = True   # finding-aware multi-positive loss (default)
USE_FINDING_AWARE_LOSS = False  # standard CLIP-style exact-pair loss
```

The selected loss is saved to `outputs/config.json` for reproducibility.

Outputs will be saved to:

```
outputs/
  best_model.pt          # best checkpoint (by avg exact R@1)
  config.json            # hyperparameters used for this run
  train.log              # full training log (also printed to console)
  metrics_epoch_*.json   # per-epoch retrieval metrics + train loss
  final_metrics.json     # final retrieval metrics from best checkpoint
```

> **Note:** Training was run overnight (~8–12 hours). GPU is strongly recommended; CPU training is not practical for the full dataset.

## Run Evaluation

After training:

```bash
python eval.py
```

This will:
- Load the saved checkpoint from `outputs/best_model.pt`
- Evaluate retrieval performance
- Save final metrics to `outputs/results/final_metrics_{finding_aware || exact_match}.json`

## Results

Evaluation on the validation set (344 samples) at epoch 16 using Recall@K:

**Standard contrastive loss:**

| Metric | @1 | @5 | @10 |
|---|---|---|---|
| Image → Text (exact) | 0.0494 | 0.1599 | 0.2384 |
| Text → Image (exact) | 0.0610 | 0.1686 | 0.2558 |
| Image → Text (finding) | 0.2849 | 0.6599 | 0.7994 |
| Text → Image (finding) | 0.2878 | 0.7006 | 0.8663 |

**Finding-aware contrastive loss (best checkpoint):**

| Metric | @1 | @5 | @10 |
|---|---|---|---|
| Image → Text (exact) | 0.0174 | 0.0494 | 0.0901 |
| Text → Image (exact) | 0.0145 | 0.0494 | 0.0727 |
| Image → Text (finding) | 0.1802 | 0.2500 | 0.2733 |
| Text → Image (finding) | 0.7209 | 0.8866 | 0.8866 |

**Exact** recall measures retrieval of the specific paired report/image. **Finding** recall measures retrieval of any report/image sharing the same radiological finding.

The standard loss achieves stronger exact-match and I2T finding recall. The finding-aware loss improves T2I finding recall at higher K (R@5: 0.84 vs 0.70), which is the clinically relevant direction when querying by image. The best-checkpoint finding-aware numbers (bottom table) reflect additional training epochs beyond epoch 16.

## Project Structure

```
.
├── train.py
├── eval.py
├── src/
│   ├── models/
│   │   └── model.py
│   ├── data/
│   │   ├── cxr_dataset.py
│   │   └── io.py
│   └── training/
│       └── train_utils.py
├── scripts/
│   └── download_images.py
├── outputs/
├── images/
├── cxr-align.json
└── IMAGE_FILENAMES
```

## Expected Files

Before training, make sure you have:
- `cxr-align.json` — paired image/report data with finding labels
- `IMAGE_FILENAMES` — mapping of image IDs to file paths
- Downloaded images inside `images/`

> **Note:** `cxr-align.json` and `IMAGE_FILENAMES` paths are currently hardcoded in the data loading scripts. If your directory layout differs, update the paths in `src/data/cxr_dataset.py` accordingly.

## Notes

- A fixed random seed is used for the train/validation split to ensure reproducibility
- GPU is required for practical training; overnight runtime should be expected on the full dataset
- If full dataset reproduction is not possible, a smaller subset can be used by modifying the dataset loader

## Full Workflow

```bash
export PHYSIONET_USERNAME=your_username
python scripts/download_images.py
python train.py
python eval.py
```

This will reproduce the full training and evaluation pipeline.