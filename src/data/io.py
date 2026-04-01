import json
from pathlib import Path
from typing import Dict, List


def load_data(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def build_samples(data: Dict, image_root: str) -> List[Dict]:
    """Build sample list from cxr-align.json, skipping missing images."""
    samples = []
    cases = data["mimic"]

    for report_id, case in cases.items():
        image_path = Path(image_root) / f"{report_id}.jpg"

        if image_path.exists():
            samples.append({
                "id": report_id,
                "report_text": case["report"],
                "image_path": str(image_path),
                "finding": case["chosen"],
                "negation_text": case["negation"],
                "omitted_text": case["omitted"],
                "location": case["location"],
            })

    return samples


def save_metrics(metrics: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
