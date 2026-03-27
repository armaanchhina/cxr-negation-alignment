from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List


BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
IMAGE_FILENAMES_PATH = Path("IMAGE_FILENAMES")
IMAGE_MAP_PATH = Path("image_id_to_path.json")
REPORTS_PATH = Path("cxr-align.json")
URLS_PATH = Path("image_urls.txt")
IMAGES_DIR = Path("images")


def create_image_map(image_filenames_path: Path) -> Dict[str, str]:
    """
    Build a mapping from image_id to relative PhysioNet JPG path.
    """
    if not image_filenames_path.exists():
        raise FileNotFoundError(f"Missing file: {image_filenames_path}")

    image_map: Dict[str, str] = {}

    with open(image_filenames_path, "r") as f:
        for line in f:
            relative_path = line.strip()
            if not relative_path:
                continue

            filename = relative_path.split("/")[-1]
            image_id = filename.replace(".jpg", "")
            image_map[image_id] = relative_path

    return image_map


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_reports(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r") as f:
        return json.load(f)


def build_image_urls(
    reports_path: Path,
    image_map_path: Path,
    base_url: str,
) -> List[str]:
    """
    Build a deduplicated list of image URLs needed for the dataset.
    """
    reports = load_reports(reports_path)

    if not image_map_path.exists():
        raise FileNotFoundError(f"Missing file: {image_map_path}")

    with open(image_map_path, "r") as f:
        image_map = json.load(f)

    urls: List[str] = []
    seen = set()

    for report_id in reports["mimic"].keys():
        if report_id in image_map:
            url = base_url + image_map[report_id]
            if url not in seen:
                urls.append(url)
                seen.add(url)

    return urls


def save_urls(urls: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for url in urls:
            f.write(url + "\n")


def download_images(
    urls_path: Path,
    output_dir: Path,
    username_env: str = "PHYSIONET_USERNAME",
) -> None:
    """
    Download images listed in urls_path using wget.

    The script expects the PhysioNet username in an environment variable.
    Password entry happens securely in the terminal through wget.
    """
    if not urls_path.exists():
        raise FileNotFoundError(f"Missing file: {urls_path}")

    username = os.getenv(username_env)
    if not username:
        raise EnvironmentError(
            f"Environment variable {username_env} is not set."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "wget",
            "--user",
            username,
            "--ask-password",
            "-i",
            str(urls_path),
            "-P",
            str(output_dir),
        ],
        check=True,
    )


def main() -> None:
    image_map = create_image_map(IMAGE_FILENAMES_PATH)
    save_json(image_map, IMAGE_MAP_PATH)

    urls = build_image_urls(
        reports_path=REPORTS_PATH,
        image_map_path=IMAGE_MAP_PATH,
        base_url=BASE_URL,
    )
    save_urls(urls, URLS_PATH)

    print(f"Saved {len(image_map)} image id mappings to {IMAGE_MAP_PATH}")
    print(f"Saved {len(urls)} image URLs to {URLS_PATH}")

    download_images(
        urls_path=URLS_PATH,
        output_dir=IMAGES_DIR,
    )

    print(f"Downloaded images into {IMAGES_DIR}")


if __name__ == "__main__":
    main()