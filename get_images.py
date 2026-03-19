import json
import subprocess
import os

BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
IMAGE_MAP_PATH = "image_id_to_path.json"
REPORTS_PATH = "cxr-align.json"
URLS_PATH = "image_urls.txt"
IMAGES_DIR = "images"

image_map = {}
urls = []


def create_image_map():
    with open("IMAGE_FILENAMES", "r") as f:
        for line in f:
            path = line.strip()
            filename = path.split("/")[-1]
            image_id = filename.replace(".jpg", "")
            image_map[image_id] = path

    return image_map


def save_image_map():
    with open(IMAGE_MAP_PATH, "w") as f:
        json.dump(image_map, f)


def images_to_fetch():

    with open(REPORTS_PATH, "r") as f:
        reports = json.load(f)

    with open(IMAGE_MAP_PATH, "r") as f:
        image_map = json.load(f)

    reports = reports["mimic"]

    urls = []
    seen = set()

    for report_id in reports.keys():
        if report_id in image_map:
            url = BASE_URL + image_map[report_id]
            if url not in seen:
                urls.append(url)
                seen.add(url)

    return urls


def save_urls_to_file():
    with open(URLS_PATH, "w") as f:
        for url in urls:
            f.write(url + "\n")


def download_images():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    subprocess.run(
        [
            "wget",
            "--user",
            os.getenv("PHYSIONET_USERNAME"),
            "--ask-password",
            "-i",
            URLS_PATH,
            "-P",
            IMAGES_DIR,
        ],
        check=True,
    )


def main():
    create_image_map()
    save_image_map()
    images_to_fetch()
    save_urls_to_file()
    download_images()


main()
    