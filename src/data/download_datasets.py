"""
src/data/download_datasets.py
Downloads datasets:
  1. Kaggle: rawsi18/military-assets-dataset-12-classes-yolo8-format
  2. Roboflow: uavmildec/mil-det
"""

import os
import subprocess
import zipfile
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_kaggle_dataset():
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise EnvironmentError("Set KAGGLE_USERNAME and KAGGLE_KEY in .env")

    dest = RAW_DIR / "kaggle_military_assets"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "rawsi18/military-assets-dataset-12-classes-yolo8-format",
            "--path", str(dest),
            "--unzip",
        ],
        check=True,
    )
    console.print(f"[green]✓ Kaggle dataset → {dest}[/green]")
    return dest


def download_roboflow_dataset(workspace: str, project: str, version: int, dest_name: str):
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ROBOFLOW_API_KEY in .env")

    dest = RAW_DIR / dest_name
    dest.mkdir(parents=True, exist_ok=True)

    import urllib.request, json
    # Use direct API — SDK v1.x silently skips download when files already exist
    api_url = f"https://api.roboflow.com/{workspace}/{project}/{version}/yolov8?api_key={api_key}"
    with urllib.request.urlopen(api_url) as r:
        link = json.loads(r.read())["export"]["link"]
    zip_path = dest / f"{project}.zip"
    urllib.request.urlretrieve(link, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    zip_path.unlink()

    console.print(f"[green]✓ {project} → {dest}[/green]")
    return dest


if __name__ == "__main__":
    try:
        download_kaggle_dataset()
    except Exception as e:
        console.print(f"[red]Kaggle failed: {e}[/red]")

    try:
        download_roboflow_dataset("uavmildec", "mil-det", 1, "roboflow_mil_det")
    except Exception as e:
        console.print(f"[red]mil-det failed: {e}[/red]")
