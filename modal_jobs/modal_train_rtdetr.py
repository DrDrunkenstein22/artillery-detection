"""
Trains RT-DETR-L on the merged dataset via Modal (T4 GPU).

Usage:
    modal run modal_jobs/modal_train_rtdetr.py
"""

import json
from pathlib import Path

import modal

app = modal.App("artillery-rtdetr")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands(
        "apt-get update -y && apt-get install -y --no-install-recommends libgl1 libglib2.0-0"
    )
    .pip_install(
        "ultralytics>=8.3.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "wandb>=0.16.0",
        "PyYAML>=6.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "pycocotools>=2.0.7",
    )
)

results_volume = modal.Volume.from_name("artillery-results", create_if_missing=True)
dataset_volume = modal.Volume.from_name("artillery-dataset", create_if_missing=True)
RESULTS_DIR = Path("/results")
DATASET_DIR = Path("/dataset")

RTDETR_CONFIG = {
    "model": "rtdetr-l.pt",
    "epochs": 50,
    "imgsz": 640,
    "batch": 8,
    "optimizer": "AdamW",
    "lr0": 0.0001,
    "lrf": 0.01,
    "weight_decay": 0.0001,
    "warmup_epochs": 3,
    "patience": 15,
    "save_period": 5,
    "exist_ok": True,
    "project": str(RESULTS_DIR / "rtdetr"),
    "name": "run",
    "workers": 4,
    "device": "0",
    "amp": True,
    "verbose": True,
}


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 4,
    volumes={
        str(RESULTS_DIR): results_volume,
        str(DATASET_DIR): dataset_volume,
    },
    secrets=[modal.Secret.from_dotenv()],
)
def train_rtdetr():
    import json
    import yaml
    from ultralytics import RTDETR

    # Fix path field in dataset.yaml to point to container location
    yaml_path = DATASET_DIR / "dataset.yaml"
    dataset_yaml = yaml.safe_load(yaml_path.read_text())
    dataset_yaml["path"] = str(DATASET_DIR)
    yaml_path.write_text(yaml.dump(dataset_yaml))

    model = RTDETR(RTDETR_CONFIG["model"])
    cfg = {**RTDETR_CONFIG, "data": str(yaml_path)}
    cfg.pop("model")
    results = model.train(**cfg)

    metrics = {
        "map50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
    }

    metrics_path = RESULTS_DIR / "rtdetr" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    results_volume.commit()
    print(f"RT-DETR done. {metrics}")
    return metrics


@app.local_entrypoint()
def main():
    from dotenv import load_dotenv

    load_dotenv()

    splits_root = Path("data/splits")
    if not (splits_root / "dataset.yaml").exists():
        print(
            "ERROR: data/splits/dataset.yaml not found. Run src/data/merge_datasets.py first."
        )
        return

    metrics = train_rtdetr.remote()
    print(f"RT-DETR metrics: {metrics}")

    out = Path("results/metrics/rtdetr_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
