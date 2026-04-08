"""
Evaluates all trained models from the Modal Volume and produces a comparison JSON.

"""

import json
from pathlib import Path

import modal

app = modal.App("artillery-evaluate")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("apt-get update -y && apt-get install -y --no-install-recommends libgl1 libglib2.0-0")
    .pip_install(
        "ultralytics>=8.3.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
        "pycocotools>=2.0.7",
    )
)

results_volume = modal.Volume.from_name("artillery-results", create_if_missing=True)
dataset_volume = modal.Volume.from_name("artillery-dataset", create_if_missing=True)
RESULTS_DIR = Path("/results")
DATASET_DIR = Path("/dataset")

CANONICAL_CLASSES = [
    "artillery", "tank", "apc", "military_truck",
    "rocket_artillery", "ifv", "military_aircraft", "other_military",
]


def _benchmark_fps(model_fn, test_imgs, warmup=10):
    import time
    for p in test_imgs[:warmup]:
        model_fn(p)
    t0 = time.perf_counter()
    for p in test_imgs:
        model_fn(p)
    return len(test_imgs) / (time.perf_counter() - t0)


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 2,
    volumes={
        str(RESULTS_DIR): results_volume,
        str(DATASET_DIR): dataset_volume,
    },
)
def evaluate_all_models():
    import time
    import torch
    import yaml
    from ultralytics import YOLO, RTDETR

    # Fix path field in dataset.yaml to point to container location
    yaml_path = DATASET_DIR / "dataset.yaml"
    dataset_yaml = yaml.safe_load(yaml_path.read_text())
    dataset_yaml["path"] = str(DATASET_DIR)
    yaml_path.write_text(yaml.dump(dataset_yaml))

    test_imgs = list((DATASET_DIR / "test" / "images").glob("*.*"))[:50]
    comparison = {}

    # YOLOv11
    yolo_ckpt = RESULTS_DIR / "yolov11" / "run" / "weights" / "best.pt"
    if yolo_ckpt.exists():
        model = YOLO(str(yolo_ckpt))
        val_results = model.val(data=str(yaml_path), split="test", verbose=False)
        fps = _benchmark_fps(lambda p: model.predict(str(p), verbose=False), test_imgs)

        per_class_ap = {}
        if hasattr(val_results, "ap_class_index") and val_results.ap_class_index is not None:
            for i, cls_idx in enumerate(val_results.ap_class_index):
                if cls_idx < len(CANONICAL_CLASSES):
                    per_class_ap[CANONICAL_CLASSES[cls_idx]] = float(val_results.box.ap50[i])

        comparison["YOLOv11m"] = {
            "map50": float(val_results.box.map50),
            "map50_95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
            "fps": round(fps, 1),
            "latency_ms": round(1000 / fps, 2),
            "params_M": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 1),
            "model_size_mb": round(yolo_ckpt.stat().st_size / 1e6, 1),
            "per_class_ap50": per_class_ap,
            "architecture": "One-stage CNN",
            "realtime": True,
        }
        print(f"YOLOv11m: mAP50={comparison['YOLOv11m']['map50']:.3f} FPS={fps:.1f}")
    else:
        print(f"YOLOv11 checkpoint not found at {yolo_ckpt} -- skipping")

    # RT-DETR
    rtdetr_ckpt = RESULTS_DIR / "rtdetr" / "run" / "weights" / "best.pt"
    if rtdetr_ckpt.exists():
        model = RTDETR(str(rtdetr_ckpt))
        val_results = model.val(data=str(yaml_path), split="test", verbose=False)
        fps = _benchmark_fps(lambda p: model.predict(str(p), verbose=False), test_imgs)

        per_class_ap = {}
        if hasattr(val_results, "ap_class_index") and val_results.ap_class_index is not None:
            for i, cls_idx in enumerate(val_results.ap_class_index):
                if cls_idx < len(CANONICAL_CLASSES):
                    per_class_ap[CANONICAL_CLASSES[cls_idx]] = float(val_results.box.ap50[i])

        comparison["RT-DETR-L"] = {
            "map50": float(val_results.box.map50),
            "map50_95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
            "fps": round(fps, 1),
            "latency_ms": round(1000 / fps, 2),
            "params_M": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 1),
            "model_size_mb": round(rtdetr_ckpt.stat().st_size / 1e6, 1),
            "per_class_ap50": per_class_ap,
            "architecture": "One-stage Transformer",
            "realtime": True,
        }
        print(f"RT-DETR-L: mAP50={comparison['RT-DETR-L']['map50']:.3f} FPS={fps:.1f}")
    else:
        print(f"RT-DETR checkpoint not found at {rtdetr_ckpt} -- skipping")

    # Faster R-CNN
    frcnn_ckpt = RESULTS_DIR / "faster_rcnn" / "best.pt"
    if frcnn_ckpt.exists():
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from PIL import Image
        import torchvision.transforms.functional as TF

        device = torch.device("cuda")
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 9)
        model.load_state_dict(torch.load(str(frcnn_ckpt), map_location=device))
        model.to(device).eval()

        def frcnn_infer(p):
            with torch.no_grad():
                img = TF.to_tensor(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                model(img)

        fps = _benchmark_fps(frcnn_infer, test_imgs)

        comparison["Faster-RCNN-R50"] = {
            "map50": None,
            "map50_95": None,
            "precision": None,
            "recall": None,
            "fps": round(fps, 1),
            "latency_ms": round(1000 / fps, 2),
            "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "model_size_mb": round(frcnn_ckpt.stat().st_size / 1e6, 1),
            "per_class_ap50": {},
            "architecture": "Two-stage CNN",
            "realtime": False,
            "note": "mAP requires COCO-style eval",
        }
        print(f"Faster-RCNN-R50: FPS={fps:.1f}")
    else:
        print(f"Faster R-CNN checkpoint not found at {frcnn_ckpt} -- skipping")

    out_path = RESULTS_DIR / "comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2))
    results_volume.commit()
    print(f"Comparison saved to {out_path}")
    return comparison


@app.local_entrypoint()
def main():
    from dotenv import load_dotenv
    load_dotenv()

    splits_root = Path("data/splits")
    if not (splits_root / "dataset.yaml").exists():
        print("ERROR: data/splits/dataset.yaml not found. Run src/data/merge_datasets.py first.")
        return

    comparison = evaluate_all_models.remote()

    out = Path("results/metrics/comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, indent=2))
    print(f"Saved to {out}")
    for model_name, metrics in comparison.items():
        print(f"\n{model_name}: " + ", ".join(f"{k}={v}" for k, v in metrics.items() if k != "per_class_ap50"))
