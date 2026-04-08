"""
modal_jobs/modal_train_faster_rcnn.py
Trains Faster R-CNN (ResNet50-FPN) on the merged dataset via Modal (T4 GPU).

Usage:
    modal run modal_jobs/modal_train_faster_rcnn.py
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("artillery-faster-rcnn")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
        "pycocotools>=2.0.7",
        "wandb>=0.16.0",
    )
)

results_volume = modal.Volume.from_name("artillery-results", create_if_missing=True)
dataset_volume = modal.Volume.from_name("artillery-dataset", create_if_missing=True)
RESULTS_DIR = Path("/results")
DATASET_DIR = Path("/dataset")

FRCNN_CONFIG = {
    "epochs": 25,
    "batch_size": 4,
    "lr": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "lr_step_size": 8,
    "lr_gamma": 0.1,
    "num_classes": 9,        # 8 canonical + background
    "imgsz": 800,
    "trainable_backbone_layers": 3,
}


class YOLODetectionDataset:
    def __init__(self, img_dir, lbl_dir, imgsz=800):
        self.imgs = sorted(Path(img_dir).glob("*.*"))
        self.lbl_dir = Path(lbl_dir)
        self.imgsz = imgsz

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image

        img_path = self.imgs[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes, labels = [], []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0]) + 1  # +1 for background at index 0
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = max(0.0, (cx - bw / 2) * w)
                y1 = max(0.0, (cy - bh / 2) * h)
                x2 = min(float(w), (cx + bw / 2) * w)
                y2 = min(float(h), (cy + bh / 2) * h)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)

        scale = self.imgsz / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))

        boxes_t = torch.tensor(boxes, dtype=torch.float32) * scale if boxes else torch.zeros((0, 4), dtype=torch.float32)
        target = {
            "boxes": boxes_t,
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return TF.to_tensor(img), target


def _collate_fn(batch):
    return tuple(zip(*batch))


def _get_model(num_classes: int, trainable_backbone_layers: int = 3):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _train_one_epoch(model, optimizer, loader, device, epoch, total_epochs):
    import torch
    model.train()
    total_loss = 0.0
    for i, (imgs, targets) in enumerate(loader):
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        losses = sum(loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += losses.item()
        if i % 50 == 0:
            print(f"  [{epoch}/{total_epochs}] batch {i}/{len(loader)} loss={losses.item():.4f}")
    return total_loss / len(loader)


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 5,
    volumes={
        str(RESULTS_DIR): results_volume,
        str(DATASET_DIR): dataset_volume,
    },
    secrets=[modal.Secret.from_dotenv()],
)
def train_faster_rcnn():
    import os
    import torch
    import yaml
    import wandb
    from torch.utils.data import DataLoader

    # Fix path field in dataset.yaml to point to container location
    yaml_path = DATASET_DIR / "dataset.yaml"
    dataset_yaml = yaml.safe_load(yaml_path.read_text())
    dataset_yaml["path"] = str(DATASET_DIR)
    yaml_path.write_text(yaml.dump(dataset_yaml))

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(project="artillery-detection", name="faster-rcnn-r50", config=FRCNN_CONFIG)

    device = torch.device("cuda")

    train_loader = DataLoader(
        YOLODetectionDataset(DATASET_DIR / "train/images", DATASET_DIR / "train/labels", FRCNN_CONFIG["imgsz"]),
        batch_size=FRCNN_CONFIG["batch_size"], shuffle=True, num_workers=4, collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        YOLODetectionDataset(DATASET_DIR / "val/images", DATASET_DIR / "val/labels", FRCNN_CONFIG["imgsz"]),
        batch_size=FRCNN_CONFIG["batch_size"], shuffle=False, num_workers=4, collate_fn=_collate_fn,
    )

    model = _get_model(FRCNN_CONFIG["num_classes"], FRCNN_CONFIG["trainable_backbone_layers"])
    model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=FRCNN_CONFIG["lr"], momentum=FRCNN_CONFIG["momentum"], weight_decay=FRCNN_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=FRCNN_CONFIG["lr_step_size"], gamma=FRCNN_CONFIG["lr_gamma"],
    )

    ckpt_dir = RESULTS_DIR / "faster_rcnn"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    history = []
    for epoch in range(1, FRCNN_CONFIG["epochs"] + 1):
        t0 = time.time()
        train_loss = _train_one_epoch(model, optimizer, train_loader, device, epoch, FRCNN_CONFIG["epochs"])
        scheduler.step()
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "train_loss": train_loss, "time_s": round(elapsed, 1)})
        print(f"Epoch {epoch}/{FRCNN_CONFIG['epochs']} loss={train_loss:.4f} ({elapsed:.1f}s)")
        if wandb_key:
            wandb.log({"train_loss": train_loss, "epoch": epoch, "lr": scheduler.get_last_lr()[0]})

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  New best (loss={best_loss:.4f})")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")

    torch.save(model.state_dict(), ckpt_dir / "last.pt")
    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))

    if wandb_key:
        wandb.summary["best_train_loss"] = best_loss
        wandb.finish()

    # Inference latency on val set
    model.eval()
    latencies = []
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = [img.to(device) for img in imgs]
            t0 = time.perf_counter()
            model(imgs)
            latencies.append((time.perf_counter() - t0) / len(imgs) * 1000)
            if len(latencies) >= 50:
                break

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    metrics = {
        "best_train_loss": round(best_loss, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "fps": round(1000 / avg_latency, 1) if avg_latency > 0 else 0,
    }

    (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    results_volume.commit()
    print(f"Faster R-CNN done. {metrics}")
    return metrics


@app.local_entrypoint()
def main():
    from dotenv import load_dotenv
    load_dotenv()

    splits_root = Path("data/splits")
    if not (splits_root / "dataset.yaml").exists():
        print("ERROR: data/splits/dataset.yaml not found. Run src/data/merge_datasets.py first.")
        return

    metrics = train_faster_rcnn.remote()
    print(f"Faster R-CNN metrics: {metrics}")

    out = Path("results/metrics/faster_rcnn_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
