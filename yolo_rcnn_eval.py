import torch
import time
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.transforms import functional as F
from ultralytics import YOLO
import os
from PIL import Image
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_WEIGHTS = (
    "/kaggle/input/yolov11-postrained-detection/pytorch/default/1/yolov11_art_best.pt"
)
RCNN_WEIGHTS = (
    "/kaggle/input/faster-rcnn-postrained/pytorch/default/1/faster_rcnn_best.pth"
)

root = "/kaggle/input/military-assets-dataset-12-classes-yolo8-format/military_object_dataset"
BATCH_SIZE = 4
WARMUP_ITERS = 5


class DetectionDataset:
    def __init__(self, root):
        self.img_dir = os.path.join(root, "val/images")
        self.label_dir = os.path.join(root, "val/labels")
        self.images = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.images[idx])).convert("RGB")
        label_path = os.path.join(
            self.label_dir, self.images[idx].replace(".jpg", ".txt")
        )

        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for l in f:
                    c, x, y, w, h = map(float, l.split())
                    W, H = img.size
                    boxes.append(
                        [
                            (x - w / 2) * W,
                            (y - h / 2) * H,
                            (x + w / 2) * W,
                            (y + h / 2) * H,
                        ]
                    )
                    labels.append(int(c) + 1)

        return (
            F.to_tensor(img),
            {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            },
        )


def benchmark_frcnn(model, dataloader, name):
    model.eval()
    metric = MeanAveragePrecision()
    torch.cuda.reset_peak_memory_stats()

    print(f"Warming up {name}...", flush=True)
    for i, (imgs, _) in zip(range(WARMUP_ITERS), dataloader):
        with torch.no_grad():
            model([img.to(DEVICE) for img in imgs])
    print(f"Warmup complete", flush=True)

    print(f"Benchmarking {name}...", flush=True)
    start = time.time()
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = [img.to(DEVICE) for img in imgs]
            outputs = model(imgs)

            for o, t in zip(outputs, targets):
                keep = o["scores"] > 0.05
                metric.update(
                    [
                        {
                            "boxes": o["boxes"][keep].cpu(),
                            "scores": o["scores"][keep].cpu(),
                            "labels": o["labels"][keep].cpu(),
                        }
                    ],
                    [{"boxes": t["boxes"], "labels": t["labels"]}],
                )

    elapsed = time.time() - start
    fps = len(dataloader.dataset) / elapsed
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"Benchmark complete", flush=True)
    results = metric.compute()

    return {
        "Model": name,
        "mAP@0.5": f"{results['map_50'].item():.4f}",
        "mAP@0.5:0.95": f"{results['map'].item():.4f}",
        "FPS": round(fps, 2),
        "VRAM": round(mem, 2),
        "Inference Time": round(elapsed, 2),
    }


def benchmark_yolo(model, dataloader, name):

    metric = MeanAveragePrecision()
    torch.cuda.reset_peak_memory_stats()

    print(f"Warming up {name}...", flush=True)
    for i, (imgs, _) in zip(range(WARMUP_ITERS), dataloader):
        with torch.no_grad():
            batch = [
                (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                for img in imgs
            ]
            model(batch, verbose=False, device=DEVICE)
    print(f"Warmup complete", flush=True)

    print(f"Benchmarking {name}...", flush=True)
    start = time.time()
    with torch.no_grad():
        for imgs, targets in dataloader:
            batch = [
                (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                for img in imgs
            ]
            results = model(batch, verbose=False, device=DEVICE)

            for result, target in zip(results, targets):
                if len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu()
                    scores = result.boxes.conf.cpu()
                    labels = result.boxes.cls.cpu().int() + 1
                else:
                    boxes = torch.empty((0, 4))
                    scores = torch.empty((0,))
                    labels = torch.empty((0,), dtype=torch.int64)

                metric.update(
                    [{"boxes": boxes, "scores": scores, "labels": labels}],
                    [{"boxes": target["boxes"], "labels": target["labels"]}],
                )

    elapsed = time.time() - start
    fps = len(dataloader.dataset) / elapsed
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"Benchmark complete", flush=True)
    results = metric.compute()

    return {
        "Model": name,
        "mAP@0.5": f"{results['map_50'].item():.4f}",
        "mAP@0.5:0.95": f"{results['map'].item():.4f}",
        "FPS": round(fps, 2),
        "VRAM (GB)": round(mem, 2),
        "Inference Time (s)": round(elapsed, 2),
    }


print("=" * 60, flush=True)
print("MILITARY OBJECT DETECTION BENCHMARK", flush=True)
print("=" * 60, flush=True)
print(f"Device: {DEVICE}", flush=True)
print(f"Batch Size: {BATCH_SIZE}", flush=True)
print(f"Warmup Iterations: {WARMUP_ITERS}", flush=True)
print("=" * 60, flush=True)

print("\nLoading validation dataset...", flush=True)
dataset = DetectionDataset(root)
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x))
)
print(f"Loaded {len(dataset)} validation images", flush=True)


print("\n" + "=" * 60, flush=True)
print("BENCHMARKING YOLO", flush=True)
print("=" * 60, flush=True)
print("Loading YOLO weights...", flush=True)
yolo = YOLO(YOLO_WEIGHTS)
print("YOLO weights loaded", flush=True)
yolo_results = benchmark_yolo(yolo, loader, "YOLOv11")


print("\n" + "=" * 60, flush=True)
print("BENCHMARKING FASTER R-CNN", flush=True)
print("=" * 60, flush=True)
print("Loading Faster R-CNN model...", flush=True)
frcnn = fasterrcnn_resnet50_fpn(weights=None)
in_feats = frcnn.roi_heads.box_predictor.cls_score.in_features
frcnn.roi_heads.box_predictor = FastRCNNPredictor(
    in_feats, 13
)  # 12 classes + background
print("Loading checkpoint...", flush=True)
ckpt = torch.load(RCNN_WEIGHTS, map_location=DEVICE)
frcnn.load_state_dict(ckpt["model_state"])
print("Moving model to device...", flush=True)
frcnn.to(DEVICE)
print("Faster R-CNN ready", flush=True)

rcnn_results = benchmark_frcnn(frcnn, loader, "Faster R-CNN")


print("\n" + "=" * 60)
print("FINAL BENCHMARK RESULTS")
print("=" * 60)
df = pd.DataFrame([yolo_results, rcnn_results])
print("\n" + df.to_string(index=False))
print("\n" + "=" * 60)

output_file = "/kaggle/working/benchmark_results.csv"
df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")
print("=" * 60)
