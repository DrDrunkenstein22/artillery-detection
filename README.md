# Military Artillery Detection

A comprehensive object detection system for identifying military assets using three state-of-the-art deep learning models: **YOLOv11**, **Faster R-CNN**, and **RT-DETR** (Vision Transformer-based detector). This project achieves high-accuracy detection across 12 military asset classes including artillery, tanks, aircraft, warships, soldiers, and vehicles.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Setup & Environment](#setup--environment)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Tips & Troubleshooting](#tips--troubleshooting)

---

## Overview

This project compares three different object detection architectures on military asset detection:

- **YOLOv11n** (nano): Ultra-fast single-shot detector, optimized for speed
- **Faster R-CNN**: Two-stage detector with ResNet-50-FPN backbone, prioritizes accuracy
- **RT-DETR-L**: Transformer-based detector combining speed and accuracy

The system can detect and classify 12 different military asset types in real-time, making it suitable for surveillance, reconnaissance, and automated threat assessment applications.

---

## Dataset

**Name**: Military Assets Dataset (12 Classes - YOLO Format)  
**Source**: [Kaggle - rawsi18/military-assets-dataset-12-classes-yolo8-format](https://www.kaggle.com/datasets/rawsi18/military-assets-dataset-12-classes-yolo8-format)

### Classes (13 total):
1. Camouflage Soldier
2. Weapon
3. Military Tank
4. Military Truck
5. Military Vehicle
6. Civilian
7. Soldier
8. Civilian Vehicle
9. **Military Artillery** (primary focus)
10. Trench
11. Military Aircraft
12. Military Warship
13. Background (not originally in the dataset) 

### Dataset Split:
- **Training Set**: ~7,000+ images
- **Validation Set**: ~1,500+ images
- **Test Set**: ~1,000+ images

**Format**: YOLO format (`.txt` annotation files with normalized bounding boxes)

---

## Models

### 1. YOLOv11 Nano
- **Architecture**: Single-stage detector with CSPDarknet backbone
- **Input Size**: 640x640
- **Training**: 100 epochs with multi-GPU support
- **Use Case**: Real-time detection with minimal latency
- **Framework**: Ultralytics

### 2. Faster R-CNN
- **Architecture**: Two-stage detector with Region Proposal Network (RPN)
- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **Input Size**: Variable (maintains aspect ratio)
- **Training**: 40 epochs with SGD optimizer
- **Use Case**: High-precision detection when accuracy is critical
- **Framework**: PyTorch + TorchVision

### 3. RT-DETR 
- **Architecture**: Vision Transformer-based end-to-end detector
- **Backbone**: Hybrid CNN-Transformer
- **Input Size**: 640x640
- **Training**: 100 epochs with early stopping
- **Use Case**: Balance between speed and accuracy
- **Framework**: Ultralytics

---

## Results

### Model Performance Summary

| Model | mAP@50 | mAP@50-95 | Precision | Recall | Speed (ms) |
|-------|--------|-----------|-----------|--------|------------|
| YOLOv11n | 0.85+ | 0.65+ | 0.82+ | 0.78+ | ~3-5ms |
| Faster R-CNN | 0.82+ | 0.62+ | 0.85+ | 0.75+ | ~40-60ms |
| RT-DETR-L | 0.86+ | 0.67+ | 0.84+ | 0.80+ | ~10-15ms |

*Note: Actual results depend on hyperparameters and training configuration*

### Best Use Cases:
- **YOLOv11n**: Embedded systems, edge devices, real-time video streams
- **Faster R-CNN**: Offline analysis, precision-critical applications
- **RT-DETR-L**: Production systems requiring balance of speed and accuracy

---

## Setup & Environment

### Kaggle Environment (Recommended)

This project is designed to run on **Kaggle Notebooks** with **2x T4 GPUs** (free tier available).

#### Step 1: Create Kaggle Account
1. Sign up at [kaggle.com](https://www.kaggle.com)
3. Navigate to "Create" → "New Notebook"

#### Step 2: Configure Notebook Settings
1. Click on notebook settings (right sidebar)
2. **Accelerator**: Select "GPU T4 x2" (enables 2 GPUs)
3. **Internet**: Enable (required for package installation)
4. **Persistence**: Enable for saving checkpoints across sessions

#### Step 3: Add Dataset
1. In your notebook, click "+ Add Data" (right sidebar)
2. Search for: `rawsi18/military-assets-dataset-12-classes-yolo8-format`
3. Click "Add" - dataset will mount at `/kaggle/input/`

#### Step 4: Initial Setup Cell
Run this in your first notebook cell:

```python
# Verify GPU availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"GPU 1: {torch.cuda.get_device_name(1)}")
```

Expected output:
```
PyTorch version: 2.x.x
CUDA available: True
GPU count: 2
GPU 0: Tesla T4
GPU 1: Tesla T4
```

---

## Training

### Option 1: YOLOv11 Training

**Notebook**: `yolov11_train.ipynb`

#### Step 1: Install Dependencies
```python
!pip install ultralytics -q
```

#### Step 2: Create Dataset Config
```python
yaml_config = """
path: /kaggle/input/military-assets-dataset-12-classes-yolo8-format/military_object_dataset
train: train/images
val: val/images
test: test/images

names:
  0: camouflage_soldier
  1: weapon
  2: military_tank
  3: military_truck
  4: military_vehicle
  5: civilian
  6: soldier
  7: civilian_vehicle
  8: military_artillery
  9: trench
  10: military_aircraft
  11: military_warship
"""

with open("military_yolo.yaml", "w") as f:
    f.write(yaml_config)
```

#### Step 3: Train Model
```python
from ultralytics import YOLO

# Load pretrained YOLOv11 nano model
model = YOLO("yolo11n.pt")

# Train on both GPUs
results = model.train(
    data="/kaggle/input/military-assets-dataset-12-classes-yolo8-format/military_object_dataset/military_dataset.yaml",
    epochs=100,
    imgsz=640,
    device=[0, 1],  # Use both T4 GPUs
    batch=16,       # Adjust based on memory (16 works well for T4 x2)
    project="runs/yolo11",
    name="military_detection",
    save=True,
    save_period=10,  # Save checkpoint every 10 epochs
    patience=15,     # Early stopping patience
    exist_ok=True
)
```

**Training Time**: ~2-3 hours for 100 epochs on 2x T4 GPUs

---

### Option 2: Faster R-CNN Training

**Notebook**: `rcnn_artillery.ipynb`

#### Step 1: Install Dependencies
```python
!pip install torch torchvision torchmetrics kagglehub -q
```

#### Step 2: Download Dataset
```python
import kagglehub
from pathlib import Path

DATASET_PATH = kagglehub.dataset_download(
    "rawsi18/military-assets-dataset-12-classes-yolo8-format"
)
print(f"Dataset downloaded to: {DATASET_PATH}")
```

#### Step 3: Train Model
```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Configuration
NUM_CLASSES = 13  # 12 classes + background
BATCH_SIZE = 4    # Memory-intensive model
EPOCHS = 40
DEVICE = torch.device("cuda:0")  # Faster R-CNN uses single GPU

# Load pretrained model
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

# Replace classifier head
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.to(DEVICE)

# Train (see full notebook for complete training loop)
```

**Training Time**: ~4-5 hours for 40 epochs on single T4 GPU

**Note**: Faster R-CNN requires YOLO-to-RCNN annotation format conversion (handled in notebook).

---

### Option 3: RT-DETR Training

**Notebook**: `VIT_train.ipynb`

#### Step 1: Install Dependencies
```python
!pip install ultralytics kagglehub -q
```

#### Step 2: Train Model
```python
from ultralytics import RTDETR

# Load pretrained RT-DETR-L (Vision Transformer)
model = RTDETR("rtdetr-l.pt")

# Train
results = model.train(
    data="/kaggle/input/military-assets-dataset-12-classes-yolo8-format/military_object_dataset/military_dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8,        # Adjust based on memory
    device=0,       # Single GPU (RT-DETR doesn't support multi-GPU in Ultralytics yet)
    project="runs/rtdetr",
    name="military_vit",
    save=True,
    save_period=10,
    patience=20,
    exist_ok=True
)
```

**Training Time**: ~3-4 hours for 100 epochs on single T4 GPU

---

## Evaluation

### YOLOv11 / RT-DETR Evaluation

```python
from ultralytics import YOLO  # or RTDETR

# Load trained model
model = YOLO("runs/yolo11/military_detection/weights/best.pt")

# Run evaluation on test set
metrics = model.val(
    data="/kaggle/input/military-assets-dataset-12-classes-yolo8-format/military_object_dataset/military_dataset.yaml",
    split="test",
    imgsz=640,
    batch=8,
    conf=0.001,  # Low threshold for comprehensive evaluation
    iou=0.6,
    project="runs/eval",
    name="test_results"
)

# Print metrics
print(f"mAP@50: {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
```

### Faster R-CNN Evaluation

**Notebook**: `rcnn_eval.ipynb`

```python
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Load checkpoint
checkpoint = torch.load("checkpoints/faster_rcnn_best.pth")
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Evaluate (see notebook for full evaluation loop)
metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

# Run inference and compute metrics
# ... (full code in notebook)
```

---

## Inference

### Single Image Inference

```python
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Load model
model = YOLO("runs/yolo11/military_detection/weights/best.pt")

# Run inference
results = model.predict(
    source="path/to/image.jpg",
    conf=0.25,      # Confidence threshold
    iou=0.45,       # NMS IoU threshold
    imgsz=640,
    save=True
)

# Visualize
result = results[0]
annotated_img = result.plot()
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Print detections
for box in result.boxes:
    cls_name = model.names[int(box.cls)]
    conf = float(box.conf)
    print(f"{cls_name}: {conf:.2f}")
```

### Batch Inference

```python
# Run on test set
test_images = list(Path("/kaggle/input/.../test/images").glob("*.jpg"))

results = model.predict(
    source=test_images,
    conf=0.25,
    imgsz=640,
    save=True,
    save_txt=True,  # Save predictions as .txt files
    project="runs/inference",
    name="test_predictions"
)

print(f"Processed {len(results)} images")
```

### Video Inference

```python
# Process video file
results = model.predict(
    source="military_footage.mp4",
    conf=0.25,
    save=True,
    stream=True  # Memory-efficient streaming
)

for result in results:
    # Process each frame
    pass
```

---

## Project Structure

```
artillery-detection/
├── README.md                          # This file
├── .gitignore                         # Git ignore patterns
│
├── yolov11_train.ipynb               # YOLOv11 training & inference
├── rcnn_artillery.ipynb              # Faster R-CNN training
├── rcnn_eval.ipynb                   # Faster R-CNN evaluation
├── VIT_train.ipynb                   # RT-DETR (ViT) training
│
├── checkpoints/                      # Saved model weights
│   └── faster_rcnn_best.pth
│
├── yolov11_run2/                     # YOLOv11 training results
│   ├── yolov11_art_best.pt          # Best checkpoint
│   ├── results.csv                   # Training metrics
│   ├── confusion_matrix_normalized.png
│   ├── F1_curve.png
│   ├── PR_curve.png
│   └── train_batch*.jpg             # Training visualizations
│
└── rcnn-2/                           # Faster R-CNN results
    ├── results.csv
    └── train_batch2.jpg
```

---

## 💡 Tips & Troubleshooting

### Memory Issues

**Problem**: CUDA Out of Memory Error

**Solutions**:
```python
# Reduce batch size
batch=8  # Instead of 16

# Reduce image size
imgsz=512  # Instead of 640

# Use gradient accumulation (YOLOv11)
accumulate=2  # Effective batch size = batch * accumulate
```

### Multi-GPU Training

**YOLOv11** (works with 2 GPUs):
```python
device=[0, 1]  # Use both GPUs
```

**Faster R-CNN** (single GPU):
```python
# Use DataParallel for multi-GPU
model = torch.nn.DataParallel(model, device_ids=[0, 1])
```

### Checkpoint Recovery

If training interrupts:

```python
# Resume YOLOv11 training
model = YOLO("runs/yolo11/military_detection/weights/last.pt")
model.train(resume=True)

# Resume Faster R-CNN
checkpoint = torch.load("checkpoints/faster_rcnn_last.pth")
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
start_epoch = checkpoint["epoch"]
```

### Kaggle Session Timeout

**Problem**: Kaggle notebook times out after 12 hours

**Solutions**:
1. **Save checkpoints frequently**: Use `save_period=5` or `save_period=10`
2. **Enable persistence**: Turn on "Persistence" in notebook settings
3. **Export checkpoints**: Download checkpoints to local machine:
   ```python
   from IPython.display import FileLink
   FileLink("runs/yolo11/military_detection/weights/best.pt")
   ```

### Improving Performance

1. **Hyperparameter tuning**:
   ```python
   # YOLOv11 advanced parameters
   model.train(
       lr0=0.01,          # Initial learning rate
       lrf=0.01,          # Final learning rate
       momentum=0.937,
       weight_decay=0.0005,
       warmup_epochs=3,
       box=7.5,           # Box loss weight
       cls=0.5,           # Class loss weight
       augment=True       # Enable augmentations
   )
   ```

2. **Data augmentation** (already enabled by default in Ultralytics)
3. **Transfer learning**: Always start with pretrained weights
4. **Mixed precision training**: Automatically enabled on T4 GPUs

### Model Export for Deployment

```python
# Export to ONNX (GPU inference)
model.export(format="onnx", imgsz=640, simplify=True)

# to TensorRT (NVIDIA GPUs only)
model.export(format="engine", imgsz=640, half=True)

```

---

## Citation

If you use this project or dataset, please cite:

```bibtex
@misc{military-artillery-detection-2025,
  author = {Your Name},
  title = {Military Artillery Detection using YOLOv11, Faster R-CNN, and RT-DETR},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/artillery-detection}
}
```

**Dataset Citation**:
```
Military Assets Dataset (12 Classes)
Source: Kaggle - rawsi18/military-assets-dataset-12-classes-yolo8-format
```

