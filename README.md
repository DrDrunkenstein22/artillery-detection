Prerequisites

Kaggle account
GPU runtime enabled

Quick Start
1. Create Kaggle Notebook

Go to Kaggle → New Notebook
Settings → Accelerator → Select GPU (T4 x2 recommended)
Enable internet access if needed

2. Add Datasets
Click "Add Data" in the right sidebar and attach:

Military assets dataset
YOLOv11 trained weights
Faster R-CNN trained weights

3. Install Dependencies
bash!pip install -q ultralytics torchmetrics
4. Update File Paths
Edit these lines in the script to match your dataset locations:
pythonYOLO_WEIGHTS = "/kaggle/input/your-yolo-dataset/yolov11_art_best.pt"
RCNN_WEIGHTS = "/kaggle/input/your-rcnn-dataset/faster_rcnn_best.pth"
root = "/kaggle/input/your-military-dataset/military_object_dataset"
```

**To find exact paths:** Check the Input section in the right sidebar and browse your attached datasets.

### 5. Run Evaluation

Copy the evaluation script into a code cell and run it. Results will be saved to `/kaggle/working/benchmark_results.csv`.

## Training Scripts

### Setup

1. Update dataset paths to `/kaggle/input/your-dataset/`
2. Set output paths to `/kaggle/working/`
3. Adjust batch size based on GPU memory

### After Training

1. Find saved models in Output section
2. Click "New Dataset" to save weights
3. Use the created dataset in future notebooks

## Common Issues

**Out of Memory**
- Reduce `BATCH_SIZE` to 2 or 1
- Verify GPU is enabled

**File Not Found**
- Check paths match your dataset locations exactly
- Ensure datasets are attached in Input section

**Slow Performance**
- Confirm GPU accelerator is enabled
- Check `DEVICE = "cuda"` in output

## Dataset Structure
```
military_object_dataset/
├── val/
│   ├── images/
│   └── labels/


Output
The benchmark produces:

Console output with metrics
benchmark_results.csv with mAP, FPS, and VRAM usage
