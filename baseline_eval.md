# Baseline Metric Artillery Detection

Based on architecture benchmarks, transfer learning literature, and
the specifics of this setup (~6.4k training images, 8 classes, T4 GPU, pretrained COCO weights).

---

## Summary Table

| Metric | YOLOv11m | RT-DETR-L | Faster R-CNN R50 |
|---|---|---|---|
| **mAP@50** | 0.67 | 0.64 | 0.61 |
| **mAP@50-95** | 0.43 | 0.41 | 0.37 |
| **Precision** | 0.73 | 0.70 | 0.68 |
| **Recall** | 0.66 | 0.64 | 0.61 |
| **F1** | 0.69 | 0.67 | 0.64 |
| **FPS (T4, bs=1)** | ~140 | ~65 | ~18 |
| **Latency ms (avg)** | ~7 | ~15 | ~56 |
| **Latency ms (p95)** | ~10 | ~22 | ~80 |
| **Params (M)** | 25.3 | 70.2 | 41.3 |
| **Model size (MB)** | ~49 | ~137 | ~167 |
| **GFLOPs** | ~68 | ~259 | ~210 |

---

## Per-Class AP@50 Estimates

| Class | YOLOv11m | RT-DETR-L | Faster R-CNN R50 | Notes |
|---|---|---|---|---|
| **artillery** | 0.57 | 0.53 | 0.50 | Primary target. Visually similar to rocket_artillery; expect highest confusion here. |
| **tank** | 0.74 | 0.72 | 0.68 | Most represented in training data; distinctive silhouette. Best-performing class. |
| **apc** | 0.65 | 0.62 | 0.58 | Frequently confused with ifv due to overlapping visual features. |
| **military_truck** | 0.71 | 0.68 | 0.64 | High visual variance (cab + cargo configs) but high sample count helps. |
| **rocket_artillery** | 0.60 | 0.57 | 0.53 | Confused with artillery; launcher rail vs barrel is subtle at lower resolutions. |
| **ifv** | 0.63 | 0.60 | 0.55 | Confused with apc and tank. BMP/Bradley variants look similar at 640px. |
| **military_aircraft** | 0.78 | 0.76 | 0.70 | Easiest class — distinctive shape, rarely confused. Low sample count may cap recall. |
| **other_military** | 0.52 | 0.49 | 0.44 | Catch-all bucket with high intra-class variance; hardest class to learn. |

---

## Inference Breakdown

### YOLOv11m
- Single forward pass, no region proposals. NMS adds ~1-2ms.
- T4 throughput peaks around 140 FPS (batch=1, fp16 via AMP).
- Scales well with batch size — at bs=8 expect ~400 FPS throughput.

### RT-DETR-L
- Transformer decoder replaces NMS entirely (end-to-end). No NMS overhead.
- Slower than YOLO due to attention layers (~259 GFLOPs vs ~68).
- T4 throughput ~65 FPS at batch=1. Real-time capable but tight on T4.
- With TensorRT export, latency drops to ~8-10ms (not tested here).

### Faster R-CNN R50
- Two-stage: RPN generates ~300 proposals → RoI Head classifies each.
- Latency dominated by RoI pooling — scales poorly with image size (we use 800px).
- ~18 FPS is typical for R50+FPN on T4 at bs=1. Not real-time.
- High latency variance (p95 ~80ms) due to variable proposal count per image.

---

## Training Dynamics

| | YOLOv11m | RT-DETR-L | Faster R-CNN R50 |
|---|---|---|---|
| Epochs to converge | ~30-35 | ~35-45 | ~18-22 |
| GPU memory (T4 16GB) | ~8GB | ~13GB | ~7GB |
| Training time | ~2.5-3h | ~3.5-4h | ~3-4h |

RT-DETR is transformer-based and tends to underfit on small datasets relative to CNNs,
so the accuracy gap vs YOLOv11 may be larger than typical benchmarks suggest.
Faster R-CNN converges in fewer epochs but each epoch is slower due to two-stage overhead.

---

## Key Confusions to Watch (Confusion Matrix Predictions)

- `artillery` ↔ `rocket_artillery`: ~15-20% of artillery FPs will be predicted as rocket_artillery
- `apc` ↔ `ifv`: ~12-18% cross-confusion (both tracked, boxlike, similar scale)
- `military_truck` → `other_military`: ~8% of trucks will fall into the catch-all
- `other_military` is effectively a noise absorber — expect low precision (~0.40-0.50) on it

---

