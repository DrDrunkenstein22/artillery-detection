"""
src/utils/viz.py

Bounding box visualization utilities.
"""

from pathlib import Path

import cv2
import numpy as np

CANONICAL_CLASSES = [
    "artillery", "tank", "apc", "military_truck",
    "rocket_artillery", "ifv", "military_aircraft", "other_military",
]

COLORS = [
    (255, 50, 50),    # artillery — red
    (50, 150, 255),   # tank — blue
    (50, 220, 50),    # apc — green
    (255, 200, 50),   # military_truck — yellow
    (200, 50, 255),   # rocket_artillery — purple
    (50, 220, 220),   # ifv — cyan
    (255, 128, 0),    # military_aircraft — orange
    (180, 180, 180),  # other_military — gray
]


def draw_yolo_labels(img_path: Path, lbl_path: Path, out_path: Path | None = None):
    """Draw YOLO bounding boxes on an image."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not load {img_path}")
        return
    h, w = img.shape[:2]

    if lbl_path.exists():
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_idx = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            color = COLORS[cls_idx % len(COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = CANONICAL_CLASSES[cls_idx] if cls_idx < len(CANONICAL_CLASSES) else str(cls_idx)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)

    if out_path:
        cv2.imwrite(str(out_path), img)
    return img
