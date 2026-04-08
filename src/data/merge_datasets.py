"""
src/data/merge_datasets.py

Merges Kaggle + Roboflow mil-det datasets into a single YOLO-format dataset
with unified class taxonomy and a stratified 80/10/10 train/val/test split.
"""

import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

console = Console()

CANONICAL_CLASSES = [
    "artillery",         # 0
    "tank",              # 1
    "apc",               # 2
    "military_truck",    # 3
    "rocket_artillery",  # 4
    "ifv",               # 5
    "military_aircraft", # 6
    "other_military",    # 7
]

CLASS2IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

KAGGLE_REMAP = {
    "artillery":          "artillery",
    "artillery-gun":      "artillery",
    "artillery_gun":      "artillery",
    "rocket-artillery":   "rocket_artillery",
    "rocket_artillery":   "rocket_artillery",
    "tank":               "tank",
    "apc":                "apc",
    "army-truck":         "military_truck",
    "military-truck":     "military_truck",
    "military_truck":     "military_truck",
    "command-vehicle":    "other_military",
    "engineer-vehicle":   "other_military",
    "hummer":             "other_military",
    "military-vehicle":   "other_military",
    "soldier":            None,
    "person":             None,
    "bmp":                "ifv",
    "ifv":                "ifv",
    "military-aircraft":  "military_aircraft",
    "military_aircraft":  "military_aircraft",
    "helicopter":         "military_aircraft",
}

MIL_DET_REMAP = {
    "artillery":                "artillery",
    "armored-fighting-vehicle": "ifv",
    "heavy vehicle":            "other_military",
    "light vehicle":            "other_military",
    "military-truck":           "military_truck",
    "mlrs":                     "rocket_artillery",
    "panzer":                   "tank",
    "trench":                   None,
}


def load_class_map(dataset_root: Path) -> dict[int, str]:
    for p in sorted(dataset_root.glob("*.yaml")):
        cfg = yaml.safe_load(p.read_text())
        names = cfg.get("names", [])
        if not names:
            continue
        if isinstance(names, list):
            return {i: n for i, n in enumerate(names)}
        if isinstance(names, dict):
            return names
    return {0: "unknown"}


def find_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for split in ["train", "val", "valid", "test", "images"]:
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        if not img_dir.exists():
            img_dir = dataset_root / "images" / split
            lbl_dir = dataset_root / "labels" / split
        if not img_dir.exists():
            continue
        for img in img_dir.glob("*.*"):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs


def remap_labels(
    lbl_path: Path,
    class_map: dict[int, str],
    remap: dict[str, str | None],
) -> list[str] | None:
    new_lines = []
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        src_name = class_map.get(int(parts[0]), "unknown").lower().replace(" ", "_")
        canonical = remap.get(src_name)
        if canonical is None:
            for k, v in remap.items():
                if k in src_name or src_name in k:
                    canonical = v
                    break
        if canonical is None:
            continue
        new_lines.append(f"{CLASS2IDX[canonical]} " + " ".join(parts[1:]))
    return new_lines or None


def collect_samples(dataset_root: Path, remap: dict, name: str) -> list[dict]:
    class_map = load_class_map(dataset_root)
    pairs = find_pairs(dataset_root)
    samples = []
    for img, lbl in pairs:
        lines = remap_labels(lbl, class_map, remap)
        if lines is None:
            continue
        dominant = Counter(int(l.split()[0]) for l in lines).most_common(1)[0][0]
        samples.append({"img": img, "labels": lines, "dominant_class": dominant, "source": name})
    print(f"  {name}: {len(samples)} samples after remap (from {len(pairs)} pairs)")
    return samples


def write_split(samples: list[dict], split_name: str, out_root: Path):
    img_out = out_root / split_name / "images"
    lbl_out = out_root / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(samples):
        stem = f"{s['source']}_{i:06d}"
        shutil.copy2(s["img"], img_out / (stem + s["img"].suffix))
        (lbl_out / (stem + ".txt")).write_text("\n".join(s["labels"]))


def merge_datasets(
    kaggle_root: Path = Path("data/raw/kaggle_military_assets/military_object_dataset"),
    mil_det_root: Path = Path("data/raw/roboflow_mil_det"),
    splits_root: Path = Path("data/splits"),
    seed: int = 42,
):
    random.seed(seed)
    print("Collecting samples...")

    all_samples = []
    for root, remap, name in [
        (kaggle_root, KAGGLE_REMAP, "kaggle"),
        (mil_det_root, MIL_DET_REMAP, "mil_det"),
    ]:
        if root.exists():
            all_samples += collect_samples(root, remap, name)
        else:
            print(f"  [skip] {name} not found at {root}")

    if not all_samples:
        print("No samples found. Run download_datasets.py first.")
        return

    print(f"Total: {len(all_samples)} samples")

    # Stratified 80/10/10 split
    by_class: dict[int, list] = defaultdict(list)
    for s in all_samples:
        by_class[s["dominant_class"]].append(s)

    train_s, val_s, test_s = [], [], []
    for cls_samples in by_class.values():
        random.shuffle(cls_samples)
        n = len(cls_samples)
        n_test = max(1, int(n * 0.10))
        n_val = max(1, int(n * 0.10))
        test_s += cls_samples[:n_test]
        val_s += cls_samples[n_test:n_test + n_val]
        train_s += cls_samples[n_test + n_val:]

    print(f"Split -> train: {len(train_s)}, val: {len(val_s)}, test: {len(test_s)}")

    splits_root.mkdir(parents=True, exist_ok=True)
    for split, samples in [("train", train_s), ("val", val_s), ("test", test_s)]:
        write_split(samples, split, splits_root)

    yaml_path = splits_root / "dataset.yaml"
    yaml_path.write_text(yaml.dump({
        "path": str(splits_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CANONICAL_CLASSES),
        "names": CANONICAL_CLASSES,
    }, default_flow_style=False))

    t = Table("Class", "Train", "Val", "Test")
    for i, cls_name in enumerate(CANONICAL_CLASSES):
        t.add_row(
            cls_name,
            str(sum(1 for s in train_s if s["dominant_class"] == i)),
            str(sum(1 for s in val_s if s["dominant_class"] == i)),
            str(sum(1 for s in test_s if s["dominant_class"] == i)),
        )
    console.print(t)
    print(f"Dataset ready at {splits_root}")
    return yaml_path


if __name__ == "__main__":
    merge_datasets()
