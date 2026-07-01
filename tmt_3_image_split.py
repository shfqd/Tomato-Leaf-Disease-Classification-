
# ══════════════════════════════════════════════════════════════════════════════
# tmt_3_image_split.py — Dataset Splitting
# ──────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Reads the raw tomato leaf disease images downloaded from Kaggle and
#   produces three independent train/test splits (70:30, 80:20, 90:10).
#   Each split is saved under dataset/dataset_split_<ratio>/ so that
#   tmt_4 (feature extraction) and tmt_5 (model training) can train and
#   evaluate the classifier under different data distribution conditions.
#
#   No image preprocessing is applied here — the images are copied as-is
#   from the Kaggle dataset folder into the split directories.
#
# INPUT : dataset_raw/<class>/*.jpg|jpeg|png
#           (the Kaggle dataset folder, one sub-folder per disease class)
# OUTPUT: dataset/dataset_split_70_30/train|test/<class>/...
#         dataset/dataset_split_80_20/train|test/<class>/...
#         dataset/dataset_split_90_10/train|test/<class>/...
#
# RUN   : python tmt_3_image_split.py
# ══════════════════════════════════════════════════════════════════════════════

import random
import shutil
import sys
from pathlib import Path

# ─── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
INPUT_DIR  = BASE_DIR / "dataset_raw"   # Kaggle dataset folder; one sub-folder per class
OUTPUT_DIR = BASE_DIR / "dataset"
CLASSES    = ["healthy", "Bacterial_spot", "Leaf_Mold", "Tomato_mosaic_virus"]
IMG_EXTS   = {".jpg", ".jpeg", ".png"}
SEED       = 42   # fixed seed so every run produces the same shuffle


def lp(p: Path) -> str:
    """Extended-length path to bypass Windows 260-char limit."""
    if sys.platform == "win32":
        return "\\\\?\\" + str(p.resolve())
    return str(p)

# SPLITS defines the three experimental conditions:
#   key   → folder name that will be created under OUTPUT_DIR
#   value → fraction of images assigned to the training set
SPLITS = {
    "dataset_split_70_30": 0.70,
    "dataset_split_80_20": 0.80,
    "dataset_split_90_10": 0.90,
}
# ───────────────────────────────────────────────────────────────────────────────


def get_image_paths(folder):
    """Return a list of all image file paths inside folder (non-recursive)."""
    return [p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS]


def split_class(cls, output_dir, train_ratio):
    """
    Copy images for one class into train/ and test/ sub-directories.

    Steps:
      1. Collect all images for the class from the Kaggle dataset folder.
      2. Shuffle randomly (seeded in main so it is reproducible).
      3. Slice at train_ratio to determine the split boundary.
      4. Copy each file to its destination folder.

    Returns (n_train, n_test) — the number of images copied to each split.
    """
    src_dir  = INPUT_DIR / cls
    train_dir = output_dir / "train" / cls
    test_dir  = output_dir / "test" / cls
    # Create destination directories; exist_ok avoids errors on re-runs
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    images = get_image_paths(src_dir)
    random.shuffle(images)   # in-place shuffle before slicing

    # Compute the index that separates train from test images
    split_idx   = int(len(images) * train_ratio)
    train_imgs  = images[:split_idx]
    test_imgs   = images[split_idx:]

    for p in train_imgs:
        shutil.copy(lp(p), lp(train_dir / p.name))
    for p in test_imgs:
        shutil.copy(lp(p), lp(test_dir  / p.name))

    return len(train_imgs), len(test_imgs)


def run_split(split_name, train_ratio):
    """
    Orchestrate the full split for one train/test ratio.

    Iterates over every class, calls split_class, and prints a summary
    table showing how many images went into train vs. test per class.
    """
    output_dir = OUTPUT_DIR / split_name
    test_ratio = round(1.0 - train_ratio, 2)
    pct_train  = int(train_ratio * 100)
    pct_test   = int(test_ratio  * 100)

    print(f"\n{'=' * 60}")
    print(f"Split: {pct_train}:{pct_test}  →  {split_name}/")
    print(f"{'=' * 60}")

    total_train = 0
    total_test  = 0
    for cls in CLASSES:
        n_train, n_test = split_class(cls, output_dir, train_ratio)
        total_train += n_train
        total_test  += n_test
        print(f"  [{cls:<25}]  train: {n_train:>5}   test: {n_test:>5}")

    print(f"  {'TOTAL':<27}   train: {total_train:>5}   test: {total_test:>5}")


def main():
    # Seed the random number generator ONCE before any shuffling so all
    # classes within a split are shuffled consistently across runs.
    random.seed(SEED)

    print("=" * 60)
    print("Dataset Splitting")
    print(f"  Input  : {INPUT_DIR}")
    print(f"  Classes: {', '.join(CLASSES)}")
    print(f"  Ratios : 70:30 / 80:20 / 90:10  (train:test)")
    print("=" * 60)

    # Run all three splits sequentially
    for split_name, train_ratio in SPLITS.items():
        run_split(split_name, train_ratio)

    # Final verification — count files in the output folders to confirm copy
    print(f"\n{'=' * 60}")
    print("All splits complete.")
    print("\nOutput folders:")
    for split_name in SPLITS:
        print(f"  {split_name}/")
        for cls in CLASSES:
            n_train = len(list((OUTPUT_DIR / split_name / "train" / cls).iterdir()))
            n_test  = len(list((OUTPUT_DIR / split_name / "test"  / cls).iterdir()))
            print(f"    {cls:<25}  train: {n_train}   test: {n_test}")


if __name__ == "__main__":
    main()
