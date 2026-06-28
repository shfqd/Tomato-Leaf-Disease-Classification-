
import random
import shutil
from pathlib import Path

# ─── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "dataset_preprocessed"
CLASSES   = ["healthy", "Bacterial_spot", "Leaf_Mold", "Tomato_mosaic_virus"]
IMG_EXTS  = {".jpg", ".jpeg", ".png"}
SEED      = 42

SPLITS = {
    "dataset_split_70_30": 0.70,
    "dataset_split_80_20": 0.80,
    "dataset_split_90_10": 0.90,
}
# ───────────────────────────────────────────────────────────────────────────────


def get_image_paths(folder):
    return [p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS]


def split_class(cls, output_dir, train_ratio):
    src_dir  = INPUT_DIR / cls
    train_dir = output_dir / cls / "train"
    test_dir  = output_dir / cls / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    images = get_image_paths(src_dir)
    random.shuffle(images)

    split_idx   = int(len(images) * train_ratio)
    train_imgs  = images[:split_idx]
    test_imgs   = images[split_idx:]

    for p in train_imgs:
        shutil.copy(p, train_dir / p.name)
    for p in test_imgs:
        shutil.copy(p, test_dir / p.name)

    return len(train_imgs), len(test_imgs)


def run_split(split_name, train_ratio):
    output_dir = BASE_DIR / split_name
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
    random.seed(SEED)

    print("=" * 60)
    print("Dataset Splitting")
    print(f"  Input  : {INPUT_DIR.relative_to(BASE_DIR.parent)}")
    print(f"  Classes: {', '.join(CLASSES)}")
    print(f"  Ratios : 70:30 / 80:20 / 90:10  (train:test)")
    print("=" * 60)

    for split_name, train_ratio in SPLITS.items():
        run_split(split_name, train_ratio)

    print(f"\n{'=' * 60}")
    print("All splits complete.")
    print("\nOutput folders:")
    for split_name in SPLITS:
        print(f"  {split_name}/")
        for cls in CLASSES:
            split_dir = BASE_DIR / split_name / cls
            n_train = len(list((split_dir / "train").iterdir()))
            n_test  = len(list((split_dir / "test").iterdir()))
            print(f"    {cls:<25}  train: {n_train}   test: {n_test}")


if __name__ == "__main__":
    main()
