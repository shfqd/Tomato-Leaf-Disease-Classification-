import random
import shutil
import sys
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
INPUT_DIR  = BASE_DIR / "tomato_data"
OUTPUT_DIR = BASE_DIR / "dataset"
CLASSES    = ["healthy", "Bacterial_spot", "Leaf_Mold", "Tomato_mosaic_virus"]
IMG_EXTS   = {".jpg", ".jpeg", ".png"}
SEED       = 42

SPLITS = {
    "dataset_split_70_30": 0.70,
    "dataset_split_80_20": 0.80,
    "dataset_split_90_10": 0.90,
}
# ───────────────────────────────────────────────────────────────────────────────


def lp(p: Path) -> str:
    """Extended-length path prefix to bypass Windows 260-char limit."""
    if sys.platform == "win32":
        return "\\\\?\\" + str(p.resolve())
    return str(p)


def get_images(folder: Path):
    return [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]


def split_and_copy(split_name, train_ratio):
    src_split  = INPUT_DIR  / split_name
    out_split  = OUTPUT_DIR / split_name
    test_ratio = round(1.0 - train_ratio, 2)
    pct_train  = int(train_ratio * 100)
    pct_test   = int(test_ratio  * 100)

    print(f"\n{'=' * 60}")
    print(f"Split {pct_train}:{pct_test}  →  dataset/{split_name}/")
    print(f"{'=' * 60}")

    total_train = total_test = 0

    for cls in CLASSES:
        src_cls   = src_split / cls
        train_cls = out_split / "train" / cls
        test_cls  = out_split / "test"  / cls
        train_cls.mkdir(parents=True, exist_ok=True)
        test_cls.mkdir(parents=True, exist_ok=True)

        images = get_images(src_cls)
        random.shuffle(images)

        split_idx   = int(len(images) * train_ratio)
        train_imgs  = images[:split_idx]
        test_imgs   = images[split_idx:]

        for p in train_imgs:
            shutil.copy(lp(p), lp(train_cls / p.name))
        for p in test_imgs:
            shutil.copy(lp(p), lp(test_cls  / p.name))

        total_train += len(train_imgs)
        total_test  += len(test_imgs)
        print(f"  [{cls:<25}]  train: {len(train_imgs):>5}   test: {len(test_imgs):>5}")

    print(f"  {'TOTAL':<27}   train: {total_train:>5}   test: {total_test:>5}")


def main():
    random.seed(SEED)

    print("=" * 60)
    print("Dataset Re-splitter")
    print(f"  Input : {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Classes: {', '.join(CLASSES)}")
    print("=" * 60)

    for split_name, train_ratio in SPLITS.items():
        split_and_copy(split_name, train_ratio)

    print(f"\n{'=' * 60}")
    print("Done. Output structure:")
    for split_name in SPLITS:
        print(f"\n  dataset/{split_name}/")
        for subset in ("train", "test"):
            for cls in CLASSES:
                cls_dir = OUTPUT_DIR / split_name / subset / cls
                n = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
                print(f"    {subset}/{cls:<25}  {n:>5} images")


if __name__ == "__main__":
    main()
