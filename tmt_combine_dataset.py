import shutil
import sys
from pathlib import Path


def lp(p: Path) -> str:
    """Return a Windows extended-length path string to bypass the 260-char limit."""
    if sys.platform == "win32":
        return "\\\\?\\" + str(p.resolve())
    return str(p)

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent / "dataset"
CLASSES    = ["healthy", "Bacterial_spot", "Leaf_Mold", "Tomato_mosaic_virus"]
IMG_EXTS   = {".jpg", ".jpeg", ".png"}

SPLITS = [
    "dataset_split_70_30",
    "dataset_split_80_20",
    "dataset_split_90_10",
]
# ───────────────────────────────────────────────────────────────────────────────


def combine_split(split_name):
    split_dir  = BASE_DIR / split_name
    output_dir = BASE_DIR / f"{split_name}_combined"

    print(f"\n{'=' * 60}")
    print(f"Combining: {split_name}")
    print(f"Output   : {output_dir.relative_to(BASE_DIR.parent)}")
    print(f"{'=' * 60}")

    for cls in CLASSES:
        out_cls = output_dir / cls
        out_cls.mkdir(parents=True, exist_ok=True)

        copied = 0
        for subset in ("train", "test"):
            src = split_dir / cls / subset
            if not src.exists():
                print(f"  WARNING: {src} not found, skipping.")
                continue
            for img in src.iterdir():
                if img.suffix.lower() in IMG_EXTS:
                    shutil.copy(lp(img), lp(out_cls / img.name))
                    copied += 1

        print(f"  [{cls:<25}]  {copied:>5} images combined")


def main():
    print("=" * 60)
    print("Dataset Combiner  (train + test → combined)")
    print(f"Base : {BASE_DIR}")
    print("=" * 60)

    for split_name in SPLITS:
        combine_split(split_name)

    print(f"\n{'=' * 60}")
    print("Done. Combined folders:")
    for split_name in SPLITS:
        out = BASE_DIR / f"{split_name}_combined"
        print(f"  {out.name}/")
        for cls in CLASSES:
            cls_dir = out / cls
            if cls_dir.exists():
                n = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
                print(f"    {cls:<25}  {n} images")


if __name__ == "__main__":
    main()
