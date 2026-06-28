import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageFile
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate partially corrupted images

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent 
INPUT_DIR  = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "dataset_augmented"
CLASSES    = ["healthy", "Bacterial_spot", "Leaf_Mold", "Tomato_mosaic_virus"]
TARGET     = 5000          # target images per class (original + augmented)
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
SEED       = 42
# ──────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)


# ─── Augmentation functions ───────────────────────────────────────────────────

def flip_horizontal(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def flip_vertical(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)

def rotate_90(img):
    return img.rotate(90, expand=True)

def rotate_180(img):
    return img.rotate(180)

def rotate_270(img):
    return img.rotate(270, expand=True)

def random_rotation(img):
    angle = random.uniform(-30, 30)
    return img.rotate(angle, resample=Image.BILINEAR, expand=False)

def adjust_brightness(img):
    factor = random.uniform(0.6, 1.4)
    return ImageEnhance.Brightness(img).enhance(factor)

def adjust_contrast(img):
    factor = random.uniform(0.6, 1.4)
    return ImageEnhance.Contrast(img).enhance(factor)

def adjust_saturation(img):
    factor = random.uniform(0.6, 1.4)
    return ImageEnhance.Color(img).enhance(factor)

def add_gaussian_noise(img):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 15, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def random_zoom(img):
    factor = random.uniform(0.8, 1.0)
    w, h = img.size
    left   = int(w * (1 - factor) / 2)
    top    = int(h * (1 - factor) / 2)
    right  = w - left
    bottom = h - top
    return img.crop((left, top, right, bottom)).resize((w, h), Image.BILINEAR)

def apply_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))


AUGMENTATIONS = [
    flip_horizontal,
    flip_vertical,
    rotate_90,
    rotate_180,
    rotate_270,
    random_rotation,
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    add_gaussian_noise,
    random_zoom,
    apply_blur,
]


def augment_image(img):
    """Apply 1–3 random augmentations to a single image."""
    ops = random.sample(AUGMENTATIONS, k=random.randint(1, 3))
    for op in ops:
        img = op(img)
    return img


# ─── Main ─────────────────────────────────────────────────────────────────────

def get_image_paths(folder):
    return [p for p in Path(folder).iterdir() if p.suffix in IMG_EXTS]


def process_class(cls):
    src_dir = Path(INPUT_DIR) / cls
    dst_dir = Path(OUTPUT_DIR) / cls
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_paths = get_image_paths(src_dir)
    original_count = len(src_paths)

    if original_count == 0:
        print(f"  [{cls}] No images found, skipping.")
        return

    # Copy originals to output folder
    print(f"  [{cls}] Copying {original_count} original images...")
    valid_paths = []
    skipped = 0
    for p in src_paths:
        try:
            img = Image.open(p).convert("RGB")
            img.save(dst_dir / p.name)
            valid_paths.append(p)
        except Exception as e:
            print(f"    [skip] {p.name} — {e}")
            skipped += 1
    if skipped:
        print(f"  [{cls}] Skipped {skipped} corrupt/unreadable files.")
    src_paths = valid_paths

    # Generate augmented images until TARGET is reached
    needed = TARGET - original_count
    if needed <= 0:
        print(f"  [{cls}] Already at or above target ({original_count}), no augmentation needed.")
        return

    print(f"  [{cls}] Generating {needed} augmented images to reach target {TARGET}...")
    generated = 0
    while generated < needed:
        src_path = random.choice(src_paths)
        try:
            img = Image.open(src_path).convert("RGB")
            aug_img = augment_image(img)
            stem = src_path.stem
            aug_name = f"{stem}_aug{generated:05d}{src_path.suffix}"
            aug_img.save(dst_dir / aug_name)
            generated += 1
        except Exception:
            continue

    total = original_count + generated
    print(f"  [{cls}] Done — {original_count} original + {generated} augmented = {total} total")


def main():
    print("=" * 60)
    print("Image Augmentation")
    print(f"  Input  : {INPUT_DIR.relative_to(BASE_DIR.parent)}")
    print(f"  Output : {OUTPUT_DIR.relative_to(BASE_DIR.parent)}")
    print(f"  Target : {TARGET} images per class")
    print("=" * 60)

    for cls in CLASSES:
        process_class(cls)

    print("=" * 60)
    print("Augmentation complete.")
    print(f"Augmented dataset saved to: {OUTPUT_DIR}")

    # Summary
    print("\nSummary:")
    for cls in CLASSES:
        count = len(list((Path(OUTPUT_DIR) / cls).iterdir()))
        print(f"  {cls:<25} {count} images")


if __name__ == "__main__":
    main()
