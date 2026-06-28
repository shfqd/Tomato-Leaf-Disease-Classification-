import numpy as np
from PIL import Image, ImageFilter, ImageFile
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
INPUT_DIR  = BASE_DIR / "dataset" / "dataset_augmented"
OUTPUT_DIR = BASE_DIR / "dataset" / "dataset_preprocessed"
CLASSES    = ["healthy", "Bacterial_spot", "Leaf_Mold", "Tomato_mosaic_virus"]
IMG_SIZE   = (224, 224)
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
SAVE_EXT   = ".jpg"
GAMMA      = 1.2
PAD_COLOR  = (114, 114, 114)  
HIST_CLIP_LIMIT = 2.0          
# ───────────────────────────────────────────────────────────────────────────────

# ─── Step 1: RGB Conversion ────────────────────────────────────────────────────
def ensure_rgb(img):
    return img.convert("RGB")


# ─── Step 3: Pre-Denoise (mild Gaussian blur before equalization) ───────────────
def apply_pre_denoise(img):
    return img.filter(ImageFilter.GaussianBlur(radius=0.5))


# ─── Step 4: Contrast-Limited Histogram Equalization ───────────────────────────
def _equalize_channel(arr, clip_limit=HIST_CLIP_LIMIT):
    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 256))
    clip_val = max(1, int(clip_limit * arr.size / 256))
    excess = int(np.sum(np.clip(hist - clip_val, 0, None)))
    hist = np.clip(hist, 0, clip_val)
    hist += excess // 256                        # redistribute clipped pixels evenly
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    lut = np.round((cdf - cdf_min) / (arr.size - cdf_min) * 255).clip(0, 255).astype(np.uint8)
    return lut[arr]


def apply_histogram_equalization(img):
    ycbcr = np.array(img.convert("YCbCr"))
    ycbcr[:, :, 0] = _equalize_channel(ycbcr[:, :, 0])
    return Image.fromarray(ycbcr, "YCbCr").convert("RGB")


# ─── Step 4: Gamma Correction ──────────────────────────────────────────────────
def apply_gamma_correction(img, gamma=GAMMA):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.power(arr, 1.0 / gamma)
    return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))


# ─── Step 7: Unsharp Masking ─────────────────────────────────────────────
def apply_unsharp_mask(img):
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=1))


# ─── Normalization Stats ────────────────────────────────────────────────
def normalize_to_float(img):
    return np.array(img).astype(np.float32) / 255.0


# ─── Step 2 : Aspect-Ratio-Preserving Resize with Letterboxing ─────────────
def resize_with_letterbox(img):
    w, h = img.size
    scale = min(IMG_SIZE[0] / w, IMG_SIZE[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", IMG_SIZE, PAD_COLOR)
    canvas.paste(img, ((IMG_SIZE[0] - new_w) // 2, (IMG_SIZE[1] - new_h) // 2))
    return canvas


# ─── Step 5: Auto White Balance — Gray World Assumption ───────────────────
def apply_white_balance(img):
    arr = np.array(img).astype(np.float32)
    for c in range(3):
        mean_c = arr[:, :, c].mean()
        if mean_c > 0:
            arr[:, :, c] = arr[:, :, c] * (128.0 / mean_c)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def preprocess(img):
    img = ensure_rgb(img)
    img = resize_with_letterbox(img)
    img = apply_pre_denoise(img)
    img = apply_histogram_equalization(img)
    img = apply_gamma_correction(img)
    img = apply_white_balance(img)
    img = apply_unsharp_mask(img)
    return img


# ─── Helpers ───────────────────────────────────────────────────────────────────
def get_image_paths(folder):
    return [p for p in Path(folder).iterdir() if p.suffix in IMG_EXTS]


def process_class(cls):
    src_dir = Path(INPUT_DIR) / cls
    dst_dir = Path(OUTPUT_DIR) / cls
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_paths = get_image_paths(src_dir)
    total = len(src_paths)
    print(f"  [{cls}] Processing {total} images...")

    processed  = 0
    skipped    = 0
    used_names = set()  

    for i, p in enumerate(src_paths, 1):
        try:
            img = preprocess(Image.open(p))

            out_stem = p.stem
            if out_stem in used_names:
                counter = 1
                while f"{out_stem}_{counter}" in used_names:
                    counter += 1
                out_stem = f"{out_stem}_{counter}"
            used_names.add(out_stem)

            img.save(dst_dir / (out_stem + SAVE_EXT), "JPEG", quality=95)
            processed += 1
            if i % 1000 == 0:
                print(f"    ... {i}/{total}")
        except Exception as e:
            print(f"    [skip] {p.name} — {e}")
            skipped += 1

    print(f"  [{cls}] Done — {processed} processed, {skipped} skipped.")
    return processed


# ─── Normalization Statistics ──────────────────────────────────────────────────
def compute_normalization_stats():
    print("\n  Computing normalization statistics...")
    ch_sum    = np.zeros(3, dtype=np.float64)
    ch_sum_sq = np.zeros(3, dtype=np.float64)
    n_pixels  = 0

    for cls in CLASSES:
        for p in (Path(OUTPUT_DIR) / cls).iterdir():
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                arr = normalize_to_float(Image.open(p).convert("RGB"))
                ch_sum    += arr.sum(axis=(0, 1))
                ch_sum_sq += (arr ** 2).sum(axis=(0, 1))
                n_pixels  += arr.shape[0] * arr.shape[1]
            except Exception:
                continue

    mean = ch_sum / n_pixels
    std  = np.sqrt(ch_sum_sq / n_pixels - mean ** 2)
    return mean, std


def save_normalization_stats(mean, std):
    stats_path = Path(OUTPUT_DIR) / "normalization_stats.txt"
    with open(stats_path, "w") as f:
        f.write("Normalization Statistics\n")
        f.write("=" * 50 + "\n")
        f.write("Computed from the full preprocessed dataset.\n")
        f.write("Use these values to standardize inputs during model training.\n\n")
        f.write(f"Mean  (R, G, B) : {mean[0]:.6f},  {mean[1]:.6f},  {mean[2]:.6f}\n")
        f.write(f"Std   (R, G, B) : {std[0]:.6f},  {std[1]:.6f},  {std[2]:.6f}\n\n")
        f.write("Usage in PyTorch:\n")
        f.write("  transforms.Normalize(\n")
        f.write(f"      mean=[{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}],\n")
        f.write(f"      std= [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]\n")
        f.write("  )\n\n")
        f.write("Usage in Keras:\n")
        f.write("  (pixel / 255.0 - mean) / std  — apply manually in preprocessing layer\n")
    print(f"  Stats saved → {stats_path.name}")


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Image Preprocessing")
    print(f"  Input    : {INPUT_DIR.relative_to(BASE_DIR.parent)}")
    print(f"  Output   : {OUTPUT_DIR.relative_to(BASE_DIR.parent)}")
    print(f"  Size     : {IMG_SIZE[0]} x {IMG_SIZE[1]} px")
    print(f"  Libraries: Pillow + NumPy only (no OpenCV needed)")
    print("=" * 60)

    total = 0
    for cls in CLASSES:
        total += process_class(cls)

    print("=" * 60)
    print(f"Preprocessing complete — {total} images saved to: dataset_preprocessed/")

    mean, std = compute_normalization_stats()
    save_normalization_stats(mean, std)

    print(f"\n  Mean (R,G,B) : {mean[0]:.4f},  {mean[1]:.4f},  {mean[2]:.4f}")
    print(f"  Std  (R,G,B) : {std[0]:.4f},  {std[1]:.4f},  {std[2]:.4f}")

    print("\nSummary:")
    for cls in CLASSES:
        count = len([p for p in (Path(OUTPUT_DIR) / cls).iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        print(f"  {cls:<25} {count} images")


if __name__ == "__main__":
    main()
