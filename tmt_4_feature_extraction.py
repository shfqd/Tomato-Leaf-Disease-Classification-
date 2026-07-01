# ══════════════════════════════════════════════════════════════════════════════
# tmt_4_feature_extraction.py — Transfer Learning Feature Extraction
# ──────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Uses a pre-trained MobileNetV2 network (ImageNet weights) as a frozen
#   feature extractor (backbone) to convert raw tomato leaf images into
#   compact 1280-dimensional feature vectors.  These vectors are then saved
#   to disk (.npy files) so that tmt_5 can train a lightweight classifier
#   on top of them — much faster than end-to-end CNN training.
#
#   The approach is called "transfer learning":
#     • MobileNetV2 has already learned general image features on ImageNet.
#     • We freeze its weights and use it purely as a feature extractor.
#     • A small Dense head (built in tmt_5) learns the tomato-specific task.
#
#   A t-SNE 2-D scatter plot is also generated to visualise how well the
#   backbone separates the four disease classes in feature space.
#
# INPUT : dataset/dataset_split_<ratio>/train|test/<class>/*.jpg|png
# OUTPUT: features_<ratio>/X_train.npy, y_train.npy, X_test.npy, y_test.npy
#         features_<ratio>/class_names.json
#         tsne_features_<ratio>.png
#
# RUN   : python tmt_4_feature_extraction.py   (run AFTER tmt_3)
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for scripts without a display
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from sklearn.manifold import TSNE
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "dataset"
IMG_SIZE  = (224, 224)    # MobileNetV2 expects 224×224 RGB input
BATCH_SIZE = 32
MAX_SAMPLES = 3000        # cap for t-SNE (too many points slows it drastically)

SPLITS = ["70_30", "80_20", "90_10"]
IMG_EXTS     = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
TF_SAFE_FMTS = {'JPEG', 'PNG', 'GIF', 'BMP'}   # formats natively supported by TensorFlow
# ───────────────────────────────────────────────────────────────────────────────


def _wp(p):
    """Return \\?\ prefixed string on Windows to bypass the 260-char path limit."""
    if sys.platform == 'win32':
        return '\\\\?\\' + str(Path(p).resolve())
    return str(p)


def clean_invalid_images(root: Path):
    """
    Walk root, validate every image with PIL.
    Files in a TF-unsupported format (e.g. WebP) are converted to JPEG in-place.
    Truly unreadable files are deleted.

    This guard is necessary because TensorFlow's image decoder crashes on
    unexpected formats, which would silently abort the extraction loop.
    """
    converted = removed = ok = 0
    for dirpath, _, filenames in os.walk(_wp(root)):
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext not in IMG_EXTS:
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with Image.open(fpath) as img:
                    img.load()  # force full decode — catches hidden corruption
                    # Convert if: format is TF-unsupported, OR a .jpg file holds non-JPEG data
                    needs_convert = (img.format not in TF_SAFE_FMTS) or \
                                    (ext in {'.jpg', '.jpeg'} and img.format != 'JPEG')
                    if needs_convert:
                        img.convert('RGB').save(fpath, format='JPEG', quality=95)
                        converted += 1
                    else:
                        ok += 1
            except Exception:
                try:
                    os.remove(fpath)
                except Exception:
                    pass
                removed += 1
    if converted or removed:
        print(f"  Image cleanup: {converted} converted, {removed} removed, {ok} ok")
    return converted + removed


def load_backbone():
    """
    Load MobileNetV2 with ImageNet weights as a frozen feature extractor.

    Key settings:
      include_top=False  — removes the ImageNet classification head; we only
                           want the convolutional feature maps.
      pooling='avg'      — applies global average pooling so the output is a
                           flat 1280-d vector per image (not a spatial grid).
      trainable=False    — freezes ALL backbone weights; only our Dense head
                           (defined in tmt_5) will be updated during training.
    """
    backbone = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        pooling='avg',
        weights='imagenet',
    )
    backbone.trainable = False
    return backbone


def extract_features(dataset, backbone, label):
    """
    Pass every batch through the frozen backbone and collect the output vectors.

    preprocess_input scales pixel values from [0, 255] to [-1, 1], which is
    the normalisation MobileNetV2 was trained with — skipping this would
    degrade feature quality significantly.

    Returns:
      features : np.ndarray of shape (N, 1280)
      labels   : np.ndarray of shape (N, num_classes)  — one-hot encoded
    """
    features, labels = [], []
    total = sum(1 for _ in dataset)
    preprocess = keras.applications.mobilenet_v2.preprocess_input
    for i, (images, batch_labels) in enumerate(dataset):
        feats = backbone(preprocess(images), training=False)
        features.append(feats.numpy())
        labels.append(batch_labels.numpy())
        print(f"  [{label}] batch {i + 1}/{total}", end='\r')
    print()
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def save_tsne_plot(X_train, y_train, X_test, y_test, class_names, split, out_path):
    """
    Reduce the 1280-d feature vectors to 2-D with t-SNE and plot a scatter.

    t-SNE (t-distributed Stochastic Neighbour Embedding) is a dimensionality
    reduction technique that preserves local structure — clusters in the 2-D
    plot reflect genuine similarity in the high-dimensional feature space.
    Well-separated clusters indicate the backbone has learned discriminative
    features for each disease class.

    Both train and test points are combined before embedding so the plot
    shows the overall distribution of the full dataset.
    If the combined size exceeds MAX_SAMPLES, a random subsample is used
    to keep t-SNE computation tractable.
    """
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.argmax(np.concatenate([y_train, y_test], axis=0), axis=1)

    # Sub-sample when the dataset is large to keep t-SNE fast
    if len(X_all) > MAX_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_all), MAX_SAMPLES, replace=False)
        X_vis, y_vis = X_all[idx], y_all[idx]
    else:
        X_vis, y_vis = X_all, y_all

    print(f"  Running t-SNE on {len(X_vis)} samples...")
    # perplexity≈30 is a good default; max_iter=1000 ensures convergence
    coords = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000).fit_transform(X_vis)

    colors = plt.cm.get_cmap('tab10', len(class_names))
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, name in enumerate(class_names):
        mask = y_vis == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=colors(i), label=name, alpha=0.6, s=15)

    ax.set_title(f't-SNE of Extracted Features (MobileNetV2) — Split {split}')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.legend(title='Classes', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  t-SNE plot saved → {out_path.name}")


def process_split(split, backbone):
    """
    Full extraction pipeline for one train/test split ratio.

    Steps:
      1. Validate and repair images in the split folder.
      2. Load images using tf.keras image_dataset_from_directory (auto-labels
         by sub-folder name → categorical one-hot encoding).
      3. Extract 1280-d MobileNetV2 feature vectors for every image.
      4. Save feature arrays (.npy) and class name mapping (.json) to disk.
      5. Generate a t-SNE scatter plot of the feature space.
    """
    dataset_dir  = DATA_DIR / f"dataset_split_{split}"
    features_dir = BASE_DIR / f"features_{split}"

    print(f"\n{'=' * 60}")
    print(f"Split: {split}  |  Dataset: {dataset_dir}")
    print(f"{'=' * 60}")

    if not (dataset_dir / 'train').exists():
        print(f"  SKIPPED — {dataset_dir / 'train'} not found.")
        return

    features_dir.mkdir(exist_ok=True)

    # Step 1: Clean invalid/unsupported images before feeding to TensorFlow
    print("  Scanning for invalid/unsupported images...")
    n_fixed = clean_invalid_images(dataset_dir)
    if n_fixed == 0:
        print("  All images OK.")

    # Step 2: Create TF datasets — shuffle=False keeps label order stable
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / 'train',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / 'test',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"  Classes ({len(class_names)}): {class_names}")

    # Save class names so tmt_5 and tmt_6 can recover the label mapping
    with open(features_dir / 'class_names.json', 'w') as f:
        json.dump(class_names, f, indent=2)

    # Steps 3–4: Extract and save features
    print("  Extracting train features...")
    X_train, y_train = extract_features(train_ds, backbone, 'train')

    print("  Extracting test features...")
    X_test, y_test = extract_features(test_ds, backbone, 'test')

    np.save(features_dir / 'X_train.npy', X_train)
    np.save(features_dir / 'y_train.npy', y_train)
    np.save(features_dir / 'X_test.npy',  X_test)
    np.save(features_dir / 'y_test.npy',  y_test)

    print(f"  Features saved → {features_dir}")
    print(f"    X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"    X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    # Step 5: Visualise the feature space
    print("  Generating t-SNE visualisation...")
    plot_path = BASE_DIR / f"tsne_features_{split}.png"
    save_tsne_plot(X_train, y_train, X_test, y_test, class_names, split, plot_path)


def main():
    print("=" * 60)
    print("Feature Extraction  (MobileNetV2 backbone)")
    print(f"Splits: {', '.join(SPLITS)}")
    print("=" * 60)

    # Load the backbone once and reuse it for all three splits to avoid
    # redundant weight downloads and GPU memory churn.
    print("\nLoading MobileNetV2 backbone (once for all splits)...")
    backbone = load_backbone()

    for split in SPLITS:
        process_split(split, backbone)

    print(f"\n{'=' * 60}")
    print("All splits complete.")
    print("\nOutputs per split:")
    for split in SPLITS:
        features_dir = BASE_DIR / f"features_{split}"
        plot_path    = BASE_DIR / f"tsne_features_{split}.png"
        print(f"  {split}:")
        print(f"    Features → {features_dir}")
        print(f"    t-SNE    → {plot_path.name}")


if __name__ == "__main__":
    main()
