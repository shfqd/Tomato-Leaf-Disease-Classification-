import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "dataset"
IMG_SIZE  = (224, 224)
BATCH_SIZE = 32
MAX_SAMPLES = 3000

SPLITS = ["70_30", "80_20", "90_10"]
# ───────────────────────────────────────────────────────────────────────────────


def load_backbone():
    backbone = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        pooling='avg',
        weights='imagenet',
    )
    backbone.trainable = False
    return backbone


def extract_features(dataset, backbone, label):
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
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.argmax(np.concatenate([y_train, y_test], axis=0), axis=1)

    if len(X_all) > MAX_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_all), MAX_SAMPLES, replace=False)
        X_vis, y_vis = X_all[idx], y_all[idx]
    else:
        X_vis, y_vis = X_all, y_all

    print(f"  Running t-SNE on {len(X_vis)} samples...")
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
    dataset_dir  = DATA_DIR / f"dataset_split_{split}"
    features_dir = BASE_DIR / f"features_{split}"

    print(f"\n{'=' * 60}")
    print(f"Split: {split}  |  Dataset: {dataset_dir}")
    print(f"{'=' * 60}")

    if not (dataset_dir / 'train').exists():
        print(f"  SKIPPED — {dataset_dir / 'train'} not found.")
        return

    features_dir.mkdir(exist_ok=True)

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

    with open(features_dir / 'class_names.json', 'w') as f:
        json.dump(class_names, f, indent=2)

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

    print("  Generating t-SNE visualisation...")
    plot_path = BASE_DIR / f"tsne_features_{split}.png"
    save_tsne_plot(X_train, y_train, X_test, y_test, class_names, split, plot_path)


def main():
    print("=" * 60)
    print("Feature Extraction  (MobileNetV2 backbone)")
    print(f"Splits: {', '.join(SPLITS)}")
    print("=" * 60)

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
