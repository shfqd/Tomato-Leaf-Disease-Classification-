import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / 'dataset'
FEATURES_DIR = Path(__file__).parent / 'features'
IMG_SIZE = (224, 224)   # MobileNetV2 native input size
BATCH_SIZE = 32

if not (DATASET_DIR / 'train').exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATASET_DIR / 'train'}.\n"
        "Run backend/models/preprocess.py first."
    )

FEATURES_DIR.mkdir(exist_ok=True)

# Load datasets (no shuffle so labels stay aligned)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / 'train',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / 'test',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False,
)

class_names = train_ds.class_names
print(f"Classes ({len(class_names)}): {class_names}")

# Save class names so train.py can restore them
with open(FEATURES_DIR / 'class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)

# Build feature extractor: MobileNetV2 pretrained on ImageNet, no top
backbone = keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    pooling='avg',
    weights='imagenet',
)
backbone.trainable = False

# Preprocessing wrapper expected by MobileNetV2
preprocess = keras.applications.mobilenet_v2.preprocess_input


def extract(dataset, split_name):
    features, labels = [], []
    total = sum(1 for _ in dataset)
    for i, (images, batch_labels) in enumerate(dataset):
        x = preprocess(images)
        feats = backbone(x, training=False)
        features.append(feats.numpy())
        labels.append(batch_labels.numpy())
        print(f"  [{split_name}] batch {i + 1}/{total}", end='\r')
    print()
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels


print("Extracting train features...")
X_train, y_train = extract(train_ds, 'train')

print("Extracting test features...")
X_test, y_test = extract(test_ds, 'test')

np.save(FEATURES_DIR / 'X_train.npy', X_train)
np.save(FEATURES_DIR / 'y_train.npy', y_train)
np.save(FEATURES_DIR / 'X_test.npy', X_test)
np.save(FEATURES_DIR / 'y_test.npy', y_test)

print(f"\nFeatures saved to: {FEATURES_DIR}")
print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")
print("\nDone. Run model_training/train.py next.")
