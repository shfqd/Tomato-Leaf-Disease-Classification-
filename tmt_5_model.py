# ══════════════════════════════════════════════════════════════════════════════
# tmt_5_model.py — Classifier Training
# ──────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Trains a classifier on top of the MobileNetV2 features extracted by tmt_4.
#   Two modes are supported automatically:
#     • Feature mode (preferred): loads X_train.npy / X_test.npy from
#       features_<split>/ and trains a small Dense + Dropout network.
#       Training is fast (~seconds/epoch) because the backbone is pre-computed.
#     • Raw-image mode (fallback): if feature files are missing, trains a
#       lightweight custom CNN directly on the image pixels.
#
#   All three train/test splits (70:30, 80:20, 90:10) are trained sequentially.
#   After training, three artefacts are saved per split:
#     • tomato_disease_<split>.h5      -- the trained Keras model
#     • training_history_<split>.json  -- per-epoch metrics (used by tmt_6)
#     • hyperparam_results_<split>.json -- accuracy snapshot at epochs 10/50/100
#
# RUN   : python tmt_5_model.py   (run AFTER tmt_4)
# ══════════════════════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "dataset"
BATCH_SIZE = 32
EPOCHS     = 100
IMG_SIZE   = (224, 224)

SPLITS = ["70_30", "80_20", "90_10"]

CHECKPOINTS  = [10, 50, 100]   # epoch numbers to record for hyperparameter comparison
INITIAL_LR   = 0.001           # starting learning rate
DECAY_STEPS  = 1000            # number of optimizer steps over which LR decays
DECAY_RATE   = 0.96            # fraction LR is multiplied by every DECAY_STEPS steps
# ───────────────────────────────────────────────────────────────────────────────

# ExponentialDecay reduces the learning rate gradually during training.
# Formula: lr = INITIAL_LR * DECAY_RATE ^ (step / DECAY_STEPS)
# This lets the model take large steps early (fast learning) and fine-tune
# with smaller steps later (stable convergence), avoiding overshooting the minimum.
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INITIAL_LR,
    decay_steps=DECAY_STEPS,
    decay_rate=DECAY_RATE,
    staircase=False,   # continuous decay (not step-wise)
)


class LearningRateLogger(keras.callbacks.Callback):
    """
    Custom Keras callback that records the current learning rate into the
    epoch logs dictionary at the end of every epoch.

    Keras does not log the LR automatically when a schedule object is used
    (rather than a plain float), so this callback extracts the value from
    the optimizer's iteration counter and injects it as 'learning_rate'.
    tmt_6 reads this value to annotate the hyperparameter bar chart.
    """
    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        if isinstance(optimizer.learning_rate,
                      keras.optimizers.schedules.LearningRateSchedule):
            lr = float(optimizer.learning_rate(optimizer.iterations))
        else:
            lr = float(keras.backend.get_value(optimizer.learning_rate))
        if logs is not None:
            logs['learning_rate'] = lr


def build_classifier(input_dim, num_classes):
    """
    Build the Dense classification head that sits on top of MobileNetV2 features.

    Architecture:
      Input (1280-d feature vector from MobileNetV2)
        -> Dense(256, ReLU)   -- learn high-level combinations of backbone features
        -> Dropout(0.4)       -- regularisation: randomly zero 40% of units each step
        -> Dense(128, ReLU)   -- second hidden layer for finer decision boundaries
        -> Dropout(0.3)       -- lighter regularisation at this depth
        -> Dense(num_classes, Softmax)  -- output probability per disease class

    Dropout is critical here: without it the small Dense network would
    memorise the training features instead of generalising.
    """
    return keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax'),
    ])


def train_split(split):
    """
    Train and save the classifier for one train/test split ratio.

    Automatically selects the training mode based on whether pre-extracted
    feature files exist for the split.
    """
    features_dir = BASE_DIR / f"features_{split}"
    dataset_dir  = DATA_DIR / f"dataset_split_{split}"

    print(f"\n{'=' * 60}")
    print(f"Training  Split: {split}")
    print(f"{'=' * 60}")

    # ── Mode selection ────────────────────────────────────────────────────────
    # Prefer feature-based training: much faster and typically more accurate
    # because the backbone has already encoded rich visual information.
    use_features = (features_dir / 'X_train.npy').exists()

    if use_features:
        # ── Feature mode: load pre-computed MobileNetV2 embeddings ───────────
        print("  Mode: pre-extracted MobileNetV2 features")
        X_train = np.load(features_dir / 'X_train.npy')
        y_train = np.load(features_dir / 'y_train.npy')
        X_test  = np.load(features_dir / 'X_test.npy')
        y_test  = np.load(features_dir / 'y_test.npy')
        with open(features_dir / 'class_names.json') as f:
            class_names = json.load(f)
        num_classes = len(class_names)
        print(f"  X_train {X_train.shape}  X_test {X_test.shape}")
        print(f"  Classes ({num_classes}): {class_names}")

        model = build_classifier(X_train.shape[1], num_classes)
        # Adam with exponential LR decay; categorical_crossentropy is standard
        # for multi-class one-hot targets.
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
            ],
        )
        model.summary()

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            callbacks=[LearningRateLogger()],
        )

    else:
        # ── Raw-image mode: train a small CNN from scratch ────────────────────
        # This path runs when tmt_4 has not been executed yet.
        # It is slower and generally yields lower accuracy, but requires no
        # pre-extracted feature files.
        print("  Mode: CNN from raw images (run tmt_4_feature_extraction.py first for better results)")
        if not (dataset_dir / 'train').exists():
            print(f"  SKIPPED — {dataset_dir / 'train'} not found.")
            return

        train_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir / 'train',
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical',
            shuffle=True,
            seed=42,
        )
        test_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir / 'test',
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical',
            shuffle=False,
        )
        class_names = train_ds.class_names
        num_classes = len(class_names)
        AUTOTUNE = tf.data.AUTOTUNE
        # cache() keeps the dataset in memory after the first epoch;
        # prefetch() overlaps data loading with GPU computation.
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Lightweight custom CNN:
        #   - Rescaling normalises pixels to [0, 1]
        #   - Augmentation layers (flip, rotate, zoom) reduce overfitting
        #     by exposing the model to varied image orientations at train time
        #   - Three Conv->MaxPool blocks extract spatial features at increasing
        #     levels of abstraction
        #   - Flatten + Dense(64) collapses spatial maps to class scores
        model = keras.Sequential([
            keras.layers.Input(shape=(*IMG_SIZE, 3)),
            keras.layers.Rescaling(1.0 / 255),          # normalise pixels to [0, 1]
            keras.layers.RandomFlip('horizontal'),        # data augmentation
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.2),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax'),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
            ],
        )
        model.summary()

        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[LearningRateLogger()],
        )

    # ── Save history and model ────────────────────────────────────────────────
    # The full history JSON is consumed by tmt_6 to draw accuracy/loss/
    # precision/recall curves over all 100 epochs.
    history_path = BASE_DIR / f"training_history_{split}.json"
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=4)
    print(f"\n  History saved → {history_path.name}")

    # Save the trained model in HDF5 format so tmt_6 can reload it for
    # inference when generating the confusion matrix.
    model_path = BASE_DIR / f"tomato_disease_{split}.h5"
    model.save(str(model_path))
    print(f"  Model  saved → {model_path.name}")

    # ── Extract checkpoint results from training history ──────────────────────
    # Snapshot val_accuracy and learning_rate at epochs 10, 50, 100 to
    # compare how much the model improved with more training time.
    val_acc_hist = history.history.get('val_accuracy', [])
    lr_hist      = history.history.get('learning_rate', [])

    hp_results = []
    for ep in CHECKPOINTS:
        idx = ep - 1   # history list is 0-indexed; epoch 10 -> index 9
        if idx < len(val_acc_hist):
            hp_results.append({
                'epochs':        ep,
                'learning_rate': lr_hist[idx] if idx < len(lr_hist) else None,
                'val_accuracy':  val_acc_hist[idx],
            })

    hp_path = BASE_DIR / f"hyperparam_results_{split}.json"
    with open(hp_path, 'w') as f:
        json.dump(hp_results, f, indent=2)
    print(f"  Hyperparam checkpoints saved → {hp_path.name}")
    for r in hp_results:
        lr_str = f"{r['learning_rate']:.2e}" if r['learning_rate'] else 'N/A'
        print(f"    epoch {r['epochs']:>3}  lr={lr_str}  val_accuracy={r['val_accuracy']:.4f}")


def main():
    print("=" * 60)
    print("Model Training  (100 epochs, all splits)")
    print(f"Splits : {', '.join(SPLITS)}")
    print(f"Epochs : {EPOCHS}")
    print("=" * 60)

    # Train each split independently; results are saved to disk so any
    # individual split can be re-trained without losing the others.
    for split in SPLITS:
        train_split(split)

    print(f"\n{'=' * 60}")
    print("All splits trained.")
    print("Run tmt_6_evaluation_visual.py next to generate graphs.")


if __name__ == "__main__":
    main()
