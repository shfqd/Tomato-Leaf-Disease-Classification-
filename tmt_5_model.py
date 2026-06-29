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

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INITIAL_LR,
    decay_steps=DECAY_STEPS,
    decay_rate=DECAY_RATE,
    staircase=False,   # continuous decay (not step-wise)
)


class LearningRateLogger(keras.callbacks.Callback):
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
    return keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax'),
    ])


def train_split(split):
    features_dir = BASE_DIR / f"features_{split}"
    dataset_dir  = DATA_DIR / f"dataset_split_{split}"

    print(f"\n{'=' * 60}")
    print(f"Training  Split: {split}")
    print(f"{'=' * 60}")

    # ── Load pre-extracted features if available ──────────────────────────────
    use_features = (features_dir / 'X_train.npy').exists()

    if use_features:
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
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        model = keras.Sequential([
            keras.layers.Input(shape=(*IMG_SIZE, 3)),
            keras.layers.Rescaling(1.0 / 255),
            keras.layers.RandomFlip('horizontal'),
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
    history_path = BASE_DIR / f"training_history_{split}.json"
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=4)
    print(f"\n  History saved → {history_path.name}")

    model_path = BASE_DIR / f"tomato_disease_{split}.h5"
    model.save(str(model_path))
    print(f"  Model  saved → {model_path.name}")

    # ── Extract checkpoint results from training history ──────────────────────
    val_acc_hist = history.history.get('val_accuracy', [])
    lr_hist      = history.history.get('learning_rate', [])

    hp_results = []
    for ep in CHECKPOINTS:
        idx = ep - 1   # history is 0-based
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

    for split in SPLITS:
        train_split(split)

    print(f"\n{'=' * 60}")
    print("All splits trained.")
    print("Run tmt_6_evaluation_visual.py next to generate graphs.")


if __name__ == "__main__":
    main()
