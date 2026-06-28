import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ── Change this for each experiment: '7030', '8020', or '9010' ───────────────
SPLIT = '8020'

# Custom callback: logs and prints learning rate each epoch
class LearningRateLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        logs['learning_rate'] = lr
        print(f"  → learning_rate: {lr:.6f}")

ROOT_DIR     = Path(__file__).resolve().parents[1]
DATASET_DIR  = ROOT_DIR / 'dataset'
FEATURES_DIR = Path(__file__).parent / f'features_{SPLIT}'
IMG_SIZE     = (256, 256)
BATCH_SIZE   = 32
NUM_CLASSES  = 10
EPOCHS       = 10

# ── Mode selection ────────────────────────────────────────────────────────────
USE_FEATURES = (FEATURES_DIR / 'X_train.npy').exists()

if USE_FEATURES:
    print(f"Split: {SPLIT}  |  Pre-extracted features found — training classifier on MobileNetV2 features.")

    X_train = np.load(FEATURES_DIR / 'X_train.npy')
    y_train = np.load(FEATURES_DIR / 'y_train.npy')
    X_test  = np.load(FEATURES_DIR / 'X_test.npy')
    y_test  = np.load(FEATURES_DIR / 'y_test.npy')

    with open(FEATURES_DIR / 'class_names.json') as f:
        class_names = json.load(f)

    print(f"Loaded  X_train {X_train.shape}  X_test {X_test.shape}")
    print(f"Classes: {class_names}")

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(NUM_CLASSES, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
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
    print(f"Split: {SPLIT}  |  No pre-extracted features found — training CNN from scratch on raw images.")
    print("Tip: run feature_extraction.py first for faster, higher-accuracy training.")

    if not (DATASET_DIR / 'train').exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_DIR / 'train'}.\n"
            "Run backend/models/preprocess.py first to download and split the dataset."
        )

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / 'train',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,
        seed=42,
    )
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / 'test',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_dataset  = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    model = keras.Sequential([
        keras.layers.Input(shape=(256, 256, 3)),
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
        keras.layers.Dense(NUM_CLASSES, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ],
    )
    model.summary()

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[LearningRateLogger()],
    )

# ── Save outputs ──────────────────────────────────────────────────────────────
history_path = Path(__file__).parent / f'training_history_{SPLIT}.json'
with open(history_path, 'w') as f:
    json.dump(history.history, f, indent=4)
print(f"Training history saved to: {history_path}")

model_path = Path(__file__).parent / f'tomato_disease_{SPLIT}.h5'
model.save(str(model_path))
print(f"Model saved to: {model_path}")

# ── Hyperparameter search ─────────────────────────────────────────────────────
if not USE_FEATURES:
    print("\nSkipping hyperparameter search — features not available (run feature_extraction.py first).")
else:
    HYPERPARAM_RESULTS = Path(__file__).parent / f'hyperparam_results_{SPLIT}.json'
    COMBOS = [
        {'learning_rate': 0.01,   'epochs': 5},
        {'learning_rate': 0.005,  'epochs': 10},
        {'learning_rate': 0.001,  'epochs': 15},
        {'learning_rate': 0.0005, 'epochs': 20},
        {'learning_rate': 0.0001, 'epochs': 25},
        {'learning_rate': 0.0001, 'epochs': 30},
        {'learning_rate': 0.0002, 'epochs': 35},
        {'learning_rate': 3e-5,   'epochs': 40},
        {'learning_rate': 5e-6,   'epochs': 50},
        {'learning_rate': 0.01,   'epochs': 50},
        {'learning_rate': 0.001,  'epochs': 100},
    ]

    print(f"\nRunning hyperparameter search ({len(COMBOS)} combinations)...")
    hp_results = []
    for combo in COMBOS:
        lr     = combo['learning_rate']
        epochs = combo['epochs']
        print(f"  lr={lr}, epochs={epochs} ...", end=' ', flush=True)

        m = keras.Sequential([
            keras.layers.Input(shape=(X_train.shape[1],)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(NUM_CLASSES, activation='softmax'),
        ])
        m.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        h = m.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  epochs=epochs, batch_size=BATCH_SIZE, verbose=0)
        val_acc = max(h.history['val_accuracy'])
        hp_results.append({'learning_rate': lr, 'epochs': epochs, 'val_accuracy': val_acc})
        print(f"val_accuracy={val_acc:.4f}")

    with open(HYPERPARAM_RESULTS, 'w') as f:
        json.dump(hp_results, f, indent=2)
    print(f"Hyperparameter results saved to: {HYPERPARAM_RESULTS}")
