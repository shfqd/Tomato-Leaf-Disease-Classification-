import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / 'dataset'
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 10

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

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / 'val',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False,
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=1)

import json
history_path = Path(__file__).parent / 'training_history.json'
with open(history_path, 'w') as f:
    json.dump(history.history, f, indent=4)
print(f"Training history saved to: {history_path}")

model_path = Path(__file__).parent / 'tomato_disease.h5'
model.save(str(model_path))
print(f"Model saved to: {model_path}")

