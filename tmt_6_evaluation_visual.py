import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json

history_path = Path(__file__).parent / 'training_history.json'
with open(history_path, 'r') as f:
    history = json.load(f)


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epoch')

plt.tight_layout()
plot_path = Path(__file__).parent / 'training_history2.png'
plt.savefig(str(plot_path))
print(f"Training plot saved to: {plot_path}")