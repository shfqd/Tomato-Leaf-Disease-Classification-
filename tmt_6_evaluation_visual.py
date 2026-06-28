import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
SPLITS     = ['7030', '8020', '9010']
COLORS     = {'7030': '#e74c3c', '8020': '#2ecc71', '9010': '#3498db'}
LABELS     = {'7030': '70/30', '8020': '80/20', '9010': '90/10'}

# ── Load available training histories ─────────────────────────────────────────
histories = {}
for split in SPLITS:
    p = SCRIPT_DIR / f'training_history_{split}.json'
    if p.exists():
        with open(p) as f:
            histories[split] = json.load(f)
        print(f"Loaded training_history_{split}.json")
    else:
        print(f"Skipping {split} — training_history_{split}.json not found")

if not histories:
    raise FileNotFoundError("No training history files found. Run train.py for at least one split first.")


def save(fig, filename):
    path = SCRIPT_DIR / filename
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ── 1. Training Graph (Accuracy & Loss) ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Training — Accuracy & Loss by Split', fontsize=13, fontweight='bold')

for split, h in histories.items():
    color = COLORS[split]
    label = LABELS[split]
    epochs = range(1, len(h['accuracy']) + 1)
    axes[0].plot(epochs, h['accuracy'],     color=color, linestyle='-',  label=f'{label} Train')
    axes[0].plot(epochs, h['val_accuracy'], color=color, linestyle='--', label=f'{label} Val')
    axes[1].plot(epochs, h['loss'],         color=color, linestyle='-',  label=f'{label} Train')
    axes[1].plot(epochs, h['val_loss'],     color=color, linestyle='--', label=f'{label} Val')

axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(fontsize=7)

axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(fontsize=7)

plt.tight_layout()
save(fig, 'training_history_comparison.png')

# ── 2. Recall Graph ───────────────────────────────────────────────────────────
if all('recall' in h for h in histories.values()):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle('Model Recall by Split', fontsize=13, fontweight='bold')

    for split, h in histories.items():
        color = COLORS[split]
        label = LABELS[split]
        epochs = range(1, len(h['recall']) + 1)
        ax.plot(epochs, h['recall'],     color=color, linestyle='-',  label=f'{label} Train')
        ax.plot(epochs, h['val_recall'], color=color, linestyle='--', label=f'{label} Val')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.legend(fontsize=7)
    plt.tight_layout()
    save(fig, 'recall_comparison.png')
else:
    print("Skipping recall graph — re-run train.py with updated metrics first.")

# ── 3. Precision Graph ────────────────────────────────────────────────────────
if all('precision' in h for h in histories.values()):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle('Model Precision by Split', fontsize=13, fontweight='bold')

    for split, h in histories.items():
        color = COLORS[split]
        label = LABELS[split]
        epochs = range(1, len(h['precision']) + 1)
        ax.plot(epochs, h['precision'],     color=color, linestyle='-',  label=f'{label} Train')
        ax.plot(epochs, h['val_precision'], color=color, linestyle='--', label=f'{label} Val')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.legend(fontsize=7)
    plt.tight_layout()
    save(fig, 'precision_comparison.png')
else:
    print("Skipping precision graph — re-run train.py with updated metrics first.")

# ── 4. Hyperparameter Bar Chart ───────────────────────────────────────────────
hyperparam_data = {}
for split in SPLITS:
    p = SCRIPT_DIR / f'hyperparam_results_{split}.json'
    if p.exists():
        with open(p) as f:
            hyperparam_data[split] = json.load(f)
        print(f"Loaded hyperparam_results_{split}.json")

if hyperparam_data:
    # Use combo labels from the first available split
    first = next(iter(hyperparam_data.values()))
    labels = [f"{{'lr': {r['learning_rate']}, 'ep': {r['epochs']}}}" for r in first]
    x = np.arange(len(labels))
    width = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle('Validation Accuracy for Hyperparameter Combinations', fontsize=13, fontweight='bold')

    available_splits = list(hyperparam_data.keys())
    for i, split in enumerate(available_splits):
        results  = hyperparam_data[split]
        val_accs = [r['val_accuracy'] for r in results]
        offset   = offsets[i] if i < len(offsets) else 0
        ax.bar(x + offset, val_accs, width, color=COLORS[split],
               alpha=0.85, label=LABELS[split])

    ax.set_ylabel('Validation Accuracy')
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.legend(title='Split')
    plt.tight_layout()
    save(fig, 'hyperparam_comparison.png')
else:
    print("Skipping hyperparameter chart — run hyperparam_search.py for at least one split first.")
