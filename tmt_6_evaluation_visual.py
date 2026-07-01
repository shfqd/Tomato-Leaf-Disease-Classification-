# ══════════════════════════════════════════════════════════════════════════════
# tmt_6_evaluation_visual.py — Evaluation & Visualisation
# ──────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Reads the artefacts produced by tmt_5 (training history JSON files and
#   saved .h5 models) and generates a comprehensive set of diagnostic graphs:
#
#   Per split (70:30 / 80:20 / 90:10):
#     - Accuracy + Loss curves at epoch checkpoints 10, 50, 100
#     - Precision curve over the full 100-epoch training run
#     - Recall curve over the full 100-epoch training run
#     - Hyperparameter bar chart (val_accuracy at each checkpoint)
#     - Confusion Matrix heatmap (model predictions vs true labels)
#
#   Cross-split:
#     - Comparison bar chart: best checkpoint accuracy for each split
#
#   All graphs are saved as PNG files in the same directory as this script.
#
# RUN   : python tmt_6_evaluation_visual.py   (run AFTER tmt_5)
# ══════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use('Agg')   # non-interactive backend -- safe for scripts without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import json
from pathlib import Path

# sklearn and keras are imported inside plot_confusion_matrix() to avoid
# loading heavy dependencies when they are not needed (e.g. if models are
# missing and the function is skipped early).

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
SPLITS      = ['70_30', '80_20', '90_10']
LABELS      = {'70_30': '70/30', '80_20': '80/20', '90_10': '90/10'}
CHECKPOINTS = [10, 50, 100]   # epoch snapshots for accuracy/loss graphs

# Colour palette (train / val / best-epoch marker)
C_TRAIN = '#1565C0'   # deep blue
C_VAL   = '#C62828'   # deep red
C_BEST  = '#2E7D32'   # dark green
C_BARS  = '#1976D2'   # mid blue (hyperparam bars)
C_BEST_BAR = '#F57F17' # amber  (best hyperparam bar)

DPI = 150
# ───────────────────────────────────────────────────────────────────────────────


def _save(fig, path: Path):
    """Save figure to disk and close it to free memory."""
    fig.savefig(str(path), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {path.name}")


def _style_ax(ax, title, ylabel, n_epochs, ylim=(0.0, 1.05)):
    """Apply consistent styling to a metric axes."""
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(1, n_epochs)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
    ax.grid(which='major', linestyle='--', linewidth=0.6, alpha=0.55)
    ax.grid(which='minor', linestyle=':',  linewidth=0.4, alpha=0.30)
    ax.minorticks_on()
    ax.tick_params(axis='both', labelsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.92, edgecolor='#cccccc')


def _annotate_best(ax, epochs_arr, values, color=C_BEST):
    """Draw a vertical line and label at the epoch with the highest value."""
    best_i   = int(np.argmax(values))
    best_ep  = epochs_arr[best_i]
    best_val = values[best_i]

    ax.axvline(x=best_ep, color=color, linestyle='--', linewidth=1.2, alpha=0.75)

    x_offset = max(len(epochs_arr) // 15, 2)
    y_offset = -0.07 if best_val > 0.5 else 0.07

    ax.annotate(
        f'Best: {best_val:.4f}\n(epoch {best_ep})',
        xy=(best_ep, best_val),
        xytext=(best_ep + x_offset, best_val + y_offset),
        fontsize=8,
        color=color,
        arrowprops=dict(arrowstyle='->', color=color, lw=1.0),
        bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=color, alpha=0.88),
    )


# ── Accuracy + Loss dual graph (one figure per epoch checkpoint) ───────────────

def plot_accuracy_loss(history, split, max_epoch):
    """
    Generate a side-by-side figure: Model Accuracy (left) | Model Loss (right).

    Shows training curves from epoch 1 up to max_epoch so the lecturer can
    compare how the model behaves at three different training lengths
    (epoch 10 = early, 50 = mid, 100 = full training).

    The grey shaded band between train and val curves highlights the
    generalisation gap -- a large gap suggests overfitting.

    Saves as: accuracy_loss_<split>_ep<max_epoch>.png
    """
    acc_key     = 'accuracy'
    val_acc_key = 'val_accuracy'
    loss_key    = 'loss'
    val_loss_key= 'val_loss'

    for key in (acc_key, val_acc_key, loss_key, val_loss_key):
        if key not in history:
            print(f"  [{split}] Skipping epoch {max_epoch} graph -- '{key}' not in history.")
            return

    n          = min(max_epoch, len(history[acc_key]))
    epochs_arr = np.arange(1, n + 1)
    label      = LABELS[split]

    train_acc  = np.array(history[acc_key][:n])
    val_acc    = np.array(history[val_acc_key][:n])
    train_loss = np.array(history[loss_key][:n])
    val_loss   = np.array(history[val_loss_key][:n])

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Model Performance up to Epoch {max_epoch}  [{label} Split]',
                 fontsize=13, fontweight='bold', y=1.01)

    # ── Accuracy subplot ──────────────────────────────────────────────────────
    ax_acc.plot(epochs_arr, train_acc, color=C_TRAIN, linewidth=2.0, label='Train Accuracy')
    ax_acc.plot(epochs_arr, val_acc,   color=C_VAL,   linewidth=2.0, label='Validation Accuracy')
    ax_acc.fill_between(epochs_arr, train_acc, val_acc, alpha=0.08, color='grey')
    _annotate_best(ax_acc, epochs_arr, val_acc)
    _style_ax(ax_acc, 'Model Accuracy', 'Accuracy', n)

    # ── Loss subplot ──────────────────────────────────────────────────────────
    # Loss should decrease over time; _annotate_best uses -val_loss so the
    # "best" point is correctly identified as the minimum (lowest) loss epoch.
    loss_max = float(np.max([train_loss, val_loss])) * 1.15
    ax_loss.plot(epochs_arr, train_loss, color=C_TRAIN, linewidth=2.0, label='Train Loss')
    ax_loss.plot(epochs_arr, val_loss,   color=C_VAL,   linewidth=2.0, label='Validation Loss')
    ax_loss.fill_between(epochs_arr, train_loss, val_loss, alpha=0.08, color='grey')
    _annotate_best(ax_loss, epochs_arr, -val_loss)   # negate so argmax finds the minimum
    _style_ax(ax_loss, 'Model Loss', 'Loss', n, ylim=(0.0, loss_max))

    plt.tight_layout()
    _save(fig, SCRIPT_DIR / f'accuracy_loss_{split}_ep{max_epoch}.png')


# ── Hyperparameter bar chart ───────────────────────────────────────────────────

def plot_hyperparam(hp_results, split):
    """
    Vertical bar chart with 3 bars -- one per epoch checkpoint (10, 50, 100).

    Each bar shows the val_accuracy recorded at that epoch from the training
    history, along with the learning rate at that point.
    The best-performing checkpoint is highlighted in amber.

    This chart helps answer: "Does training longer actually help?"
    If epoch 10 and epoch 100 have similar accuracy the model converges fast.
    """
    if not hp_results:
        print(f"  [{split}] Skipping hyperparameter chart -- no data.")
        return

    label = LABELS[split]

    # Sort by epoch number ascending: 10, 50, 100
    sorted_results = sorted(hp_results, key=lambda r: r['epochs'])
    bar_labels = [f"Epoch {r['epochs']}" for r in sorted_results]
    val_accs   = [r['val_accuracy'] for r in sorted_results]
    best_i     = int(np.argmax(val_accs))
    colors     = [C_BEST_BAR if i == best_i else C_BARS
                  for i in range(len(sorted_results))]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(range(len(bar_labels)), val_accs,
                  color=colors, alpha=0.88, edgecolor='white', linewidth=0.8, width=0.5)

    # Annotate each bar with its accuracy value and the corresponding LR
    for bar, val, r in zip(bars, val_accs, sorted_results):
        lr_str = f"lr={r['learning_rate']:.2e}" if r['learning_rate'] else ''
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2, val - 0.015,
                lr_str, ha='center', va='top', fontsize=8, color='white', fontweight='bold')

    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, fontsize=11)
    ax.set_ylabel('Validation Accuracy', fontsize=11)
    ax.set_ylim(0, min(1.0, max(val_accs) + 0.06))
    ax.set_title(f'Validation Accuracy at Epoch Checkpoints  [{label} Split]',
                 fontsize=13, fontweight='bold', pad=10)
    ax.grid(which='major', axis='y', linestyle='--', linewidth=0.6, alpha=0.55)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='both', labelsize=10)

    # Annotate the overall best checkpoint in the bottom-right corner
    best = sorted_results[best_i]
    lr_str = f"{best['learning_rate']:.2e}" if best['learning_rate'] else 'N/A'
    ax.text(0.99, 0.02,
            f"Best: Epoch {best['epochs']}  lr={lr_str}  ->  acc={best['val_accuracy']:.4f}",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color=C_BEST_BAR,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=C_BEST_BAR, alpha=0.88))

    plt.tight_layout()
    _save(fig, SCRIPT_DIR / f'hyperparam_{split}.png')


# ── Precision graph ───────────────────────────────────────────────────────────

def plot_precision(history, split):
    """
    Plot training and validation Precision over the full 100-epoch run.

    Precision = TP / (TP + FP).
    High precision means when the model predicts a disease, it is usually
    correct (few false positives).  A growing gap between train and val
    precision over epochs signals overfitting.

    Saves as: precision_<split>.png
    """
    p_key  = 'precision'
    vp_key = 'val_precision'
    for key in (p_key, vp_key):
        if key not in history:
            print(f"  [{split}] Skipping precision graph -- '{key}' not in history.")
            return

    n          = len(history[p_key])
    epochs_arr = np.arange(1, n + 1)
    label      = LABELS[split]

    train_p = np.array(history[p_key])
    val_p   = np.array(history[vp_key])

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f'Model Precision  [{label} Split]', fontsize=13, fontweight='bold', y=1.01)

    ax.plot(epochs_arr, train_p, color=C_TRAIN, linewidth=2.0, label='Train Precision')
    ax.plot(epochs_arr, val_p,   color=C_VAL,   linewidth=2.0, label='Validation Precision')
    ax.fill_between(epochs_arr, train_p, val_p, alpha=0.08, color='grey')
    _annotate_best(ax, epochs_arr, val_p)
    _style_ax(ax, 'Model Precision', 'Precision', n)

    plt.tight_layout()
    _save(fig, SCRIPT_DIR / f'precision_{split}.png')


# ── Recall graph ───────────────────────────────────────────────────────────────

def plot_recall(history, split):
    """
    Plot training and validation Recall over the full 100-epoch run.

    Recall = TP / (TP + FN).
    High recall means the model correctly identifies most disease cases
    (few false negatives).  In a plant disease detection context, high
    recall is critical because missing a diseased plant is costly.

    Saves as: recall_<split>.png
    """
    r_key  = 'recall'
    vr_key = 'val_recall'
    for key in (r_key, vr_key):
        if key not in history:
            print(f"  [{split}] Skipping recall graph -- '{key}' not in history.")
            return

    n          = len(history[r_key])
    epochs_arr = np.arange(1, n + 1)
    label      = LABELS[split]

    train_r = np.array(history[r_key])
    val_r   = np.array(history[vr_key])

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f'Model Recall  [{label} Split]', fontsize=13, fontweight='bold', y=1.01)

    ax.plot(epochs_arr, train_r, color=C_TRAIN, linewidth=2.0, label='Train Recall')
    ax.plot(epochs_arr, val_r,   color=C_VAL,   linewidth=2.0, label='Validation Recall')
    ax.fill_between(epochs_arr, train_r, val_r, alpha=0.08, color='grey')
    _annotate_best(ax, epochs_arr, val_r)
    _style_ax(ax, 'Model Recall', 'Recall', n)

    plt.tight_layout()
    _save(fig, SCRIPT_DIR / f'recall_{split}.png')


# ── Best-hyperparam cross-split comparison ────────────────────────────────────

def plot_best_hyperparam_comparison(all_best: dict):
    """
    Bar chart comparing the best checkpoint val_accuracy across all 3 splits.

    all_best = {
        '70_30': {'epochs': 100, 'learning_rate': ..., 'val_accuracy': ...},
        '80_20': {...},
        '90_10': {...},
    }

    This chart directly answers: "Which train/test split ratio gives the
    best model performance?"  A higher bar for 90:10 means more training
    data helps, which is typical for deep learning.

    Saves as: hyperparam_comparison.png
    """
    if not all_best:
        print("\nSkipping cross-split comparison -- no hyperparam data available.")
        return

    splits_present = [s for s in SPLITS if s in all_best]
    split_labels   = [LABELS[s] for s in splits_present]
    val_accs       = [all_best[s]['val_accuracy'] for s in splits_present]
    combos         = [all_best[s] for s in splits_present]

    BAR_COLORS = ['#1565C0', '#6A1B9A', '#C62828']   # blue / purple / red per split
    colors = BAR_COLORS[:len(splits_present)]

    best_acc = max(val_accs)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(split_labels, val_accs,
                  color=colors, alpha=0.88,
                  edgecolor='white', linewidth=0.8,
                  width=0.45)

    # Label each bar with its accuracy and the epoch/LR that achieved it
    for bar, val, combo in zip(bars, val_accs, combos):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.004,
            f'{val:.4f}',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold',
        )
        lr_str = f"{combo['learning_rate']:.2e}" if combo['learning_rate'] else 'N/A'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val - 0.03,
            f"Epoch {combo['epochs']}\nlr={lr_str}",
            ha='center', va='top',
            fontsize=9, color='white', fontweight='bold',
        )

    # Horizontal dashed line at the highest accuracy for easy reading
    ax.axhline(y=best_acc, color=C_BEST, linestyle='--', linewidth=1.3, alpha=0.75)

    ax.set_title('Best Epoch Checkpoint -- Cross-Split Validation Accuracy Comparison',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Dataset Split (Train / Test)', fontsize=11)
    ax.set_ylabel('Validation Accuracy at Best Epoch', fontsize=11)
    ax.set_ylim(0, min(1.0, best_acc + 0.10))
    ax.grid(which='major', axis='y', linestyle='--', linewidth=0.6, alpha=0.55)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='both', labelsize=10)

    legend_lines = [
        plt.Line2D([0], [0], color=colors[i], linewidth=6, alpha=0.88,
                   label=f"{split_labels[i]}: Epoch {combos[i]['epochs']}, acc={val_accs[i]:.4f}")
        for i in range(len(splits_present))
    ]
    ax.legend(handles=legend_lines, fontsize=9,
              framealpha=0.92, edgecolor='#cccccc', loc='lower right')

    plt.tight_layout()
    _save(fig, SCRIPT_DIR / 'hyperparam_comparison.png')


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(split):
    """
    Load the saved model and test features, run inference, then plot a
    normalised confusion matrix heatmap.

    A confusion matrix shows, for each true class (row), how many test
    samples were predicted as each class (column).  The diagonal cells
    show correct predictions; off-diagonal cells show misclassifications.

    Two values are shown in every cell:
      - Normalised fraction (0.00 -- 1.00): proportion of that true class
        predicted as the column class (row-normalised recall perspective).
      - Raw count in parentheses: actual number of test images.

    Interpreting the matrix:
      - A strong diagonal (values close to 1.0) means the model rarely
        confuses one disease for another.
      - An off-diagonal hot spot (e.g. row "Healthy", col "Bacterial_spot")
        means the model frequently mistakes one class for another, which
        may indicate visual similarity or insufficient training data.

    Saves as: confusion_matrix_<split>.png
    """
    # Heavy imports kept local so missing libraries only fail here, not at startup
    from tensorflow import keras
    from sklearn.metrics import confusion_matrix

    model_path       = SCRIPT_DIR / f'tomato_disease_{split}.h5'
    features_dir     = SCRIPT_DIR / f'features_{split}'
    class_names_path = features_dir / 'class_names.json'

    # Guard: skip gracefully if model or feature files were not produced by tmt_5
    if not model_path.exists():
        print(f"  [{split}] Skipping confusion matrix -- model file not found.")
        return
    if not (features_dir / 'X_test.npy').exists():
        print(f"  [{split}] Skipping confusion matrix -- test features not found.")
        return

    # Load test features and convert one-hot labels to integer class indices
    X_test = np.load(features_dir / 'X_test.npy')
    y_test = np.load(features_dir / 'y_test.npy')
    y_true = np.argmax(y_test, axis=1)   # one-hot -> scalar class index

    with open(class_names_path) as f:
        class_names = json.load(f)

    # Reload the trained model and predict class probabilities for each test image
    print(f"  [{split}] Loading model for confusion matrix...")
    model  = keras.models.load_model(str(model_path))
    y_prob = model.predict(X_test, verbose=0)   # shape: (N, num_classes)
    y_pred = np.argmax(y_prob, axis=1)          # pick the highest-probability class

    # Compute the raw confusion matrix (rows=true, cols=predicted)
    cm = confusion_matrix(y_true, y_pred)

    # Row-normalise: divide each row by the total count of that true class.
    # This converts raw counts to recall rates, making classes with different
    # sample sizes comparable on the same 0-1 colour scale.
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    label = LABELS[split]
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalised Count (Recall per Class)', fontsize=9)

    # Set tick labels to class names, rotated for readability
    tick_marks = np.arange(n_classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=9)

    # Annotate every cell with the normalised value and raw count
    for i in range(n_classes):
        for j in range(n_classes):
            # Use white text on dark cells, black text on light cells for readability
            text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i,
                    f'{cm_norm[i, j]:.2f}\n({cm[i, j]})',
                    ha='center', va='center',
                    fontsize=8, color=text_color, fontweight='bold')

    ax.set_title(f'Confusion Matrix  [{label} Split]',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    plt.tight_layout()
    _save(fig, SCRIPT_DIR / f'confusion_matrix_{split}.png')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('Evaluation Visualisation')
    print(f'Splits      : {", ".join(LABELS[s] for s in SPLITS)}')
    print(f'Checkpoints : epochs {CHECKPOINTS}')
    print('=' * 60)

    found_any = False
    all_best  = {}   # {split: best_checkpoint_dict}

    for split in SPLITS:
        history_path    = SCRIPT_DIR / f'training_history_{split}.json'
        hyperparam_path = SCRIPT_DIR / f'hyperparam_results_{split}.json'

        if not history_path.exists():
            print(f"\n[{split}] Skipping -- training_history_{split}.json not found.")
            continue

        found_any = True
        print(f"\n-- Split {LABELS[split]} " + "-" * 44)

        with open(history_path) as f:
            history = json.load(f)

        # Generate 3 accuracy/loss side-by-side graphs (one per epoch checkpoint)
        for ep in CHECKPOINTS:
            plot_accuracy_loss(history, split, ep)

        # Generate precision and recall line graphs for the full training run
        plot_precision(history, split)
        plot_recall(history, split)

        # Generate 1 hyperparameter bar chart (3 bars: epoch 10, 50, 100)
        hp_results = None
        if hyperparam_path.exists():
            with open(hyperparam_path) as f:
                hp_results = json.load(f)
            best = max(hp_results, key=lambda r: r['val_accuracy'])
            all_best[split] = best
        else:
            print(f"  (hyperparam_results_{split}.json not found -- skipping hyperparam chart)")

        plot_hyperparam(hp_results, split)

        # Generate confusion matrix heatmap using the saved model + test features
        plot_confusion_matrix(split)

    if not found_any:
        print('\nNo training history files found. Run tmt_5_model.py first.')
        return

    # Cross-split comparison (runs once after all splits are processed)
    print('\n-- Cross-split comparison ' + '-' * 35)
    plot_best_hyperparam_comparison(all_best)

    print(f'\n{"=" * 60}')
    print('All graphs generated.')
    n_per_split = len(CHECKPOINTS) + 1 + 2 + 1   # acc/loss + hyperparam + precision + recall + confusion
    print(f'\nOutputs ({len(SPLITS) * n_per_split + 1} graphs total):')
    for split in SPLITS:
        label = LABELS[split]
        for ep in CHECKPOINTS:
            print(f'  accuracy_loss_{split}_ep{ep}.png  [{label} -- epoch {ep}]')
        print(f'  precision_{split}.png      [{label}]')
        print(f'  recall_{split}.png         [{label}]')
        print(f'  hyperparam_{split}.png     [{label}]')
        print(f'  confusion_matrix_{split}.png  [{label}]')
    print('  hyperparam_comparison.png  [all splits]')


if __name__ == '__main__':
    main()
