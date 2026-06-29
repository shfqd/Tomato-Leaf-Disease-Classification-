import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import json
from pathlib import Path

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
    fig.savefig(str(path), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {path.name}")


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
    """Draw a vertical line and label at the epoch with highest value."""
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
    Generates a side-by-side figure:  Model Accuracy (left) | Model Loss (right)
    showing training curves from epoch 1 up to max_epoch.
    Saves as  accuracy_loss_{split}_ep{max_epoch}.png
    """
    acc_key     = 'accuracy'
    val_acc_key = 'val_accuracy'
    loss_key    = 'loss'
    val_loss_key= 'val_loss'

    for key in (acc_key, val_acc_key, loss_key, val_loss_key):
        if key not in history:
            print(f"  [{split}] Skipping epoch {max_epoch} graph — '{key}' not in history.")
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
    loss_max = float(np.max([train_loss, val_loss])) * 1.15
    ax_loss.plot(epochs_arr, train_loss, color=C_TRAIN, linewidth=2.0, label='Train Loss')
    ax_loss.plot(epochs_arr, val_loss,   color=C_VAL,   linewidth=2.0, label='Validation Loss')
    ax_loss.fill_between(epochs_arr, train_loss, val_loss, alpha=0.08, color='grey')
    _annotate_best(ax_loss, epochs_arr, -val_loss)   # best = lowest loss
    _style_ax(ax_loss, 'Model Loss', 'Loss', n, ylim=(0.0, loss_max))

    plt.tight_layout()
    _save(fig, SCRIPT_DIR / f'accuracy_loss_{split}_ep{max_epoch}.png')


# ── Hyperparameter bar chart ───────────────────────────────────────────────────

def plot_hyperparam(hp_results, split):
    """
    Vertical bar chart with 3 bars — one per epoch checkpoint (10, 50, 100).
    Each bar shows the val_accuracy recorded at that epoch from training history.
    Highlights the best epoch in amber.
    """
    if not hp_results:
        print(f"  [{split}] Skipping hyperparameter chart — no data.")
        return

    label = LABELS[split]

    # Sort by epoch number (ascending: 10, 50, 100)
    sorted_results = sorted(hp_results, key=lambda r: r['epochs'])
    bar_labels = [f"Epoch {r['epochs']}" for r in sorted_results]
    val_accs   = [r['val_accuracy'] for r in sorted_results]
    best_i     = int(np.argmax(val_accs))
    colors     = [C_BEST_BAR if i == best_i else C_BARS
                  for i in range(len(sorted_results))]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(range(len(bar_labels)), val_accs,
                  color=colors, alpha=0.88, edgecolor='white', linewidth=0.8, width=0.5)

    # Value labels above each bar
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

    # Best checkpoint label
    best = sorted_results[best_i]
    lr_str = f"{best['learning_rate']:.2e}" if best['learning_rate'] else 'N/A'
    ax.text(0.99, 0.02,
            f"Best: Epoch {best['epochs']}  lr={lr_str}  →  acc={best['val_accuracy']:.4f}",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color=C_BEST_BAR,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=C_BEST_BAR, alpha=0.88))

    plt.tight_layout()
    _save(fig, SCRIPT_DIR / f'hyperparam_{split}.png')


# ── Best-hyperparam cross-split comparison ────────────────────────────────────

def plot_best_hyperparam_comparison(all_best: dict):
    """
    all_best = {
        '70_30': {'epochs': 100, 'learning_rate': ..., 'val_accuracy': ...},
        '80_20': {...},
        '90_10': {...},
    }
    Generates a vertical bar chart comparing the best epoch checkpoint
    val_accuracy across all 3 splits.
    """
    if not all_best:
        print("\nSkipping cross-split comparison — no hyperparam data available.")
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

    ax.axhline(y=best_acc, color=C_BEST, linestyle='--', linewidth=1.3, alpha=0.75)

    ax.set_title('Best Epoch Checkpoint — Cross-Split Validation Accuracy Comparison',
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
            print(f"\n[{split}] Skipping — training_history_{split}.json not found.")
            continue

        found_any = True
        print(f"\n── Split {LABELS[split]} ──────────────────────────────────")

        with open(history_path) as f:
            history = json.load(f)

        # 3 accuracy/loss graphs — one per epoch checkpoint
        for ep in CHECKPOINTS:
            plot_accuracy_loss(history, split, ep)

        # 1 hyperparameter bar chart (3 bars: epoch 10, 50, 100)
        hp_results = None
        if hyperparam_path.exists():
            with open(hyperparam_path) as f:
                hp_results = json.load(f)
            best = max(hp_results, key=lambda r: r['val_accuracy'])
            all_best[split] = best
        else:
            print(f"  (hyperparam_results_{split}.json not found — skipping hyperparam chart)")

        plot_hyperparam(hp_results, split)

    if not found_any:
        print('\nNo training history files found. Run tmt_5_model.py first.')
        return

    # Cross-split comparison (runs once after all splits are processed)
    print('\n── Cross-split comparison ─────────────────────────────────')
    plot_best_hyperparam_comparison(all_best)

    print(f'\n{"=" * 60}')
    print('All graphs generated.')
    print(f'\nOutputs ({len(SPLITS) * (len(CHECKPOINTS) + 1) + 1} graphs total):')
    for split in SPLITS:
        label = LABELS[split]
        for ep in CHECKPOINTS:
            print(f'  accuracy_loss_{split}_ep{ep}.png  [{label} — epoch {ep}]')
        print(f'  hyperparam_{split}.png  [{label}]')
    print('  hyperparam_comparison.png  [all splits]')


if __name__ == '__main__':
    main()
