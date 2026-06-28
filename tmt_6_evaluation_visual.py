import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import json
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS     = ['70_30', '80_20', '90_10']
LABELS     = {'70_30': '70/30', '80_20': '80/20', '90_10': '90/10'}

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


# ── Per-metric graph ───────────────────────────────────────────────────────────

def plot_metric(history, split, metric, ylabel, filename_prefix):
    """
    Generates a single train-vs-validation graph for one metric.
    Saves as  <filename_prefix>_<split>.png
    """
    train_key = metric
    val_key   = f'val_{metric}'

    if train_key not in history or val_key not in history:
        print(f"  [{split}] Skipping '{metric}' — key not in history.")
        return

    train_vals = np.array(history[train_key])
    val_vals   = np.array(history[val_key])
    n          = len(train_vals)
    epochs_arr = np.arange(1, n + 1)
    label      = LABELS[split]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(epochs_arr, train_vals,
            color=C_TRAIN, linewidth=2.0, label=f'Train {ylabel}')
    ax.plot(epochs_arr, val_vals,
            color=C_VAL,   linewidth=2.0, label=f'Validation {ylabel}')

    # Shade the gap between train and val
    ax.fill_between(epochs_arr, train_vals, val_vals,
                    alpha=0.08, color='grey')

    _annotate_best(ax, epochs_arr, val_vals)
    _style_ax(ax, f'{ylabel} over {n} Epochs  [{label} Split]', ylabel, n)

    _save(fig, SCRIPT_DIR / f'{filename_prefix}_{split}.png')


# ── Hyperparameter bar chart ───────────────────────────────────────────────────

def plot_hyperparam(hp_results, split):
    """
    Horizontal bar chart sorted by val_accuracy (highest at top).
    Highlights the best combo in amber.
    """
    if not hp_results:
        print(f"  [{split}] Skipping hyperparameter chart — no data.")
        return

    label = LABELS[split]

    # Sort descending by val_accuracy
    sorted_results = sorted(hp_results, key=lambda r: r['val_accuracy'])
    bar_labels = [f"lr={r['learning_rate']:.0e}\nep={r['epochs']}" for r in sorted_results]
    val_accs   = [r['val_accuracy'] for r in sorted_results]
    colors     = [C_BEST_BAR if i == len(sorted_results) - 1 else C_BARS
                  for i in range(len(sorted_results))]

    fig, ax = plt.subplots(figsize=(12, max(5, len(sorted_results) * 0.65)))

    bars = ax.barh(range(len(bar_labels)), val_accs,
                   color=colors, alpha=0.88, edgecolor='white', linewidth=0.8)

    # Value labels at end of each bar
    for bar, val in zip(bars, val_accs):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left', fontsize=9, fontweight='bold')

    ax.set_yticks(range(len(bar_labels)))
    ax.set_yticklabels(bar_labels, fontsize=9)
    ax.set_xlabel('Validation Accuracy', fontsize=11)
    ax.set_xlim(0, min(1.0, max(val_accs) + 0.08))
    ax.set_title(f'Hyperparameter Search — Validation Accuracy  [{label} Split]',
                 fontsize=14, fontweight='bold', pad=10)
    ax.grid(which='major', axis='x', linestyle='--', linewidth=0.6, alpha=0.55)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='both', labelsize=9)

    # Best-combo label
    best = sorted_results[-1]
    ax.text(0.99, 0.02,
            f"Best: lr={best['learning_rate']:.0e}, ep={best['epochs']}  "
            f"→  acc={best['val_accuracy']:.4f}",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color=C_BEST_BAR,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=C_BEST_BAR, alpha=0.88))

    plt.tight_layout()
    _save(fig, SCRIPT_DIR / f'hyperparam_{split}.png')


# ── Best-hyperparam cross-split comparison ────────────────────────────────────

def plot_best_hyperparam_comparison(all_best: dict):
    """
    all_best = {
        '70_30': {'learning_rate': ..., 'epochs': ..., 'val_accuracy': ...},
        '80_20': {...},
        '90_10': {...},
    }
    Generates a vertical bar chart comparing the best combo from each split.
    """
    if not all_best:
        print("\nSkipping cross-split comparison — no hyperparam data available.")
        return

    splits_present = [s for s in SPLITS if s in all_best]
    split_labels   = [LABELS[s] for s in splits_present]
    val_accs       = [all_best[s]['val_accuracy'] for s in splits_present]
    combos         = [all_best[s] for s in splits_present]

    # Colour each bar differently
    BAR_COLORS = ['#1565C0', '#6A1B9A', '#C62828']   # blue / purple / red per split
    colors = BAR_COLORS[:len(splits_present)]

    overall_best_i = int(np.argmax(val_accs))

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(split_labels, val_accs,
                  color=colors, alpha=0.88,
                  edgecolor='white', linewidth=0.8,
                  width=0.45)

    # Value label above each bar
    for bar, val, combo in zip(bars, val_accs, combos):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.008,
            f'{val:.4f}',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold',
        )
        # Best combo annotation inside/below the bar top
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val - 0.04,
            f"lr={combo['learning_rate']:.0e}\nep={combo['epochs']}",
            ha='center', va='top',
            fontsize=9, color='white', fontweight='bold',
        )

    # Dashed line at overall best (no text annotation — just the line)
    best_acc = max(val_accs)
    ax.axhline(y=best_acc, color=C_BEST, linestyle='--', linewidth=1.3, alpha=0.75)

    ax.set_title('Best Hyperparameter per Dataset Split — Cross-Split Comparison',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Dataset Split (Train / Test)', fontsize=11)
    ax.set_ylabel('Best Validation Accuracy', fontsize=11)
    ax.set_ylim(0, min(1.0, best_acc + 0.12))
    ax.grid(which='major', axis='y', linestyle='--', linewidth=0.6, alpha=0.55)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='both', labelsize=10)

    # Legend showing what combo each split used
    legend_lines = [
        plt.Line2D([0], [0], color=colors[i], linewidth=6, alpha=0.88,
                   label=f"{split_labels[i]}: lr={combos[i]['learning_rate']:.0e}, ep={combos[i]['epochs']}")
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
    print(f'Splits: {", ".join(LABELS[s] for s in SPLITS)}')
    print('=' * 60)

    found_any = False
    all_best  = {}   # {split: best_combo_dict}

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

        hp_results = None
        if hyperparam_path.exists():
            with open(hyperparam_path) as f:
                hp_results = json.load(f)
            # Pick best combo for cross-split comparison
            best = max(hp_results, key=lambda r: r['val_accuracy'])
            all_best[split] = best
        else:
            print(f"  (hyperparam_results_{split}.json not found — skipping hyperparam chart)")

        plot_metric(history, split, 'accuracy',  'Accuracy',  'accuracy')
        plot_metric(history, split, 'recall',    'Recall',    'recall')
        plot_metric(history, split, 'precision', 'Precision', 'precision')
        plot_hyperparam(hp_results, split)

    if not found_any:
        print('\nNo training history files found. Run tmt_5_model.py first.')
        return

    # Cross-split comparison (runs once after all splits are processed)
    print('\n── Cross-split comparison ─────────────────────────────────')
    plot_best_hyperparam_comparison(all_best)

    print(f'\n{"=" * 60}')
    print('All graphs generated.')
    print('\nOutputs (13 graphs total):')
    for split in SPLITS:
        label = LABELS[split]
        for prefix in ('accuracy', 'recall', 'precision', 'hyperparam'):
            print(f'  {prefix}_{split}.png  [{label}]')
    print('  hyperparam_comparison.png  [all splits]')


if __name__ == '__main__':
    main()
