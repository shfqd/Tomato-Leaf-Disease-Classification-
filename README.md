# Tomato Disease Classification System

**CSC566 Group Project** — Deep learning pipeline to classify tomato leaf diseases from images using Transfer Learning (MobileNetV2).

| | |
|---|---|
| **Backbone** | MobileNetV2 (pretrained on ImageNet, frozen) |
| **Classes** | `healthy` · `Bacterial_spot` · `Leaf_Mold` · `Tomato_mosaic_virus` |
| **Best Accuracy** | 99.63% (90:10 split) |

> **Note:** Image Augmentation (`tmt_1`) and Preprocessing (`tmt_2`) are skipped. The dataset is used as-is starting from the splitting stage.

---

## Pipeline Overview

```
tmt_3_image_split.py
    → tmt_4_feature_extraction.py
        → tmt_5_model.py
            → tmt_6_evaluation_visual.py

tmt_main.py   ← inference server (run after training)
```

---

## Stage 1 — Train / Test Splitting

**Script:** `tmt_3_image_split.py`

| | |
|---|---|
| **Input** | `dataset/` (raw tomato leaf images, per-class folders) |
| **Output** | `dataset/dataset_split_70_30/`, `dataset_split_80_20/`, `dataset_split_90_10/` |

The dataset is split three times at different ratios so that Stage 3 and 4 can compare how training data volume affects final model accuracy.

| Split | Train | Test | Output Folder |
|---|---|---|---|
| A | 70% | 30% | `dataset_split_70_30/` |
| B | 80% | 20% | `dataset_split_80_20/` |
| C | 90% | 10% | `dataset_split_90_10/` |

Each split produces a `train/` and `test/` subdirectory per class. All splits use `random_state=42` for reproducibility.

---

## Stage 2 — Feature Extraction

**Script:** `tmt_4_feature_extraction.py`

| | |
|---|---|
| **Input** | `dataset/dataset_split_*/` (3 splits from Stage 1) |
| **Output** | `features_70_30/`, `features_80_20/`, `features_90_10/` |

Each output folder contains:
- `X_train.npy` — shape `(n_train, 1280)`
- `y_train.npy` — shape `(n_train,)`
- `X_test.npy`  — shape `(n_test, 1280)`
- `y_test.npy`  — shape `(n_test,)`
- `class_names.json`
- `tsne_features_{split}.png`

Instead of training a CNN from scratch, we reuse MobileNetV2 (ImageNet weights) as a fixed feature extractor — this is Transfer Learning.

**Process for each split:**

1. **`load_backbone()`** — loads MobileNetV2 with `include_top=False` + `GlobalAveragePooling2D`. All layers are frozen; weights do not change.
2. **`extract_features()`** — feeds images in batches of 32 through the frozen backbone. Input is normalised via `mobilenet_v2.preprocess_input` to `[-1, 1]`. Output per image: a **1280-dimensional feature vector**.
3. Saves feature arrays as `.npy` files.
4. **`save_tsne_plot()`** — projects the 1280-dim vectors into 2D with t-SNE, coloured by class, to visually verify class separability.

> **Why extract features separately?** Feature extraction is slow. Saving `.npy` files means Stage 3 can train and re-train the classifier many times instantly without re-running the backbone each time.

---

## Stage 3 — Model Training & Hyperparameter Search

**Script:** `tmt_5_model.py`

| | |
|---|---|
| **Input** | `features_{split}/` (`.npy` arrays from Stage 2) |
| **Output** | `tomato_disease_{split}.h5`, `training_history_{split}.json`, `hyperparam_results_{split}.json` |

### Classifier Architecture

```
Input (1280-dim feature vector)
    → Dense(256, relu) → Dropout(0.4)
    → Dense(128, relu) → Dropout(0.3)
    → Dense(4, softmax)
```

| Setting | Value |
|---|---|
| Loss | Categorical cross-entropy |
| Metrics | Accuracy, Precision, Recall |
| Optimizer | Adam |

### Hyperparameter Search

Before the final 100-epoch run, a grid search evaluates 9 combinations of `learning_rate × epochs` to find the best settings.

- **Learning rates tested:** `0.01, 0.005, 0.001, 0.0005, 0.0001, 0.0002, 3e-5, 1e-4, 1e-3`
- **Epoch counts tested:** `10, 20, 30, 40, 50`

Results are saved to `hyperparam_results_{split}.json`, sorted ascending by `val_accuracy` — the last entry is always the best.

### Final Training

The full model trains for **100 epochs** using the default learning rate. A custom `LearningRateLogger` callback records the LR at every epoch. All metrics are saved to `training_history_{split}.json` for plotting in Stage 4. This is repeated for all three splits.

---

## Stage 4 — Evaluation & Visualisation

**Script:** `tmt_6_evaluation_visual.py`

| | |
|---|---|
| **Input** | `training_history_{split}.json`, `hyperparam_results_{split}.json` |
| **Output** | 13 PNG graphs |

### Graphs Generated

**Per split × 3 splits = 12 graphs:**

| File | Content |
|---|---|
| `accuracy_{split}.png` | Training accuracy vs validation accuracy per epoch |
| `recall_{split}.png` | Training recall vs validation recall per epoch |
| `precision_{split}.png` | Training precision vs validation precision per epoch |
| `hyperparam_{split}.png` | Horizontal bar chart of all 9 HP combos, best highlighted in amber |

**Cross-split (1 graph):**

| File | Content |
|---|---|
| `hyperparam_comparison.png` | Side-by-side bars of each split's best combo — overall winner immediately visible |

**Colour scheme:** Blue = training · Red = validation · Green = best epoch · Amber = best HP combo

---

## Hyperparameter Search Results

| Split | Best LR | Best Epochs | Val Accuracy |
|---|---|---|---|
| 70:30 | 0.0001 | 50 | 98.66% |
| 80:20 | 0.001 | 50 | 98.75% |
| **90:10** | **0.001** | **50** | **99.63% ← Winner** |

**Top 3 combos for the winning 90:10 split:**

| Rank | LR | Epochs | Val Accuracy |
|---|---|---|---|
| 1st | 0.001 | 50 | 99.63% |
| 2nd | 0.0001 | 50 | 99.56% |
| 3rd | 0.0002 | 30 | 99.49% |

---

## How the Best Model is Selected

1. **Hyperparameter Grid Search** — for each of the 3 splits, 9 combos are trained and `val_accuracy` is recorded in `hyperparam_results_{split}.json`.
2. **Per-Split Winner** — within each split, the combo with the highest `val_accuracy` is selected. Results are sorted ascending; the last entry is the winner.
3. **Cross-Split Comparison** — `hyperparam_comparison.png` plots all three winners side by side. The **90:10 split (99.63%)** wins; 90% training data gives the classifier more examples while the 10% test set remains a fair held-out evaluation.
4. **Production Deployment** — `tmt_main.py` loads `tomato_disease_90_10.h5` and serves it via HTTP at `127.0.0.1:8000`.

---

## Best Model — Final Specification

| | |
|---|---|
| **Split** | 90:10 (90% train / 10% test) |
| **Model file** | `tomato_disease_90_10.h5` |
| **Backbone** | MobileNetV2 (ImageNet, frozen) → 1280-dim features |
| **Classifier** | Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Dense(4, softmax) |
| **Best LR** | 0.001 |
| **Best epochs (HP search)** | 50 |
| **Final training** | 100 epochs |
| **Batch size** | 32 |
| **Optimiser** | Adam |
| **Loss** | Categorical cross-entropy |
| **Val accuracy** | 99.63% |
| **Train accuracy** | ~99.8% (epoch 100) |

---

## Inference Server

**Script:** `tmt_main.py`  
**Run:** `python tmt_main.py`  
**URL:** `http://127.0.0.1:8000`

| Route | Description |
|---|---|
| `GET /` | Serves `test.html` — web UI for uploading leaf images |
| `POST /predict` | Accepts `{ "image": "<base64-encoded image>" }`, returns prediction JSON |

**Response format:**
```json
{
  "class": "Bacterial_spot",
  "confidence": 97.3,
  "probabilities": { "healthy": 0.01, "Bacterial_spot": 0.97, ... }
}
```

**Inference flow inside `_predict(image_bytes)`:**
1. Decode base64 image bytes → PIL Image
2. Resize to `(224, 224)`
3. Run through frozen MobileNetV2 → 1280-dim feature vector
4. Pass feature vector into the loaded Dense classifier
5. Return class name + softmax confidence scores

The server loads `tomato_disease_90_10.h5` once at startup via `_load_model()` and keeps it in memory for all subsequent requests.
