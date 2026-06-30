# Tomato Disease Classification System

**CSC566 Group Project** — Deep learning pipeline to classify tomato leaf diseases from images using Transfer Learning (MobileNetV2).

|-------------------|--------------------------------------------------------------------|
|    **Backbone**   | MobileNetV2 (pretrained on ImageNet, frozen)                       |
|    **Classes**    | `healthy` · `Bacterial_spot` · `Leaf_Mold` · `Tomato_mosaic_virus` |
| **Best Accuracy** | 99.63% (90:10 split)                                               |

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

|------------|--------------------------------------------------------------------------------|
| **Input**  | `dataset/` (raw tomato leaf images, per-class folders)                         |
| **Output** | `dataset/dataset_split_70_30/`, `dataset_split_80_20/`, `dataset_split_90_10/` |

The dataset is split three times at different ratios so that Stage 3 and 4 can compare how training data volume affects final model accuracy.

| Split | Train | Test |     Output Folder      |
|-------|-------|------|------------------------|
|   A   |  70%  |  30% | `dataset_split_70_30/` |
|   B   |  80%  |  20% | `dataset_split_80_20/` |
|   C   |  90%  |  10% | `dataset_split_90_10/` |

Each split produces a `train/` and `test/` subdirectory per class. All splits use `random_state=42` for reproducibility.

---

## Stage 2 — Feature Extraction

**Script:** `tmt_4_feature_extraction.py`

|------------|---------------------------------------------------------|
| **Input**  | `dataset/dataset_split_*/` (3 splits from Stage 1)      |
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

## Stage 3 — Model Training

**Script:** `tmt_5_model.py`

|------------|-------------------------------------------------------------------------------------------------|
| **Input**  | `features_{split}/` (`.npy` arrays from Stage 2)                                                |
| **Output** | `tomato_disease_{split}.h5`, `training_history_{split}.json`, `hyperparam_results_{split}.json` |

### Classifier Architecture

```
Input (1280-dim feature vector)
    → Dense(256, relu) → Dropout(0.4)
    → Dense(128, relu) → Dropout(0.3)
    → Dense(4, softmax)
```

|  Setting  |           Value             |
|-----------|-----------------------------|
|   Loss    |  Categorical cross-entropy  |
|  Metrics  | Accuracy, Precision, Recall |
| Optimizer |           Adam              |

### Training

The model trains for **100 epochs**. A custom `LearningRateLogger` callback **records the learning rate at every epoch** — the LR is not predefined, it is observed and logged from whatever Adam uses at each step.

All metrics per epoch (accuracy, val_accuracy, loss, val_loss, precision, recall, learning_rate) are saved to `training_history_{split}.json`.

### Hyperparameter Results (generated after training)

Once training is complete, the saved history is used to extract the model's performance at **3 epoch checkpoints: epoch 10, epoch 50, and epoch 100**. These become the hyperparameter comparison points — the varying "hyperparameter" here is the epoch count, and the learning rate at each checkpoint is the value recorded by `LearningRateLogger` at that epoch.

Results are saved to `hyperparam_results_{split}.json` with the val_accuracy at each of the 3 checkpoints.

This entire process is repeated for all three splits (70_30, 80_20, 90_10).

---

## Stage 4 — Evaluation & Visualisation

**Script:** `tmt_6_evaluation_visual.py`

|------------|--------------------------------------------------------------------|
| **Input**  | `training_history_{split}.json`, `hyperparam_results_{split}.json` |
| **Output** | 12 PNG graphs (4 per split × 3 splits) + 1 cross-split comparison  |

### Graphs Generated

**Per split — 6 graphs × 3 splits = 18 graphs:**

Each split produces 3 accuracy/loss graph pairs, 1 precision graph, 1 recall graph, and 1 hyperparameter bar chart.

|           Graph          | Epoch Checkpoint |                             Content                                 |
|--------------------------|------------------|---------------------------------------------------------------------|
| Accuracy/Loss graph 1    |     Epoch 10     | Train Accuracy & Loss vs Validation Accuracy & Loss up to epoch 10  |
| Accuracy/Loss graph 2    |     Epoch 50     | Train Accuracy & Loss vs Validation Accuracy & Loss up to epoch 50  |
| Accuracy/Loss graph 3    |     Epoch 100    | Train Accuracy & Loss vs Validation Accuracy & Loss up to epoch 100 |
| `precision_{split}.png`  |         —        | Train Precision vs Validation Precision over full 100 epochs        |
| `recall_{split}.png`     |         —        | Train Recall vs Validation Recall over full 100 epochs              |
| `hyperparam_{split}.png` |         —        | Bar chart with 3 bars showing val_accuracy at epoch 10, 50, and 100 |

**Cross-split (1 graph):**

| File                        |                                                 Content                                                  |
|-----------------------------|----------------------------------------------------------------------------------------------------------|
| `hyperparam_comparison.png` | Compares the best epoch checkpoint val_accuracy across all 3 splits — overall winner immediately visible |

**Total: 19 graphs (18 per-split + 1 cross-split)**

**Colour scheme:** Blue = Train · Orange = Validation

---

## Hyperparameter Results (Val Accuracy at Each Epoch Checkpoint)

|   Split   | Epoch 10 |  Epoch 50  | Epoch 100  |          Best         |
|-----------|----------|------------|------------|-----------------------|
|   70:30   |  98.05%  | **98.71%** |   98.66%   |     **Epoch 50**      |
|   80:20   |  97.95%  |   98.46%   | **98.79%** |    **Epoch 100**      |
| **90:10** |  98.61%  | **99.63%** |   99.49%   | **Epoch 50 ← Winner** |

---

## How the Best Model is Selected

1. **Train the model** — each split trains for 100 epochs; `LearningRateLogger` records the LR and all metrics at every epoch into `training_history_{split}.json`.
2. **Extract epoch checkpoints** — val_accuracy is read from the history at epoch 10, 50, and 100. These 3 values are saved to `hyperparam_results_{split}.json`.
3. **Per-split winner** — the epoch checkpoint with the highest val_accuracy is the best configuration for that split. Epoch 50 wins for the 70:30 and 90:10 splits (validation accuracy peaks before overfitting sets in); Epoch 100 wins only for the 80:20 split.
4. **Cross-split comparison** — `hyperparam_comparison.png` plots the best checkpoint val_accuracy from all 3 splits side by side. The **90:10 split** wins with the highest accuracy — 90% training data gives the classifier more examples to learn from.
5. **Production deployment** — `tmt_main.py` loads `tomato_disease_90_10.h5` and serves it via HTTP at `127.0.0.1:8000`.

---

## Best Model — Final Specification

|---------------------------|---------------------------------------------------------------------------|
| **Split**                 | 90:10 (90% train / 10% test)                                              |
| **Model file**            | `tomato_disease_90_10.h5`                                                 |
| **Backbone**              | MobileNetV2 (ImageNet, frozen) → 1280-dim features                        |
| **Classifier**            | Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Dense(4, softmax) |
| **LR**                    | Recorded by `LearningRateLogger` (not predefined)                         |
| **Best epoch checkpoint** | 50                                                                        |
| **Final training**        | 100 epochs                                                                |
| **Batch size**            | 32                                                                        |
| **Optimiser**             | Adam                                                                      |
| **Loss**                  | Categorical cross-entropy                                                 |
| **Val accuracy**          | 99.63%                                                                    |
| **Train accuracy**        | ~99.8% (epoch 50)                                                         |

---

## Inference Server

**Script:** `tmt_main.py`  
**Run:** `python tmt_main.py`  
**URL:** `http://127.0.0.1:8000`

|      Route      |                              Description                                 |
|-----------------|--------------------------------------------------------------------------|
|     `GET /`     | Serves `test.html` — web UI for uploading leaf images                    |
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
