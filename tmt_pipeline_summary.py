# =============================================================================
# TOMATO DISEASE CLASSIFICATION SYSTEM — FULL PIPELINE SUMMARY
# =============================================================================
#
# Project  : CSC566 Group Project
# Task     : Classify tomato leaf disease from images using deep learning
# Backbone : MobileNetV2 (pretrained on ImageNet, frozen)
# Classes  : healthy | Bacterial_spot | Leaf_Mold | Tomato_mosaic_virus
#
# NOTE: Image Augmentation (tmt_1) and Preprocessing (tmt_2) are SKIPPED.
#       The dataset is used as-is, starting directly from Stage 3.
#
# Active pipeline execution order:
#
#   tmt_3_image_split.py
#       → tmt_4_feature_extraction.py
#           → tmt_5_model.py
#               → tmt_6_evaluation_visual.py
#
# After training is complete:
#   tmt_main.py  →  runs the inference web server on http://127.0.0.1:8000
#
# =============================================================================


# -----------------------------------------------------------------------------
# STAGE 1 — TRAIN / TEST SPLITTING
# Script  : tmt_3_image_split.py
# Input   : dataset/                 (raw tomato leaf images, per-class folders)
# Output  : dataset/dataset_split_70_30/   (train/ + test/)
#           dataset/dataset_split_80_20/   (train/ + test/)
#           dataset/dataset_split_90_10/   (train/ + test/)
# -----------------------------------------------------------------------------
#
# The dataset is split THREE times at different train/test ratios.
# Running all three lets Stage 3 & 4 compare how training data volume
# affects final model accuracy.
#
#   Split A — 70% train, 30% test   →  dataset_split_70_30/
#   Split B — 80% train, 20% test   →  dataset_split_80_20/
#   Split C — 90% train, 10% test   →  dataset_split_90_10/
#
# All splits use random_state=42 so results are fully reproducible.
# Each split produces a train/ and test/ subdirectory per class.


# -----------------------------------------------------------------------------
# STAGE 2 — FEATURE EXTRACTION
# Script  : tmt_4_feature_extraction.py
# Input   : dataset/dataset_split_*/   (the 3 splits from Stage 1)
# Output  : features_70_30/            (X_train.npy, y_train.npy,
#           features_80_20/             X_test.npy,  y_test.npy,
#           features_90_10/             class_names.json,
#                                       tsne_features_{split}.png)
# -----------------------------------------------------------------------------
#
# Instead of training a CNN from scratch, we reuse MobileNetV2 (ImageNet
# weights) as a fixed feature extractor. This is called Transfer Learning.
#
# Process for each split:
#   1. load_backbone()
#      Loads MobileNetV2 with include_top=False + GlobalAveragePooling2D.
#      ALL layers are frozen — weights do not change during this stage.
#
#   2. extract_features(dataset, backbone)
#      Feeds images in batches of 32 through the frozen backbone.
#      Input image is normalised by mobilenet_v2.preprocess_input → [-1, 1].
#      Output per image: a 1280-dimensional feature vector.
#
#   3. Saves the feature arrays as .npy files:
#         X_train.npy  shape: (n_train, 1280)
#         y_train.npy  shape: (n_train,)
#         X_test.npy   shape: (n_test,  1280)
#         y_test.npy   shape: (n_test,)
#
#   4. save_tsne_plot()
#      Runs t-SNE to project 1280-dim vectors into 2D and plots them
#      coloured by class — visually verifies that classes are separable.
#      Output: tsne_features_{split}.png
#
# Why extract features separately?
#   Feature extraction is slow (GPU-bound). Saving .npy files means
#   Stage 3 can train and re-train the classifier many times instantly
#   without re-running the backbone each time.


# -----------------------------------------------------------------------------
# STAGE 3 — MODEL TRAINING & HYPERPARAMETER SEARCH
# Script  : tmt_5_model.py
# Input   : features_{split}/   (.npy arrays from Stage 2)
# Output  : tomato_disease_{split}.h5        (trained Keras model)
#           training_history_{split}.json    (epoch-by-epoch metrics)
#           hyperparam_results_{split}.json  (HP search results)
# -----------------------------------------------------------------------------
#
# build_classifier(input_dim=1280, num_classes=4) defines the network:
#
#   Input  →  Dense(256, activation='relu')  →  Dropout(0.4)
#          →  Dense(128, activation='relu')  →  Dropout(0.3)
#          →  Dense(4,   activation='softmax')
#
#   Loss      : categorical_crossentropy
#   Metrics   : accuracy, precision, recall
#   Optimizer : Adam
#
# ── Hyperparameter Search ────────────────────────────────────────────────────
#
# Before the final 100-epoch run, a grid search tries 9 combinations of
# learning_rate × epochs to find which settings generalise best.
#
#   Learning rates tested : 0.01, 0.005, 0.001, 0.0005, 0.0001,
#                           0.0002, 3e-5, 1e-4, 1e-3
#   Epoch counts tested   : 10, 20, 30, 40, 50
#
# For each combination, the model trains and validation accuracy is recorded.
# All results are saved to hyperparam_results_{split}.json, sorted ascending
# by val_accuracy so the best combo is always the last entry.
#
# ── Final Training ───────────────────────────────────────────────────────────
#
# The full model trains for 100 epochs using the default learning rate.
# A custom LearningRateLogger callback records the LR at every epoch.
# All metrics (accuracy, val_accuracy, precision, recall, etc.) are saved
# to training_history_{split}.json for plotting in Stage 4.
#
# This is repeated for all three splits (70_30, 80_20, 90_10).


# -----------------------------------------------------------------------------
# STAGE 4 — EVALUATION & VISUALISATION
# Script  : tmt_6_evaluation_visual.py
# Input   : training_history_{split}.json   (from Stage 3)
#           hyperparam_results_{split}.json  (from Stage 3)
# Output  : 13 PNG graphs (see below)
# -----------------------------------------------------------------------------
#
# Generates the following charts:
#
#   Per split (× 3 splits = 12 graphs):
#     accuracy_{split}.png    training accuracy vs validation accuracy per epoch
#     recall_{split}.png      training recall   vs validation recall   per epoch
#     precision_{split}.png   training precision vs validation precision per epoch
#     hyperparam_{split}.png  horizontal bar chart of all 9 HP combos,
#                              best combo highlighted in amber
#
#   Cross-split (1 graph):
#     hyperparam_comparison.png   side-by-side bars of each split's best combo,
#                                  making the overall winner immediately obvious
#
# Colour scheme used across all charts:
#   Blue  = training metric
#   Red   = validation metric
#   Green = best epoch marker
#   Amber = best hyperparameter combination


# =============================================================================
# HYPERPARAMETER SEARCH RESULTS — ALL SPLITS
# =============================================================================
#
#   Split 70:30 — best combo: lr=0.0001, epochs=50  →  val_accuracy = 98.66%
#   Split 80:20 — best combo: lr=0.001,  epochs=50  →  val_accuracy = 98.75%
#   Split 90:10 — best combo: lr=0.001,  epochs=50  →  val_accuracy = 99.63%  ← WINNER
#
# Top 3 combos for the winning 90:10 split:
#   1st  lr=0.001,  ep=50  →  99.63%
#   2nd  lr=0.0001, ep=50  →  99.56%
#   3rd  lr=0.0002, ep=30  →  99.49%


# =============================================================================
# HOW THE BEST MODEL IS SELECTED — 4-STEP PROCESS
# =============================================================================
#
#   Step 1 — Hyperparameter Grid Search
#             For each of the 3 splits, 9 combos are trained and
#             val_accuracy is recorded in hyperparam_results_{split}.json.
#
#   Step 2 — Per-Split Winner
#             Within each split, the combo with the highest val_accuracy
#             is selected. Results are sorted ascending; last entry = winner.
#
#   Step 3 — Cross-Split Comparison
#             tmt_6_evaluation_visual.py plots hyperparam_comparison.png.
#             The 90:10 split (99.63%) outperforms both other splits.
#             Reason: 90% training data gives the classifier more examples
#             while the 10% test set is still a fair held-out evaluation.
#
#   Step 4 — Production Deployment
#             tmt_main.py is hardcoded to load tomato_disease_90_10.h5.
#             This model is served via HTTP at 127.0.0.1:8000.


# =============================================================================
# BEST MODEL — FINAL SPECIFICATION
# =============================================================================
#
#   Split              : 90:10  (90% train / 10% test)
#   Model file         : tomato_disease_90_10.h5
#   Backbone           : MobileNetV2 (ImageNet, frozen) → 1280-dim features
#   Classifier         : Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3)
#                        → Dense(4, softmax)
#   Best LR            : 0.001
#   Best epochs (HP)   : 50
#   Final training     : 100 epochs
#   Batch size         : 32
#   Optimiser          : Adam
#   Loss               : Categorical cross-entropy
#   Val accuracy       : 99.63%
#   Train accuracy     : ~99.8%  (epoch 100)


# =============================================================================
# INFERENCE SERVER
# Script  : tmt_main.py
# Run     : python tmt_main.py
# URL     : http://127.0.0.1:8000
# =============================================================================
#
#   GET  /            → serves test.html (web UI for uploading leaf images)
#   POST /predict     → accepts JSON { "image": "<base64-encoded image>" }
#                        returns JSON { "class": "...",
#                                       "confidence": 97.3,
#                                       "probabilities": { ... } }
#
# Inference flow inside _predict(image_bytes):
#   1. Decode base64 image bytes → PIL Image
#   2. Resize to (224, 224)
#   3. Run through frozen MobileNetV2 → 1280-dim feature vector
#   4. Pass feature vector into loaded Dense classifier
#   5. Return class name + softmax confidence scores
#
# The server loads tomato_disease_90_10.h5 once at startup (_load_model())
# and keeps it in memory for all subsequent requests.
