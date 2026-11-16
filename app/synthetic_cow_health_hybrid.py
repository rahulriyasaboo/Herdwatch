#!/usr/bin/env python3
"""Synthetic dairy-cow health classification with an LSTM + XGBoost model.

Usage: python synthetic_cow_health_hybrid.py

Dependencies are installed on demand if missing. TensorFlow can be a large
install; a compatible Python/OS build and internet connection may be needed.
This is a reproducible synthetic demonstration, not a reproduction of the
Kaur & Virk dataset or a validated veterinary diagnostic model.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from typing import Any

SEED = 42
N_ROWS = 10_000
N_COWS = 150
PAPER_DOI = "https://doi.org/10.1007/s43926-025-00109-5"


def ensure_dependencies() -> None:
    """Install the requested third-party packages if they are not importable."""
    packages = {
        "numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn",
        "imblearn": "imbalanced-learn", "tensorflow": "tensorflow",
        "xgboost": "xgboost", "shap": "shap",
    }
    missing = [package for module, package in packages.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        print("Installing missing dependencies: " + ", ".join(missing), file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def generate_dataset(n_rows: int = N_ROWS, seed: int = SEED):
    """Generate daily records with a 15% sick prevalence and realistic signals.

    Activity is ordinal (1 low, 2 moderate, 3 high) as a compact proxy for
    collar activity categories. Health-related values are correlated with the
    label, with overlap/noise to avoid a perfectly separable toy dataset.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    cow_ids = rng.integers(1, N_COWS + 1, n_rows)
    sick = (rng.random(n_rows) < 0.15).astype(np.int32)
    thi = rng.uniform(60, 85, n_rows)
    ventilation = rng.beta(4, 2, n_rows)
    forage = rng.beta(5, 2, n_rows)

    # Mild environmental stress effects make the generated variables related.
    heat_stress = np.clip((thi - 72) / 13, 0, 1)
    temp = rng.normal(38.85 + 0.10 * heat_stress + 0.95 * sick, 0.28, n_rows)
    pulse = rng.normal(65 + 5 * heat_stress + 23 * sick, 8.0, n_rows)
    activity_score = rng.normal(2.2 - 0.75 * sick - 0.15 * heat_stress, 0.55, n_rows)
    activity = np.clip(np.rint(activity_score), 1, 3).astype(int)
    rumination_slope = rng.normal(-0.5 - 7.0 * sick - 1.0 * heat_stress, 2.8, n_rows)
    gait = np.clip(rng.normal(0.10 + 0.27 * sick, 0.11, n_rows), 0, 1)

    frame = pd.DataFrame({
        "cow_id": cow_ids,
        "avg_temp": np.clip(temp, 37.0, 42.0),
        "avg_pulse": np.clip(pulse, 35, 150),
        "avg_activity": activity,
        "rumination_slope_6h": rumination_slope,
        "gait_asymmetry": gait,
        "THI": thi,
        "ventilation_index": ventilation,
        "forage_index": forage,
        "target": sick,
    })
    # Build a valid per-cow date order for sequence windows, then retain exactly
    # the requested number of rows. Dates are metadata and excluded from inputs.
    frame["_original_order"] = np.arange(n_rows)
    frame["day"] = frame.groupby("cow_id").cumcount()
    frame = frame.sort_values(["cow_id", "day"], kind="stable").reset_index(drop=True)
    return frame


def make_sequences(frame, feature_columns):
    """Construct 3-day windows within each cow; first two days are warm-up."""
    import numpy as np

    values = frame[feature_columns].to_numpy(dtype=np.float32)
    labels = frame["target"].to_numpy(dtype=np.int32)
    cow_ids = frame["cow_id"].to_numpy()
    sequences, row_indices = [], []
    for cow in np.unique(cow_ids):
        idx = np.flatnonzero(cow_ids == cow)  # already ordered chronologically
        for pos in range(2, len(idx)):
            window_idx = idx[pos - 2:pos + 1]
            sequences.append(values[window_idx])
            row_indices.append(idx[pos])
    row_indices = np.asarray(row_indices, dtype=int)
    return np.asarray(sequences, dtype=np.float32), labels[row_indices], row_indices


def score_predictions(y_true, y_pred, y_prob) -> dict[str, float]:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
    }


def main() -> None:
    ensure_dependencies()
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    from xgboost import XGBClassifier

    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    data = generate_dataset()
    feature_columns = ["cow_id", "avg_temp", "avg_pulse", "avg_activity",
                       "rumination_slope_6h", "gait_asymmetry", "THI",
                       "ventilation_index", "forage_index"]
    sequences, y, row_ids = make_sequences(data, feature_columns)
    aligned = data.iloc[row_ids].reset_index(drop=True)

    # Split by cow so no cow's windows leak from train into test. Test rows stay
    # at original prevalence and are preserved unmodified for final evaluation.
    cows = aligned["cow_id"].to_numpy()
    unique_cows = np.unique(cows)
    train_cows, test_cows = train_test_split(unique_cows, test_size=0.30,
                                              random_state=SEED)
    train_mask = np.isin(cows, train_cows)
    test_mask = np.isin(cows, test_cows)
    X_train_seq, y_train = sequences[train_mask], y[train_mask]
    X_test_seq, y_test = sequences[test_mask], y[test_mask]
    test_frame = aligned.loc[test_mask].copy()  # untouched, un-SMOTE'd test set

    # Scale each feature using training data only; apply the same transform to
    # all sequence time steps. SMOTE is then applied only to flattened training
    # windows. This preserves the holdout distribution and avoids leakage.
    scaler = StandardScaler()
    scaler.fit(X_train_seq.reshape(-1, len(feature_columns)))
    X_train_seq = scaler.transform(X_train_seq.reshape(-1, len(feature_columns))).reshape(X_train_seq.shape)
    X_test_seq = scaler.transform(X_test_seq.reshape(-1, len(feature_columns))).reshape(X_test_seq.shape)
    smote = SMOTE(random_state=SEED)
    X_train_flat, y_train_bal = smote.fit_resample(
        X_train_seq.reshape(len(X_train_seq), -1), y_train)
    X_train_bal = X_train_flat.reshape(-1, 3, len(feature_columns))

    # LSTM encoder: sequence-to-vector temporal representation (16 dimensions).
    inp = tf.keras.Input(shape=(3, len(feature_columns)), name="daily_window")
    x = tf.keras.layers.LSTM(32, name="lstm_encoder")(inp)
    embedding = tf.keras.layers.Dense(16, activation="relu", name="temporal_embedding")(x)
    encoder = tf.keras.Model(inp, embedding, name="lstm_encoder_model")
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="health_probability")(embedding)
    classifier = tf.keras.Model(inp, output, name="lstm_sequence_classifier")
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    classifier.fit(X_train_bal, y_train_bal, epochs=10, batch_size=64,
                   validation_split=0.1, verbose=0,
                   callbacks=[tf.keras.callbacks.EarlyStopping(
                       patience=2, restore_best_weights=True)])
    # Convert trained network to an embedding extractor.
    embedding_model = tf.keras.Model(encoder.input,
                                     encoder.get_layer("temporal_embedding").output)
    train_emb = embedding_model.predict(X_train_bal, verbose=0)
    test_emb = embedding_model.predict(X_test_seq, verbose=0)

    # Fuse the temporal embedding with static environmental/feeding indicators.
    static_cols = ["THI", "forage_index"]
    static_train = aligned.loc[train_mask, static_cols].to_numpy(dtype=np.float32)
    static_test = test_frame[static_cols].to_numpy(dtype=np.float32)
    # SMOTE created synthetic windows, so use corresponding nearest-neighbor
    # interpolation for static inputs by oversampling the complete fused vector.
    # Embeddings are extracted first, then SMOTE is applied only to fused train.
    real_train_emb = embedding_model.predict(X_train_seq, verbose=0)
    fused_real = np.concatenate([real_train_emb, static_train], axis=1)
    fused_test = np.concatenate([test_emb, static_test], axis=1)
    fused_train, y_train_bal = smote.fit_resample(fused_real, y_train)

    # Resampled data are balanced; ratio remains safe and explicit for cases
    # where SMOTE cannot exactly balance due to custom sampling modifications.
    positives = max(int(np.sum(y_train_bal == 1)), 1)
    negatives = max(int(np.sum(y_train_bal == 0)), 1)
    hybrid = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                           subsample=0.9, colsample_bytree=0.9,
                           scale_pos_weight=negatives / positives,
                           objective="binary:logistic", eval_metric="logloss",
                           random_state=SEED, n_jobs=-1)
    hybrid.fit(fused_train, y_train_bal)
    hybrid_prob = hybrid.predict_proba(fused_test)[:, 1]
    hybrid_pred = (hybrid_prob >= 0.5).astype(int)
    hybrid_metrics = score_predictions(y_test, hybrid_pred, hybrid_prob)

    # Standard tabular XGBoost benchmark with no LSTM features. It uses the
    # same cow-held-out partition and original, unresampled training rows.
    baseline = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9,
                             scale_pos_weight=float((y_train == 0).sum()) /
                                                max(int((y_train == 1).sum()), 1),
                             objective="binary:logistic", eval_metric="logloss",
                             random_state=SEED, n_jobs=-1)
    raw_train = aligned.loc[train_mask, feature_columns].to_numpy(dtype=np.float32)
    raw_test = test_frame[feature_columns].to_numpy(dtype=np.float32)
    baseline.fit(raw_train, y_train)
    baseline_prob = baseline.predict_proba(raw_test)[:, 1]
    baseline_pred = (baseline_prob >= 0.5).astype(int)
    baseline_metrics = score_predictions(y_test, baseline_pred, baseline_prob)

    names = [f"lstm_embedding_{i + 1}" for i in range(16)] + static_cols
    top = sorted(zip(names, hybrid.feature_importances_), key=lambda z: z[1], reverse=True)[:5]
    report = {
        "dataset_rows": int(len(data)),
        "sequence_samples": int(len(y)),
        "train_samples_after_smote": int(len(y_train_bal)),
        "test_samples_untouched": int(len(y_test)),
        "test_sick_prevalence": float(np.mean(y_test)),
        "hybrid_lstm_xgboost": hybrid_metrics,
        "standard_xgboost_baseline": baseline_metrics,
    }
    print("\nEvaluation metrics (JSON):")
    print(json.dumps(report, indent=2))
    print("\nTop 5 XGBoost fused-input feature importances:")
    for rank, (name, importance) in enumerate(top, start=1):
        print(f"{rank}. {name}: {importance:.6f}")

    print("\nMarkdown comparison table (metrics are proportions except paper values):")
    print("| Model | Accuracy | Precision | F1 | DOI / source |")
    print("|---|---:|---:|---:|---|")
    print(f"| Hybrid LSTM + XGBoost (synthetic test) | {hybrid_metrics['accuracy']:.4f} | {hybrid_metrics['precision']:.4f} | {hybrid_metrics['f1']:.4f} | Synthetic experiment |")
    print(f"| Standard XGBoost (synthetic test) | {baseline_metrics['accuracy']:.4f} | {baseline_metrics['precision']:.4f} | {baseline_metrics['f1']:.4f} | Synthetic experiment |")
    print(f"| Random Forest (Kaur & Virk, 2025; reported) | 0.9246 | Not supplied | 0.9236 | {PAPER_DOI} |")
    print("\nNote: Kaur & Virk report RF precision of 92.36% alongside accuracy 92.46% and F1 92.36%; the table above includes the requested columns and source DOI.")


if __name__ == "__main__":
    main()
