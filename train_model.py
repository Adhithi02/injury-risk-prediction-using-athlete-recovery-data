import os
import json
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from metrics import concordance_index


PROCESSED_PATH = os.path.join("data", "processed", "master_dataset.csv")
MODELS_DIR = os.path.join("models")
REPORTS_DIR = os.path.join("reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")

# Threshold tuning config: maximize precision subject to minimum recall
MIN_RECALL_FLOOR = 0.20


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if "label_next_10_15" not in df.columns:
        raise ValueError("Processed dataset missing 'label_next_10_15' column. Run preprocessing first.")

    # Date handling
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Select numeric feature columns (exclude label and injury column to prevent data leakage)
    feature_cols = [c for c in df.columns if c not in ["date", "label_next_10_15", "injury"]]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].copy()
    y = df["label_next_10_15"].astype(int).copy()

    # Fill NaNs with column medians (robust for time-series gaps)
    X = X.fillna(X.median())

    return X, y, feature_cols


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    n = len(df)
    split_idx = int(n * (1 - test_fraction))
    train_idx = np.arange(0, split_idx)
    test_idx = np.arange(split_idx, n)
    return train_idx, test_idx


def _select_threshold_for_precision(y_true: np.ndarray, y_score: np.ndarray, min_recall: float) -> float:
    # Use precision-recall curve thresholds to choose threshold with highest precision given recall constraint
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns precision/recall for len(thresholds)+1; align by trimming first entry
    precision_arr = precision_arr[1:]
    recall_arr = recall_arr[1:]
    if len(thresholds) == 0:
        return 0.5
    # Mask by recall constraint
    mask = recall_arr >= min_recall
    if not np.any(mask):
        # fallback: use threshold at maximum precision regardless of recall
        best_idx = int(np.argmax(precision_arr))
        return float(thresholds[best_idx])
    candidate_idxs = np.where(mask)[0]
    # Among candidates, pick max precision; tie-breaker: higher threshold
    best_idx = candidate_idxs[np.argmax(precision_arr[candidate_idxs])]
    return float(thresholds[best_idx])


def _plot_roc_pr_confusion(y_true: np.ndarray, y_score: np.ndarray, threshold: float, model_name: str) -> Dict[str, str]:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    saved_paths: Dict[str, str] = {}

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    roc_path = os.path.join(PLOTS_DIR, f"{model_name}_roc.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()
    saved_paths["roc"] = roc_path

    # PR Curve
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(recall_arr, precision_arr, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend(loc="upper right")
    pr_path = os.path.join(PLOTS_DIR, f"{model_name}_pr.png")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()
    saved_paths["pr"] = pr_path

    # Confusion Matrix at threshold
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.title(f"Confusion Matrix @ thr={threshold:.2f} - {model_name}")
    cm_path = os.path.join(PLOTS_DIR, f"{model_name}_confusion.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    saved_paths["confusion"] = cm_path

    return saved_paths


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, dates: pd.Series, feature_cols: List[str]):
    train_idx, test_idx = time_based_split(X, test_fraction=0.2)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Pipelines
    scaler = StandardScaler()

    lr_clf = Pipeline([
        ("scaler", scaler),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=1, class_weight="balanced")),
    ])

    rf_clf = Pipeline([
        ("scaler", scaler),  # scaling is harmless; RF is tree-based
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )),
    ])

    models = {
        "logreg": lr_clf,
        "random_forest": rf_clf,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    results = {}
    best_name = None
    best_auc = -1.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        # Metrics independent of threshold
        roc_auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else float("nan")
        c_index = concordance_index(y_test.values, proba)
        ap = average_precision_score(y_test, proba)

        # Choose threshold to maximize precision with recall floor
        threshold = _select_threshold_for_precision(y_test.values, proba, MIN_RECALL_FLOOR)
        y_pred = (proba >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )

        # Plot curves and confusion
        plot_paths = _plot_roc_pr_confusion(y_test.values, proba, threshold, name)

        results[name] = {
            "roc_auc": float(roc_auc),
            "c_index": float(c_index),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": float(threshold),
            "plots": plot_paths,
        }

        if not np.isnan(roc_auc) and roc_auc > best_auc:
            best_auc = roc_auc
            best_name = name

        # Save per-model report with additional convenience fields for UI
        per_model = {
            **results[name],
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "average_precision": float(ap),
            "model": name,
        }
        with open(os.path.join(REPORTS_DIR, f"metrics_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(per_model, f, indent=2)

        joblib.dump(model, os.path.join(MODELS_DIR, f"model_{name}.joblib"))

    # Save best model alias and metadata
    assert best_name is not None, "Model selection failed."
    best_model_path = os.path.join(MODELS_DIR, "model_best.joblib")
    joblib.dump(models[best_name], best_model_path)

    metadata = {
        "best_model": best_name,
        "feature_columns": feature_cols,
        "best_threshold": results[best_name]["threshold"],
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Also save a concise overall report
    with open(os.path.join(REPORTS_DIR, "metrics_overall.json"), "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, **results[best_name]}, f, indent=2)

    return best_name, results


def main():
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(
            f"Processed dataset not found at {PROCESSED_PATH}. Run data_preprocessing.py first."
        )

    df = pd.read_csv(PROCESSED_PATH)
    X, y, feature_cols = prepare_features(df)
    dates = pd.to_datetime(df["date"]) if "date" in df.columns else pd.Series(np.arange(len(df)))

    best_name, results = train_and_evaluate(X, y, dates, feature_cols)
    print(f"Training complete. Best model: {best_name}")
    print(json.dumps(results[best_name], indent=2))


if __name__ == "__main__":
    main()


