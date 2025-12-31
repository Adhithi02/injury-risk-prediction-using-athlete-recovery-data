import numpy as np
from typing import Tuple, Dict, Any
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix


def concordance_index(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes the Concordance Index (C-index) for binary outcomes.

    For a pair of samples (i, j), the prediction is concordant if the sample
    with higher true outcome (injury==1) has a higher predicted risk score.
    Ties count as 0.5.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_score: Predicted risk scores (continuous values)
        
    Returns:
        C-index value between 0 and 1 (0.5 = random, 1.0 = perfect)
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    n = len(y_true)
    concordant = 0.0
    comparable = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            comparable += 1
            if y_score[i] == y_score[j]:
                concordant += 0.5
            elif (y_true[i] > y_true[j] and y_score[i] > y_score[j]) or (
                y_true[j] > y_true[i] and y_score[j] > y_score[i]
            ):
                concordant += 1.0

    if comparable == 0:
        return float("nan")
    return concordant / comparable


def compute_all_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for injury prediction.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_score: Predicted risk scores (continuous values 0-1)
        threshold: Classification threshold for binary predictions
        
    Returns:
        Dictionary containing all metrics
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    
    metrics = {}
    
    # Primary metric: C-index (Concordance Index)
    metrics['c_index'] = concordance_index(y_true, y_score)
    
    # Secondary metrics
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics['roc_auc'] = float('nan')
    
    # Classification metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    metrics['precision'] = float(precision)
    metrics['recall'] = float(recall)
    metrics['f1_score'] = float(f1)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Additional derived metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # Injury-specific metrics
    total_injuries = int(np.sum(y_true))
    total_predictions = len(y_true)
    metrics['injury_rate'] = total_injuries / total_predictions if total_predictions > 0 else 0.0
    metrics['predicted_injury_rate'] = int(np.sum(y_pred)) / total_predictions if total_predictions > 0 else 0.0
    
    return metrics


def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """Print a formatted summary of evaluation metrics."""
    print("=" * 50)
    print("INJURY PREDICTION MODEL EVALUATION")
    print("=" * 50)
    
    print(f"\n📊 PRIMARY METRICS:")
    print(f"  C-index (Concordance): {metrics['c_index']:.4f}")
    print(f"  ROC-AUC:               {metrics['roc_auc']:.4f}")
    
    print(f"\n🎯 CLASSIFICATION METRICS:")
    print(f"  Accuracy:              {metrics['accuracy']:.4f}")
    print(f"  Precision:             {metrics['precision']:.4f}")
    print(f"  Recall (Sensitivity):  {metrics['recall']:.4f}")
    print(f"  F1-Score:              {metrics['f1_score']:.4f}")
    print(f"  Specificity:           {metrics['specificity']:.4f}")
    
    print(f"\n📈 CONFUSION MATRIX:")
    print(f"  True Positives:        {metrics['true_positives']}")
    print(f"  False Positives:       {metrics['false_positives']}")
    print(f"  True Negatives:        {metrics['true_negatives']}")
    print(f"  False Negatives:       {metrics['false_negatives']}")
    
    print(f"\n🏥 INJURY STATISTICS:")
    print(f"  Actual Injury Rate:    {metrics['injury_rate']:.4f} ({metrics['injury_rate']*100:.2f}%)")
    print(f"  Predicted Injury Rate: {metrics['predicted_injury_rate']:.4f} ({metrics['predicted_injury_rate']*100:.2f}%)")
    
    print("=" * 50)


def demo_metrics():
    """Demo function to show what metrics look like with sample data."""
    print("Running metrics demo with sample injury prediction data...")
    
    # Sample data: 100 days, 10% injury rate
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # Simulate predictions: good model (C-index ~0.7)
    y_score = np.random.beta(2, 5, size=n_samples)  # Skewed towards low values
    y_score[y_true == 1] += 0.3  # Injured days get higher scores
    
    metrics = compute_all_metrics(y_true, y_score)
    print_metrics_summary(metrics)
    
    print("\n💡 This is what your actual model metrics will look like!")
    print("   Run 'python train_model.py' to see real results.")


if __name__ == "__main__":
    demo_metrics()
