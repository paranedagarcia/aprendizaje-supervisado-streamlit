"""Cálculo de métricas para evaluación de modelos de clasificación y regresión."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "R²": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if y_score is not None:
        try:
            classes = np.unique(y_true)
            if len(classes) == 2:
                score_pos = y_score[:, 1] if y_score.ndim == 2 else y_score
                metrics["ROC-AUC"] = float(roc_auc_score(y_true, score_pos))
            elif y_score.ndim == 2 and y_score.shape[1] == len(classes):
                metrics["ROC-AUC"] = float(
                    roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
                )
        except Exception:
            pass

    return metrics


def get_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    target_names = None
    if class_names is not None and len(class_names) >= len(labels):
        target_names = [class_names[i] if i < len(class_names) else str(i) for i in labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T
    return df.round(4)
