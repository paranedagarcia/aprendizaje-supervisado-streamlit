"""Gráficos interactivos con Plotly para EDA, clasificación, regresión y fronteras de decisión."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure | None:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None
    # Limit to top 15 columns for readability
    if num_df.shape[1] > 15:
        num_df = num_df.iloc[:, :15]
    corr = num_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Matriz de Correlación (Variables Numéricas)",
    )
    fig.update_layout(height=460, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_target_distribution(y: np.ndarray, task: str, target_name: str, class_names: list[str] | None = None) -> go.Figure:
    if task == "classification":
        labels = [class_names[int(v)] if class_names and int(v) < len(class_names) else str(v) for v in y]
        counts = pd.Series(labels).value_counts().reset_index()
        counts.columns = [target_name, "Cantidad"]
        fig = px.bar(
            counts,
            x=target_name,
            y="Cantidad",
            color=target_name,
            text="Cantidad",
            title=f"Distribución de Clases ({target_name})",
        )
        fig.update_layout(showlegend=False, height=360)
        return fig

    fig = px.histogram(
        x=y,
        nbins=30,
        marginal="box",
        labels={"x": target_name},
        title=f"Distribución de la Variable Objetivo ({target_name})",
    )
    fig.update_layout(height=360)
    return fig


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Real vs. Predicho (Test)") -> go.Figure:
    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name="Predicciones",
            marker=dict(size=8, opacity=0.75, color="#2563EB"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Predicción perfecta (y = ŷ)",
            line=dict(color="#DC2626", dash="dash", width=2),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Valor Real (y)",
        yaxis_title="Valor Predicho (ŷ)",
        height=400,
    )
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            name="Residuos",
            marker=dict(size=8, opacity=0.75, color="#0D9488"),
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#DC2626")
    fig.update_layout(
        title="Gráfico de Residuos (y - ŷ vs. ŷ)",
        xaxis_title="Valor Predicho (ŷ)",
        yaxis_title="Residuo (y - ŷ)",
        height=400,
    )
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> go.Figure:
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    display_labels = [
        class_names[i] if class_names and i < len(class_names) else str(i) for i in labels
    ]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(
        cm,
        x=display_labels,
        y=display_labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Clase Predicha", y="Clase Real", color="Muestras"),
        title="Matriz de Confusión (Conjunto de Prueba)",
    )
    fig.update_layout(height=400)
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str] | None = None,
) -> go.Figure | None:
    classes = np.unique(y_true)
    fig = go.Figure()

    if len(classes) == 2:
        score_pos = y_score[:, 1] if y_score.ndim == 2 else y_score
        fpr, tpr, _ = roc_curve(y_true, score_pos, pos_label=classes[1])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC (AUC = {roc_auc:.3f})",
                line=dict(width=2.5, color="#2563EB"),
            )
        )
    elif y_score.ndim == 2 and y_score.shape[1] == len(classes):
        y_bin = label_binarize(y_true, classes=classes)
        for idx, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, idx], y_score[:, idx])
            roc_auc = auc(fpr, tpr)
            label = class_names[int(cls)] if class_names and int(cls) < len(class_names) else f"Clase {cls}"
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{label} (AUC = {roc_auc:.3f})",
                )
            )
    else:
        return None

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Azar (AUC = 0.50)",
            line=dict(color="gray", dash="dash"),
        )
    )
    fig.update_layout(
        title="Curva ROC (Receiver Operating Characteristic)",
        xaxis_title="Tasa de Falsos Positivos (FPR)",
        yaxis_title="Tasa de Verdaderos Positivos (TPR)",
        height=400,
    )
    return fig


def plot_feature_importance(
    names: list[str],
    values: np.ndarray,
    title: str = "Importancia / Coeficientes de Variables",
    top_n: int = 15,
) -> go.Figure:
    df_imp = pd.DataFrame({"Variable": names, "Valor": values})
    df_imp["Abs"] = df_imp["Valor"].abs()
    df_imp = df_imp.sort_values("Abs", ascending=False).head(top_n).sort_values("Valor")
    colors = ["#2563EB" if v >= 0 else "#DC2626" for v in df_imp["Valor"]]

    fig = go.Figure(
        go.Bar(
            x=df_imp["Valor"],
            y=df_imp["Variable"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in df_imp["Valor"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Magnitud",
        yaxis_title="Variable",
        height=max(320, 28 * len(df_imp) + 100),
        margin=dict(l=20, r=40, t=50, b=20),
    )
    return fig


def plot_decision_boundary_2d(
    estimator,
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    X_test_proc: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    class_names: list[str] | None = None,
    highlight_support_vectors: bool = False,
) -> go.Figure | None:
    """Visualiza la frontera de decisión en 2D (directamente si hay 2 variables o vía PCA 2D)."""
    if X_train_proc.shape[1] < 2:
        return None

    if X_train_proc.shape[1] == 2:
        X_tr_2d = X_train_proc
        X_te_2d = X_test_proc
        axis_x = feature_names[0] if len(feature_names) >= 1 else "Dim 1"
        axis_y = feature_names[1] if len(feature_names) >= 2 else "Dim 2"
        title_suffix = ""
    else:
        pca = PCA(n_components=2, random_state=42)
        X_tr_2d = pca.fit_transform(X_train_proc)
        X_te_2d = pca.transform(X_test_proc)
        var_exp = pca.explained_variance_ratio_.sum() * 100
        axis_x = f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)"
        axis_y = f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"
        title_suffix = f" (Proyección PCA 2D: {var_exp:.1f}% varianza)"

    model_2d = clone(estimator)
    model_2d.fit(X_tr_2d, y_train)

    X_all = np.vstack([X_tr_2d, X_te_2d])
    x_min, x_max = X_all[:, 0].min() - 0.6, X_all[:, 0].max() + 0.6
    y_min, y_max = X_all[:, 1].min() - 0.6, X_all[:, 1].max() + 0.6

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 140), np.linspace(y_min, y_max, 140))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model_2d.predict(grid).reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, 140),
            y=np.linspace(y_min, y_max, 140),
            z=Z,
            colorscale="Portland",
            opacity=0.35,
            showscale=False,
            hoverinfo="skip",
            contours=dict(showlines=False),
        )
    )

    palette = px.colors.qualitative.Set1
    classes = np.unique(np.concatenate([y_train, y_test]))
    for idx, cls in enumerate(classes):
        color = palette[idx % len(palette)]
        label = class_names[int(cls)] if class_names and int(cls) < len(class_names) else f"Clase {cls}"

        mask_tr = y_train == cls
        fig.add_trace(
            go.Scatter(
                x=X_tr_2d[mask_tr, 0],
                y=X_tr_2d[mask_tr, 1],
                mode="markers",
                name=f"{label} (Train)",
                marker=dict(size=7, color=color, opacity=0.65, symbol="circle"),
            )
        )

        mask_te = y_test == cls
        fig.add_trace(
            go.Scatter(
                x=X_te_2d[mask_te, 0],
                y=X_te_2d[mask_te, 1],
                mode="markers",
                name=f"{label} (Test)",
                marker=dict(size=10, color=color, symbol="diamond", line=dict(width=1.5, color="black")),
            )
        )

    if highlight_support_vectors and hasattr(model_2d, "support_vectors_"):
        sv = model_2d.support_vectors_
        fig.add_trace(
            go.Scatter(
                x=sv[:, 0],
                y=sv[:, 1],
                mode="markers",
                name=f"Vectores de Soporte ({len(sv)})",
                marker=dict(size=13, color="rgba(0,0,0,0)", line=dict(width=2, color="black")),
            )
        )

    fig.update_layout(
        title=f"Frontera de Decisión 2D{title_suffix}",
        xaxis_title=axis_x,
        yaxis_title=axis_y,
        height=470,
    )
    return fig
