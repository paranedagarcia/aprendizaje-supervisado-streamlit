"""Componentes de interfaz reutilizables: encabezado, EDA, métricas, simulador What-If y ejercicios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils.datasets import DatasetBundle
from utils.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    get_classification_report_df,
)
from utils.visualization import (
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_residuals,
    plot_roc_curve,
    plot_target_distribution,
)


def configure_page(title: str, icon: str = "📘") -> None:
    st.set_page_config(page_title=f"{title} | Aprendizaje Supervisado", page_icon=icon, layout="wide")


def render_split_sidebar(page_key: str, default_scale: bool = True) -> tuple[float, int, bool]:
    """Controles comunes en el sidebar para train/test split y escalado."""
    st.sidebar.subheader("2. Partición y Preprocesamiento")
    test_size = st.sidebar.slider(
        "Proporción de prueba (test_size)",
        min_value=0.10,
        max_value=0.50,
        value=0.25,
        step=0.05,
        key=f"{page_key}_test_size",
        help="Fracción del dataset reservada para evaluar la generalización del modelo.",
    )
    random_state = st.sidebar.number_input(
        "Semilla de partición",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
        key=f"{page_key}_split_seed",
    )
    scale_features = st.sidebar.checkbox(
        "Estandarizar variables numéricas (StandardScaler)",
        value=default_scale,
        key=f"{page_key}_scale",
        help="Centra las variables numéricas con media 0 y desviación estándar 1.",
    )
    return float(test_size), int(random_state), bool(scale_features)


def render_eda_tab(bundle: DatasetBundle) -> None:
    """Área de exploración de datos (EDA)."""
    st.subheader(f"Dataset activo: {bundle.name}")
    st.write(bundle.description)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas (muestras)", f"{bundle.df.shape[0]}")
    c2.metric("Variables predictoras (X)", f"{len(bundle.feature_names)}")
    c3.metric("Valores nulos", f"{int(bundle.df.isna().sum().sum())}")
    c4.metric(
        "Tipo de tarea",
        "Clasificación" if bundle.task == "classification" else "Regresión",
    )

    with st.expander("Vista previa de los datos (primeras 15 filas)", expanded=True):
        st.dataframe(bundle.df.head(15), width="stretch")

    with st.expander("Estadísticas descriptivas"):
        st.dataframe(bundle.df.describe(include="all").T, width="stretch")

    col_left, col_right = st.columns(2)
    with col_left:
        fig_target = plot_target_distribution(
            bundle.y, bundle.task, bundle.target_name, bundle.class_names
        )
        st.plotly_chart(fig_target, width="stretch")

    with col_right:
        fig_corr = plot_correlation_heatmap(bundle.df)
        if fig_corr is not None:
            st.plotly_chart(fig_corr, width="stretch")
        else:
            st.info("Se necesitan al menos 2 variables numéricas para mostrar la matriz de correlación.")


def render_metrics_comparison(
    task: str,
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    y_train_score: np.ndarray | None = None,
    y_test_score: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Muestra métricas de Train vs Test con diagnóstico rápido de sobreajuste."""
    if task == "regression":
        tr = compute_regression_metrics(y_train, y_train_pred)
        te = compute_regression_metrics(y_test, y_test_pred)
    else:
        tr = compute_classification_metrics(y_train, y_train_pred, y_train_score)
        te = compute_classification_metrics(y_test, y_test_pred, y_test_score)

    st.markdown("#### Métricas de Evaluación (Prueba vs. Entrenamiento)")
    cols = st.columns(len(te))
    for col, (metric_name, test_val) in zip(cols, te.items()):
        train_val = tr.get(metric_name, test_val)
        delta = test_val - train_val
        col.metric(
            label=f"{metric_name} (Test)",
            value=f"{test_val:.4f}",
            delta=f"{delta:+.4f} vs Train ({train_val:.4f})",
            delta_color="inverse" if metric_name in {"RMSE", "MAE", "MSE"} else "normal",
        )

    # Educational overfitting/underfitting hint
    if task == "classification":
        gap = tr["Accuracy"] - te["Accuracy"]
        if gap > 0.10:
            st.warning(
                f"Posible **sobreajuste (overfitting)**: Accuracy en Train ({tr['Accuracy']:.3f}) "
                f"supera por {gap:.3f} al de Test ({te['Accuracy']:.3f}). Prueba regularizar o reducir la complejidad."
            )
        elif te["Accuracy"] < 0.65 and tr["Accuracy"] < 0.65:
            st.info(
                "Posible **subajuste (underfitting)**: el desempeño es bajo tanto en Train como en Test. "
                "Prueba aumentar la flexibilidad del modelo o revisar las variables."
            )
    else:
        gap = tr["R²"] - te["R²"]
        if gap > 0.15:
            st.warning(
                f"Posible **sobreajuste (overfitting)**: R² en Train ({tr['R²']:.3f}) "
                f"es notablemente mayor que en Test ({te['R²']:.3f})."
            )
        elif te["R²"] < 0.40 and tr["R²"] < 0.40:
            st.info(
                "Posible **subajuste (underfitting)**: R² bajo en ambos conjuntos. "
                "Prueba un modelo no lineal o revisa los hiperparámetros."
            )

    return tr, te


def render_standard_eval_charts(
    task: str,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_score: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> None:
    if task == "regression":
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_actual_vs_predicted(y_test, y_test_pred), width="stretch")
        with c2:
            st.plotly_chart(plot_residuals(y_test, y_test_pred), width="stretch")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                plot_confusion_matrix(y_test, y_test_pred, class_names),
                width="stretch",
            )
        with c2:
            if y_test_score is not None:
                fig_roc = plot_roc_curve(y_test, y_test_score, class_names)
                if fig_roc is not None:
                    st.plotly_chart(fig_roc, width="stretch")
                else:
                    st.info("Curva ROC no disponible para esta configuración de clases.")
            else:
                st.info("Este modelo no expone probabilidades ni función de decisión para trazar ROC.")

        with st.expander("Reporte detallado de clasificación (Precision, Recall, F1)"):
            st.dataframe(
                get_classification_report_df(y_test, y_test_pred, class_names),
                width="stretch",
            )


def render_what_if_predictor(pipeline, bundle: DatasetBundle, page_key: str) -> None:
    """Simulador interactivo para ingresar valores y obtener la predicción del modelo entrenado."""
    st.subheader("Simulador de Predicción Manual (What-If)")
    st.write(
        "Modifica los valores de las variables predictoras para observar cómo responde el modelo entrenado."
    )

    input_values: dict[str, object] = {}
    cols = st.columns(min(3, max(1, len(bundle.feature_names))))

    for idx, feat in enumerate(bundle.feature_names):
        col = cols[idx % len(cols)]
        series = bundle.X[feat]
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            min_v = float(clean.min())
            max_v = float(clean.max())
            mean_v = float(clean.mean())
            if min_v == max_v:
                max_v = min_v + 1.0
            input_values[feat] = col.slider(
                label=feat,
                min_value=min_v,
                max_value=max_v,
                value=mean_v,
                key=f"{page_key}_whatif_{feat}",
            )
        else:
            options = series.dropna().astype(str).unique().tolist()
            if not options:
                options = [""]
            input_values[feat] = col.selectbox(
                label=feat,
                options=options,
                key=f"{page_key}_whatif_{feat}",
            )

    sample_df = pd.DataFrame([input_values])
    pred = pipeline.predict(sample_df)[0]

    st.divider()
    if bundle.task == "regression":
        st.success(f"Predicción para **{bundle.target_name}**: `{float(pred):.4f}`")
    else:
        pred_idx = int(pred)
        pred_label = (
            bundle.class_names[pred_idx]
            if bundle.class_names and pred_idx < len(bundle.class_names)
            else str(pred_idx)
        )
        st.success(f"Clase predicha para **{bundle.target_name}**: **{pred_label}**")

        if hasattr(pipeline, "predict_proba"):
            try:
                probs = pipeline.predict_proba(sample_df)[0]
                labels = (
                    bundle.class_names
                    if bundle.class_names and len(bundle.class_names) == len(probs)
                    else [f"Clase {i}" for i in range(len(probs))]
                )
                prob_df = pd.DataFrame({"Clase": labels, "Probabilidad": probs})
                st.bar_chart(prob_df.set_index("Clase"))
            except Exception:
                pass


def render_practice_tab(
    challenges: list[str],
    quiz_questions: list[dict[str, object]],
    page_key: str,
) -> None:
    """Área de ejercicios guiados y autoevaluación interactiva."""
    st.subheader("Desafíos Prácticos de Experimentación")
    st.write("Prueba estos experimentos ajustando los controles del sidebar y observa el impacto en métricas y gráficos:")
    for i, challenge in enumerate(challenges, start=1):
        st.checkbox(f"**Reto {i}:** {challenge}", key=f"{page_key}_challenge_{i}")

    st.divider()
    st.subheader("Autoevaluación Rápida")
    for idx, q in enumerate(quiz_questions):
        st.markdown(f"**Pregunta {idx + 1}:** {q['question']}")
        choice = st.radio(
            "Selecciona una respuesta:",
            options=q["options"],
            index=None,
            key=f"{page_key}_quiz_{idx}",
            label_visibility="collapsed",
        )
        if choice is not None:
            if choice == q["answer"]:
                st.success(f"¡Correcto! {q['explanation']}")
            else:
                st.error(f"No exactamente. {q['explanation']}")
        st.write("")
