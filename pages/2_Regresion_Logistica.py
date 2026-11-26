"""Página 2: Regresión Logística (Clasificación binaria y multiclase)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import sklearn
import streamlit as st
from packaging.version import Version
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils.datasets import render_dataset_sidebar
from utils.preprocessing import build_preprocessor, get_transformed_feature_names, split_dataset
from utils.ui import (
    configure_page,
    render_eda_tab,
    render_metrics_comparison,
    render_practice_tab,
    render_split_sidebar,
    render_standard_eval_charts,
    render_what_if_predictor,
)
from utils.visualization import plot_decision_boundary_2d, plot_feature_importance

configure_page("Regresión Logística", icon="🎯")

st.title("🎯 Regresión Logística")
st.caption("Área de aprendizaje supervisado: **Clasificación** (Probabilidades, Función Sigmoide y Regularización)")

PAGE_KEY = "logistic_reg"

# --- SIDEBAR ---
bundle = render_dataset_sidebar(
    task="classification", default_dataset_key="breast_cancer", page_key=PAGE_KEY
)
test_size, random_state, scale_features = render_split_sidebar(PAGE_KEY, default_scale=True)

st.sidebar.subheader("3. Hiperparámetros del Modelo")
solver = st.sidebar.selectbox(
    "Optimizador (solver)",
    options=["lbfgs", "liblinear", "saga"],
    key=f"{PAGE_KEY}_solver",
    help="lbfgs es rápido para L2; liblinear soporta L1/L2 en datasets pequeños; saga soporta L1, L2 y ElasticNet.",
)

penalty_options_map = {
    "lbfgs": ["l2", "Ninguna"],
    "liblinear": ["l2", "l1"],
    "saga": ["l2", "l1", "elasticnet", "Ninguna"],
}
penalty_choice = st.sidebar.selectbox(
    "Regularización (penalty)",
    options=penalty_options_map[solver],
    key=f"{PAGE_KEY}_penalty_{solver}",
)

C_val = st.sidebar.select_slider(
    "Inverso de regularización (C)",
    options=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    value=1.0,
    key=f"{PAGE_KEY}_C",
    help="Valores pequeños de C aplican regularización fuerte; valores grandes relajan la regularización.",
)

l1_ratio = 0.5
if penalty_choice == "elasticnet":
    l1_ratio = st.sidebar.slider(
        "Proporción L1 (l1_ratio)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key=f"{PAGE_KEY}_l1_ratio",
    )

max_iter = st.sidebar.slider(
    "Máximo de iteraciones (max_iter)",
    min_value=100,
    max_value=3000,
    value=1000,
    step=100,
    key=f"{PAGE_KEY}_max_iter",
)

class_weight_opt = st.sidebar.selectbox(
    "Balanceo de clases (class_weight)",
    options=["None", "balanced"],
    key=f"{PAGE_KEY}_class_weight",
    help="'balanced' ajusta pesos inversamente proporcionales a la frecuencia de cada clase.",
)

threshold = st.sidebar.slider(
    "Umbral de decisión (solo clasificación binaria)",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    key=f"{PAGE_KEY}_threshold",
    help="Probabilidad mínima para asignar la clase positiva (por defecto 0.50).",
)

# --- TABS ---
tab_theory, tab_eda, tab_train, tab_predict, tab_practice = st.tabs(
    [
        "📘 Teoría y Conceptos",
        "🔍 Exploración (EDA)",
        "⚙️ Entrenamiento y Evaluación",
        "🧪 Predicción Manual",
        "🎯 Ejercicios y Práctica",
    ]
)

with tab_theory:
    st.subheader("¿Qué es la Regresión Logística?")
    st.markdown(
        r"""
        A pesar de su nombre, la **Regresión Logística** es un algoritmo de **clasificación**.
        Estima la probabilidad de pertenencia a una clase aplicando la **función sigmoide** sobre una combinación lineal:

        $$P(y = 1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \beta_0 + \mathbf{w}^\top \mathbf{x}$$

        Para problemas **multiclase**, generaliza mediante la función **Softmax**.
        """
    )

    # Interactive sigmoid illustration
    z_vals = np.linspace(-8, 8, 200)
    sig_vals = 1.0 / (1.0 + np.exp(-z_vals))
    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(x=z_vals, y=sig_vals, mode="lines", name="σ(z)", line=dict(width=3, color="#2563EB")))
    fig_sig.add_hline(y=threshold, line_dash="dash", line_color="#DC2626", annotation_text=f"Umbral = {threshold:.2f}")
    fig_sig.update_layout(
        title="Función Sigmoide y Umbral de Clasificación",
        xaxis_title="Combinación lineal (z)",
        yaxis_title="Probabilidad P(y=1|x)",
        height=300,
    )
    st.plotly_chart(fig_sig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            #### Hiperparámetros Clave
            - **`C`**: controla la regularización ($C = 1/\\lambda$). Menor `C` = mayor regularización.
            - **`penalty`**: `l2` reduce pesos de forma suave; `l1` promueve coeficientes exactamente en cero.
            - **Umbral**: en problemas desbalanceados o críticos (ej. diagnóstico médico), ajustar el umbral permite priorizar *Recall* o *Precision*.
            """
        )
    with c2:
        st.markdown(
            """
            #### Ventajas y Limitaciones
            - **Ventajas**: entrega probabilidades calibradas, es interpretable (log-odds), rápido y eficiente.
            - **Limitaciones**: su frontera de decisión es **lineal** en el espacio original de variables.
            """
        )

if bundle is None:
    st.stop()

with tab_eda:
    render_eda_tab(bundle)

X_train, X_test, y_train, y_test = split_dataset(
    bundle.X, bundle.y, task="classification", test_size=test_size, random_state=random_state
)

cw = None if class_weight_opt == "None" else "balanced"
lr_kwargs: dict[str, object] = {
    "solver": solver,
    "max_iter": int(max_iter),
    "class_weight": cw,
    "random_state": random_state,
}

if Version(sklearn.__version__) >= Version("1.8"):
    if penalty_choice == "Ninguna":
        lr_kwargs["C"] = np.inf
    else:
        lr_kwargs["C"] = float(C_val)
        if penalty_choice == "l1":
            lr_kwargs["l1_ratio"] = 1.0
        elif penalty_choice == "l2":
            lr_kwargs["l1_ratio"] = 0.0
        elif penalty_choice == "elasticnet":
            lr_kwargs["l1_ratio"] = float(l1_ratio)
else:
    if penalty_choice == "Ninguna":
        lr_kwargs["penalty"] = None
    else:
        lr_kwargs["penalty"] = penalty_choice
        lr_kwargs["C"] = float(C_val)
        if penalty_choice == "elasticnet":
            lr_kwargs["l1_ratio"] = float(l1_ratio)

estimator = LogisticRegression(**lr_kwargs)

preprocessor = build_preprocessor(bundle.X, scale_numeric=scale_features)
pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
pipeline.fit(X_train, y_train)

y_train_proba = pipeline.predict_proba(X_train)
y_test_proba = pipeline.predict_proba(X_test)
is_binary = len(np.unique(bundle.y)) == 2

if is_binary:
    classes_sorted = pipeline.named_steps["model"].classes_
    y_train_pred = np.where(y_train_proba[:, 1] >= threshold, classes_sorted[1], classes_sorted[0])
    y_test_pred = np.where(y_test_proba[:, 1] >= threshold, classes_sorted[1], classes_sorted[0])
else:
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

with tab_train:
    if is_binary and not np.isclose(threshold, 0.50):
        st.info(f"Clasificación binaria evaluada con umbral personalizado: **{threshold:.2f}**.")

    render_metrics_comparison(
        "classification",
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        y_train_score=y_train_proba,
        y_test_score=y_test_proba,
    )
    render_standard_eval_charts(
        "classification", y_test, y_test_pred, y_test_score=y_test_proba, class_names=bundle.class_names
    )

    # 2D Decision Boundary
    X_train_proc = pipeline.named_steps["preprocessor"].transform(X_train)
    X_test_proc = pipeline.named_steps["preprocessor"].transform(X_test)
    feat_names_out = get_transformed_feature_names(pipeline.named_steps["preprocessor"], bundle.feature_names)

    fig_boundary = plot_decision_boundary_2d(
        pipeline.named_steps["model"],
        X_train_proc,
        y_train,
        X_test_proc,
        y_test,
        feat_names_out,
        bundle.class_names,
    )
    if fig_boundary is not None:
        st.plotly_chart(fig_boundary, width="stretch")

    # Coefficients
    fitted = pipeline.named_steps["model"]
    if hasattr(fitted, "coef_"):
        coef_arr = fitted.coef_[0] if fitted.coef_.shape[0] == 1 else np.mean(np.abs(fitted.coef_), axis=0)
        title_coef = (
            "Coeficientes Log-Odds (Clase Positiva)"
            if fitted.coef_.shape[0] == 1
            else "Magnitud Media Absoluta de Coeficientes (Multiclase)"
        )
        if len(coef_arr) == len(feat_names_out):
            st.plotly_chart(
                plot_feature_importance(feat_names_out, coef_arr, title=title_coef),
                width="stretch",
            )

with tab_predict:
    render_what_if_predictor(pipeline, bundle, PAGE_KEY)

with tab_practice:
    render_practice_tab(
        challenges=[
            "En **Breast Cancer**, mueve el **umbral de decisión** de 0.50 a 0.25 y luego a 0.80. Observa el intercambio entre Falsos Positivos y Falsos Negativos en la Matriz de Confusión.",
            "Cambia al dataset **Moons 2D** o **Circles 2D**. ¿Por qué la Regresión Logística no logra separar bien las clases incluso con `C = 100`?",
            "En **Breast Cancer**, selecciona `solver = liblinear`, `penalty = l1` y reduce `C` a `0.05`. Observa qué variables conservan coeficientes distintos de cero.",
        ],
        quiz_questions=[
            {
                "question": "¿Qué ocurre cuando disminuyes el hiperparámetro `C` en LogisticRegression?",
                "options": [
                    "Aumenta la fuerza de la regularización (modelo más simple).",
                    "Disminuye la fuerza de la regularización (modelo más complejo).",
                    "Cambia el modelo a un árbol de decisión.",
                ],
                "answer": "Aumenta la fuerza de la regularización (modelo más simple).",
                "explanation": "En scikit-learn, C es el inverso de la fuerza de regularización (C = 1/λ).",
            },
            {
                "question": "¿Qué forma tiene la frontera de decisión de la Regresión Logística sin ingeniería de características?",
                "options": [
                    "Una frontera no lineal en forma de árbol.",
                    "Un hiperplano lineal.",
                    "Círculos concéntricos.",
                ],
                "answer": "Un hiperplano lineal.",
                "explanation": "La decisión depende del signo de w·x + b = 0, que define una recta o hiperplano lineal.",
            },
        ],
        page_key=PAGE_KEY,
    )
