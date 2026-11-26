"""Página 5: Support Vector Machine (SVC y SVR)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

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
from utils.visualization import plot_decision_boundary_2d

configure_page("Support Vector Machine", icon="🧭")

st.title("🧭 Support Vector Machine (SVM)")
st.caption("Área de aprendizaje supervisado: **Clasificación (SVC) y Regresión (SVR)** (Margen máximo y Kernels)")

PAGE_KEY = "svm"

# --- SIDEBAR ---
st.sidebar.subheader("0. Tipo de Problema")
task_label = st.sidebar.radio(
    "Selecciona la tarea",
    options=["Clasificación (SVC)", "Regresión (SVR)"],
    key=f"{PAGE_KEY}_task",
)
task = "classification" if "SVC" in task_label else "regression"
default_ds = "moons" if task == "classification" else "poly_1d"

bundle = render_dataset_sidebar(task=task, default_dataset_key=default_ds, page_key=f"{PAGE_KEY}_{task}")
test_size, random_state, scale_features = render_split_sidebar(PAGE_KEY, default_scale=True)

if not scale_features:
    st.sidebar.warning(
        "SVM es muy sensible a la escala de las variables. Se recomienda mantener activado `StandardScaler`."
    )

st.sidebar.subheader("3. Hiperparámetros de SVM")
kernel = st.sidebar.selectbox(
    "Kernel",
    options=["rbf", "linear", "poly", "sigmoid"],
    key=f"{PAGE_KEY}_kernel",
    help="rbf proyecta a un espacio no lineal de dimensión infinita; linear busca un hiperplano recto; poly usa polinomios.",
)

C_val = st.sidebar.select_slider(
    "Parámetro de regularización (C)",
    options=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
    value=1.0,
    key=f"{PAGE_KEY}_C",
    help="Controla el equilibrio entre margen ancho (C bajo) y penalizar errores de entrenamiento (C alto).",
)

gamma: str | float = "scale"
if kernel in {"rbf", "poly", "sigmoid"}:
    gamma_mode = st.sidebar.radio(
        "Modo de gamma",
        options=["scale", "auto", "Manual"],
        key=f"{PAGE_KEY}_gamma_mode",
        help="Define el alcance de influencia de cada vector de soporte (mayor gamma = frontera más ajustada).",
    )
    if gamma_mode == "Manual":
        gamma = float(
            st.sidebar.select_slider(
                "Valor de gamma",
                options=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                value=0.5,
                key=f"{PAGE_KEY}_gamma_val",
            )
        )
    else:
        gamma = gamma_mode

degree = 3
if kernel == "poly":
    degree = st.sidebar.slider("Grado polinomial (degree)", 2, 6, 3, key=f"{PAGE_KEY}_degree")

epsilon = 0.1
if task == "regression":
    epsilon = st.sidebar.slider(
        "Tubo de tolerancia (epsilon)",
        min_value=0.01,
        max_value=2.0,
        value=0.1,
        step=0.05,
        key=f"{PAGE_KEY}_epsilon",
        help="Margen de error dentro del cual SVR no penaliza las desviaciones.",
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
    st.subheader("¿Qué son las Máquinas de Vectores de Soporte (SVM)?")
    st.markdown(
        """
        Una **SVM** busca el hiperplano que separa las clases maximizando el **margen** (la distancia hacia los
        puntos más cercanos de cada clase, llamados **vectores de soporte**).

        - **Truco del Kernel (*Kernel Trick*)**: permite transformar implícitamente los datos a un espacio de mayor
          dimensión (por ejemplo con kernel `rbf` o `poly`) para resolver problemas no lineales como **Moons** o **Circles**.
        - **SVR (Regresión)**: busca una función lo más plana posible tal que la mayoría de los puntos caigan
          dentro de un tubo de ancho $\\pm \\varepsilon$ (*epsilon-insensitive tube*).
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            #### Hiperparámetros Principales
            - **`C`**: C pequeño = margen más ancho y mayor tolerancia a errores (menos overfitting); C alto = ajusta estrictamente los datos de entrenamiento.
            - **`gamma`** (en RBF/Poly): gamma bajo = frontera suave; gamma alto = "islas" alrededor de cada punto (alto riesgo de overfitting).
            - **`epsilon`** (en SVR): define el ancho de la zona sin penalización.
            """
        )
    with c2:
        st.markdown(
            """
            #### Ventajas y Limitaciones
            - **Ventajas**: muy efectivo en alta dimensionalidad y con fronteras no lineales complejas.
            - **Limitaciones**: **requiere escalado** de variables, es costoso computacionalmente en datasets muy grandes y sus hiperparámetros (`C`, `gamma`) requieren ajuste cuidadoso.
            """
        )

if bundle is None:
    st.stop()

with tab_eda:
    render_eda_tab(bundle)

X_train, X_test, y_train, y_test = split_dataset(
    bundle.X, bundle.y, task=task, test_size=test_size, random_state=random_state
)

if task == "classification":
    estimator = SVC(
        C=float(C_val),
        kernel=kernel,
        gamma=gamma,
        degree=int(degree),
        probability=True,
        random_state=random_state,
    )
else:
    estimator = SVR(
        C=float(C_val),
        kernel=kernel,
        gamma=gamma,
        degree=int(degree),
        epsilon=float(epsilon),
    )

preprocessor = build_preprocessor(bundle.X, scale_numeric=scale_features)
pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
y_train_score = pipeline.predict_proba(X_train) if task == "classification" else None
y_test_score = pipeline.predict_proba(X_test) if task == "classification" else None

fitted_svm = pipeline.named_steps["model"]
feat_names_out = get_transformed_feature_names(pipeline.named_steps["preprocessor"], bundle.feature_names)

with tab_train:
    render_metrics_comparison(
        task, y_train, y_train_pred, y_test, y_test_pred, y_train_score=y_train_score, y_test_score=y_test_score
    )

    n_sv = len(fitted_svm.support_)
    pct_sv = (n_sv / len(X_train)) * 100
    st.info(
        f"**Vectores de soporte utilizados:** `{n_sv}` de `{len(X_train)}` muestras de entrenamiento ({pct_sv:.1f}%)."
    )

    render_standard_eval_charts(
        task, y_test, y_test_pred, y_test_score=y_test_score, class_names=bundle.class_names
    )

    if task == "classification":
        X_tr_proc = pipeline.named_steps["preprocessor"].transform(X_train)
        X_te_proc = pipeline.named_steps["preprocessor"].transform(X_test)
        fig_b = plot_decision_boundary_2d(
            fitted_svm,
            X_tr_proc,
            y_train,
            X_te_proc,
            y_test,
            feat_names_out,
            bundle.class_names,
            highlight_support_vectors=True,
        )
        if fig_b is not None:
            st.plotly_chart(fig_b, width="stretch")
    elif len(bundle.feature_names) == 1 and pd.api.types.is_numeric_dtype(bundle.X.iloc[:, 0]):
        feat_name = bundle.feature_names[0]
        x_grid = np.linspace(float(bundle.X[feat_name].min()), float(bundle.X[feat_name].max()), 250)
        y_grid = pipeline.predict(pd.DataFrame({feat_name: x_grid}))
        fig_1d = go.Figure()
        fig_1d.add_trace(go.Scatter(x=X_train[feat_name], y=y_train, mode="markers", name="Train", opacity=0.65))
        fig_1d.add_trace(
            go.Scatter(
                x=X_test[feat_name],
                y=y_test,
                mode="markers",
                name="Test",
                marker=dict(symbol="diamond", size=9),
            )
        )
        fig_1d.add_trace(
            go.Scatter(x=x_grid, y=y_grid, mode="lines", name=f"SVR ({kernel})", line=dict(color="#DC2626", width=3))
        )
        fig_1d.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_grid + epsilon,
                mode="lines",
                name=f"+ε ({epsilon})",
                line=dict(color="#94A3B8", dash="dot"),
            )
        )
        fig_1d.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_grid - epsilon,
                mode="lines",
                name=f"-ε ({epsilon})",
                line=dict(color="#94A3B8", dash="dot"),
            )
        )
        fig_1d.update_layout(
            title=f"Curva SVR 1D con Tubo ±ε (kernel={kernel}, C={C_val})",
            xaxis_title=feat_name,
            yaxis_title=bundle.target_name,
            height=430,
        )
        st.plotly_chart(fig_1d, width="stretch")

with tab_predict:
    render_what_if_predictor(pipeline, bundle, f"{PAGE_KEY}_{task}")

with tab_practice:
    render_practice_tab(
        challenges=[
            "En **Circles 2D**, compara el kernel `linear` frente a `rbf`. ¿Por qué el kernel `rbf` separa los círculos concéntricos casi perfectamente?",
            "En **Moons 2D** con kernel `rbf`, pon `gamma` en modo **Manual** y prueba `0.05`, `1.0` y `10.0`. Observa la frontera de decisión y el número de vectores de soporte.",
            "Cambia a **Regresión (SVR)** con **Curva Polinomial 1D** y aumenta `epsilon` de `0.1` a `1.5`. Observa cómo se ensancha el tubo ±ε y disminuye el número de vectores de soporte.",
        ],
        quiz_questions=[
            {
                "question": "¿Por qué es crucial escalar las variables antes de entrenar una SVM?",
                "options": [
                    "Porque SVM optimiza distancias y productos internos; variables con rangos mayores dominarían el margen y el kernel.",
                    "Porque SVM solo acepta números entre 0 y 1.",
                    "No es necesario; SVM es invariante a la escala.",
                ],
                "answer": "Porque SVM optimiza distancias y productos internos; variables con rangos mayores dominarían el margen y el kernel.",
                "explanation": "El cálculo del margen y del kernel RBF depende directamente de la norma euclidiana entre muestras.",
            },
            {
                "question": "Con kernel RBF, ¿qué sucede si eliges un valor muy alto de `gamma` y `C`?",
                "options": [
                    "El modelo se vuelve muy rígido y subajusta.",
                    "La frontera rodea estrechamente los puntos de entrenamiento, causando sobreajuste.",
                    "El modelo se convierte en una regresión lineal.",
                ],
                "answer": "La frontera rodea estrechamente los puntos de entrenamiento, causando sobreajuste.",
                "explanation": "Un gamma muy grande restringe la influencia de cada soporte a su vecindad inmediata, memorizando el ruido.",
            },
        ],
        page_key=PAGE_KEY,
    )
