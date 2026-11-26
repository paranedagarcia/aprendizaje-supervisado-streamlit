"""Página 7: K-Nearest Neighbors (KNN - Clasificación y Regresión)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
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
from utils.visualization import plot_decision_boundary_2d

configure_page("KNN (K-Nearest Neighbors)", icon="📍")

st.title("📍 K-Nearest Neighbors (KNN)")
st.caption("Área de aprendizaje supervisado: **Clasificación y Regresión** (Aprendizaje basado en instancias y distancias)")

PAGE_KEY = "knn"

# --- SIDEBAR ---
st.sidebar.subheader("0. Tipo de Problema")
task_label = st.sidebar.radio(
    "Selecciona la tarea",
    options=["Clasificación", "Regresión"],
    key=f"{PAGE_KEY}_task",
)
task = "classification" if task_label == "Clasificación" else "regression"
default_ds = "iris" if task == "classification" else "poly_1d"

bundle = render_dataset_sidebar(task=task, default_dataset_key=default_ds, page_key=f"{PAGE_KEY}_{task}")
test_size, random_state, scale_features = render_split_sidebar(PAGE_KEY, default_scale=True)

if not scale_features:
    st.sidebar.warning(
        "KNN se basa en distancias entre puntos. Sin `StandardScaler`, las variables con mayor rango dominan el cálculo."
    )

st.sidebar.subheader("3. Hiperparámetros de KNN")
n_neighbors = st.sidebar.slider(
    "Número de vecinos (n_neighbors / K)",
    min_value=1,
    max_value=35,
    value=5,
    step=1,
    key=f"{PAGE_KEY}_k",
    help="K pequeño produce fronteras muy flexibles (alta varianza); K grande suaviza la decisión (mayor sesgo).",
)

weights = st.sidebar.selectbox(
    "Ponderación de vecinos (weights)",
    options=["uniform", "distance"],
    key=f"{PAGE_KEY}_weights",
    help="'uniform' da igual peso a todos los vecinos; 'distance' pesa más a los vecinos más cercanos.",
)

metric = st.sidebar.selectbox(
    "Métrica de distancia (metric)",
    options=["minkowski", "euclidean", "manhattan", "chebyshev"],
    key=f"{PAGE_KEY}_metric",
    help="Define cómo se mide la cercanía entre dos observaciones.",
)

p_val = 2
if metric == "minkowski":
    p_val = st.sidebar.slider(
        "Parámetro p de Minkowski",
        min_value=1,
        max_value=5,
        value=2,
        key=f"{PAGE_KEY}_p",
        help="p=1 equivale a distancia Manhattan; p=2 equivale a distancia Euclidiana.",
    )

algorithm = st.sidebar.selectbox(
    "Algoritmo de búsqueda (algorithm)",
    options=["auto", "ball_tree", "kd_tree", "brute"],
    key=f"{PAGE_KEY}_algo",
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
    st.subheader("¿Qué es K-Nearest Neighbors (KNN)?")
    st.markdown(
        r"""
        **KNN** es un algoritmo **no paramétrico** y de aprendizaje perezoso (*lazy learner*): no aprende una fórmula
        explícita durante el entrenamiento, sino que almacena los datos y, para predecir un nuevo punto $\mathbf{x}$,
        busca sus **$K$ vecinos más cercanos** según una métrica de distancia (por ejemplo, Euclidiana):

        $$d(\mathbf{x}, \mathbf{x}') = \sqrt{\sum_{j=1}^{p} (x_j - x'_j)^2}$$

        - En **Clasificación**: asigna la clase mayoritaria entre los $K$ vecinos.
        - En **Regresión**: predice el promedio (simple o ponderado por distancia) de los $K$ vecinos.
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            #### Elección de K (Sesgo vs. Varianza)
            - **$K = 1$**: memoriza el entrenamiento (error en Train = 0, pero muy sensible al ruido).
            - **$K$ muy grande**: frontera excesivamente suave que puede ignorar patrones locales (subajuste).
            - En clasificación binaria suele convenir un **$K$ impar** para evitar empates.
            """
        )
    with c2:
        st.markdown(
            """
            #### Ventajas y Limitaciones
            - **Ventajas**: intuitivo, cero supuestos sobre la distribución de datos, se adapta naturalmente a fronteras no lineales.
            - **Limitaciones**: **muy sensible al escalado** y a variables irrelevantes (*maldición de la dimensionalidad*), y predicción lenta en datasets muy grandes.
            """
        )

if bundle is None:
    st.stop()

with tab_eda:
    render_eda_tab(bundle)

X_train, X_test, y_train, y_test = split_dataset(
    bundle.X, bundle.y, task=task, test_size=test_size, random_state=random_state
)

max_valid_k = max(1, len(X_train))
effective_k = min(n_neighbors, max_valid_k)
if effective_k < n_neighbors:
    st.warning(f"`n_neighbors` se ajustó a {effective_k} porque el conjunto de entrenamiento tiene {len(X_train)} filas.")

knn_kwargs = {
    "n_neighbors": effective_k,
    "weights": weights,
    "metric": metric,
    "algorithm": algorithm,
}
if metric == "minkowski":
    knn_kwargs["p"] = p_val

estimator = KNeighborsClassifier(**knn_kwargs) if task == "classification" else KNeighborsRegressor(**knn_kwargs)

preprocessor = build_preprocessor(bundle.X, scale_numeric=scale_features)
pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
y_train_score = pipeline.predict_proba(X_train) if task == "classification" else None
y_test_score = pipeline.predict_proba(X_test) if task == "classification" else None

fitted_knn = pipeline.named_steps["model"]
feat_names_out = get_transformed_feature_names(pipeline.named_steps["preprocessor"], bundle.feature_names)
X_tr_proc = pipeline.named_steps["preprocessor"].transform(X_train)
X_te_proc = pipeline.named_steps["preprocessor"].transform(X_test)

with tab_train:
    render_metrics_comparison(
        task, y_train, y_train_pred, y_test, y_test_pred, y_train_score=y_train_score, y_test_score=y_test_score
    )
    render_standard_eval_charts(
        task, y_test, y_test_pred, y_test_score=y_test_score, class_names=bundle.class_names
    )

    # Validation curve across K
    max_k_scan = min(30, max_valid_k)
    k_values = list(range(1, max_k_scan + 1))
    tr_scores, te_scores = [], []
    for k in k_values:
        scan_kwargs = {**knn_kwargs, "n_neighbors": k}
        m = KNeighborsClassifier(**scan_kwargs) if task == "classification" else KNeighborsRegressor(**scan_kwargs)
        m.fit(X_tr_proc, y_train)
        if task == "classification":
            tr_scores.append(accuracy_score(y_train, m.predict(X_tr_proc)))
            te_scores.append(accuracy_score(y_test, m.predict(X_te_proc)))
        else:
            tr_scores.append(r2_score(y_train, m.predict(X_tr_proc)))
            te_scores.append(r2_score(y_test, m.predict(X_te_proc)))

    best_k = k_values[int(np.argmax(te_scores))]
    metric_lbl = "Accuracy" if task == "classification" else "R²"

    fig_k = go.Figure()
    fig_k.add_trace(go.Scatter(x=k_values, y=tr_scores, mode="lines+markers", name=f"{metric_lbl} Train"))
    fig_k.add_trace(go.Scatter(x=k_values, y=te_scores, mode="lines+markers", name=f"{metric_lbl} Test"))
    fig_k.add_vline(
        x=effective_k,
        line_dash="dash",
        line_color="#2563EB",
        annotation_text=f"K actual = {effective_k}",
    )
    fig_k.add_vline(
        x=best_k,
        line_dash="dot",
        line_color="#16A34A",
        annotation_text=f"Mejor K en Test = {best_k}",
        annotation_position="bottom right",
    )
    fig_k.update_layout(
        title=f"Curva de Selección de K ({metric_lbl} Train vs. Test)",
        xaxis_title="Número de vecinos (K)",
        yaxis_title=metric_lbl,
        height=390,
    )
    st.plotly_chart(fig_k, width="stretch")

    if task == "classification":
        fig_b = plot_decision_boundary_2d(
            fitted_knn, X_tr_proc, y_train, X_te_proc, y_test, feat_names_out, bundle.class_names
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
            go.Scatter(
                x=x_grid,
                y=y_grid,
                mode="lines",
                name=f"KNN Regresión (K={effective_k}, {weights})",
                line=dict(color="#DC2626", width=3),
            )
        )
        fig_1d.update_layout(
            title=f"Curva de Regresión KNN 1D (K={effective_k}, weights={weights})",
            xaxis_title=feat_name,
            yaxis_title=bundle.target_name,
            height=420,
        )
        st.plotly_chart(fig_1d, width="stretch")

with tab_predict:
    render_what_if_predictor(pipeline, bundle, f"{PAGE_KEY}_{task}")

with tab_practice:
    render_practice_tab(
        challenges=[
            "En **Moons 2D**, prueba `K = 1`, `K = 7` y `K = 30`. Observa cómo cambia la suavidad de la frontera de decisión y el Accuracy en Train vs. Test.",
            "En **Wine**, desactiva **Estandarizar variables numéricas** y compara el Accuracy con el resultado estandarizado. ¿Por qué cae tanto sin escalado?",
            "En **Curva Polinomial 1D (Regresión)**, compara `weights = 'uniform'` frente a `weights = 'distance'` con `K = 10`.",
        ],
        quiz_questions=[
            {
                "question": "Con `K = 1` en KNN, ¿cuál es típicamente el Accuracy sobre el conjunto de entrenamiento (Train)?",
                "options": [
                    "Cercano o igual a 1.0 (100%), porque cada punto es su propio vecino más cercano.",
                    "Siempre 0.5.",
                    "Siempre menor que el de Test.",
                ],
                "answer": "Cercano o igual a 1.0 (100%), porque cada punto es su propio vecino más cercano.",
                "explanation": "Con K=1 el modelo memoriza perfectamente el entrenamiento, lo que suele provocar alto sobreajuste en presencia de ruido.",
            },
            {
                "question": "¿Cuál es la equivalencia entre la distancia de Minkowski con `p = 1` y `p = 2`?",
                "options": [
                    "p = 1 es Manhattan y p = 2 es Euclidiana.",
                    "p = 1 es Coseno y p = 2 es Hamming.",
                    "Ambas son idénticas siempre.",
                ],
                "answer": "p = 1 es Manhattan y p = 2 es Euclidiana.",
                "explanation": "La norma L1 corresponde a la distancia Manhattan (suma de diferencias absolutas) y la L2 a la distancia Euclidiana.",
            },
        ],
        page_key=PAGE_KEY,
    )
