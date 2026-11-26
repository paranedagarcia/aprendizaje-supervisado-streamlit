"""Página 3: Árboles de Decisión (Clasificación y Regresión)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

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

configure_page("Árboles de Decisión", icon="🌳")

st.title("🌳 Árboles de Decisión")
st.caption("Área de aprendizaje supervisado: **Clasificación y Regresión** (Particionamiento recursivo y poda)")

PAGE_KEY = "decision_tree"

# --- SIDEBAR ---
st.sidebar.subheader("0. Tipo de Problema")
task_label = st.sidebar.radio(
    "Selecciona la tarea",
    options=["Clasificación", "Regresión"],
    key=f"{PAGE_KEY}_task",
)
task = "classification" if task_label == "Clasificación" else "regression"
default_ds = "iris" if task == "classification" else "diabetes"

bundle = render_dataset_sidebar(task=task, default_dataset_key=default_ds, page_key=f"{PAGE_KEY}_{task}")
test_size, random_state, scale_features = render_split_sidebar(PAGE_KEY, default_scale=False)

st.sidebar.subheader("3. Hiperparámetros del Árbol")
if task == "classification":
    criterion = st.sidebar.selectbox(
        "Criterio de división (criterion)",
        options=["gini", "entropy", "log_loss"],
        key=f"{PAGE_KEY}_crit_clf",
        help="Mide la impureza de los nodos: Gini es rápido; Entropy/Log-Loss se basan en ganancia de información.",
    )
else:
    criterion = st.sidebar.selectbox(
        "Criterio de división (criterion)",
        options=["squared_error", "friedman_mse", "absolute_error"],
        key=f"{PAGE_KEY}_crit_reg",
        help="Mide el error dentro de cada hoja (varianza/MSE o error absoluto/MAE).",
    )

limit_depth = st.sidebar.checkbox("Limitar profundidad máxima (max_depth)", value=True, key=f"{PAGE_KEY}_lim_depth")
max_depth = (
    st.sidebar.slider("Profundidad máxima (max_depth)", min_value=1, max_value=20, value=4, key=f"{PAGE_KEY}_max_depth")
    if limit_depth
    else None
)

min_samples_split = st.sidebar.slider(
    "Mínimo muestras para dividir (min_samples_split)",
    min_value=2,
    max_value=30,
    value=2,
    key=f"{PAGE_KEY}_min_split",
)
min_samples_leaf = st.sidebar.slider(
    "Mínimo muestras por hoja (min_samples_leaf)",
    min_value=1,
    max_value=20,
    value=1,
    key=f"{PAGE_KEY}_min_leaf",
)
max_features_opt = st.sidebar.selectbox(
    "Máximo de variables por división (max_features)",
    options=["Todas (None)", "sqrt", "log2"],
    key=f"{PAGE_KEY}_max_features",
)
max_features = None if max_features_opt == "Todas (None)" else max_features_opt

ccp_alpha = st.sidebar.slider(
    "Poda por complejidad de costo (ccp_alpha)",
    min_value=0.0,
    max_value=0.05,
    value=0.0,
    step=0.002,
    format="%.3f",
    key=f"{PAGE_KEY}_ccp_alpha",
    help="Valores mayores podan ramas que aportan poca mejora de impureza.",
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
    st.subheader("¿Cómo funcionan los Árboles de Decisión?")
    st.markdown(
        """
        Un **Árbol de Decisión** aprende una jerarquía de reglas `si ... entonces ...` dividiendo el espacio de
        características de forma recursiva para maximizar la homogeneidad en cada nodo hijo.

        - En **Clasificación**, busca reducir la impureza (**Índice de Gini** o **Entropía**).
        - En **Regresión**, busca reducir el error cuadrático (**MSE**) o absoluto (**MAE**), prediciendo el promedio (o mediana) en cada hoja.
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            #### Hiperparámetros para controlar el sobreajuste
            - **`max_depth`**: limita cuántos niveles puede crecer el árbol.
            - **`min_samples_split`** y **`min_samples_leaf`**: evitan crear hojas con muy pocos ejemplos.
            - **`ccp_alpha`**: aplica poda posterior (*Cost-Complexity Pruning*).
            """
        )
    with c2:
        st.markdown(
            """
            #### Ventajas y Limitaciones
            - **Ventajas**: muy interpretables, capturan relaciones no lineales, **no requieren escalado** de variables numéricas.
            - **Limitaciones**: alta varianza (pequeños cambios en datos generan árboles distintos) y propensos al **sobreajuste** si crecen sin restricción.
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
    estimator = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        ccp_alpha=ccp_alpha,
        random_state=random_state,
    )
else:
    estimator = DecisionTreeRegressor(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        ccp_alpha=ccp_alpha,
        random_state=random_state,
    )

preprocessor = build_preprocessor(bundle.X, scale_numeric=scale_features)
pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
y_train_score = pipeline.predict_proba(X_train) if task == "classification" else None
y_test_score = pipeline.predict_proba(X_test) if task == "classification" else None

fitted_tree = pipeline.named_steps["model"]
feat_names_out = get_transformed_feature_names(pipeline.named_steps["preprocessor"], bundle.feature_names)

with tab_train:
    render_metrics_comparison(
        task, y_train, y_train_pred, y_test, y_test_pred, y_train_score=y_train_score, y_test_score=y_test_score
    )

    k1, k2 = st.columns(2)
    k1.metric("Profundidad real alcanzada", f"{fitted_tree.get_depth()}")
    k2.metric("Número de hojas", f"{fitted_tree.get_n_leaves()}")

    render_standard_eval_charts(
        task, y_test, y_test_pred, y_test_score=y_test_score, class_names=bundle.class_names
    )

    # Depth validation curve
    X_tr_proc = pipeline.named_steps["preprocessor"].transform(X_train)
    X_te_proc = pipeline.named_steps["preprocessor"].transform(X_test)
    depths = list(range(1, 16))
    tr_scores, te_scores = [], []
    for d in depths:
        if task == "classification":
            m = DecisionTreeClassifier(criterion=criterion, max_depth=d, random_state=random_state)
            m.fit(X_tr_proc, y_train)
            tr_scores.append(accuracy_score(y_train, m.predict(X_tr_proc)))
            te_scores.append(accuracy_score(y_test, m.predict(X_te_proc)))
        else:
            m = DecisionTreeRegressor(criterion=criterion, max_depth=d, random_state=random_state)
            m.fit(X_tr_proc, y_train)
            tr_scores.append(r2_score(y_train, m.predict(X_tr_proc)))
            te_scores.append(r2_score(y_test, m.predict(X_te_proc)))

    metric_lbl = "Accuracy" if task == "classification" else "R²"
    fig_depth = go.Figure()
    fig_depth.add_trace(go.Scatter(x=depths, y=tr_scores, mode="lines+markers", name=f"{metric_lbl} Train"))
    fig_depth.add_trace(go.Scatter(x=depths, y=te_scores, mode="lines+markers", name=f"{metric_lbl} Test"))
    if max_depth is not None:
        fig_depth.add_vline(x=max_depth, line_dash="dash", line_color="gray", annotation_text=f"max_depth={max_depth}")
    fig_depth.update_layout(
        title=f"Curva de Complejidad: {metric_lbl} según Profundidad (max_depth)",
        xaxis_title="max_depth",
        yaxis_title=metric_lbl,
        height=380,
    )
    st.plotly_chart(fig_depth, width="stretch")

    col_imp, col_extra = st.columns(2)
    with col_imp:
        st.plotly_chart(
            plot_feature_importance(
                feat_names_out,
                fitted_tree.feature_importances_,
                title="Importancia de Variables (Gini/MSE)",
            ),
            width="stretch",
        )
    with col_extra:
        if task == "classification":
            fig_b = plot_decision_boundary_2d(
                fitted_tree, X_tr_proc, y_train, X_te_proc, y_test, feat_names_out, bundle.class_names
            )
            if fig_b is not None:
                st.plotly_chart(fig_b, width="stretch")
        else:
            st.info(
                "En regresión, un árbol aproxima la función mediante una función escalonada (constante por regiones)."
            )

    # Visual plot of the decision tree structure
    st.subheader("Visualización de la Estructura del Árbol")
    plot_depth = min(fitted_tree.get_depth(), 4)
    if fitted_tree.get_depth() > 4:
        st.caption("Se muestran los primeros 4 niveles para mantener la legibilidad del diagrama.")
    fig_tree, ax = plt.subplots(figsize=(14, 6))
    plot_tree(
        fitted_tree,
        max_depth=plot_depth,
        feature_names=feat_names_out,
        class_names=bundle.class_names if task == "classification" else None,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    st.pyplot(fig_tree, clear_figure=True)
    plt.close(fig_tree)

with tab_predict:
    render_what_if_predictor(pipeline, bundle, f"{PAGE_KEY}_{task}")

with tab_practice:
    render_practice_tab(
        challenges=[
            "Desmarca **Limitar profundidad máxima** y observa la diferencia entre Train y Test en **Moons 2D** o **Diabetes**.",
            "Usa la **Curva de Complejidad** para identificar la profundidad óptima donde Test alcanza su máximo antes de empezar a caer.",
            "Con el árbol sin límite de profundidad, aumenta progresivamente `ccp_alpha` (poda) y observa cómo disminuyen las hojas mientras mejora la generalización.",
        ],
        quiz_questions=[
            {
                "question": "¿Por qué los árboles de decisión normalmente no requieren estandarizar (escalar) las variables numéricas?",
                "options": [
                    "Porque dividen los datos mediante umbrales independientes por variable (orden relativo, no distancias).",
                    "Porque scikit-learn siempre escala automáticamente cualquier dataset.",
                    "Porque solo aceptan variables binarias.",
                ],
                "answer": "Porque dividen los datos mediante umbrales independientes por variable (orden relativo, no distancias).",
                "explanation": "Las particiones x_j <= t son invariantes ante transformaciones monótonas crecientes como el escalado.",
            },
            {
                "question": "¿Qué efecto tiene aumentar `min_samples_leaf`?",
                "options": [
                    "Hace el árbol más profundo y propenso al sobreajuste.",
                    "Regulariza el árbol evitando hojas con muy pocas observaciones.",
                    "Convierte el árbol en un Random Forest.",
                ],
                "answer": "Regulariza el árbol evitando hojas con muy pocas observaciones.",
                "explanation": "Exigir más muestras por hoja suaviza las predicciones y reduce la varianza.",
            },
        ],
        page_key=PAGE_KEY,
    )
