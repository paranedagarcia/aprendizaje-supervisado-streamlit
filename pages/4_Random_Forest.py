"""Página 4: Random Forest (Clasificación y Regresión)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree

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

configure_page("Random Forest", icon="🌲")

st.title("🌲 Random Forest")
st.caption("Área de aprendizaje supervisado: **Clasificación y Regresión** (Ensamble Bagging de Árboles)")

PAGE_KEY = "random_forest"

# --- SIDEBAR ---
st.sidebar.subheader("0. Tipo de Problema")
task_label = st.sidebar.radio(
    "Selecciona la tarea",
    options=["Clasificación", "Regresión"],
    key=f"{PAGE_KEY}_task",
)
task = "classification" if task_label == "Clasificación" else "regression"
default_ds = "wine" if task == "classification" else "friedman1"

bundle = render_dataset_sidebar(task=task, default_dataset_key=default_ds, page_key=f"{PAGE_KEY}_{task}")
test_size, random_state, scale_features = render_split_sidebar(PAGE_KEY, default_scale=False)

st.sidebar.subheader("3. Hiperparámetros de Random Forest")
n_estimators = st.sidebar.slider(
    "Número de árboles (n_estimators)",
    min_value=10,
    max_value=300,
    value=100,
    step=10,
    key=f"{PAGE_KEY}_n_est",
    help="Más árboles reducen la varianza del ensamble a costa de mayor tiempo de cómputo.",
)

if task == "classification":
    criterion = st.sidebar.selectbox(
        "Criterio (criterion)",
        options=["gini", "entropy", "log_loss"],
        key=f"{PAGE_KEY}_crit_clf",
    )
else:
    criterion = st.sidebar.selectbox(
        "Criterio (criterion)",
        options=["squared_error", "friedman_mse", "absolute_error"],
        key=f"{PAGE_KEY}_crit_reg",
    )

limit_depth = st.sidebar.checkbox("Limitar profundidad (max_depth)", value=False, key=f"{PAGE_KEY}_lim_depth")
max_depth = (
    st.sidebar.slider("Profundidad máxima (max_depth)", 1, 25, 8, key=f"{PAGE_KEY}_max_depth")
    if limit_depth
    else None
)

min_samples_split = st.sidebar.slider("min_samples_split", 2, 20, 2, key=f"{PAGE_KEY}_min_split")
min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 15, 1, key=f"{PAGE_KEY}_min_leaf")

max_features_opt = st.sidebar.selectbox(
    "Variables por división (max_features)",
    options=["sqrt", "log2", "Todas (None)"],
    key=f"{PAGE_KEY}_max_features",
    help="Seleccionar un subconjunto aleatorio de variables en cada división descorrelaciona los árboles.",
)
max_features = None if max_features_opt == "Todas (None)" else max_features_opt

bootstrap = st.sidebar.checkbox(
    "Muestreo Bootstrap (bootstrap)",
    value=True,
    key=f"{PAGE_KEY}_bootstrap",
    help="Cada árbol se entrena con una muestra con reemplazo del conjunto de entrenamiento.",
)
oob_score = st.sidebar.checkbox(
    "Calcular Out-of-Bag Score (oob_score)",
    value=True,
    disabled=not bootstrap,
    key=f"{PAGE_KEY}_oob",
    help="Evalúa cada árbol usando las muestras que quedaron fuera de su muestra bootstrap.",
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
    st.subheader("¿Qué es Random Forest?")
    st.markdown(
        """
        **Random Forest** (Bosque Aleatorio) es un método de **ensamble basado en Bagging** que entrena múltiples
        árboles de decisión independientes combinando dos fuentes de aleatoriedad:

        1. **Bootstrap de muestras**: cada árbol se entrena en una submuestra con reemplazo de los datos.
        2. **Subconjunto aleatorio de variables (`max_features`)**: en cada división solo se considera una fracción de las características.

        La predicción final se obtiene por **votación mayoritaria** (Clasificación) o **promedio** (Regresión),
        reduciendo drásticamente la **varianza** de un árbol individual.
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            #### Out-of-Bag (OOB) Score
            Con `bootstrap=True`, aproximadamente un 36.8% de las muestras no son seleccionadas para cada árbol.
            Esas muestras actúan como un conjunto de validación interno gratuito (**OOB Score**).
            """
        )
    with c2:
        st.markdown(
            """
            #### Ventajas y Limitaciones
            - **Ventajas**: excelente rendimiento fuera de la caja, robusto frente a ruido y outliers, estima importancia de variables.
            - **Limitaciones**: menos interpretable que un solo árbol, mayor consumo de memoria y más lento en inferencia con muchos árboles.
            """
        )

if bundle is None:
    st.stop()

with tab_eda:
    render_eda_tab(bundle)

X_train, X_test, y_train, y_test = split_dataset(
    bundle.X, bundle.y, task=task, test_size=test_size, random_state=random_state
)

effective_oob = bool(oob_score and bootstrap)

if task == "classification":
    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        oob_score=effective_oob,
        random_state=random_state,
        n_jobs=-1,
    )
else:
    estimator = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        oob_score=effective_oob,
        random_state=random_state,
        n_jobs=-1,
    )

preprocessor = build_preprocessor(bundle.X, scale_numeric=scale_features)
pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
y_train_score = pipeline.predict_proba(X_train) if task == "classification" else None
y_test_score = pipeline.predict_proba(X_test) if task == "classification" else None

fitted_rf = pipeline.named_steps["model"]
feat_names_out = get_transformed_feature_names(pipeline.named_steps["preprocessor"], bundle.feature_names)

with tab_train:
    render_metrics_comparison(
        task, y_train, y_train_pred, y_test, y_test_pred, y_train_score=y_train_score, y_test_score=y_test_score
    )

    if effective_oob and hasattr(fitted_rf, "oob_score_"):
        st.info(
            f"**Out-of-Bag (OOB) Score:** `{fitted_rf.oob_score_:.4f}` "
            f"({'Accuracy' if task == 'classification' else 'R²'} estimado sobre muestras fuera de bolsa)."
        )

    render_standard_eval_charts(
        task, y_test, y_test_pred, y_test_score=y_test_score, class_names=bundle.class_names
    )

    X_tr_proc = pipeline.named_steps["preprocessor"].transform(X_train)
    X_te_proc = pipeline.named_steps["preprocessor"].transform(X_test)

    c_imp, c_bound = st.columns(2)
    with c_imp:
        st.plotly_chart(
            plot_feature_importance(
                feat_names_out,
                fitted_rf.feature_importances_,
                title="Importancia de Variables en el Bosque",
            ),
            width="stretch",
        )
    with c_bound:
        if task == "classification":
            fig_b = plot_decision_boundary_2d(
                fitted_rf, X_tr_proc, y_train, X_te_proc, y_test, feat_names_out, bundle.class_names
            )
            if fig_b is not None:
                st.plotly_chart(fig_b, width="stretch")
        else:
            st.markdown("#### Inspección de un Árbol Individual del Bosque")
            tree_idx = st.slider(
                "Índice del árbol a inspeccionar",
                min_value=0,
                max_value=len(fitted_rf.estimators_) - 1,
                value=0,
                key=f"{PAGE_KEY}_inspect_tree_reg",
            )
            fig_t, ax = plt.subplots(figsize=(9, 5))
            plot_tree(
                fitted_rf.estimators_[tree_idx],
                max_depth=3,
                feature_names=feat_names_out,
                filled=True,
                rounded=True,
                fontsize=7,
                ax=ax,
            )
            st.pyplot(fig_t, clear_figure=True)
            plt.close(fig_t)

    if task == "classification":
        with st.expander("Inspeccionar un árbol individual dentro del bosque"):
            tree_idx = st.slider(
                "Índice del árbol",
                min_value=0,
                max_value=len(fitted_rf.estimators_) - 1,
                value=0,
                key=f"{PAGE_KEY}_inspect_tree_clf",
            )
            fig_t, ax = plt.subplots(figsize=(13, 5))
            plot_tree(
                fitted_rf.estimators_[tree_idx],
                max_depth=3,
                feature_names=feat_names_out,
                class_names=bundle.class_names,
                filled=True,
                rounded=True,
                fontsize=8,
                ax=ax,
            )
            st.pyplot(fig_t, clear_figure=True)
            plt.close(fig_t)

with tab_predict:
    render_what_if_predictor(pipeline, bundle, f"{PAGE_KEY}_{task}")

with tab_practice:
    render_practice_tab(
        challenges=[
            "Compara `n_estimators = 10` frente a `n_estimators = 200` en **Moons 2D** y observa cómo la frontera de decisión se vuelve más suave y estable.",
            "Activa `oob_score` y compáralo con el desempeño en **Test**. ¿Qué tan parecidos son?",
            "En **Friedman #1 (Regresión)**, compara `max_features = 'sqrt'` vs `'Todas (None)'` y observa las importancias de las variables `x1..x6` (recuerda que `x6` es ruido puro).",
        ],
        quiz_questions=[
            {
                "question": "¿Por qué Random Forest selecciona un subconjunto aleatorio de variables (`max_features`) en cada división?",
                "options": [
                    "Para descorrelacionar los árboles entre sí y reducir mejor la varianza del ensamble.",
                    "Para obligar a que todos los árboles sean idénticos.",
                    "Para eliminar la necesidad de datos de entrenamiento.",
                ],
                "answer": "Para descorrelacionar los árboles entre sí y reducir mejor la varianza del ensamble.",
                "explanation": "Si todos los árboles usaran siempre la misma variable dominante en la raíz, serían muy similares y promediarlos reduciría menos la varianza.",
            },
            {
                "question": "¿Qué representa el OOB Score (Out-of-Bag)?",
                "options": [
                    "La evaluación sobre las muestras que no entraron en la muestra bootstrap de cada árbol.",
                    "El error calculado exclusivamente sobre el conjunto de prueba externo.",
                    "La profundidad promedio del bosque.",
                ],
                "answer": "La evaluación sobre las muestras que no entraron en la muestra bootstrap de cada árbol.",
                "explanation": "Permite estimar la capacidad de generalización durante el propio entrenamiento con bootstrap.",
            },
        ],
        page_key=PAGE_KEY,
    )
