"""Página 1: Regresión Lineal (OLS, Ridge, Lasso, ElasticNet y extensión Polinomial)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

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
from utils.visualization import plot_feature_importance

configure_page("Regresión Lineal", icon="📈")

st.title("📈 Regresión Lineal")
st.caption("Área de aprendizaje supervisado: **Regresión** (OLS, Regularización L1/L2 y Características Polinomiales)")

PAGE_KEY = "linear_reg"

# --- SIDEBAR ---
bundle = render_dataset_sidebar(task="regression", default_dataset_key="diabetes", page_key=PAGE_KEY)
test_size, random_state, scale_features = render_split_sidebar(PAGE_KEY, default_scale=True)

st.sidebar.subheader("3. Hiperparámetros del Modelo")
variant = st.sidebar.selectbox(
    "Variante de Regresión Lineal",
    options=["LinearRegression (OLS)", "Ridge (L2)", "Lasso (L1)", "ElasticNet (L1 + L2)"],
    key=f"{PAGE_KEY}_variant",
    help="OLS no regulariza; Ridge encoge coeficientes; Lasso puede llevar coeficientes a 0; ElasticNet combina ambas.",
)

fit_intercept = st.sidebar.checkbox(
    "Ajustar intercepto (fit_intercept)",
    value=True,
    key=f"{PAGE_KEY}_fit_intercept",
)

alpha = 1.0
l1_ratio = 0.5
if variant != "LinearRegression (OLS)":
    alpha = st.sidebar.slider(
        "Fuerza de regularización (alpha)",
        min_value=0.001,
        max_value=100.0,
        value=1.0,
        step=0.1,
        key=f"{PAGE_KEY}_alpha",
        help="Mayor alpha implica mayor penalización sobre la magnitud de los coeficientes.",
    )
if variant == "ElasticNet (L1 + L2)":
    l1_ratio = st.sidebar.slider(
        "Proporción L1 (l1_ratio)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key=f"{PAGE_KEY}_l1_ratio",
        help="0 equivale a Ridge (L2) y 1 equivale a Lasso (L1).",
    )

poly_degree = st.sidebar.slider(
    "Grado polinomial (PolynomialFeatures)",
    min_value=1,
    max_value=6,
    value=1,
    step=1,
    key=f"{PAGE_KEY}_poly_degree",
    help="Grado 1 es regresión lineal pura. Grados mayores permiten ajustar curvas no lineales (cuidado con el sobreajuste).",
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
    st.subheader("¿Qué es la Regresión Lineal?")
    st.markdown(
        r"""
        La **Regresión Lineal** modela una variable objetivo continua $y$ como una combinación lineal de las variables predictoras $x_1, x_2, \dots, x_p$:

        $$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p$$

        El modelo clásico (**OLS - Mínimos Cuadrados Ordinarios**) minimiza la suma de errores cuadráticos:
        $$\min_{\beta} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            #### Variantes Regularizadas
            - **Ridge ($L_2$)**: añade $\\alpha \\sum \\beta_j^2$. Reduce la varianza y maneja bien la multicolinealidad.
            - **Lasso ($L_1$)**: añade $\\alpha \\sum |\\beta_j|$. Realiza **selección automática de variables** al anular coeficientes irrelevantes.
            - **ElasticNet**: combina penalización $L_1$ y $L_2$ controlada por `l1_ratio`.
            """
        )
    with c2:
        st.markdown(
            """
            #### Ventajas y Limitaciones
            - **Ventajas**: rápido, altamente interpretable (cada coeficiente indica el efecto de una variable), buena línea base.
            - **Limitaciones**: asume linealidad (salvo que agregues transformaciones polinomiales), sensible a valores atípicos (*outliers*) y a escalas muy distintas cuando se usa regularización.
            """
        )

if bundle is None:
    st.stop()

with tab_eda:
    render_eda_tab(bundle)

# Guard against combinatorial explosion with PolynomialFeatures on high-dimensional datasets
effective_degree = poly_degree
if poly_degree > 2 and bundle.X.shape[1] > 10:
    st.warning(
        f"El dataset tiene {bundle.X.shape[1]} variables; para evitar explosión combinatoria, "
        "el grado polinomial se limitó automáticamente a 2."
    )
    effective_degree = 2

X_train, X_test, y_train, y_test = split_dataset(
    bundle.X, bundle.y, task="regression", test_size=test_size, random_state=random_state
)

if variant == "LinearRegression (OLS)":
    estimator = LinearRegression(fit_intercept=fit_intercept)
elif variant == "Ridge (L2)":
    estimator = Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=random_state)
elif variant == "Lasso (L1)":
    estimator = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=10000, random_state=random_state)
else:
    estimator = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        max_iter=10000,
        random_state=random_state,
    )

preprocessor = build_preprocessor(bundle.X, scale_numeric=scale_features)
steps = [("preprocessor", preprocessor)]
if effective_degree > 1:
    steps.append(("poly", PolynomialFeatures(degree=effective_degree, include_bias=False)))
steps.append(("model", estimator))
pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

with tab_train:
    render_metrics_comparison("regression", y_train, y_train_pred, y_test, y_test_pred)
    render_standard_eval_charts("regression", y_test, y_test_pred)

    # 1D Curve visualization if single numeric feature
    if len(bundle.feature_names) == 1 and pd.api.types.is_numeric_dtype(bundle.X.iloc[:, 0]):
        feat_name = bundle.feature_names[0]
        x_grid = np.linspace(float(bundle.X[feat_name].min()), float(bundle.X[feat_name].max()), 250)
        y_grid = pipeline.predict(pd.DataFrame({feat_name: x_grid}))
        fig_1d = go.Figure()
        fig_1d.add_trace(
            go.Scatter(x=X_train[feat_name], y=y_train, mode="markers", name="Train", opacity=0.65)
        )
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
                name=f"Ajuste (grado {effective_degree})",
                line=dict(color="#DC2626", width=3),
            )
        )
        fig_1d.update_layout(
            title=f"Curva de Ajuste 1D ({variant}, grado {effective_degree})",
            xaxis_title=feat_name,
            yaxis_title=bundle.target_name,
            height=420,
        )
        st.plotly_chart(fig_1d, width="stretch")

    # Coefficients visualization
    fitted_model = pipeline.named_steps["model"]
    base_names = get_transformed_feature_names(pipeline.named_steps["preprocessor"], bundle.feature_names)
    if effective_degree > 1:
        feat_names_out = list(pipeline.named_steps["poly"].get_feature_names_out(base_names))
    else:
        feat_names_out = base_names

    coefs = np.ravel(fitted_model.coef_)
    if len(coefs) == len(feat_names_out):
        st.plotly_chart(
            plot_feature_importance(
                feat_names_out,
                coefs,
                title=f"Coeficientes Estimados ({variant}) — Intercepto: {float(fitted_model.intercept_):.4f}",
            ),
            width="stretch",
        )
        zero_count = int(np.sum(np.isclose(coefs, 0.0, atol=1e-6)))
        if variant in {"Lasso (L1)", "ElasticNet (L1 + L2)"}:
            st.info(
                f"Coeficientes anulados (≈ 0): **{zero_count} de {len(coefs)}**. "
                "Aumenta `alpha` para observar cómo Lasso descarta variables menos informativas."
            )

with tab_predict:
    render_what_if_predictor(pipeline, bundle, PAGE_KEY)

with tab_practice:
    render_practice_tab(
        challenges=[
            "Selecciona el dataset **Curva Polinomial 1D** y prueba grados polinomiales 1, 3 y 6. Observa cómo cambia el R² de Train vs. Test.",
            "En **Diabetes**, cambia entre **OLS**, **Ridge** y **Lasso** aumentando `alpha` hasta 20. ¿Qué ocurre con la magnitud de los coeficientes y cuántos se vuelven exactamente 0 en Lasso?",
            "Desactiva la opción **Estandarizar variables numéricas** en **Linnerud** con Ridge (`alpha=10`) y observa cómo cambia la escala relativa de los coeficientes.",
        ],
        quiz_questions=[
            {
                "question": "¿Cuál es la principal diferencia entre Ridge (L2) y Lasso (L1)?",
                "options": [
                    "Lasso puede llevar coeficientes exactamente a cero, realizando selección de variables.",
                    "Ridge solo funciona para clasificación binaria.",
                    "Lasso no necesita el hiperparámetro alpha.",
                ],
                "answer": "Lasso puede llevar coeficientes exactamente a cero, realizando selección de variables.",
                "explanation": "La geometría de la norma L1 produce soluciones dispersas donde varios coeficientes se anulan exactamente.",
            },
            {
                "question": "Si aumentas el grado polinomial a 6 y el R² de Train es 0.99 pero el de Test cae bruscamente, ¿qué está ocurriendo?",
                "options": [
                    "Subajuste (underfitting)",
                    "Sobreajuste (overfitting)",
                    "Fuga de datos imposible de corregir",
                ],
                "answer": "Sobreajuste (overfitting)",
                "explanation": "El modelo memoriza el ruido del entrenamiento y pierde capacidad de generalizar en datos nuevos.",
            },
        ],
        page_key=PAGE_KEY,
    )
