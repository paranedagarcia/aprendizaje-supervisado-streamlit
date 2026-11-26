"""Página 6: Gradient Boosting (Clasificación y Regresión)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score
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

configure_page("Gradient Boosting", icon="🚀")

st.title("🚀 Gradient Boosting")
st.caption("Área de aprendizaje supervisado: **Clasificación y Regresión** (Ensamble secuencial por descenso de gradiente)")

PAGE_KEY = "grad_boost"

# --- SIDEBAR ---
st.sidebar.subheader("0. Tipo de Problema")
task_label = st.sidebar.radio(
    "Selecciona la tarea",
    options=["Clasificación", "Regresión"],
    key=f"{PAGE_KEY}_task",
)
task = "classification" if task_label == "Clasificación" else "regression"
default_ds = "breast_cancer" if task == "classification" else "friedman1"

bundle = render_dataset_sidebar(task=task, default_dataset_key=default_ds, page_key=f"{PAGE_KEY}_{task}")
test_size, random_state, scale_features = render_split_sidebar(PAGE_KEY, default_scale=False)

st.sidebar.subheader("3. Hiperparámetros de Gradient Boosting")
n_estimators = st.sidebar.slider(
    "Número de etapas / árboles (n_estimators)",
    min_value=10,
    max_value=300,
    value=100,
    step=10,
    key=f"{PAGE_KEY}_n_est",
    help="Cantidad de árboles secuenciales que corrigen los residuos anteriores.",
)

learning_rate = st.sidebar.select_slider(
    "Tasa de aprendizaje (learning_rate)",
    options=[0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    value=0.1,
    key=f"{PAGE_KEY}_lr",
    help="Escala la contribución de cada árbol. Tasas menores suelen generalizar mejor pero requieren más n_estimators.",
)

max_depth = st.sidebar.slider(
    "Profundidad máxima por árbol (max_depth)",
    min_value=1,
    max_value=10,
    value=3,
    key=f"{PAGE_KEY}_max_depth",
    help="En boosting se prefieren árboles poco profundos (aprendices débiles, típicos entre 2 y 5).",
)

subsample = st.sidebar.slider(
    "Submuestreo estocástico (subsample)",
    min_value=0.3,
    max_value=1.0,
    value=1.0,
    step=0.1,
    key=f"{PAGE_KEY}_subsample",
    help="Valores < 1.0 activan Stochastic Gradient Boosting, reduciendo la varianza.",
)

min_samples_split = st.sidebar.slider("min_samples_split", 2, 20, 2, key=f"{PAGE_KEY}_min_split")
min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 15, 1, key=f"{PAGE_KEY}_min_leaf")

if task == "classification":
    is_binary_bundle = bundle is not None and len(np.unique(bundle.y)) == 2
    loss_opts = ["log_loss", "exponential"] if is_binary_bundle else ["log_loss"]
    loss = st.sidebar.selectbox("Función de pérdida (loss)", options=loss_opts, key=f"{PAGE_KEY}_loss_clf")
else:
    loss = st.sidebar.selectbox(
        "Función de pérdida (loss)",
        options=["squared_error", "absolute_error", "huber"],
        key=f"{PAGE_KEY}_loss_reg",
        help="huber y absolute_error son más robustas ante outliers que squared_error.",
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
    st.subheader("¿Qué es Gradient Boosting?")
    st.markdown(
        r"""
        A diferencia de Random Forest (donde los árboles se entrenan en paralelo e independientemente),
        **Gradient Boosting** construye los árboles **de forma secuencial**: cada nuevo árbol intenta predecir
        los **errores (gradientes negativos de la función de pérdida)** cometidos por el ensamble acumulado hasta ese momento:

        $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

        donde $\eta$ es el `learning_rate` y $h_m$ es el árbol entrenado en la iteración $m$.
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            #### Equilibrio Clave
            - **`learning_rate` vs. `n_estimators`**: una tasa baja con más árboles suele dar mejor generalización.
            - **`max_depth`**: árboles pequeños (2–4) capturan interacciones sin memorizar el ruido.
            - **`subsample < 1.0`**: introduce aleatoriedad estilo *bagging* para reducir varianza.
            """
        )
    with c2:
        st.markdown(
            """
            #### Ventajas y Limitaciones
            - **Ventajas**: altísimo poder predictivo en datos tabulares, maneja relaciones no lineales complejas y distintos tipos de pérdida.
            - **Limitaciones**: más sensible al sobreajuste que Random Forest si `n_estimators` o `learning_rate` son muy altos, y entrenamiento secuencial más lento.
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
    estimator = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=float(learning_rate),
        max_depth=max_depth,
        subsample=float(subsample),
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        loss=loss,
        random_state=random_state,
    )
else:
    estimator = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=float(learning_rate),
        max_depth=max_depth,
        subsample=float(subsample),
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        loss=loss,
        random_state=random_state,
    )

preprocessor = build_preprocessor(bundle.X, scale_numeric=scale_features)
pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
y_train_score = pipeline.predict_proba(X_train) if task == "classification" else None
y_test_score = pipeline.predict_proba(X_test) if task == "classification" else None

fitted_gb = pipeline.named_steps["model"]
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

    # Staged learning curve across boosting iterations
    iters = list(range(1, len(fitted_gb.estimators_) + 1))
    if task == "classification":
        tr_curve = [accuracy_score(y_train, p) for p in fitted_gb.staged_predict(X_tr_proc)]
        te_curve = [accuracy_score(y_test, p) for p in fitted_gb.staged_predict(X_te_proc)]
        metric_lbl = "Accuracy"
    else:
        tr_curve = [r2_score(y_train, p) for p in fitted_gb.staged_predict(X_tr_proc)]
        te_curve = [r2_score(y_test, p) for p in fitted_gb.staged_predict(X_te_proc)]
        metric_lbl = "R²"

    best_iter = int(np.argmax(te_curve)) + 1
    fig_stage = go.Figure()
    fig_stage.add_trace(go.Scatter(x=iters, y=tr_curve, mode="lines", name=f"{metric_lbl} Train"))
    fig_stage.add_trace(go.Scatter(x=iters, y=te_curve, mode="lines", name=f"{metric_lbl} Test"))
    fig_stage.add_vline(
        x=best_iter,
        line_dash="dash",
        line_color="#16A34A",
        annotation_text=f"Mejor Test: iter {best_iter} ({te_curve[best_iter - 1]:.3f})",
    )
    fig_stage.update_layout(
        title=f"Evolución del Desempeño por Iteración de Boosting ({metric_lbl})",
        xaxis_title="Número de árboles (etapa de boosting)",
        yaxis_title=metric_lbl,
        height=390,
    )
    st.plotly_chart(fig_stage, width="stretch")

    c_imp, c_bound = st.columns(2)
    with c_imp:
        st.plotly_chart(
            plot_feature_importance(
                feat_names_out,
                fitted_gb.feature_importances_,
                title="Importancia de Variables (Gradient Boosting)",
            ),
            width="stretch",
        )
    with c_bound:
        if task == "classification":
            fig_b = plot_decision_boundary_2d(
                fitted_gb, X_tr_proc, y_train, X_te_proc, y_test, feat_names_out, bundle.class_names
            )
            if fig_b is not None:
                st.plotly_chart(fig_b, width="stretch")
        else:
            st.info(
                "Observa la curva superior: si la línea de **Test** alcanza un máximo y luego desciende "
                "mientras **Train** sigue subiendo, estás viendo el punto exacto donde comienza el sobreajuste."
            )

with tab_predict:
    render_what_if_predictor(pipeline, bundle, f"{PAGE_KEY}_{task}")

with tab_practice:
    render_practice_tab(
        challenges=[
            "Fija `n_estimators = 250` y compara `learning_rate = 0.03` frente a `learning_rate = 0.5` observando la **Evolución por Iteración**.",
            "En **Diabetes (Regresión)**, prueba `max_depth = 2` vs `max_depth = 8`. ¿Qué sucede con la brecha entre Train y Test?",
            "Reduce `subsample` de `1.0` a `0.7` (Stochastic Gradient Boosting) y verifica si mejora el desempeño en Test.",
        ],
        quiz_questions=[
            {
                "question": "¿Cuál es la diferencia fundamental entre Random Forest (Bagging) y Gradient Boosting?",
                "options": [
                    "Random Forest entrena árboles independientes en paralelo; Gradient Boosting entrena árboles secuencialmente corrigiendo errores previos.",
                    "Gradient Boosting solo sirve para regresión y Random Forest solo para clasificación.",
                    "Random Forest requiere learning_rate y Gradient Boosting no.",
                ],
                "answer": "Random Forest entrena árboles independientes en paralelo; Gradient Boosting entrena árboles secuencialmente corrigiendo errores previos.",
                "explanation": "Bagging reduce varianza promediando modelos independientes; Boosting reduce principalmente el sesgo de forma aditiva secuencial.",
            },
            {
                "question": "¿Qué suele ocurrir si usas un `learning_rate` muy pequeño (ej. 0.01) con pocos árboles (`n_estimators = 20`)?",
                "options": [
                    "Sobreajuste extremo.",
                    "Subajuste (el modelo no alcanza a corregir suficientemente los errores).",
                    "El modelo ignora las etiquetas de entrenamiento.",
                ],
                "answer": "Subajuste (el modelo no alcanza a corregir suficientemente los errores).",
                "explanation": "Pasos muy pequeños requieren más iteraciones (`n_estimators`) para converger.",
            },
        ],
        page_key=PAGE_KEY,
    )
