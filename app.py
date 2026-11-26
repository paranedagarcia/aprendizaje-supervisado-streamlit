"""Página principal: Laboratorio Interactivo de Aprendizaje Supervisado."""

from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Laboratorio de Aprendizaje Supervisado",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Laboratorio de Aprendizaje Supervisado")
st.markdown(
    """
    Bienvenido a esta aplicación multipágina diseñada para **aprender, experimentar y ejercitar**
    los principales algoritmos de **Aprendizaje Supervisado** utilizando `scikit-learn` y `Streamlit`.

    Usa el **menú de navegación en la barra lateral izquierda** para acceder a cada tema.
    En cada página encontrarás **5 áreas de trabajo**:
    1. **Teoría y Conceptos**: intuición, formulación, ventajas, limitaciones y guía de hiperparámetros.
    2. **Exploración de Datos (EDA)**: inspección del dataset, estadísticas, distribución del objetivo y correlaciones.
    3. **Entrenamiento y Evaluación**: ajuste de hiperparámetros en tiempo real en el sidebar, métricas *Train vs. Test* y visualizaciones interactivas.
    4. **Predicción Interactiva (What-If)**: simulador manual para probar predicciones con nuevos valores.
    5. **Ejercicios y Práctica**: retos guiados de experimentación y preguntas de autoevaluación.
    """
)

st.divider()

st.subheader("🗺️ Mapa de Temas Disponibles")

topics_df = pd.DataFrame(
    [
        {
            "Página": "1. Regresión Lineal",
            "Tipo de Tarea": "Regresión",
            "Modelos incluidos": "OLS, Ridge (L2), Lasso (L1), ElasticNet + Polinomial",
            "Datasets ejemplo (sklearn)": "Diabetes, Linnerud, Lineal Sintética, Curva 1D",
            "Concepto clave": "Relación lineal, regularización y sesgo-varianza",
        },
        {
            "Página": "2. Regresión Logística",
            "Tipo de Tarea": "Clasificación",
            "Modelos incluidos": "LogisticRegression (Binaria y Multiclase)",
            "Datasets ejemplo (sklearn)": "Breast Cancer, Iris, Wine, Moons 2D",
            "Concepto clave": "Función sigmoide, probabilidades, umbral y ROC-AUC",
        },
        {
            "Página": "3. Árboles de Decisión",
            "Tipo de Tarea": "Clasificación y Regresión",
            "Modelos incluidos": "DecisionTreeClassifier / DecisionTreeRegressor",
            "Datasets ejemplo (sklearn)": "Iris, Wine, Moons, Diabetes, Friedman #1",
            "Concepto clave": "Impureza (Gini/Entropía/MSE), profundidad y poda (ccp_alpha)",
        },
        {
            "Página": "4. Random Forest",
            "Tipo de Tarea": "Clasificación y Regresión",
            "Modelos incluidos": "RandomForestClassifier / RandomForestRegressor",
            "Datasets ejemplo (sklearn)": "Breast Cancer, Wine, Diabetes, Friedman #1",
            "Concepto clave": "Bagging, ensamble de árboles, OOB Score e importancia",
        },
        {
            "Página": "5. Support Vector Machine",
            "Tipo de Tarea": "Clasificación y Regresión",
            "Modelos incluidos": "SVC / SVR (Kernels: RBF, Linear, Poly, Sigmoid)",
            "Datasets ejemplo (sklearn)": "Moons 2D, Circles 2D, Iris, Diabetes, Curva 1D",
            "Concepto clave": "Margen máximo, vectores de soporte y truco del kernel",
        },
        {
            "Página": "6. Gradient Boosting",
            "Tipo de Tarea": "Clasificación y Regresión",
            "Modelos incluidos": "GradientBoostingClassifier / GradientBoostingRegressor",
            "Datasets ejemplo (sklearn)": "Breast Cancer, Moons, Diabetes, Friedman #1",
            "Concepto clave": "Boosting secuencial, learning rate y curva por iteración",
        },
        {
            "Página": "7. KNN (K-Nearest Neighbors)",
            "Tipo de Tarea": "Clasificación y Regresión",
            "Modelos incluidos": "KNeighborsClassifier / KNeighborsRegressor",
            "Datasets ejemplo (sklearn)": "Iris, Moons 2D, Wine, Diabetes, Curva 1D",
            "Concepto clave": "Aprendizaje basado en instancias, distancias y selección de K",
        },
    ]
)

st.dataframe(topics_df, width="stretch", hide_index=True)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("🧭 Clasificación vs. Regresión")
    st.markdown(
        """
        - **Regresión**: la variable objetivo ($y$) es **continua** (ej. precio, temperatura, progresión médica).
          Se evalúa con métricas como **$R^2$**, **RMSE** y **MAE**.
        - **Clasificación**: la variable objetivo ($y$) es **categórica o discreta** (ej. sano/enfermo, especie de flor).
          Se evalúa con **Accuracy**, **Precision**, **Recall**, **F1-Score**, **Matriz de Confusión** y **ROC-AUC**.
        """
    )

with col2:
    st.subheader("📂 Cargar tu Propio Dataset (CSV)")
    st.markdown(
        """
        Puedes cargar tu propio archivo `.csv` aquí o directamente desde la barra lateral de **cualquier página**.
        El archivo quedará guardado en la sesión para que puedas comparar cómo se comportan distintos modelos sobre tus propios datos.
        """
    )

    sep_label = st.selectbox(
        "Separador del archivo CSV",
        options=["Coma (,)", "Punto y coma (;)", "Tabulación (\\t)"],
        key="home_csv_sep",
    )
    sep_map = {"Coma (,)": ",", "Punto y coma (;)": ";", "Tabulación (\\t)": "\t"}

    uploaded = st.file_uploader(
        "Subir CSV para usar en todas las páginas",
        type=["csv"],
        key="home_csv_uploader",
    )
    if uploaded is not None:
        try:
            df_user = pd.read_csv(uploaded, sep=sep_map[sep_label])
            st.session_state["custom_csv_df"] = df_user
            st.session_state["custom_csv_name"] = uploaded.name
            st.success(
                f"Dataset `{uploaded.name}` cargado ({df_user.shape[0]} filas, {df_user.shape[1]} columnas). "
                "Ahora ve a cualquier página del menú lateral y selecciona **Cargar dataset propio (CSV)**."
            )
            st.dataframe(df_user.head(10), width="stretch")
        except Exception as exc:
            st.error(f"Error al leer el archivo CSV: {exc}")
    elif st.session_state.get("custom_csv_df") is not None:
        df_saved = st.session_state["custom_csv_df"]
        st.info(
            f"Dataset propio activo en sesión: **{st.session_state.get('custom_csv_name')}** "
            f"({df_saved.shape[0]} filas, {df_saved.shape[1]} columnas)."
        )

st.divider()
st.caption(
    "Consejo pedagógico: al ajustar los hiperparámetros en cada página, compara siempre las métricas de "
    "**Train** frente a **Test** para identificar cuándo un modelo está subajustando o sobreajustando."
)
