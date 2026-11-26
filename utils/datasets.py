"""Carga de datasets de scikit-learn y carga de datasets propios (CSV)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_linnerud,
    load_wine,
    make_circles,
    make_classification,
    make_friedman1,
    make_moons,
    make_regression,
)
from sklearn.preprocessing import LabelEncoder

TaskType = Literal["classification", "regression"]


@dataclass
class DatasetBundle:
    name: str
    description: str
    task: TaskType
    df: pd.DataFrame
    feature_names: list[str]
    target_name: str
    X: pd.DataFrame
    y: np.ndarray
    class_names: list[str] | None = None
    is_custom: bool = False


CLASSIFICATION_DATASETS: dict[str, str] = {
    "Iris (3 clases - Clásico)": "iris",
    "Breast Cancer (Binaria - Diagnóstico médico)": "breast_cancer",
    "Wine (3 clases - Química de vinos)": "wine",
    "Moons 2D (No lineal - Binaria)": "moons",
    "Circles 2D (Concéntricos - Binaria)": "circles",
    "Clasificación Sintética (Multivariable)": "synthetic_clf",
}

REGRESSION_DATASETS: dict[str, str] = {
    "Diabetes (Progresión de enfermedad)": "diabetes",
    "Linnerud (Ejercicio físico vs. Peso)": "linnerud",
    "Regresión Lineal Sintética": "synthetic_reg",
    "Regresión No Lineal (Friedman #1)": "friedman1",
    "Curva Polinomial 1D (Sintética)": "poly_1d",
}


def load_sklearn_dataset(
    dataset_key: str,
    task: TaskType,
    random_state: int = 42,
    n_samples: int = 300,
    noise: float = 0.2,
) -> DatasetBundle:
    """Carga un dataset incluido o generado con scikit-learn."""
    if task == "classification":
        if dataset_key == "iris":
            raw = load_iris(as_frame=True)
            df = raw.frame.copy()
            target_name = "especie"
            df = df.rename(columns={"target": target_name})
            feature_names = list(raw.feature_names)
            class_names = [str(c) for c in raw.target_names]
            desc = (
                "Dataset clásico de 150 flores Iris con 4 medidas de sépalos y pétalos "
                "distribuidas en 3 especies (setosa, versicolor, virginica)."
            )
            return DatasetBundle(
                name="Iris (scikit-learn)",
                description=desc,
                task="classification",
                df=df,
                feature_names=feature_names,
                target_name=target_name,
                X=df[feature_names].copy(),
                y=df[target_name].to_numpy(),
                class_names=class_names,
            )

        if dataset_key == "breast_cancer":
            raw = load_breast_cancer(as_frame=True)
            df = raw.frame.copy()
            target_name = "diagnostico"
            df = df.rename(columns={"target": target_name})
            feature_names = list(raw.feature_names)
            class_names = [str(c) for c in raw.target_names]
            desc = (
                "Dataset de diagnóstico de cáncer de mama (569 muestras, 30 características "
                "numéricas). Clasificación binaria: malignant vs benign."
            )
            return DatasetBundle(
                name="Breast Cancer Wisconsin (scikit-learn)",
                description=desc,
                task="classification",
                df=df,
                feature_names=feature_names,
                target_name=target_name,
                X=df[feature_names].copy(),
                y=df[target_name].to_numpy(),
                class_names=class_names,
            )

        if dataset_key == "wine":
            raw = load_wine(as_frame=True)
            df = raw.frame.copy()
            target_name = "cultivar"
            df = df.rename(columns={"target": target_name})
            feature_names = list(raw.feature_names)
            class_names = [str(c) for c in raw.target_names]
            desc = (
                "Análisis químico de 178 vinos italianos con 13 variables fisicoquímicas "
                "clasificados en 3 cultivares distintos."
            )
            return DatasetBundle(
                name="Wine (scikit-learn)",
                description=desc,
                task="classification",
                df=df,
                feature_names=feature_names,
                target_name=target_name,
                X=df[feature_names].copy(),
                y=df[target_name].to_numpy(),
                class_names=class_names,
            )

        if dataset_key == "moons":
            X_arr, y_arr = make_moons(
                n_samples=n_samples, noise=noise, random_state=random_state
            )
            feature_names = ["x1", "x2"]
            target_name = "clase"
            df = pd.DataFrame(X_arr, columns=feature_names)
            df[target_name] = y_arr
            return DatasetBundle(
                name="Make Moons 2D (scikit-learn)",
                description=(
                    "Dos medias lunas entrelazadas en 2D. Ideal para comparar modelos "
                    "lineales frente a fronteras de decisión no lineales."
                ),
                task="classification",
                df=df,
                feature_names=feature_names,
                target_name=target_name,
                X=df[feature_names].copy(),
                y=y_arr,
                class_names=["Luna 0", "Luna 1"],
            )

        if dataset_key == "circles":
            X_arr, y_arr = make_circles(
                n_samples=n_samples,
                noise=max(0.05, noise * 0.6),
                factor=0.5,
                random_state=random_state,
            )
            feature_names = ["x1", "x2"]
            target_name = "clase"
            df = pd.DataFrame(X_arr, columns=feature_names)
            df[target_name] = y_arr
            return DatasetBundle(
                name="Make Circles 2D (scikit-learn)",
                description=(
                    "Dos círculos concéntricos en 2D. Excelente para visualizar por qué "
                    "los kernels RBF, árboles o KNN superan a modelos estrictamente lineales."
                ),
                task="classification",
                df=df,
                feature_names=feature_names,
                target_name=target_name,
                X=df[feature_names].copy(),
                y=y_arr,
                class_names=["Exterior (0)", "Interior (1)"],
            )

        # Default: synthetic_clf
        X_arr, y_arr = make_classification(
            n_samples=n_samples,
            n_features=6,
            n_informative=4,
            n_redundant=1,
            n_classes=2,
            class_sep=max(0.4, 1.5 - noise * 2),
            random_state=random_state,
        )
        feature_names = [f"var_{i + 1}" for i in range(X_arr.shape[1])]
        target_name = "clase"
        df = pd.DataFrame(X_arr, columns=feature_names)
        df[target_name] = y_arr
        return DatasetBundle(
            name="Clasificación Sintética (scikit-learn)",
            description=(
                "Dataset binario generado con make_classification (4 variables informativas, "
                "1 redundante y 1 de ruido)."
            ),
            task="classification",
            df=df,
            feature_names=feature_names,
            target_name=target_name,
            X=df[feature_names].copy(),
            y=y_arr,
            class_names=["Clase 0", "Clase 1"],
        )

    # Regression datasets
    if dataset_key == "diabetes":
        raw = load_diabetes(as_frame=True)
        df = raw.frame.copy()
        target_name = "progresion"
        df = df.rename(columns={"target": target_name})
        feature_names = list(raw.feature_names)
        return DatasetBundle(
            name="Diabetes (scikit-learn)",
            description=(
                "442 pacientes con 10 variables clínicas normalizadas (edad, sexo, IMC, "
                "presión arterial y 6 mediciones sanguíneas) para predecir la progresión "
                "de la diabetes al cabo de un año."
            ),
            task="regression",
            df=df,
            feature_names=feature_names,
            target_name=target_name,
            X=df[feature_names].copy(),
            y=df[target_name].to_numpy(dtype=float),
        )

    if dataset_key == "linnerud":
        raw = load_linnerud(as_frame=True)
        df = raw.data.copy()
        target_name = "Weight"
        df[target_name] = raw.target[target_name]
        feature_names = list(raw.feature_names)
        return DatasetBundle(
            name="Linnerud (scikit-learn)",
            description=(
                "Dataset pequeño (20 atletas) que relaciona ejercicios físicos "
                "(Chins, Situps, Jumps) con el peso corporal (Weight). Útil para ver "
                "el efecto del tamaño muestral reducido y la varianza."
            ),
            task="regression",
            df=df,
            feature_names=feature_names,
            target_name=target_name,
            X=df[feature_names].copy(),
            y=df[target_name].to_numpy(dtype=float),
        )

    if dataset_key == "friedman1":
        X_arr, y_arr = make_friedman1(
            n_samples=n_samples,
            n_features=6,
            noise=noise * 5.0,
            random_state=random_state,
        )
        feature_names = [f"x{i + 1}" for i in range(X_arr.shape[1])]
        target_name = "objetivo_y"
        df = pd.DataFrame(X_arr, columns=feature_names)
        df[target_name] = y_arr
        return DatasetBundle(
            name="Friedman #1 No Lineal (scikit-learn)",
            description=(
                "Problema de regresión no lineal clásico (make_friedman1): las primeras "
                "5 variables tienen relaciones sinusoidales y cuadráticas con la salida."
            ),
            task="regression",
            df=df,
            feature_names=feature_names,
            target_name=target_name,
            X=df[feature_names].copy(),
            y=y_arr.astype(float),
        )

    if dataset_key == "poly_1d":
        rng = np.random.default_rng(random_state)
        X_arr = np.sort(rng.uniform(-3.0, 3.0, size=(n_samples, 1)), axis=0)
        y_arr = (
            0.5 * X_arr[:, 0] ** 3
            - 1.2 * X_arr[:, 0] ** 2
            + 0.8 * X_arr[:, 0]
            + 2.0
            + rng.normal(0.0, max(0.2, noise * 4.0), size=n_samples)
        )
        feature_names = ["x"]
        target_name = "y"
        df = pd.DataFrame({"x": X_arr[:, 0], "y": y_arr})
        return DatasetBundle(
            name="Curva Polinomial 1D (Sintética)",
            description=(
                "Relación no lineal 1D (polinomio cúbico con ruido). Perfecta para "
                "visualizar subajuste (underfitting) y sobreajuste (overfitting)."
            ),
            task="regression",
            df=df,
            feature_names=feature_names,
            target_name=target_name,
            X=df[feature_names].copy(),
            y=y_arr.astype(float),
        )

    # Default: synthetic_reg
    X_arr, y_arr = make_regression(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        noise=noise * 35.0,
        random_state=random_state,
    )
    feature_names = [f"var_{i + 1}" for i in range(X_arr.shape[1])]
    target_name = "objetivo_y"
    df = pd.DataFrame(X_arr, columns=feature_names)
    df[target_name] = y_arr
    return DatasetBundle(
        name="Regresión Lineal Sintética (scikit-learn)",
        description=(
            "Dataset generado con make_regression (5 variables, 3 informativas y "
            "ruido gaussiano configurable)."
        ),
        task="regression",
        df=df,
        feature_names=feature_names,
        target_name=target_name,
        X=df[feature_names].copy(),
        y=y_arr.astype(float),
    )


def render_dataset_sidebar(
    task: TaskType,
    default_dataset_key: str,
    page_key: str,
) -> DatasetBundle | None:
    """Renderiza en el sidebar la selección de dataset de scikit-learn o carga de CSV propio."""
    st.sidebar.subheader("1. Fuente de Datos")

    source_options = ["Dataset de scikit-learn", "Cargar dataset propio (CSV)"]
    has_saved_csv = "custom_csv_df" in st.session_state and st.session_state["custom_csv_df"] is not None
    default_source_idx = 0

    source = st.sidebar.radio(
        "Origen de los datos",
        options=source_options,
        index=default_source_idx,
        key=f"{page_key}_data_source",
        help="Puedes practicar con datasets incluidos de scikit-learn o subir tu propio archivo CSV.",
    )

    if source == "Dataset de scikit-learn":
        catalog = CLASSIFICATION_DATASETS if task == "classification" else REGRESSION_DATASETS
        labels = list(catalog.keys())
        keys = list(catalog.values())
        default_idx = keys.index(default_dataset_key) if default_dataset_key in keys else 0

        selected_label = st.sidebar.selectbox(
            "Dataset de scikit-learn",
            options=labels,
            index=default_idx,
            key=f"{page_key}_sk_dataset",
        )
        selected_key = catalog[selected_label]

        n_samples = 300
        noise = 0.2
        if selected_key in {"moons", "circles", "synthetic_clf", "synthetic_reg", "friedman1", "poly_1d"}:
            with st.sidebar.expander("Parámetros del dataset sintético", expanded=False):
                n_samples = st.slider(
                    "Número de muestras",
                    min_value=80,
                    max_value=1000,
                    value=300,
                    step=20,
                    key=f"{page_key}_n_samples",
                )
                noise = st.slider(
                    "Nivel de ruido",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.25,
                    step=0.05,
                    key=f"{page_key}_noise",
                )

        seed = st.sidebar.number_input(
            "Semilla aleatoria (random_state)",
            min_value=0,
            max_value=9999,
            value=42,
            step=1,
            key=f"{page_key}_dataset_seed",
        )

        return load_sklearn_dataset(
            dataset_key=selected_key,
            task=task,
            random_state=int(seed),
            n_samples=int(n_samples),
            noise=float(noise),
        )

    # Custom CSV flow
    sep_label = st.sidebar.selectbox(
        "Separador del CSV",
        options=["Coma (,)", "Punto y coma (;)", "Tabulación (\\t)"],
        key=f"{page_key}_csv_sep",
    )
    sep_map = {"Coma (,)": ",", "Punto y coma (;)": ";", "Tabulación (\\t)": "\t"}
    sep = sep_map[sep_label]

    uploaded_file = st.sidebar.file_uploader(
        "Sube un archivo CSV",
        type=["csv"],
        key=f"{page_key}_csv_uploader",
        help="El archivo permanecerá disponible mientras navegues entre páginas.",
    )

    if uploaded_file is not None:
        try:
            df_custom = pd.read_csv(uploaded_file, sep=sep)
            st.session_state["custom_csv_df"] = df_custom
            st.session_state["custom_csv_name"] = uploaded_file.name
        except Exception as exc:
            st.sidebar.error(f"No se pudo leer el archivo CSV: {exc}")
            return None
    elif not has_saved_csv:
        st.info(
            "Sube un archivo `.csv` en la barra lateral izquierda (o cárgalo desde la página **Inicio**) "
            "para analizar tu propio dataset. También encontrarás ejemplos listos en la carpeta `data/`."
        )
        return None

    df_custom = st.session_state["custom_csv_df"].copy()
    csv_name = st.session_state.get("custom_csv_name", "dataset_propio.csv")
    st.sidebar.success(f"Archivo activo: {csv_name} ({df_custom.shape[0]} filas, {df_custom.shape[1]} cols)")

    if st.sidebar.button("Limpiar CSV cargado", key=f"{page_key}_clear_csv"):
        st.session_state["custom_csv_df"] = None
        st.session_state["custom_csv_name"] = None
        st.rerun()

    if df_custom.shape[1] < 2:
        st.error("El dataset necesita al menos 2 columnas (1 variable predictora y 1 objetivo).")
        return None

    cols = list(df_custom.columns)
    target_col = st.sidebar.selectbox(
        "Columna objetivo (y)",
        options=cols,
        index=len(cols) - 1,
        key=f"{page_key}_target_col",
    )

    available_features = [c for c in cols if c != target_col]
    selected_features = st.sidebar.multiselect(
        "Variables predictoras (X)",
        options=available_features,
        default=available_features,
        key=f"{page_key}_feature_cols",
    )

    if not selected_features:
        st.warning("Selecciona al menos una variable predictora (X) en la barra lateral.")
        return None

    # Drop rows with missing target
    valid_mask = df_custom[target_col].notna()
    dropped = int((~valid_mask).sum())
    if dropped > 0:
        st.sidebar.warning(f"Se descartaron {dropped} filas con valor nulo en '{target_col}'.")
    df_clean = df_custom.loc[valid_mask, selected_features + [target_col]].copy()

    if len(df_clean) < 10:
        st.error("El dataset filtrado tiene menos de 10 filas válidas. Agrega más datos.")
        return None

    X = df_clean[selected_features].copy()
    y_series = df_clean[target_col]

    if task == "regression":
        y_num = pd.to_numeric(y_series, errors="coerce")
        if y_num.isna().any():
            st.error(
                f"Para **Regresión**, la columna objetivo `{target_col}` debe ser numérica. "
                "Selecciona otra columna o cambia a una tarea de Clasificación."
            )
            return None
        return DatasetBundle(
            name=f"Dataset propio: {csv_name}",
            description=f"Dataset cargado por el usuario (`{csv_name}`) con objetivo numérico `{target_col}`.",
            task="regression",
            df=df_clean,
            feature_names=selected_features,
            target_name=target_col,
            X=X,
            y=y_num.to_numpy(dtype=float),
            is_custom=True,
        )

    # Classification target encoding
    unique_count = y_series.nunique()
    if unique_count < 2:
        st.error(f"La columna objetivo `{target_col}` debe tener al menos 2 clases distintas.")
        return None
    if unique_count > 25:
        st.error(
            f"La columna `{target_col}` tiene {unique_count} valores únicos (parece continua). "
            "Para clasificación selecciona una columna categórica o discreta (≤ 25 clases)."
        )
        return None

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_series.astype(str))
    class_names = [str(c) for c in le.classes_]

    return DatasetBundle(
        name=f"Dataset propio: {csv_name}",
        description=(
            f"Dataset cargado por el usuario (`{csv_name}`) con objetivo categórico "
            f"`{target_col}` ({len(class_names)} clases)."
        ),
        task="classification",
        df=df_clean,
        feature_names=selected_features,
        target_name=target_col,
        X=X,
        y=y_encoded,
        class_names=class_names,
        is_custom=True,
    )
