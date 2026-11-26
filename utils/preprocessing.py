"""Pipeline de preprocesamiento y división train/test segura para datasets numéricos y mixtos."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """Crea un ColumnTransformer que imputa nulos, codifica categóricas y opcionalmente escala numéricas."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []

    if numeric_cols:
        num_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), numeric_cols))

    if categorical_cols:
        cat_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
        transformers.append(("cat", Pipeline(cat_steps), categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_transformed_feature_names(preprocessor: ColumnTransformer, original_cols: list[str]) -> list[str]:
    """Obtiene nombres legibles de las características tras el ColumnTransformer."""
    try:
        names = list(preprocessor.get_feature_names_out())
        return [n.replace("num__", "").replace("cat__", "") for n in names]
    except Exception:
        return list(original_cols)


def split_dataset(
    X: pd.DataFrame,
    y: np.ndarray,
    task: str,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """Realiza train_test_split estratificado para clasificación cuando es posible."""
    stratify = None
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        if np.all(counts >= 2) and int(len(y) * test_size) >= len(counts):
            stratify = y
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
