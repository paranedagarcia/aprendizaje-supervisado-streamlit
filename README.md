# 🎓 Laboratorio Interactivo de Aprendizaje Supervisado (Streamlit)

Aplicación multipágina en Python con **Streamlit** y **scikit-learn** orientada al aprendizaje y práctica de algoritmos de **Aprendizaje Supervisado**.

## Estructura del Proyecto

```text
aprendizaje-supervisado-streamlit/
├── app.py                                # Página de inicio y gestor de CSV propio
├── pages/
│   ├── 1_Regresion_Lineal.py             # Regresión (OLS, Ridge, Lasso, ElasticNet + Polinomial)
│   ├── 2_Regresion_Logistica.py          # Clasificación (Binaria y Multiclase)
│   ├── 3_Arboles_de_Decision.py          # Clasificación y Regresión
│   ├── 4_Random_Forest.py                # Clasificación y Regresión
│   ├── 5_Support_Vector_Machine.py       # Clasificación (SVC) y Regresión (SVR)
│   ├── 6_Gradient_Boosting.py            # Clasificación y Regresión
│   └── 7_KNN.py                          # Clasificación y Regresión
├── utils/
│   ├── datasets.py                       # Datasets de scikit-learn y carga de CSV propios
│   ├── preprocessing.py                  # Imputación, OneHotEncoder, StandardScaler y split
│   ├── metrics.py                        # Métricas de clasificación y regresión
│   ├── visualization.py                  # Gráficos Plotly (ROC, CM, fronteras 2D, residuos, etc.)
│   └── ui.py                             # Componentes comunes (EDA, What-If, ejercicios)
├── data/
│   ├── ejemplo_clasificacion_clientes.csv
│   └── ejemplo_regresion_viviendas.csv
└── requirements.txt
```

## Funcionalidades Principales

- **7 páginas independientes en `pages/`**, una por cada tema de aprendizaje supervisado.
- **Datasets incluidos de `scikit-learn`** (Iris, Breast Cancer, Wine, Moons, Circles, Diabetes, Linnerud, Friedman #1, etc.) sin necesidad de descargas externas.
- **Carga de datasets propios (`.csv`)** desde la página de inicio o el sidebar de cualquier página, con soporte automático para valores nulos y variables categóricas.
- **Ajuste de hiperparámetros generales en el sidebar** en tiempo real para cada algoritmo.
- **5 áreas de trabajo por página**: Teoría y Conceptos, Exploración (EDA), Entrenamiento y Evaluación, Predicción Manual (*What-If*) y Ejercicios con Autoevaluación.

## Instalación y Ejecución

### Ejecución rápida (recomendada en Windows)

Si Windows Control de aplicaciones bloquea el ejecutable `streamlit.exe`, inicia la aplicación con `python -m streamlit` o mediante el script incluido `run_app.ps1`:

```powershell
# Opción 1: Script lanzador directo
.\run_app.ps1

# Opción 2: Con el entorno virtual activado
.\.venv\Scripts\Activate.ps1
streamlit run .\app.py

# Opción 3: Como módulo de Python
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

### Reinstalación desde cero

```powershell
# Con pip
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

O usando `uv`:

```powershell
uv venv
uv pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```
