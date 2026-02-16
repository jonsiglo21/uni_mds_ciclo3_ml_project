# Proyecto MLOps: "Análisis y creación de modelo para detectar deserción de clientes"

## 📋 Descripción del Proyecto
Proyecto final del curso "Introduction to MLOps" que implementa un modelo de Machine Learning para predecir la deserción de clientes de una entidad financiera peruana.

## 🎯 Objetivo
Desarrollar un modelo de clasificación para predecir la probabilidad de deserción de clientes, permitiendo implementar estrategias de retención proactivas.

## 📊 Dataset
- **train_clientes.csv**: 70,000 clientes, 60 variables
- **train_requerimientos.csv**: 51,417 registros de requerimientos

## 🛠️ Tecnologías
- Python 3.12
- Pandas 2.2.3 / NumPy 2.2.4
- Scikit-learn 1.6.1
- FastAPI 0.115.11
- MLflow 2.20.2

## 📁 Estructura del Proyecto
uni_mds_ciclo3_ml_project/
├── .github/                     # Configuración de GitHub (opcional)
├── data/
│   ├── raw/                     # Datos originales CSV
│   └── processed/               # Datos procesados para entrenamiento
├── notebooks/                   # Jupyter notebooks para experimentación
├── src/
│   ├── __init__.py
│   ├── data_preparation.py      # Transformación de datos
│   ├── train.py                 # Entrenamiento del modelo
│   └── serving.py               # API para predicciones
├── models/                      # Modelos serializados (.pkl)
├── reports/                     # Reportes, gráficas, resultados
│   ├── figures/
│   └── metrics/
├── experiments/                 # Experimentos con MLflow (opcional)
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Documentación principal