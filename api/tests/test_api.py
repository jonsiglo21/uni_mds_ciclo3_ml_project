"""
Pruebas para la API de predicción de churn.
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Determinar la ruta raíz del proyecto (donde está api/, src/, etc.)
current_file = Path(__file__).resolve()  # Ruta absoluta a este archivo
project_root = current_file.parent.parent.parent  # Sube: tests/ -> api/ -> raíz

# Agregar raíz del proyecto al path para imports
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Ahora importamos desde api.serving (funciona porque la raíz está en sys.path)
from api.serving import app

# Crear cliente de prueba
client = TestClient(app)

def test_health_endpoint():
    """Prueba el endpoint de health."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_predict_endpoint_no_data():
    """Prueba el endpoint de predicción sin datos (debe fallar)."""
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Unprocessable Entity (error de validación)

def test_predict_endpoint_invalid_data():
    """Prueba el endpoint con datos inválidos."""
    invalid_data = {
        "cliente_id": 123,
        "features": [1, 2, 3]  # Esto no es válido según el modelo esperado
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Unprocessable Entity

# NOTA: Para probar con datos reales, necesitarás:
# 1. Tener un modelo entrenado en models/random_forest_champion.pkl
# 2. Tener el archivo feature_columns.txt en data/processed/training/

@pytest.mark.skip(reason="Requiere modelo entrenado")
def test_predict_endpoint_with_real_data():
    """Prueba el endpoint con datos reales (requiere modelo entrenado)."""
    # Esta prueba se saltará hasta que tengamos un modelo entrenado
    real_data = {
        "cliente_id": 64216,
        "features": {
            "FLG_BANCARIZADO": 1,
            "EDAD": 56,
            "ANTIGUEDAD": 0,
            # ... más features según tu modelo
        }
    }
    response = client.post("/predict", json=real_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediccion" in data
    assert "probabilidad" in data
    assert "cliente_id" in data