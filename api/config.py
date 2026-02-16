# api/config.py
from pydantic_settings import BaseSettings
from pathlib import Path
import sys

# ============================================================
# FIX: Obtener la raíz del proyecto
# ============================================================
current_file = Path(__file__).resolve()  # api/config.py
project_root = current_file.parent.parent  # sube: api/ -> raíz

class Settings(BaseSettings):
    """Configuración de la API."""
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Model paths (USANDO PROJECT_ROOT)
    MODEL_PATH: Path = project_root / "models" / "random_forest_champion_grid.pkl"
    FEATURES_PATH: Path = project_root / "data" / "processed" / "training" / "feature_columns.txt"
    MAPPINGS_PATH: Path = project_root / "data" / "processed" / "training" / "category_mappings.pkl"
    
    # Model config
    PREDICTION_THRESHOLD: float = 0.5
    
    # Optional: usar .env file en la raíz
    class Config:
        env_file = project_root / ".env"
        case_sensitive = True

# Instancia global
settings = Settings()

# Para verificar (opcional)
if __name__ == "__main__":
    print("🔧 Configuración cargada:")
    print(f"   📁 Proyecto: {project_root}")
    print(f"   📦 Modelo: {settings.MODEL_PATH}")
    print(f"   ✅ Existe: {settings.MODEL_PATH.exists()}")
    print(f"   📋 Features: {settings.FEATURES_PATH}")
    print(f"   ✅ Existe: {settings.FEATURES_PATH.exists()}")