# api/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ClienteInput(BaseModel):
    """Estructura de datos de entrada para predicción."""
    FLG_BANCARIZADO: int = Field(..., ge=0, le=1, description="Flag bancarizado")
    FLAG_LIMA_PROVINCIA: str = Field(..., description="Lima o Provincia")
    EDAD: float = Field(..., ge=0, le=120, description="Edad del cliente")
    ANTIGUEDAD: float = Field(..., ge=0, description="Antigüedad en meses")
    RANG_INGRESO: Optional[str] = Field(None, description="Rango de ingresos")

class PrediccionResponse(BaseModel):
    """Estructura de respuesta de predicción."""
    probabilidad_desercion: float
    clase_predicha: int
    umbral: float = 0.5
    modelo_version: str = "random_forest_champion"

class BatchPrediccionRequest(BaseModel):
    """Solicitud de predicción por lote."""
    clientes: List[ClienteInput]

class BatchPrediccionResponse(BaseModel):
    """Respuesta de predicción por lote."""
    predicciones: List[PrediccionResponse]

class HealthResponse(BaseModel):
    """Respuesta del health check."""
    status: str
    model_loaded: bool
    timestamp: str

class ModelInfoResponse(BaseModel):
    """Información del modelo."""
    model_type: str
    features: List[str]
    total_features: int
    parameters: Dict[str, Any]