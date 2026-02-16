"""
API para servir modelo de predicción de deserción de clientes.
Versión con umbral ÓPTIMO AUTOMÁTICO desde metadatos.
Rutas absolutas y preprocesamiento sin errores de tipos.
Incluye model_file en respuestas y endpoints de debug.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Optional, Any
import uvicorn
from pathlib import Path
import time
from contextlib import asynccontextmanager
from datetime import datetime
import traceback

# ============================================================
# CONFIGURACIÓN DE RUTAS ABSOLUTAS
# ============================================================
import sys
current_file = Path(__file__).resolve()  # api/serving.py
project_root = current_file.parent.parent  # sube: api/ -> raíz
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"📁 Raíz del proyecto: {project_root}")

# Importar modelos Pydantic
# from api.models import ClienteInput, PrediccionResponse, BatchPrediccionRequest, BatchPrediccionResponse, HealthResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# ============================================================
# NUEVOS MODELOS PARA FEATURES RAW
# ============================================================

class ClienteRawInput(BaseModel):
    """Acepta cualquier feature que el modelo necesite."""
    class Config:
        extra = "allow"  # Permite campos adicionales
    
    # No definimos campos fijos, acepta cualquier key-value

class PrediccionResponse(BaseModel):
    """Respuesta de predicción para un cliente."""
    probabilidad_desercion: float = Field(..., ge=0, le=1, description="Probabilidad de deserción")
    clase_predicha: int = Field(..., ge=0, le=1, description="Clase predicha (0=no deserta, 1=deserta)")
    umbral: float = Field(..., ge=0, le=1, description="Umbral utilizado para la clasificación")
    modelo_version: str = Field(..., description="Versión del modelo")
    tiempo_procesamiento_ms: Optional[float] = Field(None, description="Tiempo de procesamiento en ms")
    model_file: Optional[str] = Field(None, description="Nombre del archivo del modelo")
    timestamp_prediccion: Optional[str] = Field(default_factory=lambda: datetime.now().isoformat())

class BatchRawRequest(BaseModel):
    """Solicitud de predicción para múltiples clientes con features raw."""
    clientes: List[Dict[str, Any]] = Field(..., max_items=100, description="Lista de clientes (máx 100)")

class BatchPrediccionResponse(BaseModel):
    """Respuesta de predicción para múltiples clientes."""
    predicciones: List[PrediccionResponse]

class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud."""
    status: str
    model_loaded: bool
    timestamp: str
    features_loaded: int
    threshold_actual: float

# Configuración logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURACIÓN DE RUTAS DE ARCHIVOS
# ============================================================
MODEL_PATH = project_root / "models" / "random_forest_champion_bayesian.pkl"
FEATURES_PATH = project_root / "data" / "processed" / "training" / "feature_columns.txt"
MAPPINGS_PATH = project_root / "data" / "processed" / "training" / "category_mappings.pkl"

print(f"📦 Modelo: {MODEL_PATH}")
print(f"📋 Features: {FEATURES_PATH}")
print(f"🗺️ Mappings: {MAPPINGS_PATH}")

# Variables globales
model = None
feature_cols = []
category_mappings = {}
model_metadata = {}
optimal_threshold = 0.5

def find_latest_model() -> Path:
    """
    Busca automáticamente el modelo más reciente en la carpeta models/.
    """
    models_dir = project_root / "models"
    if not models_dir.exists():
        logger.warning(f"⚠️ Directorio models/ no encontrado en: {models_dir}")
        return MODEL_PATH
    
    # Buscar modelos champion (prioridad)
    champion_models = list(models_dir.glob("*champion*.pkl"))
    if champion_models:
        latest = max(champion_models, key=lambda p: p.stat().st_mtime)
        logger.info(f"🔍 Modelo champion encontrado: {latest.name}")
        return latest
    
    # Si no hay champion, buscar cualquier modelo .pkl
    all_models = list(models_dir.glob("*.pkl"))
    if all_models:
        latest = max(all_models, key=lambda p: p.stat().st_mtime)
        logger.info(f"🔍 Modelo más reciente encontrado: {latest.name}")
        return latest
    
    logger.warning(f"⚠️ No se encontraron modelos en {models_dir}")
    return MODEL_PATH

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Maneja el ciclo de vida de la aplicación.
    Carga el modelo al iniciar y libera recursos al final.
    """
    global model, feature_cols, category_mappings, model_metadata, optimal_threshold, MODEL_PATH
    
    logger.info("="*60)
    logger.info("🚀 Iniciando API de Predicción de Churn (MODO RAW FEATURES)")
    logger.info("="*60)
    
    # Buscar el modelo más reciente
    MODEL_PATH = find_latest_model()
    
    if not MODEL_PATH.exists():
        logger.error(f"❌ No se encontró modelo en {MODEL_PATH}")
        logger.error("   La API funcionará en modo degradado (solo health check)")
    else:
        try:
            logger.info(f"📦 Cargando modelo desde: {MODEL_PATH}")
            
            # Cargar modelo y metadatos
            model_data = joblib.load(MODEL_PATH)
            
            if isinstance(model_data, dict) and 'model' in model_data:
                model = model_data['model']
                model_metadata = model_data.get('metadata', {})
                logger.info("✅ Modelo cargado desde formato diccionario")
            else:
                model = model_data
                model_metadata = {}
                logger.info("✅ Modelo cargado directamente")
            
            logger.info(f"   📊 Tipo de modelo: {type(model).__name__}")
            
            # Extraer umbral óptimo
            if model_metadata and 'best_threshold' in model_metadata:
                optimal_threshold = float(model_metadata['best_threshold'])
                logger.info(f"   🎯 Umbral óptimo desde metadatos: {optimal_threshold:.4f}")
            elif model_metadata and 'metrics' in model_metadata:
                metrics = model_metadata['metrics']
                if 'best_threshold' in metrics:
                    optimal_threshold = float(metrics['best_threshold'])
                    logger.info(f"   🎯 Umbral óptimo desde métricas: {optimal_threshold:.4f}")
            elif hasattr(model, 'best_threshold'):
                optimal_threshold = float(model.best_threshold)
                logger.info(f"   🎯 Umbral óptimo desde atributo: {optimal_threshold:.4f}")
            else:
                logger.info(f"   ℹ️ Usando umbral por defecto: {optimal_threshold}")
            
            # Cargar features
            if FEATURES_PATH.exists():
                with open(FEATURES_PATH, 'r') as f:
                    feature_cols = [line.strip() for line in f.readlines()]
                logger.info(f"   📋 Features desde archivo: {len(feature_cols)}")
            elif model_metadata and 'feature_columns' in model_metadata:
                feature_cols = model_metadata['feature_columns']
                logger.info(f"   📋 Features desde metadata: {len(feature_cols)}")
            elif hasattr(model, 'feature_names_in_'):
                feature_cols = list(model.feature_names_in_)
                logger.info(f"   📋 Features inferidas: {len(feature_cols)}")
            
            # Mostrar información del modelo
            if model_metadata and 'best_params' in model_metadata:
                logger.info("   🎯 Mejores parámetros:")
                for param, value in model_metadata['best_params'].items():
                    logger.info(f"      {param}: {value}")
            
            if model_metadata and 'metrics' in model_metadata:
                metrics = model_metadata['metrics']
                if 'f1_optimized' in metrics:
                    logger.info(f"   📊 F1 optimizado: {metrics['f1_optimized']:.4f}")
                if 'recall_class_1' in metrics:
                    logger.info(f"   📊 Recall clase 1: {metrics['recall_class_1']:.2%}")
            
            logger.info("✅ API lista para recibir peticiones (formato RAW)")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {str(e)}")
            logger.error(traceback.format_exc())
            model = None
    
    yield
    
    logger.info("🔄 Cerrando API - Liberando recursos")
    model = None

# Inicializar app
app = FastAPI(
    title="API de Predicción de Deserción de Clientes (RAW)",
    description="Acepta features raw del modelo (FLG_SEGURO_MENOS0, NRO_ACCES_CANAL3_MENOS0, etc.)",
    version="2.0.0",
    lifespan=lifespan,
    contact={
        "name": "Equipo de Machine Learning",
        "email": "ml@empresa.com",
    }
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz con información básica."""
    return {
        "message": "API de Predicción de Churn (MODO RAW)",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "info": "/info",
        "threshold": "/threshold",
        "status": "operational" if model is not None else "degraded"
    }

@app.get("/health", response_model=HealthResponse, tags=["Salud"])
async def health():
    """Verifica el estado de la API y el modelo."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        features_loaded=len(feature_cols) if feature_cols else 0,
        threshold_actual=optimal_threshold
    )

@app.get("/info", tags=["Modelo"])
async def model_info():
    """Obtiene información detallada del modelo."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )
    
    info = {
        "model_type": type(model).__name__,
        "features": feature_cols[:15],  # Primeras 15 para no saturar
        "total_features": len(feature_cols),
        "optimal_threshold": optimal_threshold,
        "model_loaded": True,
        "model_file": MODEL_PATH.name,
        "model_path": str(MODEL_PATH)
    }
    
    if model_metadata:
        info["metadata"] = {
            "best_params": model_metadata.get('best_params', {}),
            "metrics": model_metadata.get('metrics', {}),
            "timestamp": model_metadata.get('timestamp', 'unknown')
        }
    
    return info

@app.post("/predict", response_model=PrediccionResponse, tags=["Predicción"])
async def predict_raw(cliente: Dict[Any, Any]):
    """
    Predice la probabilidad de deserción para un cliente usando features RAW.
    Acepta cualquier diccionario con las features que espera el modelo.
    """
    global model, feature_cols, optimal_threshold
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )
    
    start_time = time.time()
    
    try:
        logger.info("📥 Recibida petición para cliente (RAW)")
        
        # Convertir input a DataFrame
        df = pd.DataFrame([cliente])
        
        # ====================================================
        # VALIDACIÓN Y PREPARACIÓN
        # ====================================================
        
        # Verificar que tenemos todas las features necesarias
        if feature_cols:
            missing_cols = set(feature_cols) - set(df.columns)
            if missing_cols:
                logger.warning(f"⚠️ Features faltantes: {missing_cols}")
                # Completar con 0 las features faltantes
                for col in missing_cols:
                    df[col] = 0
            
            # Ordenar columnas como espera el modelo
            df = df[feature_cols]
        else:
            # Si no tenemos lista de features, usar las que vienen
            logger.warning("⚠️ No hay lista de features definida, usando las del input")
        
        # ====================================================
        # PREDICCIÓN
        # ====================================================
        
        proba = model.predict_proba(df)[0, 1]
        pred_opt = int(proba >= optimal_threshold)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Predicción completada en {processing_time:.2f}ms")
        logger.info(f"   Probabilidad: {proba:.4f}")
        logger.info(f"   Umbral: {optimal_threshold:.4f}")
        logger.info(f"   Clase: {pred_opt}")
        logger.info(f"   Modelo: {MODEL_PATH.name}")
        
        return PrediccionResponse(
            probabilidad_desercion=float(proba),
            clase_predicha=pred_opt,
            umbral=optimal_threshold,
            modelo_version=MODEL_PATH.stem,
            tiempo_procesamiento_ms=processing_time,
            model_file=MODEL_PATH.name,
            timestamp_prediccion=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error en predicción: {str(e)}")
        logger.error(f"   Tipo: {type(e).__name__}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en predicción: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPrediccionResponse, tags=["Predicción"])
async def predict_batch_raw(request: BatchRawRequest):
    """
    Predice la probabilidad de deserción para múltiples clientes usando features RAW.
    """
    global model, feature_cols, optimal_threshold
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )
    
    if len(request.clientes) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Máximo 100 clientes por petición batch"
        )
    
    start_time = time.time()
    
    try:
        logger.info(f"📥 Recibida petición batch RAW con {len(request.clientes)} clientes")
        
        resultados = []
        
        for i, cliente_dict in enumerate(request.clientes):
            try:
                # Procesar cada cliente
                df = pd.DataFrame([cliente_dict])
                
                if feature_cols:
                    missing_cols = set(feature_cols) - set(df.columns)
                    for col in missing_cols:
                        df[col] = 0
                    df = df[feature_cols]
                
                proba = model.predict_proba(df)[0, 1]
                pred = int(proba >= optimal_threshold)
                
                resultados.append(PrediccionResponse(
                    probabilidad_desercion=float(proba),
                    clase_predicha=pred,
                    umbral=optimal_threshold,
                    modelo_version=MODEL_PATH.stem,
                    model_file=MODEL_PATH.name
                ))
                
            except Exception as e:
                logger.error(f"❌ Error procesando cliente {i}: {str(e)}")
                resultados.append(PrediccionResponse(
                    probabilidad_desercion=0.0,
                    clase_predicha=0,
                    umbral=optimal_threshold,
                    modelo_version=MODEL_PATH.stem,
                    model_file=MODEL_PATH.name
                ))
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"✅ Batch completado en {processing_time:.2f}ms")
        
        return BatchPrediccionResponse(predicciones=resultados)
        
    except Exception as e:
        logger.error(f"❌ Error en batch: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en predicción batch: {str(e)}"
        )

@app.get("/threshold", tags=["Modelo"])
async def get_threshold():
    """Obtiene el umbral óptimo actual del modelo."""
    return {
        "optimal_threshold": optimal_threshold,
        "model": MODEL_PATH.name,
        "description": "Umbral que maximiza F1-score en validación"
    }

# ============================================================
# ENDPOINTS DE DIAGNÓSTICO
# ============================================================

@app.get("/debug/feature-list", tags=["Debug"])
async def debug_feature_list():
    """Devuelve la lista completa de features que espera el modelo."""
    if model is None:
        return {"error": "Modelo no cargado"}
    
    return {
        "total_features": len(feature_cols),
        "features": feature_cols,
        "model_file": MODEL_PATH.name
    }

@app.get("/debug/random-forest-params", tags=["Debug"])
async def debug_rf_params():
    """Endpoint específico para ver parámetros del Random Forest."""
    if model is None:
        return {"error": "Modelo no cargado"}
    
    if "RandomForest" not in str(type(model)):
        return {
            "mensaje": f"El modelo no es RandomForest, es: {type(model).__name__}",
            "modelo": MODEL_PATH.name
        }
    
    # Obtener todos los parámetros del modelo
    params = {}
    if hasattr(model, 'get_params'):
        params = model.get_params()
    
    rf_info = {
        "tipo_optimizacion": "bayesiana" if "bayesian" in MODEL_PATH.name.lower() else "grid",
        "archivo": MODEL_PATH.name,
        "params": params,
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "min_samples_split": model.min_samples_split,
        "min_samples_leaf": model.min_samples_leaf,
        "criterion": model.criterion,
        "n_features_in_": getattr(model, 'n_features_in_', 'N/A'),
    }
    
    return rf_info

# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 INICIANDO API DE CHURN PREDICTION (MODO RAW)")
    print("="*60)
    print(f"📦 Modelo: {MODEL_PATH}")
    print(f"📊 Umbral óptimo: {optimal_threshold}")
    print("="*60)
    print("\n📌 Endpoints disponibles:")
    print("   - GET  /                 → Información general")
    print("   - GET  /health            → Estado de la API")
    print("   - GET  /info              → Información del modelo")
    print("   - GET  /threshold         → Umbral óptimo")
    print("   - POST /predict           → Predicción individual (RAW)")
    print("   - POST /predict/batch     → Predicción múltiple (RAW)")
    print("   - GET  /debug/feature-list → Lista completa de features")
    print("\n" + "="*60)
    
    uvicorn.run(
        "api.serving:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )