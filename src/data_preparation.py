"""
Módulo de preparación de datos para el modelo de deserción de clientes.
Versión final con manejo correcto de tipos de datos y fillna.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Any, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuración logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreparator:
    """Clase para preparar datos de clientes y requerimientos."""
    
    def __init__(self, raw_data_path: Optional[str] = None, 
                 processed_data_path: Optional[str] = None):
        """
        Args:
            raw_data_path: Ruta a datos crudos (opcional)
            processed_data_path: Ruta para guardar datos procesados (opcional)
        """
        # Determinar la raíz del proyecto automáticamente
        self.current_file = Path(__file__).resolve()  # src/data_preparation.py
        self.project_root = self.current_file.parent.parent  # sube: src/ -> raíz
        
        logger.info(f"📁 Raíz del proyecto detectada: {self.project_root}")
        
        # Configurar rutas de datos
        if raw_data_path is None:
            self.raw_path = self.project_root / 'data' / 'raw'
        else:
            self.raw_path = Path(raw_data_path)
            
        if processed_data_path is None:
            self.processed_path = self.project_root / 'data' / 'processed'
        else:
            self.processed_path = Path(processed_data_path)
        
        # Crear directorios si no existen
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # Crear subdirectorios
        (self.processed_path / 'training').mkdir(exist_ok=True)
        
        logger.info(f"📁 Raw data path: {self.raw_path}")
        logger.info(f"📁 Processed data path: {self.processed_path}")
        
        # Identificar top productos y submotivos
        self.top_products = None
        self.top_submotivos = None
        self.category_mappings: Dict[str, Any] = {}
        
    def detect_separator_and_encoding(self, file_path: Path) -> Tuple[str, str]:
        """
        Detecta automáticamente el separador y la codificación del archivo.
        """
        logger.info(f"🔍 Detectando formato de archivo: {file_path.name}")
        
        # Probamos diferentes codificaciones
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        separators_to_try = [';', ',', '\t', '|']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    first_line = f.readline().strip()
                
                # Intentar detectar el separador
                for sep in separators_to_try:
                    if sep in first_line:
                        # Verificar que el separador funciona
                        sample = first_line.split(sep)
                        if len(sample) > 1:  # Al menos 2 columnas
                            logger.info(f"✅ Detectado: separador='{sep}', encoding={encoding}")
                            return sep, encoding
                
                # Si no se detecta separador, probar con pandas
                try:
                    df_sample = pd.read_csv(file_path, encoding=encoding, nrows=5)
                    if df_sample.shape[1] > 1:  # Más de una columna
                        logger.info(f"✅ Detectado por pandas: encoding={encoding}")
                        return 'infer', encoding
                except:
                    continue
                    
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error probando encoding {encoding}: {e}")
                continue
        
        # Si todo falla, usar valores por defecto
        logger.warning("⚠️ No se pudo detectar formato, usando defaults: sep=';', encoding='utf-8'")
        return ';', 'utf-8'
    
    def load_csv_robust(self, file_path: Path) -> pd.DataFrame:
        """
        Carga un CSV de manera robusta, intentando diferentes separadores y codificaciones.
        """
        logger.info(f"📂 Cargando archivo: {file_path.name}")
        
        # Detectar formato
        sep, encoding = self.detect_separator_and_encoding(file_path)
        
        # Intentar carga con diferentes estrategias
        errors = []
        
        # Estrategia 1: Usar separador detectado
        try:
            df = pd.read_csv(
                file_path,
                sep=sep,
                encoding=encoding,
                engine='python'  # Más tolerante a errores
            )
            logger.info(f"✅ Cargado con sep='{sep}', encoding={encoding}, shape={df.shape}")
            return df
        except Exception as e:
            errors.append(f"Estrategia 1: {e}")
        
        # Estrategia 2: Dejar que pandas infiera
        try:
            df = pd.read_csv(
                file_path,
                encoding='utf-8',
                engine='python'
            )
            logger.info(f"✅ Cargado con inferencia automática, shape={df.shape}")
            return df
        except Exception as e:
            errors.append(f"Estrategia 2: {e}")
        
        # Estrategia 3: Leer línea por línea
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
            
            # Intentar detectar separador de las primeras líneas
            sample = lines[0].strip()
            for sep_candidate in [';', ',', '\t', '|']:
                if sep_candidate in sample:
                    data = [line.strip().split(sep_candidate) for line in lines]
                    df = pd.DataFrame(data[1:], columns=data[0])
                    logger.info(f"✅ Cargado manualmente con sep='{sep_candidate}', shape={df.shape}")
                    return df
            
            # Si no hay separador claro, asumir una sola columna
            df = pd.DataFrame({'contenido': [line.strip() for line in lines]})
            logger.warning(f"⚠️ Cargado como una sola columna, shape={df.shape}")
            return df
            
        except Exception as e:
            errors.append(f"Estrategia 3: {e}")
        
        # Si todo falla
        raise ValueError(f"❌ No se pudo cargar {file_path}. Errores: {errors}")
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carga datos crudos de clientes y requerimientos."""
        logger.info("Cargando datos crudos...")
        
        # Mostrar directorio actual para debug
        current_dir = Path.cwd()
        logger.info(f"Directorio actual de ejecución: {current_dir}")
        
        # Archivos a cargar
        clientes_file = 'train_clientes.csv'
        requerimientos_file = 'train_requerimientos.csv'
        
        # Construir rutas completas
        clientes_path = self.raw_path / clientes_file
        requerimientos_path = self.raw_path / requerimientos_file
        
        logger.info(f"Buscando clientes en: {clientes_path}")
        logger.info(f"Buscando requerimientos en: {requerimientos_path}")
        
        # Verificar que los archivos existen
        if not clientes_path.exists():
            logger.error(f"❌ No se encuentra {clientes_path}")
            logger.info("Archivos disponibles en el directorio raw:")
            if self.raw_path.exists():
                files = list(self.raw_path.glob("*"))
                if files:
                    for f in files:
                        logger.info(f"  - {f.name}")
                else:
                    logger.warning("  El directorio está vacío")
            else:
                logger.error(f"❌ El directorio {self.raw_path} no existe")
            raise FileNotFoundError(f"No se encuentra {clientes_path}")
            
        if not requerimientos_path.exists():
            logger.error(f"❌ No se encuentra {requerimientos_path}")
            raise FileNotFoundError(f"No se encuentra {requerimientos_path}")
        
        # Cargar archivos de manera robusta
        logger.info("✅ Archivos encontrados, cargando datos...")
        
        try:
            client_df = self.load_csv_robust(clientes_path)
            requirement_df = self.load_csv_robust(requerimientos_path)
        except Exception as e:
            logger.error(f"❌ Error cargando archivos: {e}")
            raise
        
        # Limpiar nombres de columnas (quitar espacios)
        client_df.columns = client_df.columns.str.strip()
        requirement_df.columns = requirement_df.columns.str.strip()
        
        logger.info(f"✅ Clientes: {client_df.shape}, Requerimientos: {requirement_df.shape}")
        logger.info(f"📊 Clientes columnas: {client_df.columns.tolist()}")
        logger.info(f"📊 Requerimientos columnas: {requirement_df.columns.tolist()}")
        
        # Validar datos cargados
        self.validate_data(client_df, requirement_df)
        
        return client_df, requirement_df
    
    def validate_data(self, client_df: pd.DataFrame, requirement_df: pd.DataFrame):
        """Valida integridad de datos cargados."""
        logger.info("Validando datos...")
        
        # Verificar que la columna ID_CORRELATIVO existe
        required_cols = ['ID_CORRELATIVO']
        
        for col in required_cols:
            if col not in client_df.columns:
                logger.error(f"❌ Columna '{col}' no encontrada en clientes")
                logger.info(f"Columnas disponibles: {client_df.columns.tolist()}")
                raise KeyError(f"Columna '{col}' no encontrada en clientes")
            
            if col not in requirement_df.columns:
                logger.error(f"❌ Columna '{col}' no encontrada en requerimientos")
                logger.info(f"Columnas disponibles: {requirement_df.columns.tolist()}")
                raise KeyError(f"Columna '{col}' no encontrada en requerimientos")
        
        # Verificar IDs consistentes
        client_ids = set(client_df['ID_CORRELATIVO'].unique())
        req_ids = set(requirement_df['ID_CORRELATIVO'].unique())
        
        logger.info(f"👥 Clientes únicos: {len(client_ids)}")
        logger.info(f"📝 Clientes con requerimientos: {len(req_ids)}")
        logger.info(f"⚠️ Clientes sin requerimientos: {len(client_ids - req_ids)}")
        
        # Verificar target
        if 'ATTRITION' in client_df.columns:
            attrition_counts = client_df['ATTRITION'].value_counts()
            logger.info(f"🎯 Distribución target:")
            for valor, count in attrition_counts.items():
                porcentaje = count / len(client_df) * 100
                logger.info(f"   Clase {valor}: {count} clientes ({porcentaje:.1f}%)")
    
    def analyze_missing_values(self, df: pd.DataFrame, name: str) -> pd.Series:
        """Analiza valores faltantes en el dataframe."""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Columna': missing.index,
            'Valores Faltantes': missing.values,
            'Porcentaje': missing_pct.values
        })
        missing_df = missing_df[missing_df['Valores Faltantes'] > 0].sort_values('Valores Faltantes', ascending=False)
        
        if len(missing_df) > 0:
            logger.info(f"Valores faltantes en {name}:")
            for _, row in missing_df.iterrows():
                logger.info(f"  {row['Columna']}: {row['Valores Faltantes']} ({row['Porcentaje']:.2f}%)")
        
        return missing
    
    def clean_requirements(self, requirement_df: pd.DataFrame) -> pd.DataFrame:
        """Limpia datos de requerimientos."""
        logger.info("Limpiando requerimientos...")
        
        # Analizar valores faltantes
        self.analyze_missing_values(requirement_df, "requerimientos")
        
        # Eliminar nulos en DICTAMEN
        initial_shape = requirement_df.shape
        requirement_df = requirement_df.dropna(subset=['DICTAMEN'])
        
        eliminated = initial_shape[0] - requirement_df.shape[0]
        if eliminated > 0:
            logger.info(f"Eliminados {eliminated} registros con DICTAMEN nulo")
        
        return requirement_df
    
    def get_top_categories(self, requirement_df: pd.DataFrame):
        """Identifica top productos y submotivos."""
        # Top productos
        product_counts = requirement_df['PRODUCTO_SERVICIO_2'].value_counts()
        self.top_products = product_counts.head(3).index.tolist()
        
        # Top submotivos
        submotivo_counts = requirement_df['SUBMOTIVO_2'].value_counts()
        self.top_submotivos = submotivo_counts.head(3).index.tolist()
        
        logger.info(f"📊 Top productos: {self.top_products} (frecuencias: {product_counts.head(3).values})")
        logger.info(f"📊 Top submotivos: {self.top_submotivos} (frecuencias: {submotivo_counts.head(3).values})")
        
        # Guardar mappings
        self.category_mappings['top_products'] = self.top_products
        self.category_mappings['top_submotivos'] = self.top_submotivos
        
    def create_requirement_features(self, requirement_df: pd.DataFrame) -> pd.DataFrame:
        """Crea features a partir de requerimientos usando pandas."""
        logger.info("Creando features de requerimientos...")
        
        # Identificar top categorías si no existe
        if self.top_products is None:
            self.get_top_categories(requirement_df)
        
        # Tipo de requerimiento
        type_req = pd.crosstab(
            requirement_df['ID_CORRELATIVO'], 
            requirement_df['TIPO_REQUERIMIENTO2']
        ).reset_index()
        
        # Dictamen
        dictamen = pd.crosstab(
            requirement_df['ID_CORRELATIVO'], 
            requirement_df['DICTAMEN']
        ).reset_index()
        
        # Meses
        codmes = pd.crosstab(
            requirement_df['ID_CORRELATIVO'], 
            requirement_df['CODMES']
        ).reset_index()
        
        # Productos top
        productos_crosstab = pd.crosstab(
            requirement_df['ID_CORRELATIVO'], 
            requirement_df['PRODUCTO_SERVICIO_2']
        )
        
        productos = pd.DataFrame()
        productos['ID_CORRELATIVO'] = productos_crosstab.index
        for product in self.top_products:
            if product in productos_crosstab.columns:
                productos[product] = productos_crosstab[product].values
            else:
                productos[product] = 0
        
        # Submotivos top
        submotivos_crosstab = pd.crosstab(
            requirement_df['ID_CORRELATIVO'], 
            requirement_df['SUBMOTIVO_2']
        )
        
        submotivos = pd.DataFrame()
        submotivos['ID_CORRELATIVO'] = submotivos_crosstab.index
        for submotivo in self.top_submotivos:
            if submotivo in submotivos_crosstab.columns:
                submotivos[submotivo] = submotivos_crosstab[submotivo].values
            else:
                submotivos[submotivo] = 0
        
        # Merge de todas las features
        features = type_req
        for df in [dictamen, codmes, productos, submotivos]:
            features = features.merge(df, on='ID_CORRELATIVO', how='outer')
        
        # Rellenar NaN con 0 (para columnas numéricas)
        # Identificar columnas numéricas y de texto
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        string_cols = features.select_dtypes(include=['object', 'string']).columns
        
        # Para numéricas: fillna con 0
        features[numeric_cols] = features[numeric_cols].fillna(0)
        
        # Para strings: fillna con '0' (como string) o '' según convenga
        for col in string_cols:
            if col != 'ID_CORRELATIVO':  # No modificar ID
                features[col] = features[col].fillna('0')
        
        logger.info(f"✅ Features de requerimientos shape: {features.shape}")
        logger.info(f"📊 Columnas generadas: {features.columns.tolist()}")
        
        return features
    
    def process_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa variables categóricas."""
        logger.info("Procesando variables categóricas...")
        
        df_processed = df.copy()
        
        # One-hot encoding para FLAG_LIMA_PROVINCIA
        if 'FLAG_LIMA_PROVINCIA' in df_processed.columns:
            # Manejar nulos antes de one-hot encoding
            df_processed['FLAG_LIMA_PROVINCIA'] = df_processed['FLAG_LIMA_PROVINCIA'].fillna('Desconocido')
            dummies = pd.get_dummies(
                df_processed['FLAG_LIMA_PROVINCIA'], 
                prefix='FLAG_LIMA', 
                drop_first=True,
                dtype=int
            )
            df_processed = pd.concat([df_processed, dummies], axis=1)
            df_processed.drop(columns=['FLAG_LIMA_PROVINCIA'], inplace=True)
        
        # Procesar rangos de ingresos
        if 'RANG_INGRESO' in df_processed.columns:
            df_processed['RANG_INGRESO'] = df_processed['RANG_INGRESO'].fillna('Desconocido')
            df_processed['RANG_INGRESO'] = df_processed['RANG_INGRESO'].replace('', 'Desconocido')
            
            ingreso_dummies = pd.get_dummies(
                df_processed['RANG_INGRESO'],
                prefix='INGRESO',
                drop_first=True,
                dtype=int
            )
            df_processed = pd.concat([df_processed, ingreso_dummies], axis=1)
            df_processed.drop(columns=['RANG_INGRESO'], inplace=True)
        
        # Procesar ANTIGUEDAD (puede tener valores vacíos)
        if 'ANTIGUEDAD' in df_processed.columns:
            df_processed['ANTIGUEDAD'] = pd.to_numeric(df_processed['ANTIGUEDAD'], errors='coerce').fillna(0)
        
        return df_processed
    
    def prepare_features(self, client_df: pd.DataFrame, 
                        requirement_features: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepara el dataset final de features."""
        logger.info("Preparando dataset final...")
        
        # Merge clientes con features de requerimientos
        final_df = client_df.merge(requirement_features, on='ID_CORRELATIVO', how='left')
        
        # Identificar tipos de columnas para fillna apropiado
        numeric_cols = final_df.select_dtypes(include=[np.number]).columns
        string_cols = final_df.select_dtypes(include=['object', 'string']).columns
        
        # Para numéricas: fillna con 0
        final_df[numeric_cols] = final_df[numeric_cols].fillna(0)
        
        # Para strings: fillna con '' (vacío) para no mezclar tipos
        for col in string_cols:
            if col != 'ID_CORRELATIVO':  # No modificar ID
                final_df[col] = final_df[col].fillna('')
        
        # Procesar variables categóricas
        final_df = self.process_categorical_variables(final_df)
        
        # Identificar columnas a excluir
        exclude_cols = ['ID_CORRELATIVO', 'CODMES', 'ATTRITION']
        
        # Seleccionar solo columnas numéricas para features
        feature_cols = []
        for col in final_df.columns:
            if col not in exclude_cols:
                if pd.api.types.is_numeric_dtype(final_df[col]):
                    feature_cols.append(col)
        
        logger.info(f"✅ Total features seleccionadas: {len(feature_cols)}")
        
        # Verificar valores infinitos
        for col in feature_cols:
            if not np.isfinite(final_df[col]).all():
                logger.warning(f"⚠️ Columna {col} tiene valores no finitos, reemplazando...")
                final_df[col] = final_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return final_df, feature_cols
    
    def save_processed_data(self, final_df: pd.DataFrame, feature_cols: List[str]):
        """Guarda datos procesados."""
        logger.info("Guardando datos procesados...")
        
        training_path = self.processed_path / 'training'
        
        # Guardar en parquet
        parquet_path = training_path / 'clientes_procesados.parquet'
        final_df.to_parquet(parquet_path, index=False, compression='snappy')
        logger.info(f"✅ Datos guardados en: {parquet_path}")
        
        # Guardar en CSV
        csv_path = training_path / 'clientes_procesados.csv'
        final_df.to_csv(csv_path, index=False)
        logger.info(f"✅ CSV guardado en: {csv_path}")
        
        # Guardar lista de features
        features_path = training_path / 'feature_columns.txt'
        with open(features_path, 'w') as f:
            for col in feature_cols:
                f.write(f"{col}\n")
        logger.info(f"✅ Features guardadas en: {features_path}")
        
        # Guardar mappings
        self.category_mappings['feature_columns'] = feature_cols
        mappings_path = training_path / 'category_mappings.pkl'
        joblib.dump(self.category_mappings, mappings_path)
        logger.info(f"✅ Mappings guardados en: {mappings_path}")
    
    def run_pipeline(self) -> Tuple[pd.DataFrame, List[str]]:
        """Ejecuta pipeline completo de preparación."""
        logger.info("="*60)
        logger.info("🚀 Iniciando pipeline de preparación de datos")
        logger.info("="*60)
        
        try:
            # Cargar datos
            client_df, requirement_df = self.load_raw_data()
            
            # Análisis de valores nulos
            self.analyze_missing_values(client_df, "clientes")
            
            # Limpiar requerimientos
            requirement_df = self.clean_requirements(requirement_df)
            
            # Identificar top categorías
            self.get_top_categories(requirement_df)
            
            # Crear features de requerimientos
            req_features = self.create_requirement_features(requirement_df)
            
            # Preparar dataset final
            final_df, feature_cols = self.prepare_features(client_df, req_features)
            
            # Guardar resultados
            self.save_processed_data(final_df, feature_cols)
            
            logger.info("="*60)
            logger.info("✅ Pipeline de preparación completado exitosamente")
            logger.info("="*60)
            
            return final_df, feature_cols
            
        except Exception as e:
            logger.error(f"❌ Error en el pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    print("\n" + "="*60)
    print("📊 PREPARACIÓN DE DATOS - CHURN PREDICTION")
    print("="*60)
    
    # Crear instancia y ejecutar pipeline
    preparator = DataPreparator()
    final_df, feature_cols = preparator.run_pipeline()
    
    print(f"\n✅ Dataset final shape: {final_df.shape}")
    print(f"✅ Features seleccionadas: {len(feature_cols)}")
    if 'ATTRITION' in final_df.columns:
        print(f"\n📊 Distribución target:")
        for valor, count in final_df['ATTRITION'].value_counts().items():
            porcentaje = count / len(final_df) * 100
            print(f"   Clase {valor}: {count} registros ({porcentaje:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print("="*60)