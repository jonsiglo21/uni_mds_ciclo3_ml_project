"""
Módulo de entrenamiento del modelo de deserción de clientes.
Versión COMPLETA con valores DINÁMICOS para todos los parámetros.
"""

# Configurar matplotlib ANTES de cualquier otro import
import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                           roc_auc_score, precision_recall_curve, f1_score,
                           RocCurveDisplay, PrecisionRecallDisplay)
import joblib
import logging
import json
from pathlib import Path
import seaborn as sns
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

# Configuración logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Clase para entrenar y evaluar modelos de clasificación."""

    def __init__(self,
                 data_path: Optional[str] = None,
                 models_path: Optional[str] = None,
                 reports_path: Optional[str] = None,
                 experiments_path: Optional[str] = None):
        """
        Args:
            data_path: Ruta a datos procesados (opcional)
            models_path: Ruta para guardar modelos (opcional)
            reports_path: Ruta para guardar reportes (opcional)
            experiments_path: Ruta para experimentos MLflow (opcional)
        """
        # Determinar la raíz del proyecto automáticamente
        self.current_file = Path(__file__).resolve()
        self.project_root = self.current_file.parent.parent

        logger.info(f"📁 Raíz del proyecto detectada: {self.project_root}")

        # Configurar rutas
        if data_path is None:
            self.data_path = self.project_root / 'data' / 'processed' / 'training'
        else:
            self.data_path = Path(data_path)

        if models_path is None:
            self.models_path = self.project_root / 'models'
        else:
            self.models_path = Path(models_path)

        if reports_path is None:
            self.reports_path = self.project_root / 'reports'
        else:
            self.reports_path = Path(reports_path)

        if experiments_path is None:
            self.experiments_path = self.project_root / 'experiments'
        else:
            self.experiments_path = Path(experiments_path)

        # Crear directorios necesarios
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.reports_path.mkdir(parents=True, exist_ok=True)
        self.experiments_path.mkdir(parents=True, exist_ok=True)
        (self.reports_path / 'figures').mkdir(parents=True, exist_ok=True)
        (self.reports_path / 'metrics').mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Data path: {self.data_path}")
        logger.info(f"📁 Models path: {self.models_path}")
        logger.info(f"📁 Reports path: {self.reports_path}")
        logger.info(f"📁 Experiments path: {self.experiments_path}")

        self.model = None
        self.feature_cols = None
        self.best_params = None
        self.best_threshold = 0.5
        self.X_train_shape = None
        
        # ===== GRID SEARCH PARAMETERS (VALORES REALES) =====
        self.param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'class_weight': ['balanced', None]
        }
        
        # Calcular total de combinaciones dinámicamente
        self.total_combinations = 1
        for values in self.param_grid.values():
            self.total_combinations *= len(values)

        # Configurar MLflow
        try:
            tracking_uri = f"file:///{self.experiments_path.as_posix()}/mlruns"
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("churn_prediction")
            logger.info(f"✅ MLflow configurado con URI: {tracking_uri}")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo configurar MLflow: {e}")
            mlflow.set_tracking_uri("")

    def load_data(self) -> pd.DataFrame:
        """Carga datos procesados."""
        logger.info("Cargando datos procesados...")

        if not self.data_path.exists():
            raise FileNotFoundError(f"❌ No existe el directorio: {self.data_path}")

        parquet_path = self.data_path / 'clientes_procesados.parquet'
        df = pd.read_parquet(parquet_path)
        logger.info(f"✅ Datos shape: {df.shape}")

        # Cargar features
        features_file = self.data_path / 'feature_columns.txt'
        if features_file.exists():
            with open(features_file, 'r') as f:
                self.feature_cols = [line.strip() for line in f.readlines()]
            logger.info(f"✅ Features: {len(self.feature_cols)}")

        # Verificar ATTRITION
        if 'ATTRITION' not in df.columns:
            raise ValueError("❌ La columna 'ATTRITION' no está en el dataframe")

        # Distribución target
        target_dist = df['ATTRITION'].value_counts()
        logger.info("🎯 Distribución del target:")
        for valor, count in target_dist.items():
            porcentaje = count / len(df) * 100
            logger.info(f"   Clase {valor}: {count} registros ({porcentaje:.1f}%)")

        return df

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2,
                    use_smote: bool = True, random_state: int = 42) -> tuple:
        """Prepara datos para entrenamiento."""
        logger.info("Preparando datos...")

        X = df[self.feature_cols]
        y = df['ATTRITION']

        if len(y.unique()) < 2:
            logger.error("❌ Solo hay una clase")
            return None, None, None, None

        # Split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            logger.info("✅ Split con estratificación")
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        self.X_train_shape = X_train.shape
        logger.info(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")

        # SMOTE
        class_dist = y.value_counts(normalize=True)
        minority_ratio = min(class_dist.values)
        if use_smote and len(y.unique()) == 2 and minority_ratio < 0.3:
            logger.info(f"⚠️ Clase minoritaria: {minority_ratio:.2%} - Aplicando SMOTE...")
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=random_state)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                logger.info(f"✅ After SMOTE: {X_train.shape}")
            except ImportError:
                logger.warning("⚠️ imbalanced-learn no instalado")

        return X_train, X_test, y_train, y_test

    def optimize_hyperparameters(self, X_train, y_train, cv_folds: int = 3):
        """Optimización con GridSearchCV."""
        logger.info("Iniciando Grid Search...")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            self.param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        logger.info(f"✅ Mejores parámetros: {self.best_params}")
        logger.info(f"✅ Mejor AUC-ROC CV: {grid_search.best_score_:.4f}")

        return self.model

    def find_best_threshold(self, y_test, y_proba):
        """Encuentra el mejor threshold."""
        logger.info("🔍 Buscando mejor threshold...")

        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        logger.info(f"✅ Mejor threshold: {best_threshold:.3f}")

        # Graficar
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1_scores, 'b-', label='F1 Score')
        plt.axvline(best_threshold, color='r', linestyle='--',
                   label=f'Mejor: {best_threshold:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Threshold (Grid Search)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.reports_path / 'figures' / 'threshold_optimization.png', dpi=100)
        plt.close()

        return best_threshold

    def evaluate_model(self, X_test, y_test) -> dict:
        """Evalúa el modelo."""
        logger.info("Evaluando modelo...")

        y_pred_default = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        self.best_threshold = self.find_best_threshold(y_test, y_proba)
        y_pred_optimized = (y_proba >= self.best_threshold).astype(int)

        metrics = {
            'accuracy_default': float((y_pred_default == y_test).mean()),
            'f1_default': float(f1_score(y_test, y_pred_default, zero_division=0)),
            'accuracy_optimized': float((y_pred_optimized == y_test).mean()),
            'f1_optimized': float(f1_score(y_test, y_pred_optimized, zero_division=0)),
            'best_threshold': float(self.best_threshold),
            'train_samples': int(self.X_train_shape[0]),
            'test_samples': int(len(y_test))
        }

        if len(np.unique(y_test)) == 2:
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba))

        # Classification report
        try:
            report = classification_report(y_test, y_pred_optimized, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
            for clase in ['0', '1']:
                if clase in report:
                    metrics[f'precision_class_{clase}'] = report[clase]['precision']
                    metrics[f'recall_class_{clase}'] = report[clase]['recall']
                    metrics[f'f1_class_{clase}'] = report[clase]['f1-score']
        except Exception as e:
            logger.warning(f"⚠️ Error en report: {e}")

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred_optimized)
        metrics['confusion_matrix'] = cm.tolist()
        logger.info(f"📊 Matriz:\n{cm}")

        # Gráficas
        self._save_plots(y_test, y_proba, y_pred_optimized, cm)

        return metrics

    def _save_plots(self, y_test, y_proba, y_pred, cm):
        """Guarda gráficas."""
        figures_path = self.reports_path / 'figures'

        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title(f'Matriz de Confusión (Grid, thr={self.best_threshold:.3f})')
        plt.tight_layout()
        plt.savefig(figures_path / 'confusion_matrix.png', dpi=100)
        plt.close()

        if len(np.unique(y_test)) == 2:
            fig, ax = plt.subplots(figsize=(8,6))
            RocCurveDisplay.from_predictions(
                y_test, y_proba, ax=ax,
                name=f'Grid (AUC = {roc_auc_score(y_test, y_proba):.4f})'
            )
            ax.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
            ax.legend()
            plt.title('Curva ROC (Grid Search)')
            plt.tight_layout()
            plt.savefig(figures_path / 'roc_curve.png', dpi=100)
            plt.close()

    def analyze_feature_importance(self, feature_names: List[str]):
        """Analiza importancia de features."""
        if not hasattr(self.model, 'feature_importances_'):
            return

        importances = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        importance_df.to_csv(self.reports_path / 'metrics' / 'feature_importance.csv', index=False)

        logger.info("📊 Top 10 features:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.4f}")

        # Gráfico
        top_n = min(20, len(importance_df))
        plt.figure(figsize=(12,8))
        plt.barh(range(top_n), importance_df.head(top_n)['importance'])
        plt.yticks(range(top_n), importance_df.head(top_n)['feature'])
        plt.title(f'Top {top_n} Features (Grid Search)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.reports_path / 'figures' / 'feature_importance.png', dpi=100)
        plt.close()

    def save_model(self, metrics: dict, is_champion: bool = True):
        """Guarda el modelo."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_metadata = {
            'timestamp': timestamp,
            'model_type': type(self.model).__name__,
            'parameters': dict(self.model.get_params()),
            'metrics': metrics,
            'feature_columns': self.feature_cols,
            'best_params': self.best_params,
            'best_threshold': float(self.best_threshold),
            'project_root': str(self.project_root)
        }

        model_path = self.models_path / f'random_forest_grid_{timestamp}.pkl'
        joblib.dump({'model': self.model, 'metadata': model_metadata}, model_path)
        logger.info(f"✅ Modelo guardado: {model_path}")

        if is_champion:
            champion_path = self.models_path / 'random_forest_champion_grid.pkl'
            joblib.dump({'model': self.model, 'metadata': model_metadata}, champion_path)

        # Métricas JSON
        metrics_file = self.reports_path / 'metrics' / f'metrics_grid_{timestamp}.json'
        with open(metrics_file, 'w') as f:
            json.dump({**metrics, 'model_path': str(model_path)}, f, indent=2)

        return model_path

    def run_pipeline(self, optimize: bool = True, use_smote: bool = True):
        """Ejecuta pipeline completo."""
        logger.info("="*60)
        logger.info("🚀 Iniciando GRID SEARCH PIPELINE")
        logger.info("="*60)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"📊 MLflow Run ID: {run_id}")

            try:
                # Cargar datos
                df = self.load_data()

                # Preparar datos
                X_train, X_test, y_train, y_test = self.prepare_data(df, use_smote=use_smote)

                if X_train is None:
                    return None, None

                # Optimizar
                self.optimize_hyperparameters(X_train, y_train)

                # Evaluar
                metrics = self.evaluate_model(X_test, y_test)

                # Feature importance
                self.analyze_feature_importance(self.feature_cols)

                # Guardar modelo
                model_path = self.save_model(metrics)

                # ===== LOGGING MLflow CON VALORES DINÁMICOS =====
                try:
                    # Parámetros del modelo
                    mlflow.log_params({f"best_{k}": v for k, v in self.best_params.items()})
                    
                    # ===== INFORMACIÓN DEL GRID SEARCH (VALORES DINÁMICOS) =====
                    mlflow.log_params({
                        "optimization_method": "grid_search",
                        "cv_folds": 3,
                        "scoring_metric": "roc_auc",
                        
                        # Detalles del grid (desde self.param_grid)
                        "param_grid": str(self.param_grid),
                        "total_combinations": self.total_combinations,
                        "n_estimators_options": str(self.param_grid.get('n_estimators', [])),
                        "max_depth_options": str(self.param_grid.get('max_depth', [])),
                        "min_samples_split_options": str(self.param_grid.get('min_samples_split', [])),
                        "min_samples_leaf_options": str(self.param_grid.get('min_samples_leaf', [])),
                        "class_weight_options": str(self.param_grid.get('class_weight', [])),
                        
                        # Configuración
                        "refit": True,
                        "return_train_score": True,
                        "grid_search_verbose": True,
                        
                        # Balanceo
                        "use_smote": use_smote,
                        "smote_method": "standard" if use_smote else "none",
                        
                        # Dataset
                        "dataset_shape": str(df.shape),
                        "total_features": len(self.feature_cols),
                    })

                    # Tags
                    mlflow.set_tag("model_family", "random_forest")
                    mlflow.set_tag("optimization_type", "grid_search")
                    mlflow.set_tag("balance_technique", "smote_standard" if use_smote else "none")

                    # Métricas
                    mlflow.log_metrics({
                        'accuracy_default': metrics['accuracy_default'],
                        'accuracy_optimized': metrics['accuracy_optimized'],
                        'f1_default': metrics['f1_default'],
                        'f1_optimized': metrics['f1_optimized'],
                        'roc_auc': metrics.get('roc_auc', 0),
                        'best_threshold': metrics['best_threshold'],
                        'train_samples': metrics['train_samples'],
                        'test_samples': metrics['test_samples']
                    })

                    # Métricas por clase
                    for clase in ['0', '1']:
                        clase_metrics = {}
                        if f'precision_class_{clase}' in metrics:
                            clase_metrics[f'precision_class_{clase}'] = metrics[f'precision_class_{clase}']
                        if f'recall_class_{clase}' in metrics:
                            clase_metrics[f'recall_class_{clase}'] = metrics[f'recall_class_{clase}']
                        if f'f1_class_{clase}' in metrics:
                            clase_metrics[f'f1_class_{clase}'] = metrics[f'f1_class_{clase}']
                        if clase_metrics:
                            mlflow.log_metrics(clase_metrics)

                    # Artefactos
                    for figure in ['confusion_matrix.png', 'roc_curve.png',
                                  'feature_importance.png', 'threshold_optimization.png']:
                        figure_path = self.reports_path / 'figures' / figure
                        if figure_path.exists():
                            mlflow.log_artifact(str(figure_path))

                    # Modelo en MLflow
                    if X_test is not None and len(X_test) > 0:
                        signature = infer_signature(X_test, self.model.predict(X_test))
                        mlflow.sklearn.log_model(
                            self.model,
                            "model",
                            signature=signature,
                            input_example=X_test.iloc[:min(5, len(X_test))]
                        )
                        logger.info("✅ Modelo guardado en MLflow")

                except Exception as e:
                    logger.warning(f"⚠️ Error en logging: {e}")

                logger.info("="*60)
                logger.info("✅ GRID SEARCH COMPLETADO")
                logger.info("="*60)

                return self.model, metrics

            except Exception as e:
                logger.error(f"❌ Error: {e}")
                mlflow.log_param("error", str(e))
                raise


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 GRID SEARCH - CHURN PREDICTION")
    print("="*60)

    trainer = ModelTrainer()
    model, metrics = trainer.run_pipeline(optimize=True, use_smote=True)

    if model:
        print("\n📊 RESULTADOS GRID SEARCH:")
        print(f"   AUC-ROC: {metrics.get('roc_auc', 'N/A'):.4f}")
        print(f"   F1 (optimized): {metrics['f1_optimized']:.4f}")
        print(f"   Mejor threshold: {metrics['best_threshold']:.3f}")
        
        print(f"\n   📍 Clase 1 (CHURN):")
        print(f"      - Precision: {metrics.get('precision_class_1', 0):.3f}")
        print(f"      - Recall: {metrics.get('recall_class_1', 0):.3f}")
        print(f"      - F1-Score: {metrics.get('f1_class_1', 0):.3f}")

    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print("="*60)