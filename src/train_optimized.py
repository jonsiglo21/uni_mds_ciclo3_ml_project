"""
train_optimized.py - Versión BAYESIAN OPTIMIZATION con SMOTE-ENN
VALORES DINÁMICOS para todos los parámetros.
"""

# Configurar matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
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

# ===== OPTIMIZACIÓN BAYESIANA =====
from imblearn.combine import SMOTEENN
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real

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

class BayesianModelTrainer:
    """Versión BAYESIAN OPTIMIZATION con SMOTE-ENN."""

    def __init__(self,
                 data_path: Optional[str] = None,
                 models_path: Optional[str] = None,
                 reports_path: Optional[str] = None,
                 experiments_path: Optional[str] = None):
        """
        Args:
            data_path: Ruta a datos procesados
            models_path: Ruta para guardar modelos
            reports_path: Ruta para guardar reportes
            experiments_path: Ruta para experimentos MLflow
        """
        self.current_file = Path(__file__).resolve()
        self.project_root = self.current_file.parent.parent

        logger.info(f"📁 Raíz del proyecto: {self.project_root}")

        # Configurar rutas
        self.data_path = self.project_root / 'data' / 'processed' / 'training' if data_path is None else Path(data_path)
        self.models_path = self.project_root / 'models' if models_path is None else Path(models_path)
        self.reports_path = self.project_root / 'reports' if reports_path is None else Path(reports_path)
        self.experiments_path = self.project_root / 'experiments' if experiments_path is None else Path(experiments_path)

        # Crear directorios
        for path in [self.models_path, self.reports_path, self.experiments_path]:
            path.mkdir(parents=True, exist_ok=True)
        (self.reports_path / 'figures').mkdir(parents=True, exist_ok=True)
        (self.reports_path / 'metrics').mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Models: {self.models_path}")
        logger.info(f"📁 Reports: {self.reports_path}")

        self.model = None
        self.feature_cols = None
        self.best_params = None
        self.best_threshold = 0.5
        self.X_train_shape = None
        
        # ===== BAYESIAN SEARCH SPACE (VALORES REALES) =====
        self.search_spaces = {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(5, 50),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical(['sqrt', 'log2', None]),
            'class_weight': Categorical(['balanced', 'balanced_subsample', None]),
            'bootstrap': Categorical([True, False]),
            'min_impurity_decrease': Real(0.0, 0.1, 'uniform')
        }

        # Configurar MLflow
        try:
            tracking_uri = f"file:///{self.experiments_path.as_posix()}/mlruns"
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("churn_prediction")
            logger.info(f"✅ MLflow configurado")
        except Exception as e:
            logger.warning(f"⚠️ MLflow: {e}")

    def load_data(self) -> pd.DataFrame:
        """Carga datos procesados."""
        logger.info("Cargando datos...")

        parquet_path = self.data_path / 'clientes_procesados.parquet'
        df = pd.read_parquet(parquet_path)
        logger.info(f"✅ Datos shape: {df.shape}")

        features_file = self.data_path / 'feature_columns.txt'
        with open(features_file, 'r') as f:
            self.feature_cols = [line.strip() for line in f.readlines()]
        logger.info(f"✅ Features: {len(self.feature_cols)}")

        target_dist = df['ATTRITION'].value_counts()
        logger.info("🎯 Target:")
        for valor, count in target_dist.items():
            logger.info(f"   Clase {valor}: {count} ({count/len(df)*100:.1f}%)")

        return df

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2,
                    use_smote_enn: bool = True, random_state: int = 42) -> tuple:
        """Prepara datos con SMOTE-ENN."""
        logger.info("Preparando datos...")

        X = df[self.feature_cols]
        y = df['ATTRITION']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"📊 Train original: {X_train.shape}")

        self.X_train_shape = X_train.shape

        if use_smote_enn:
            logger.info("🔄 Aplicando SMOTE-ENN...")
            smote_enn = SMOTEENN(random_state=random_state)
            X_train, y_train = smote_enn.fit_resample(X_train, y_train)
            logger.info(f"✅ After SMOTE-ENN: {X_train.shape}")

        return X_train, X_test, y_train, y_test

    def optimize_with_bayes(self, X_train, y_train, n_iter: int = 50):
        """Optimización con Bayesian Optimization."""
        logger.info(f"🔍 Bayesian Optimization ({n_iter} iteraciones)...")

        bayes_search = BayesSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            self.search_spaces,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )

        bayes_search.fit(X_train, y_train)

        self.best_params = bayes_search.best_params_
        self.model = bayes_search.best_estimator_

        logger.info(f"✅ Mejores parámetros:")
        for param, value in self.best_params.items():
            logger.info(f"   {param}: {value}")
        logger.info(f"✅ Mejor AUC-ROC CV: {bayes_search.best_score_:.4f}")

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
        plt.title('F1 Score vs Threshold (Bayesian)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.reports_path / 'figures' / 'threshold_optimization_v2.png', dpi=100)
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
        """Guarda gráficas con sufijo _v2."""
        figures_path = self.reports_path / 'figures'

        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title(f'Matriz de Confusión (Bayesian, thr={self.best_threshold:.3f})')
        plt.tight_layout()
        plt.savefig(figures_path / 'confusion_matrix_v2.png', dpi=100)
        plt.close()

        if len(np.unique(y_test)) == 2:
            fig, ax = plt.subplots(figsize=(8,6))
            RocCurveDisplay.from_predictions(
                y_test, y_proba, ax=ax,
                name=f'Bayesian (AUC = {roc_auc_score(y_test, y_proba):.4f})'
            )
            ax.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
            ax.legend()
            plt.title('Curva ROC (Bayesian)')
            plt.tight_layout()
            plt.savefig(figures_path / 'roc_curve_v2.png', dpi=100)
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

        importance_df.to_csv(self.reports_path / 'metrics' / 'feature_importance_v2.csv', index=False)

        logger.info("📊 Top 10 features (Bayesian):")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.4f}")

        # Gráfico
        top_n = min(20, len(importance_df))
        plt.figure(figsize=(12,8))
        plt.barh(range(top_n), importance_df.head(top_n)['importance'])
        plt.yticks(range(top_n), importance_df.head(top_n)['feature'])
        plt.title(f'Top {top_n} Features (Bayesian)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.reports_path / 'figures' / 'feature_importance_v2.png', dpi=100)
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
            'version': 'bayesian'
        }

        model_path = self.models_path / f'random_forest_bayesian_{timestamp}.pkl'
        joblib.dump({'model': self.model, 'metadata': model_metadata}, model_path)
        logger.info(f"✅ Modelo guardado: {model_path}")

        if is_champion:
            champion_path = self.models_path / 'random_forest_champion_bayesian.pkl'
            joblib.dump({'model': self.model, 'metadata': model_metadata}, champion_path)

        # Métricas JSON
        metrics_file = self.reports_path / 'metrics' / f'metrics_bayesian_{timestamp}.json'
        with open(metrics_file, 'w') as f:
            json.dump({**metrics, 'model_path': str(model_path)}, f, indent=2)

        return model_path

    def run_pipeline(self, n_iter: int = 50):
        """Ejecuta pipeline bayesiano."""
        logger.info("="*60)
        logger.info("🚀 Iniciando BAYESIAN OPTIMIZATION PIPELINE")
        logger.info("="*60)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"📊 MLflow Run ID: {run_id}")

            try:
                # Cargar datos
                df = self.load_data()

                # Preparar datos con SMOTE-ENN
                X_train, X_test, y_train, y_test = self.prepare_data(df, use_smote_enn=True)

                if X_train is None:
                    return None, None

                # Optimización bayesiana
                self.optimize_with_bayes(X_train, y_train, n_iter=n_iter)

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
                    
                    # ===== INFORMACIÓN DE BAYESIAN SEARCH (VALORES DINÁMICOS) =====
                    # Crear resumen legible del espacio de búsqueda
                    search_summary = {
                        'n_estimators': '100-500',
                        'max_depth': '5-50',
                        'min_samples_split': '2-20',
                        'min_samples_leaf': '1-10',
                        'max_features': ['sqrt', 'log2', None],
                        'class_weight': ['balanced', 'balanced_subsample', None],
                        'bootstrap': [True, False],
                        'min_impurity_decrease': '0.0-0.1'
                    }

                    mlflow.log_params({
                        # Método de optimización
                        "optimization_method": "bayesian",
                        "bayesian_n_iter": n_iter,
                        "bayesian_random_state": 42,
                        "bayesian_cv_folds": 3,
                        "bayesian_scoring": "roc_auc",
                        
                        # Espacio de búsqueda (desde self.search_spaces)
                        "search_space_n_estimators": str(self.search_spaces['n_estimators']),
                        "search_space_max_depth": str(self.search_spaces['max_depth']),
                        "search_space_min_samples_split": str(self.search_spaces['min_samples_split']),
                        "search_space_min_samples_leaf": str(self.search_spaces['min_samples_leaf']),
                        "search_space_max_features": str(self.search_spaces['max_features']),
                        "search_space_class_weight": str(self.search_spaces['class_weight']),
                        "search_space_bootstrap": str(self.search_spaces['bootstrap']),
                        "search_space_min_impurity_decrease": str(self.search_spaces['min_impurity_decrease']),
                        
                        # Resumen legible
                        "search_space_summary": str(search_summary),
                        
                        # Balanceo
                        "smote_method": "enn",
                        "smote_enn_used": True,
                        
                        # Dataset
                        "dataset_shape": str(df.shape),
                        "total_features": len(self.feature_cols),
                    })

                    # Tags
                    mlflow.set_tag("model_family", "random_forest")
                    mlflow.set_tag("optimization_type", "bayesian")
                    mlflow.set_tag("balance_technique", "smote_enn")

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
                    for figure in ['confusion_matrix_v2.png', 'roc_curve_v2.png',
                                  'feature_importance_v2.png', 'threshold_optimization_v2.png']:
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
                logger.info("✅ BAYESIAN OPTIMIZATION COMPLETADO")
                logger.info("="*60)

                return self.model, metrics

            except Exception as e:
                logger.error(f"❌ Error: {e}")
                mlflow.log_param("error", str(e))
                raise


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 BAYESIAN OPTIMIZATION - CHURN PREDICTION")
    print("="*60)

    trainer = BayesianModelTrainer()
    model, metrics = trainer.run_pipeline(n_iter=50)

    if model:
        print("\n📊 RESULTADOS BAYESIANOS:")
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