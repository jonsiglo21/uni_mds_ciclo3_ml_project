"""
API package for Churn Prediction Model.

This package contains the FastAPI application and related modules
for serving the machine learning model predictions.
"""

__version__ = "1.0.0"
__author__ = "jonathan chavez"

# Exportar clases principales para facilitar imports
from api.models import ClienteInput, PrediccionResponse
from api.config import settings

__all__ = [
    'ClienteInput',
    'PrediccionResponse',
    'settings',
    '__version__'
]