import torch
import joblib
import os
from typing import Any, Dict, Optional
from pathlib import Path
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """
    Manages loading and caching of ML models (PyTorch & LightGBM).
    Singleton pattern to avoid reloading heavy models.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.models = {}
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            self.model_paths = {
                "ltlf_recursive": settings.LTLF_RECURSIVE_PATH,
                "decom_engine": settings.DECOMP_MODEL_PATH
            }
            self.initialized = True
            
    def load_all_models(self) -> Dict[str, Any]:
        """Load all configured models into memory."""
        for name, path in self.model_paths.items():
            self.load_model(name, path)
        return self.models

    def load_model(self, name: str, path: str) -> Optional[Any]:
        """
        Load a single model from disk.
        Supports .pt (PyTorch) and .pkl (Joblib/Pickle).
        """
        if name in self.models:
            return self.models[name]

        if not os.path.exists(path):
            logger.warning(f"Model file not found at {path}. Model '{name}' will be unavailable.")
            return None

        try:
            logger.info(f"Loading model '{name}' from {path}...")
            
            if name == "decom_engine":
                from app.ml.decom_engine import DecomEngine
                model = DecomEngine()
                model.load(path)
            elif name == "ltlf_recursive":
                from app.ml.ltlf_recursive import LTLFRecursiveEngine
                model = LTLFRecursiveEngine()
                model.load()
            elif path.endswith(".pt"):
                # Load PyTorch model
                # map_location='cpu' ensuring we can run anywhere
                model = torch.load(path, map_location=torch.device('cpu'))
                if hasattr(model, 'eval'):
                    model.eval()
            elif path.endswith(".pkl") or path.endswith(".joblib"):
                # Load Scikit-Learn / LightGBM model
                model = joblib.load(path)
            else:
                logger.error(f"Unsupported file extension for model '{name}': {path}")
                return None

            self.models[name] = model
            logger.info(f"Successfully loaded model '{name}'")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model '{name}' from {path}: {str(e)}")
            return None

    def get_model(self, name: str) -> Optional[Any]:
        """Retrieve a loaded model."""
        return self.models.get(name) or self.load_model(name, self.model_paths.get(name, ""))
