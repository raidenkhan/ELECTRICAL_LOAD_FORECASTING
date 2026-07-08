import json
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from app.core.logging import get_logger

logger = get_logger(__name__)

MODEL_METADATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
METADATA_FILE = os.path.join(MODEL_METADATA_DIR, "metadata.json")


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class ModelMetadata:
    """
    Model version metadata tracker.
    Tracks training date, metrics, parameters, and data hash.
    """
    
    def __init__(
        self,
        model_name: str,
        version: str = "1.0.0",
        model_type: str = "stlf",
        training_date: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        training_data_hash: Optional[str] = None,
        feature_names: Optional[List[str]] = None
    ):
        self.model_name = model_name
        self.version = version
        self.model_type = model_type
        self.training_date = training_date or datetime.now().isoformat()
        self.metrics = metrics or {}
        self.params = params or {}
        self.training_data_hash = training_data_hash or ""
        self.feature_names = feature_names or []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "model_type": self.model_type,
            "training_date": self.training_date,
            "metrics": self.metrics,
            "params": self.params,
            "training_data_hash": self.training_data_hash,
            "feature_names": self.feature_names
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        return cls(
            model_name=data["model_name"],
            version=data.get("version", "1.0.0"),
            model_type=data.get("model_type", "stlf"),
            training_date=data.get("training_date"),
            metrics=data.get("metrics", {}),
            params=data.get("params", {}),
            training_data_hash=data.get("training_data_hash"),
            feature_names=data.get("feature_names", [])
        )


def save_model_metadata(metadata: ModelMetadata) -> bool:
    """
    Save model metadata to metadata.json.
    Creates or updates the file with all model versions.
    """
    try:
        all_metadata = load_all_metadata()
        all_metadata[metadata.model_name] = metadata.to_dict()
        
        with open(METADATA_FILE, "w") as f:
            json.dump(all_metadata, f, indent=2)
        
        logger.info(f"Saved metadata for model '{metadata.model_name}' v{metadata.version}")
        return True
    except Exception as e:
        logger.error(f"Failed to save metadata: {str(e)}")
        return False


def load_all_metadata() -> Dict[str, Any]:
    """Load all model metadata from metadata.json."""
    if not os.path.exists(METADATA_FILE):
        return {}
    
    try:
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata: {str(e)}")
        return {}


def get_model_metadata(model_name: str) -> Optional[ModelMetadata]:
    """Get metadata for a specific model."""
    all_metadata = load_all_metadata()
    if model_name in all_metadata:
        return ModelMetadata.from_dict(all_metadata[model_name])
    return None


def list_models() -> List[Dict[str, str]]:
    """List all registered models with versions."""
    all_metadata = load_all_metadata()
    return [
        {"name": name, "version": data.get("version"), "type": data.get("model_type")}
        for name, data in all_metadata.items()
    ]