
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Any
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class LTLFRecursiveEngine:
    """
    Long-Term Load Forecasting Engine (Recursive).
    Predicts Daily Peak Load for 1-30 days ahead using LightGBM Quantiles.
    """
    
    def __init__(self):
        self.model_path = settings.LTLF_RECURSIVE_PATH
        self.artifact = None
        self.models = {}
        self.features = []
        self.last_peak_history = [] # For recursion
        
    def load(self):
        """Load model artifact."""
        if self.artifact:
            return

        if not os.path.exists(self.model_path):
            logger.error(f"LTLF model not found at {self.model_path}")
            return
            
        try:
            self.artifact = joblib.load(self.model_path)
            self.models = self.artifact.get("models", {})
            self.features = self.artifact.get("features", [])
            self.last_peak_history = self.artifact.get("last_peak_history", []) # Last 7 days
            logger.info("Successfully loaded LTLF Recursive Model")
        except Exception as e:
            logger.error(f"Failed to load LTLF model: {e}")
            
    def predict(self, start_date: pd.Timestamp, horizon_days: int = 30) -> Dict[str, Any]:
        """
        Generate Daily Peak forecast recursively.
        """
        self.load()
        
        if not self.models:
            raise ValueError("LTLF models not loaded")
            
        preds = {0.1: [], 0.5: [], 0.9: []}
        
        # Initialize history
        # In a real real-time system, we should update this from DB
        # For now, use the one saved in artifact as fallback, or update if provided
        # Ideally, we query the DB for the last 7 days of actual peaks.
        # But for this implementation, we'll assume the artifact is reasonably fresh
        # or we rely on the caller to provide context (which we don't handle yet).
        
        # Simplification: Use artifact history. In prod, we'd query `ValidatedData`.
        current_history = list(self.last_peak_history) # Copy
        
        # Start from tomorrow relative to last known data?
        # The training script saved state at the end of training.
        # Ideally `start_date` should align.
        
        current_date = start_date
        timestamps = []
        
        for _ in range(horizon_days):
            timestamps.append(current_date)
            
            # Construct feature row
            row = pd.DataFrame(index=[current_date])
            row['DayOfWeek'] = current_date.dayofweek
            row['Month'] = current_date.month
            row['DayOfYear'] = current_date.dayofyear
            row['Lag_1'] = current_history[-1]
            row['Lag_7'] = current_history[-7] if len(current_history) >= 7 else current_history[0]
            
            # Predict
            step_pred_p50 = 0
            for alpha in [0.1, 0.5, 0.9]:
                if alpha in self.models:
                    model = self.models[alpha]
                    # Check feature columns
                    p = model.predict(row[self.features])[0]
                    preds[alpha].append(p)
                    if alpha == 0.5:
                        step_pred_p50 = p
            
            # Recurse using P50
            current_history.append(step_pred_p50)
            current_date += pd.Timedelta(days=1)
            
        return {
            "timestamps": timestamps,
            "forecast_mw": preds[0.5],
            "p10": preds[0.1],
            "p90": preds[0.9],
            "metadata": {
                "model_type": "LightGBM Recursive",
                "horizon_days": horizon_days
            }
        }
