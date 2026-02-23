
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Optional, Tuple
from app.core.config import settings
from app.core.logging import get_logger
from app.ml.model_loader import ModelLoader

logger = get_logger(__name__)

class STLFEnsemble:
    """
    Short-Term Load Forecasting Ensemble Engine.
    Combines Autoformer (Deep Learning) and LightGBM (Gradient Boosting)
    using Adaptive Kalman Fusion.
    """
    
    def __init__(self):
        self.loader = ModelLoader()
        self.horizon = 24 # 6 hours * 4 steps/hour (Matched to trained Autoformer)
        self.input_len = 96 # 24 hours lookback
        
        # Feature Definitions (Matched to Research Pipeline)
        self.lgbm_features = [
            "Hour", "DayOfWeek", "Month", "IsWeekend", "Hour_Sin", "Hour_Cos",
            "Lag_1", "Lag_4", "Lag_96", "Lag_672",
            "Rolling_Mean_24h", "Rolling_Std_24h", "Rolling_Min_24h", "Rolling_Max_24h",
            "NY6ZA_Flow", "NY6ZA_Lag_1", "T2_Generation", "Freq_Deviation"
        ]
        
        self.af_features = ["Community_Load_MW"] + self.lgbm_features
        
        # Stats from feature_engineering_report.md for Standardization
        # Used by Autoformer which was trained on Scaled data.
        self.feature_stats = {
            "Community_Load_MW": {"mean": 83.6134, "std": 21.1097},
            "Hour": {"mean": 11.50, "std": 6.92},
            "Hour_Sin": {"mean": -0.00, "std": 0.71},
            "Hour_Cos": {"mean": -0.00, "std": 0.71},
            "DayOfWeek": {"mean": 3.02, "std": 2.00},
            "Month": {"mean": 5.50, "std": 3.39},
            "IsWeekend": {"mean": 0.29, "std": 0.45},
            "Lag_1": {"mean": 81.64, "std": 23.31},
            "Lag_4": {"mean": 81.65, "std": 23.31},
            "Lag_96": {"mean": 81.77, "std": 23.29},
            "Lag_672": {"mean": 82.65, "std": 22.59},
            "Rolling_Mean_24h": {"mean": 81.70, "std": 18.68},
            "Rolling_Std_24h": {"mean": 11.34, "std": 8.23},
            "Rolling_Min_24h": {"mean": 53.59, "std": 27.43},
            "Rolling_Max_24h": {"mean": 101.61, "std": 24.02},
            "NY6ZA_Flow": {"mean": 124.44, "std": 34.41},
            "NY6ZA_Lag_1": {"mean": 124.44, "std": 34.41},
            "T2_Generation": {"mean": 13.91, "std": 9.94},
            "Freq_Deviation": {"mean": 0.24, "std": 0.08}
        }
        
        # Benchmark Performance (MAE) -> Variance estimate
        self.sigma_autoformer = 8.30 * 1.25
        self.sigma_lightgbm = 20.31 * 1.25
        
    def predict(self, df_history: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate 6-hour forecast from history. (Limited by model horizon)
        
        Args:
            df_history: DataFrame with at least 24h (96 steps) of data.
            
        Returns:
            Dictionary with timestamps, p50 forecast, confident intervals, and metadata.
        """
        models = self.loader.load_all_models()
        model_af = models.get("autoformer_stlf")
        model_lgb = models.get("lightgbm_stlf")
        
        if model_af is None and model_lgb is None:
            raise ValueError("No STLF models enabled or loaded.")
            
        # Prepare inputs
        last_timestamp = df_history.index[-1]
        future_timestamps = [last_timestamp + pd.Timedelta(minutes=15 * (i+1)) for i in range(self.horizon)]
        
        pred_af = None
        pred_lgb = None
        
        # 1. LightGBM Inference
        if model_lgb:
            try:
                pred_lgb = self._predict_lightgbm(model_lgb, df_history)
            except Exception as e:
                logger.error(f"LightGBM inference failed: {e}")
                
        # 2. Autoformer Inference
        if model_af:
            try:
                pred_af = self._predict_autoformer(model_af, df_history)
            except Exception as e:
                logger.error(f"Autoformer inference failed: {e}")
        
        # 3. Kalman Fusion
        ensemble_pred, uncertainty = self._kalman_fusion(pred_af, pred_lgb)
        
        # Replace NaN/Inf with 0.0 for JSON compliance
        ensemble_pred = np.nan_to_num(ensemble_pred, nan=0.0, posinf=0.0, neginf=0.0)
        uncertainty = np.nan_to_num(uncertainty, nan=0.0, posinf=0.0, neginf=0.0)
        
        return {
            "timestamps": future_timestamps,
            "forecast_mw": ensemble_pred.tolist(),
            "p10": (ensemble_pred - 1.28 * uncertainty).tolist(), # 10th percentile (assuming normal)
            "p90": (ensemble_pred + 1.28 * uncertainty).tolist(), # 90th percentile
            "contributions": {
                "autoformer": pred_af.tolist() if pred_af is not None else [],
                "lightgbm": pred_lgb.tolist() if pred_lgb is not None else []
            }
        }
        
    def _predict_lightgbm(self, model, df: pd.DataFrame) -> np.ndarray:
        """
        Run LightGBM Direct Forecast.
        Handles Dictionary of models (per step) or Single Multi-output model.
        """
        # Feature selection for LightGBM (Model expects 18 features)
        missing = [c for c in self.lgbm_features if c not in df.columns]
        if missing:
             logger.warning(f"LightGBM missing features: {missing}. Filling with 0.0")
             for c in missing:
                  df[c] = 0.0
                  
        # Select and Reorder to match model expectations
        input_row = df.iloc[[-1]][self.lgbm_features]
        
        preds = []
        
        if isinstance(model, dict):
            # Dict of {step: model}
            sorted_steps = sorted([k for k in model.keys() if isinstance(k, int)])
            if not sorted_steps:
                sorted_steps = sorted(model.keys())
                
            for step in sorted_steps:
                try:
                    m = model[step]
                    p = m.predict(input_row)
                    preds.append(p[0])
                except Exception as e:
                    logger.error(f"Inference failed for step {step}: {e}")
                    preds.append(0.0)
                
            if len(preds) < self.horizon:
                preds.extend([preds[-1] if preds else 0.0] * (self.horizon - len(preds)))
            
            return np.array(preds[:self.horizon])
            
        elif hasattr(model, 'predict'):
            # Single model
            p = model.predict(input_row)
            if p.ndim == 1:
                return p[:self.horizon]
            elif p.ndim == 2:
                return p[0, :self.horizon]
            else:
                 return np.full(self.horizon, p[0])

        return np.zeros(self.horizon)

    def _predict_autoformer(self, model, df: pd.DataFrame) -> np.ndarray:
        """
        Run Autoformer Inference.
        Expects (Batch, Seq_Len, Features).
        """
        # Check available columns vs expected (19 features)
        missing = [c for c in self.af_features if c not in df.columns]
        if missing:
            logger.warning(f"Autoformer missing features: {missing}. Filling with 0.")
            for c in missing:
                df[c] = 0.0
                
        # Select and Reorder
        df_enc = df.iloc[-self.input_len:][self.af_features].copy()
        
        # Standardization (Z-score)
        # Deep Learning models are sensitive to scale.
        for col in self.af_features:
            stats = self.feature_stats.get(col, {"mean": 0.0, "std": 1.0})
            df_enc[col] = (df_enc[col] - stats["mean"]) / stats["std"]
            
        # Check for NaNs
        nan_cols = df_enc.columns[df_enc.isna().any()].tolist()
        if nan_cols:
            logger.warning(f"Autoformer input NaNs in columns: {nan_cols}")
            df_enc = df_enc.fillna(0.0)
        
        # Batch dim, float32
        data_enc = df_enc.values.astype(np.float32)
        x_enc = torch.tensor(data_enc).float().unsqueeze(0) # [1, 96, 19]
        
        import torch.nn as nn
        from app.ml.architectures.autoformer import OptimizedAutoformer
        
        inference_model = None
        
        if isinstance(model, dict) and "enc_embedding.weight" in model:
            # It's a state dict
            # Deduce horizon from fc_seasonal if needed, or use self.horizon
            
            # Use params found in inspection and error log
            # horizon=24 is critical!
            # future_dim=4
            
            arch = OptimizedAutoformer(input_dim=19, future_dim=4, seq_len=96, horizon=24, d_model=32, n_heads=2, e_layers=1)
            try:
                arch.load_state_dict(model)
                inference_model = arch
            except Exception as e:
                logger.error(f"Failed to load Autoformer state dict: {e}")
                pass
        elif isinstance(model, nn.Module):
            inference_model = model
            
        if inference_model is None:
             return np.zeros(self.horizon)
             
        inference_model.eval()
        
        with torch.no_grad():
            # Extract target for RevIN (first column)
            target_for_revin = x_enc[:, :, 0:1]
            
            # Forward pass with RevIN context
            # We don't have future_feats readily available in this vector, so passing None.
            output = inference_model(x_enc, future_feats=None, target_for_revin=target_for_revin)
            
            # Output is [Batch, Horizon]
            pred_scaled = output[0].numpy()
            
            # Inverse Transform (Un-scale)
            # Autoformer predicts scaled values if trained with scaled target.
            target_stats = self.feature_stats["Community_Load_MW"]
            pred = (pred_scaled * target_stats["std"]) + target_stats["mean"]
            
        return pred[:self.horizon]

    def _kalman_fusion(self, pred1, pred2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse predictions using Inverse Variance Weighting (Static Kalman Filter).
        """
        # Handle None or NaN
        if pred1 is not None and np.isnan(pred1).any():
            logger.warning("Autoformer prediction contains NaNs. Ignoring for fusion.")
            pred1 = None
            
        if pred2 is not None and np.isnan(pred2).any():
            logger.warning("LightGBM prediction contains NaNs. Ignoring for fusion.")
            pred2 = None

        if pred1 is None and pred2 is None:
            return np.zeros(self.horizon), np.zeros(self.horizon)
            
        if pred1 is None:
            return pred2, np.full(self.horizon, self.sigma_lightgbm)
            
        if pred2 is None:
            return pred1, np.full(self.horizon, self.sigma_autoformer)
            
        # Fusion
        # w1 = var2 / (var1 + var2)
        v1 = self.sigma_autoformer ** 2
        v2 = self.sigma_lightgbm ** 2
        
        w1 = v2 / (v1 + v2)
        w2 = v1 / (v1 + v2)
        
        fused = w1 * pred1 + w2 * pred2
        
        # Combined variance
        v_fused = (v1 * v2) / (v1 + v2)
        std_fused = np.sqrt(v_fused)
        
        return fused, np.full(self.horizon, std_fused)
