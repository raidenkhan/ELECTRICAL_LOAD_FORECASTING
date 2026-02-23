
from typing import Dict, Any, List, Optional
import pandas as pd
from app.services.feature_engine import FeatureEngine
from app.ml.stlf_ensemble import STLFEnsemble
from app.ml.ltlf_recursive import LTLFRecursiveEngine
from app.api.deps import get_database
from sqlalchemy.future import select
from app.db.models.data import ValidatedData
from app.core.logging import get_logger

logger = get_logger(__name__)

class ForecastService:
    """
    Orchestrator for Load Forecasting.
    Routes requests to STLF or LTLF engines.
    """
    
    def __init__(self):
        self.feature_engine = FeatureEngine()
        self.stlf_engine = STLFEnsemble()
        self.ltlf_engine = LTLFRecursiveEngine()
        
    async def generate_forecast(self, session, horizon_hours: int = 24, model_type: str = "stlf") -> Dict[str, Any]:
        """
        Main entry point for generating forecasts.
        """
        if model_type.lower() == "stlf":
            return await self._run_stlf(session, horizon_hours)
        elif model_type.lower() == "ltlf":
            return await self._run_ltlf(int(horizon_hours / 24))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    async def run_simulation(
        self, 
        session, 
        horizon_hours: int, 
        temp_offset: float, 
        inflow_offset_pct: float, 
        industrial_load_offset_pct: float
    ) -> Dict[str, Any]:
        """
        Run a 'What-If' simulation by adjusting input features.
        """
        # 1. Fetch historical data as base
        df = await self._fetch_historical_data(session)

        # 2. Apply Scenario Offsets
        if temp_offset != 0:
            df["temperature_c"] = df["temperature_c"] + temp_offset
            logger.info(f"Applied temperature offset: {temp_offset}°C")
        
        if inflow_offset_pct != 0:
            # NJ6ZA_Flow proxy is line1_mw
            df["line1_mw"] = df["line1_mw"] * (1 + inflow_offset_pct / 100)
            logger.info(f"Applied grid inflow offset: {inflow_offset_pct}%")

        if industrial_load_offset_pct != 0:
            # Adjust total load directly
            df["total_load_mw"] = df["total_load_mw"] * (1 + industrial_load_offset_pct / 100)
            logger.info(f"Applied industrial load offset: {industrial_load_offset_pct}%")

        # 3. Feature Engineering & Inference
        df.rename(columns={"total_load_mw": "TOTAL_LOAD_MW", "frequency_hz": "FREQ_HZ"}, inplace=True)
        df_features = self.feature_engine.transform(df)
        prediction = self.stlf_engine.predict(df_features)

        return {
            "model_type": "simulation",
            "offsets": {
                "temp": temp_offset,
                "inflow": inflow_offset_pct,
                "industrial": industrial_load_offset_pct
            },
            **prediction
        }

    async def get_shap_values(self) -> Dict[str, Any]:
        """
        Return SHAP feature importance for the latest STLF ensemble.
        """
        # Mocking SHAP values for Stage 6 demo
        features = ["Lag_96_Load", "Rolling_Mean_24h", "Hour_Sin", "Hour_Cos", "Temp_T1_Winding", "NY6ZA_Flow", "T2_Generation"]
        values = [45.2, 28.5, -12.4, 8.2, 15.6, -5.3, 3.1]
        
        return {
            "features": features,
            "values": values,
            "base_value": 1420.5
        }

    async def get_performance_metrics(self) -> List[Dict[str, Any]]:
        """
        Return performance metrics across different horizons.
        """
        # Mocking metrics based on project benchmarks
        return [
            {"horizon": "STLF (24h)", "mae": 15.4, "rmse": 22.1, "mape": 1.2, "sample_count": 1440},
            {"horizon": "MTLF (168h)", "mae": 42.8, "rmse": 58.4, "mape": 3.5, "sample_count": 524},
            {"horizon": "LTLF (720h)", "mae": 85.2, "rmse": 112.7, "mape": 6.8, "sample_count": 120}
        ]

    async def _fetch_historical_data(self, session) -> pd.DataFrame:
        """Helper to fetch and prepare historical data."""
        needed_steps = 672 + 96 
        stmt = select(ValidatedData).order_by(ValidatedData.timestamp.desc()).limit(needed_steps)
        result = await session.execute(stmt)
        data = result.scalars().all()
        
        if not data:
            raise ValueError("Insufficient data history")
            
        rows = [d.__dict__ for d in data][::-1]
        df = pd.DataFrame(rows)
        if "timestamp" in df.columns:
            df.set_index("timestamp", inplace=True)
        
        numeric_cols = ['total_load_mw', 'line1_mw', 'line2_mw', 'line3_mw', 
                       'voltage_kv', 'current_a', 'frequency_hz', 'temperature_c']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
            
    async def _run_stlf(self, session, horizon_hours: int) -> Dict[str, Any]:
        """Run Short-Term Load Forecast."""
        # 1. Fetch recent data from DB for context
        # We need at least 672 (7 days) + buffer steps history
        df = await self._fetch_historical_data(session)
        
        logger.info(f"Loaded {len(df)} rows from database. Columns: {df.columns.tolist()}")
            
        # 2. Feature Engineering
        # Rename columns if needed to match FeatureEngine expectation
        # FeatureEngine expects: "TOTAL_LOAD_MW", "line1_mw" etc.
        # ValidatedData has: total_load_mw, line1_mw... (snake_case)
        # We map them.
        rename_map = {
            "total_load_mw": "TOTAL_LOAD_MW",
            "frequency_hz": "FREQ_HZ" # or frequency_hz depending on FeatureEngine
        }
        df.rename(columns=rename_map, inplace=True)
        # Ensure we have "TOTAL_LOAD_MW"
        if "TOTAL_LOAD_MW" not in df.columns and "total_load_mw" in df.columns:
             df["TOTAL_LOAD_MW"] = df["total_load_mw"]
             
        df_features = self.feature_engine.transform(df)
        
        # 3. Model Inference
        # We pass the last 96 steps of features
        prediction = self.stlf_engine.predict(df_features)
        
        # 4. Calculate Regime Distribution from forecast
        regime_dist = self._calculate_regime_distribution(prediction.get("forecast_mw", []))
        
        return {
            "model_type": "stlf",
            "inputs_used": len(df),
            "regime_distribution": regime_dist,
            **prediction
        }
        
    def _calculate_regime_distribution(self, forecast_mw: List[float]) -> List[Dict[str, Any]]:
        """
        Classify forecast steps into operating regimes and aggregate into 4-hour blocks.
        Regimes:
          - Standard (regime0): load < p50
          - Transition (regime1): p50 <= load < p80
          - Peak (regime2): load >= p80
        """
        import numpy as np
        if not forecast_mw:
            return []
        
        arr = np.array(forecast_mw)
        p50 = np.percentile(arr, 50)
        p80 = np.percentile(arr, 80)
        
        # 4-hour blocks: 6 blocks of 16 steps (at 15-min intervals)
        # If horizon is 24 steps (6h at 15-min), use 4-step blocks
        steps_per_hour = 4  # 15-min intervals
        block_size = 4 * steps_per_hour  # 4-hour block = 16 steps
        
        # Adjust block_size if forecast is shorter (e.g. 24 steps = 6h)
        n = len(arr)
        if n <= 24:
            # Treat as hourly-ish: 4 blocks of ~6 steps
            block_size = max(1, n // 6)
        
        block_labels = ['00-04', '04-08', '08-12', '12-16', '16-20', '20-24']
        result = []
        
        for i, label in enumerate(block_labels):
            start = i * block_size
            end = start + block_size
            block = arr[start:end]
            if len(block) == 0:
                break
            
            total = len(block)
            standard = int(np.sum(block < p50))
            peak = int(np.sum(block >= p80))
            transition = total - standard - peak
            
            result.append({
                "hour": label,
                "regime0": round(standard / total * 100, 1),
                "regime1": round(transition / total * 100, 1),
                "regime2": round(peak / total * 100, 1),
            })
        
        return result

    async def _run_ltlf(self, days: int) -> Dict[str, Any]:
        """Run Long-Term Load Forecast."""
        start_date = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
        return self.ltlf_engine.predict(start_date, days)
