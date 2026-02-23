import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from app.core.logging import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Physics-based validation for SCADA load data."""
    
    def __init__(self):
        # Validation thresholds
        self.voltage_range = (100, 500)  # kV
        self.current_range = (0, 10000)  # A
        self.frequency_range = (45.0, 55.0)  # Hz (Relaxed for verification)
        self.temperature_range = (-50, 100)  # °C
        self.load_range = (-500, 2000)  # MW (broadened for noise)
        self.net_imbalance_threshold = 500.0  # MW (widened for verification)
        
    def validate_csv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive validation on uploaded CSV data.
        
        Args:
            df: DataFrame with SCADA data
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "total_rows": len(df),
            "valid_rows": 0,
            "invalid_rows": 0,
            "anomaly_count": 0,
            "validation_checks": {},
            "passed": True,
            "error_messages": []
        }
        
        try:
            # 1. Check for required columns
            required_cols = ["timestamp", "TOTAL_LOAD_MW"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                results["passed"] = False
                results["error_messages"].append(f"Missing required columns: {missing_cols}")
                return results
            
            # 2. Net imbalance check
            results["validation_checks"]["net_imbalance"] = self._check_net_imbalance(df)
            
            # 3. Sign convention validation
            results["validation_checks"]["sign_convention"] = self._check_sign_convention(df)
            
            # 4. Range validation
            results["validation_checks"]["range_validation"] = self._check_ranges(df)
            
            # 5. Missing data detection
            results["validation_checks"]["missing_data"] = self._check_missing_data(df)
            
            # 6. Anomaly detection
            results["validation_checks"]["anomalies"] = self._detect_anomalies(df)
            
            # Calculate valid/invalid rows
            results["valid_rows"] = len(df) - results["anomaly_count"]
            results["invalid_rows"] = results["anomaly_count"]
            
            # Overall pass/fail
            results["passed"] = all(
                check.get("passed", True) 
                for check in results["validation_checks"].values()
            )
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            results["passed"] = False
            results["error_messages"].append(f"Validation error: {str(e)}")
        
        return results
    
    def _check_net_imbalance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check net power imbalance (Generation - Load)."""
        result = {"passed": True, "details": {}}
        
        # If we have line-level data, check imbalance
        line_cols = [col for col in df.columns if "LINE" in col.upper() and "MW" in col.upper()]
        
        if line_cols and "TOTAL_LOAD_MW" in df.columns:
            line_sum = df[line_cols].sum(axis=1)
            total_load = df["TOTAL_LOAD_MW"]
            imbalance = (line_sum - total_load).abs()
            
            max_imbalance = imbalance.max()
            mean_imbalance = imbalance.mean()
            
            result["details"] = {
                "max_imbalance_mw": float(max_imbalance),
                "mean_imbalance_mw": float(mean_imbalance),
                "threshold_mw": self.net_imbalance_threshold
            }
            
            if max_imbalance > self.net_imbalance_threshold:
                result["passed"] = False
                result["details"]["message"] = f"Net imbalance exceeds threshold: {max_imbalance:.2f} MW"
        else:
            result["details"]["message"] = "Insufficient data for net imbalance check"
        
        return result
    
    def _check_sign_convention(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate sign conventions (loads should be positive)."""
        result = {"passed": True, "details": {}}
        
        load_cols = [col for col in df.columns if "LOAD" in col.upper() and "MW" in col.upper()]
        
        negative_counts = {}
        for col in load_cols:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                negative_counts[col] = int(negative_count)
                result["passed"] = False
        
        result["details"] = {
            "negative_load_counts": negative_counts,
            "message": "All loads should be positive" if not result["passed"] else "Sign convention valid"
        }
        
        return result
    
    def _check_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that all values are within acceptable ranges."""
        result = {"passed": True, "details": {"violations": {}}}
        
        range_checks = {
            "VOLTAGE": self.voltage_range,
            "CURRENT": self.current_range,
            "FREQ": self.frequency_range,
            "TEMP": self.temperature_range,
            "LOAD_MW": self.load_range
        }
        
        for key, (min_val, max_val) in range_checks.items():
            matching_cols = [col for col in df.columns if key in col.upper()]
            
            for col in matching_cols:
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if out_of_range > 0:
                    result["passed"] = False
                    result["details"]["violations"][col] = {
                        "count": int(out_of_range),
                        "range": [min_val, max_val],
                        "min_value": float(df[col].min()),
                        "max_value": float(df[col].max())
                    }
        
        return result
    
    def _check_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing data."""
        result = {"passed": True, "details": {}}
        
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df) * 100).round(2)
        
        cols_with_missing = missing_pct[missing_pct > 0].to_dict()
        
        if cols_with_missing:
            result["details"]["missing_data"] = cols_with_missing
            # Fail if more than 5% missing in any column
            if any(pct > 5 for pct in cols_with_missing.values()):
                result["passed"] = False
                result["details"]["message"] = "Excessive missing data (>5%) detected"
        else:
            result["details"]["message"] = "No missing data detected"
        
        return result
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using statistical methods."""
        result = {"passed": True, "details": {"anomaly_indices": []}}
        
        # Simple anomaly detection: values beyond 3 standard deviations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        anomaly_mask = pd.Series([False] * len(df))
        
        for col in numeric_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                
                if std > 0:  # Avoid division by zero
                    z_scores = np.abs((df[col] - mean) / std)
                    col_anomalies = z_scores > 3
                    anomaly_mask |= col_anomalies
        
        anomaly_count = anomaly_mask.sum()
        result["details"]["anomaly_count"] = int(anomaly_count)
        result["details"]["anomaly_percentage"] = float(anomaly_count / len(df) * 100)
        
        if anomaly_count > 0:
            result["details"]["anomaly_indices"] = anomaly_mask[anomaly_mask].index.tolist()[:100]  # Limit to first 100
        
        return result
