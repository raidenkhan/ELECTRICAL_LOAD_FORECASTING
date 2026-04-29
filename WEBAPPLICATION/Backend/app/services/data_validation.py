import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from app.core.logging import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Physics-based validation for SCADA load data."""
    
    def __init__(self):
        # Validation thresholds
        self.voltage_range = (10, 500)  # kV
        self.current_range = (0, 10000)  # A
        self.frequency_range = (45.0, 55.0)  # Hz
        self.temperature_range = (-50, 100)  # °C
        self.load_range = (25.0, 2000)  # MW (Below 25 MW is an outage for Community Load)
        self.net_imbalance_threshold = 500.0  # MW
        self.MIN_LOAD_OUTAGE = 25.0
        
    def validate_csv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive validation and return a health grade.
        """
        df = df.copy()
        df.columns = [col.upper() for col in df.columns]
        
        results = {
            "total_rows": len(df),
            "valid_rows": 0,
            "invalid_rows": 0,
            "anomaly_count": 0,
            "validation_checks": {},
            "passed": True,
            "error_messages": [],
            "health_grade": "F",
            "impact_summary": ""
        }
        
        try:
            # 1. Column Check
            required_cols = ["TIMESTAMP", "TOTAL_LOAD_MW"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                results["passed"] = False
                results["error_messages"].append(f"Missing required columns: {missing_cols}")
                return results
            
            # 2. Perform Checks
            results["validation_checks"]["outage_check"] = self._check_outages(df)
            results["validation_checks"]["net_imbalance"] = self._check_net_imbalance(df)
            results["validation_checks"]["sign_convention"] = self._check_sign_convention(df)
            results["validation_checks"]["range_validation"] = self._check_ranges(df)
            results["validation_checks"]["missing_data"] = self._check_missing_data(df)
            results["validation_checks"]["anomalies"] = self._detect_anomalies(df)
            results["validation_checks"]["physics_correlation"] = self._check_physics_correlation(df)
            
            # 3. Grade Calculation
            # Count critical failures (Net imbalance or massive range violations)
            critical_checks = ["net_imbalance", "range_validation", "missing_data"]
            fail_count = sum(1 for k, c in results["validation_checks"].items() if k in critical_checks and not c.get("passed", True))
            
            anomaly_pct = results["validation_checks"]["anomalies"]["details"].get("anomaly_percentage", 0)
            outage_pct = results["validation_checks"]["outage_check"]["details"].get("outage_percentage", 0)
            
            if fail_count == 0 and anomaly_pct < 1 and outage_pct < 2: results["health_grade"] = "A"
            elif fail_count <= 1 and anomaly_pct < 5 and outage_pct < 10: results["health_grade"] = "B"
            elif fail_count <= 2 and anomaly_pct < 10: results["health_grade"] = "C"
            else: results["health_grade"] = "D"
            
            # 4. Impact Summary
            start_date = pd.to_datetime(df["TIMESTAMP"]).min().date()
            end_date = pd.to_datetime(df["TIMESTAMP"]).max().date()
            results["impact_summary"] = f"Ingesting {len(df)} points from {start_date} to {end_date}. "
            
            outage_count = results["validation_checks"]["outage_check"]["details"].get("outage_count", 0)
            if outage_count > 0:
                results["impact_summary"] += f"Detected {outage_count} points below 25MW (Grid Outage). "

            if results["health_grade"] in ["A", "B"]:
                results["impact_summary"] += "High quality data: Expect improved forecast stability."
            else:
                results["impact_summary"] += "Caution: Data contains anomalies that may skew performance."

            results["anomaly_count"] = int(results["validation_checks"]["anomalies"]["details"].get("anomaly_count", 0))
            results["valid_rows"] = len(df) - results["anomaly_count"] - outage_count
            results["passed"] = results["health_grade"] != "F"
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            results["passed"] = False
            results["error_messages"].append(f"Validation error: {str(e)}")
        
        return results

    def _check_outages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Flag records where Community Load is below 25 MW."""
        result = {"passed": True, "details": {}}
        if "TOTAL_LOAD_MW" in df.columns:
            outage_mask = df["TOTAL_LOAD_MW"] < self.MIN_LOAD_OUTAGE
            outage_count = int(outage_mask.sum())
            outage_pct = float(outage_count / len(df) * 100)
            
            result["details"] = {
                "outage_count": outage_count,
                "outage_percentage": outage_pct,
                "threshold_mw": self.MIN_LOAD_OUTAGE
            }
            if outage_pct > 20: # Over 20% outage is a bad batch
                result["passed"] = False
                result["details"]["message"] = "High percentage of outage data detected (>20%)."
        
        return result

    def _check_physics_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verify physics-based correlations (e.g. Load vs Current)."""
        result = {"passed": True, "details": {}}
        
        if "TOTAL_LOAD_MW" in df.columns and "CURRENT_A" in df.columns:
            # P = sqrt(3) * V * I * pf
            # Simple check: Correlation should be positive and high
            load = pd.to_numeric(df["TOTAL_LOAD_MW"], errors='coerce').dropna()
            curr = pd.to_numeric(df["CURRENT_A"], errors='coerce').dropna()
            
            if len(load) > 10:
                corr = np.corrcoef(load, curr)[0, 1]
                result["details"]["load_current_corr"] = float(corr)
                if corr < 0.7: # Weak correlation for power physics
                    result["passed"] = False
                    result["details"]["message"] = "Weak Load-Current correlation detected. Possible sensor failure."
        
        return result
    
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
