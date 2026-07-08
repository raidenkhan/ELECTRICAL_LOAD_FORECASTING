import pandas as pd
import numpy as np
from typing import Dict, Any
from sqlalchemy import create_engine, text
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

DB_PATH = settings.DATABASE_URL.replace("+aiosqlite", "")


def get_monthly_insights() -> Dict[str, Any]:
    """
    Analyze monthly patterns and provide insights for high peaks and low dips.
    """
    try:
        engine = create_engine(DB_PATH)
        
        # Get recent data
        query = text('''
            SELECT timestamp, total_load_mw, temperature_c 
            FROM validated_data 
            ORDER BY timestamp DESC 
            LIMIT 10000
        ''')
        
        with engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()
        
        if len(rows) < 100:
            return {"error": "Insufficient data"}
        
        df = pd.DataFrame(rows, columns=['timestamp', 'total_load_mw', 'temperature_c'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Monthly stats
        df['month'] = df['timestamp'].dt.month
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['dow'] = df['timestamp'].dt.dayofweek
        
        monthly = df.groupby('month')['total_load_mw'].agg(['mean', 'max', 'min', 'std', 'count'])
        monthly['range'] = monthly['max'] - monthly['min']
        
        # Find highest peaks by day
        daily_max = df.groupby('date')['total_load_mw'].max()
        peak_days = daily_max.nlargest(5)
        
        # Find lowest dips by day
        dip_days = daily_max.nsmallest(5)
        
        # DOW pattern
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_avg = df.groupby('dow')['total_load_mw'].mean()
        
        # Temperature correlation
        temp_corr = df['temperature_c'].corr(df['total_load_mw'])
        
        # Identify drivers
        drivers = []
        
        # Temperature driver
        temp_driver = "cooling" if temp_corr > 0.1 else "heating" if temp_corr < -0.1 else "minimal"
        drivers.append({
            "name": "Temperature",
            "impact": temp_driver,
            "correlation": round(temp_corr, 3),
            "description": f"Temp has {temp_driver} effect on load (correlation: {temp_corr:.3f})"
        })
        
        # Day of week driver
        dow_values = [dow_avg[i] for i in range(7)]
        max_dow = dow_avg.idxmax()
        min_dow = dow_avg.idxmin()
        drivers.append({
            "name": "Day of Week",
            "impact": "moderate",
            "highest": dow_names[max_dow],
            "lowest": dow_names[min_dow],
            "description": f"{dow_names[max_dow]} is highest, {dow_names[min_dow]} is lowest"
        })
        
        # Identify extreme months
        highest_month = monthly['mean'].idxmax()
        lowest_month = monthly['mean'].idxmin()
        
        month_names = {1: "January", 2: "February", 3: "March", 4: "April",
                  5: "May", 6: "June", 7: "July", 8: "August",
                  9: "September", 10: "October", 11: "November", 12: "December"}
        
        return {
            "summary": {
                "total_records": len(df),
                "date_range": f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}",
                "overall_mean": round(df['total_load_mw'].mean(), 1),
                "overall_max": round(df['total_load_mw'].max(), 1),
                "overall_min": round(df['total_load_mw'].min(), 1)
            },
            "monthly_patterns": {
                month_names.get(m, str(m)): {
                    "mean": round(v['mean'], 1),
                    "max": round(v['max'], 1),
                    "range": round(v['range'], 1)
                }
                for m, v in monthly.iterrows() if v['count'] > 50
            },
            "extreme_days": {
                "highest_peaks": [
                    {"date": str(d), "load": round(l, 1)}
                    for d, l in peak_days.items()
                ],
                "lowest_dips": [
                    {"date": str(d), "load": round(l, 1)}
                    for d, l in dip_days.items()
                ]
            },
            "dow_pattern": {
                dow_names[i]: round(v, 1)
                for i, v in dow_avg.items()
            },
            "drivers": drivers,
            "recommendations": [
                f"Expect high peaks on {dow_names[max_dow]} days between 14:00-18:00",
                f"Temperature effect is {temp_driver} - prepare for {'high' if abs(temp_corr) > 0.3 else 'moderate'} temp sensitivity",
                f"Highest load month: {month_names.get(highest_month, str(highest_month))} (~{monthly.loc[highest_month, 'mean']:.0f} MW)",
                f"Lowest load month: {month_names.get(lowest_month, str(lowest_month))} (~{monthly.loc[lowest_month, 'mean']:.0f} MW)"
            ]
        }
        
    except Exception as e:
        logger.error(f"Monthly analysis failed: {str(e)}")
        return {"error": str(e)}


# Run if called directly
if __name__ == "__main__":
    result = get_monthly_insights()
    import json
    print(json.dumps(result, indent=2))