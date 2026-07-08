import pandas as pd, json
df = pd.read_csv("Backend/report_forecast_data.csv")
for _, row in df.iterrows():
    pred = json.loads(row["pred_mw"])
    actual = json.loads(row["actual_mw"])
    print(f"{row['test_date']} {row['horizon']}: pred={len(pred)}h actual={len(actual)}h n_hours={row['n_hours']}")
