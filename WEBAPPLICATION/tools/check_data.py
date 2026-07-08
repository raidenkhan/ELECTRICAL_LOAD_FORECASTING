import pandas as pd, json
df = pd.read_csv("Backend/report_forecast_data.csv")
for _, row in df.iterrows():
    pred = json.loads(row["pred_mw"])
    actual = json.loads(row["actual_mw"])
    mae = sum(abs(p-a) for p,a in zip(pred,actual)) / len(pred)
    print(f"{row['test_date']} {row['horizon']}: n={len(pred)}h, mae={mae:.1f} MW")
