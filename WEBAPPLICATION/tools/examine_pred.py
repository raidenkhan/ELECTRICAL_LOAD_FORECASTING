import pandas as pd, json
df = pd.read_csv("Backend/report_forecast_data.csv")
row = df.iloc[0]
pred = json.loads(row["pred_mw"])
actual = json.loads(row["actual_mw"])
print("First 24 hours of D+1 prediction (2026-04-01):")
for h in range(24):
    print(f"  H{h}: pred={pred[h]:.1f}, actual={actual[h]:.1f}")
print(f"\nPred range: {min(pred):.1f} - {max(pred):.1f}")
print(f"Actual range: {min(actual):.1f} - {max(actual):.1f}")
