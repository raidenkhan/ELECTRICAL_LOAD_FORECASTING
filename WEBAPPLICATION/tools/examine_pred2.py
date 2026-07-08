import pandas as pd, json
df = pd.read_csv("Backend/report_forecast_data.csv")
for _, row in df.iterrows():
    pred = json.loads(row["pred_mw"])
    actual = json.loads(row["actual_mw"])
    mae = sum(abs(p-a) for p,a in zip(pred,actual))/len(pred)
    bias = sum(p-a for p,a in zip(pred,actual))/len(pred)
    print(f"{row['test_date']} {row['horizon']}: mae={mae:.1f}MW bias={bias:.1f}MW n={len(pred)}")
    # Show first 6 hours
    for h in range(min(6, len(pred))):
        print(f"  H{h}: pred={pred[h]:.0f} actual={actual[h]:.0f}")
    print()
