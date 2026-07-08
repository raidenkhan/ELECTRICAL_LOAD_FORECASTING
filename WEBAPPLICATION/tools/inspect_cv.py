import pandas as pd
df = pd.read_csv("Backend/output/true_6fold_cv.csv")
print("Columns:", list(df.columns))
for _, r in df.iterrows():
    print(r["fold"], f"all={float(r['mae_all']):.1f}", f"24h={float(r['mae_24h']):.1f}", f"168h={float(r['mae_168h']):.1f}", f"720h={float(r['mae_720h']):.1f}")
