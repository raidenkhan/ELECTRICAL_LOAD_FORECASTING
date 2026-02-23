import pandas as pd
import numpy as np

try:
    df = pd.read_csv(r'c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\MODEL_BUILDING\results_final_ltlf\final_ltlf_benchmarks.csv')
    df = df.dropna(subset=['Actual', 'Pred_P50'])
    mae = np.mean(np.abs(df['Actual'] - df['Pred_P50']))
    print(f"MAE: {mae}")
except Exception as e:
    print(e)
