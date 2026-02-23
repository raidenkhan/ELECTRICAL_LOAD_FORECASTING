import json
import numpy as np

def find_value(data, target, tolerance=0.1, path=""):
    if isinstance(data, dict):
        for k, v in data.items():
            find_value(v, target, tolerance, f"{path}.{k}")
    elif isinstance(data, list):
        for i, v in enumerate(data):
            find_value(v, target, tolerance, f"{path}[{i}]")
    elif isinstance(data, (int, float)):
        if abs(data - target) < tolerance:
            print(f"Match found at {path}: {data}")

path = r'c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\MODEL_BUILDING\results_dl\dl_metrics.json'
with open(path, 'r') as f:
    data = json.load(f)

print("Searching for 3.81...")
find_value(data, 3.81)
print("\nSearching for 7.41...")
find_value(data, 7.41)
