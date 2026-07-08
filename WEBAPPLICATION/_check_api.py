import requests
import json

r = requests.post('http://localhost:8000/api/v1/forecast/dispatch/refresh', timeout=10)
print('CACHE CLEAR:', r.json())

r = requests.get('http://localhost:8000/api/v1/forecast/dispatch/tomorrow', timeout=30)
data = r.json()
fw = data.get('forecast_mw', [])
print(f'engine: {data.get("engine")}')
print(f'forecast_date: {data.get("forecast_date")}')
print()
for i, v in enumerate(fw, 1):
    min_val = min(fw)
    marker = ' <<< MIN' if v == min_val else ''
    print(f'  H{i:02d}: {v:.0f} MW{marker}')

print(f'\nMIN: {min(fw):.0f} MW at H{fw.index(min(fw))+1:02d}')
print(f'MAX: {max(fw):.0f} MW at H{fw.index(max(fw))+1:02d}')
