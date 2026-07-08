import requests
r = requests.get('http://localhost:8000/api/v1/forecast/dispatch/tomorrow', timeout=30)
data = r.json()
fw = data.get('forecast_mw', [])
p10 = data.get('p10_mw', [])
p90 = data.get('p90_mw', [])
for i in range(24):
    if p10 and p90:
        band = p90[i] - p10[i]
        print(f'H{i+1:02d}: forecast={fw[i]:.0f}  P10={p10[i]:.0f}  P90={p90[i]:.0f}  band={band:.0f} MW')
