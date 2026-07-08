import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine('sqlite:///loadforecast.db')

# Get March data
query = text('''
    SELECT timestamp, total_load_mw, temperature_c 
    FROM validated_data 
    WHERE timestamp >= '2025-03-01' AND timestamp < '2025-04-01'
    ORDER BY total_load_mw DESC
    LIMIT 20
''')

with engine.connect() as conn:
    result = conn.execute(query)
    rows = result.fetchall()

print('=== MARCH TOP 20 HIGHEST LOADS ===')
print('Date/Time                | Load (MW) | Temp (C)')
print('-' * 50)
for r in rows[:20]:
    print(f'{r[0]} | {r[1]:8.1f} | {r[2]:6.1f}')

# Get correlation with temperature
query2 = text('''
    SELECT temperature_c, total_load_mw
    FROM validated_data
    WHERE temperature_c IS NOT NULL AND total_load_mw IS NOT NULL
    LIMIT 5000
''')

with engine.connect() as conn:
    result = conn.execute(query2)
    rows2 = result.fetchall()

temps = [r[0] for r in rows2]
loads = [r[1] for r in rows2]

corr = pd.Series(temps).corr(pd.Series(loads))
print(f'\n=== TEMP-LOAD CORRELATION ===')
print(f'Correlation: {corr:.3f}')
print(f'Interpretation: {"+" if corr > 0 else ""}{corr} = {"positive" if corr > 0 else "negative"} correlation')

# If positive, higher temp = higher load (cooling AC)
# If negative, higher temp = lower load (heating OFF)
print(f'\nIf > 0: Hot days = More AC = Higher load')
print(f'If < 0: Cold days = More heating = Higher load')