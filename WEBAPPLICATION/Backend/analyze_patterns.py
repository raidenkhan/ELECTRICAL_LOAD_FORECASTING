import pandas as pd

# Real historical data patterns from analysis
actual_dow = {
    "Monday": 79.7,
    "Tuesday": 81.0,
    "Wednesday": 82.2,
    "Thursday": 83.4,
    "Friday": 80.6,
    "Saturday": 79.2,
    "Sunday": 79.5
}

# Our forecast patterns
forecast_dow = {
    "Saturday": 71.5,
    "Sunday": 76.4,
    "Monday": 74.9,
    "Tuesday": 75.8,
    "Wednesday": 82.2,
    "Thursday": 91.5,
    "Friday": 81.2
}

dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

print("=" * 55)
print("FORECAST vs ACTUAL - DAY OF WEEK PATTERNS")
print("=" * 55)
print("Day          | Actual | Forecast | Diff")
print("-" * 45)

total_diff = 0
for name in dow_names:
    actual = actual_dow.get(name, 0)
    forecast = forecast_dow.get(name, 0)
    diff = forecast - actual
    total_diff += abs(diff)
    marker = "HIGH" if diff > 5 else "LOW" if diff < -5 else "OK"
    print(f"{name:<12} | {actual:6.1f} | {forecast:7.1f} | {diff:+6.1f} [{marker}]")

avg_diff = total_diff / 7
print("-" * 45)
print(f"Average difference: {avg_diff:.1f} MW")

# Correlation
actual_arr = [actual_dow[n] for n in dow_names]
forecast_arr = [forecast_dow[n] for n in dow_names]
corr = pd.Series(actual_arr).corr(pd.Series(forecast_arr))

print("\n" + "=" * 55)
print("ANALYSIS")
print("=" * 55)
print(f"Pattern correlation: {corr:.3f}")

if corr > 0.7:
    print("Status: Patterns are well correlated")
elif corr > 0.3:
    print("Status: Patterns moderately correlated")
else:
    print("Status: Patterns need adjustment")

print("\n" + "=" * 55)
print("FINDINGS")
print("=" * 55)
print("1. Forecast SAT (71.5) is much lower than actual SAT (79.2)")
print("2. Forecast THU (91.5) is higher than actual THU (83.4)")
print("3. Weekend pattern is underestimating actual load")
print("\nThis explains why 1-month view shows repeating pattern")
print("that differs from historical patterns")