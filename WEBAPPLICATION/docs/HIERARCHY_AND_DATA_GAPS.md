# Architecture Plan: Hierarchical Forecasting + Data Gaps

## The Problem

Current system forecasts ECG system-wide demand only. A national grid operator
needs forecasts at multiple levels of aggregation:

```
Ghana National Grid (ECG Total)
  +-- Northern Zone Substations
  |     +-- Substation A (500 kV)
  |     +-- Substation B (161 kV)
  +-- Southern Zone Substations
  |     +-- Substation C (500 kV)
  |     +-- Substation D (225 kV)
  +-- Generation (supply-side)
        +-- Hydro (Akosombo, Kpong, Bui)
        +-- Thermal (Tema, Takoradi)
        +-- Solar + Imports
```

## Required Data (Currently Missing)

| Data | Use Case | Source |
|------|----------|--------|
| Substation-level demand (kV level) | Regional forecasts, loss calculations | ECG SCADA/EMS |
| Generation unit output (MW/h) | Supply forecast, hydro scheduling | GRIDCo dispatch logs |
| Reservoir levels (Akosombo, Bui) | Hydro generation capacity | VRA/GRIDCo |
| Solar farm irradiance + output | Solar forecasting | Volta River Authority |
| Rainfall data | Hydro inflow forecasting | Ghana Meteorological Agency |
| Scheduled outages (generation + transmission) | Supply availability | GRIDCo maintenance dept |
| Economic indicators (GDP, industrial index) | Long-term demand growth | Ghana Statistical Service |
| Population by region | Spatial demand growth | Ghana Statistical Service |
| Fuel prices (gas, crude) | Thermal generation cost | Ministry of Energy |

Without these, the engine is blind to:
- Regional demand patterns (Accra + coastal vs Northern savanna)
- Hydro availability during dry season (major supply risk)
- Solar ramp rates during cloud cover (grid stability)

## Reconciliation Strategy

### Approach: MinT (Minimum Trace) Optimal Combination

Once regional/substation forecasts exist, they must reconcile:

```
Forecast hierarchy:
    y_total[t] = sum(y_zones[t]) = sum(y_subs[t])
    
MinT reconciliation:
    y_tilde = S @ (S' @ W^{-1} @ S)^{-1} @ S' @ W^{-1} @ y_hat
    
    where S = summation matrix (zones -> total)
          W = covariance matrix of forecast errors
          y_hat = base (incoherent) forecasts
          y_tilde = reconciled forecasts
```

### Implementation phases:

**Phase 1 (current):** System-wide only — no reconciliation needed

**Phase 2 (data dependent):** Zone-level + MinT reconciliation when zone data arrives.
Add optional disaggregation via proportional allocation based on historical shares.

**Phase 3 (future):** Full hierarchy with generation mix + bottom-up supply forecasting.
Requires the 8+ data sources listed above.

## How to Add Regional Forecasting (Template)

```python
class HierarchicalReconciliation:
    """MinT optimal combination for hierarchical forecasts."""

    def __init__(
        self, 
        summation_matrix: np.ndarray,  # S: (n_bottom x n_total)
        error_covariance: np.ndarray,  # W: (n_total x n_total)
    ):
        self.S = summation_matrix
        self.W = error_covariance
        # Pre-compute reconciliation matrix for efficiency
        self.P = self._compute_reconciliation_matrix()

    def _compute_reconciliation_matrix(self) -> np.ndarray:
        # P = (S' W^{-1} S)^{-1} S' W^{-1}
        SW_inv = self.S.T @ np.linalg.inv(self.W)
        return np.linalg.inv(SW_inv @ self.S) @ SW_inv

    def reconcile(self, forecasts: np.ndarray) -> np.ndarray:
        # y_tilde = S @ P @ y_hat
        return self.S @ (self.P @ forecasts)
```

The engine currently has no hierarchical components. This is tracked for future
data-dependent phases. Progress cannot proceed without substation/regional data
from ECG SCADA.

## Data Quality Grading

| Dim | Grade | Why |
|-----|-------|-----|
| Demand coverage | A | 96.4% days, gap-filled |
| Temperature | A | 100% covered (Open-Meteo backfill) |
| Holidays | B+ | 126 marked, need public holiday calendar validation |
| Generation mix | F | Not tracked |
| Substation loads | F | Not tracked |
| Outage schedules | F | Not tracked |
| Rainfall/hydro | F | Not tracked |
| Solar irradiance | F | Not tracked |
| Economic indicators | D | GDP growth rate only (not at forecast granularity) |

## Next Steps (Data-Dependent)

1. Engage ECG SCADA team for substation-level data access
2. Engage GRIDCo for generation dispatch logs
3. Engage VRA for reservoir levels and hydro scheduling
4. Integrate Ghana Met Agency rainfall forecasts (available on request)
5. Build data pipeline for the above (4-8 weeks per data source)
6. Once zone-level data exists: implement MinT reconciliation (1-2 weeks)
7. Once generation data exists: add supply-constrained forecasting for hydro (2-4 weeks)
