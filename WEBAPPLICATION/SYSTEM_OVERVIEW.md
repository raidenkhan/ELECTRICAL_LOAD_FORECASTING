# GRIDCo Dispatch Scheduling System — Overview

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        OPERATOR (Browser)                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │  DispatchSchedule │  │  Control Room     │  │  System Overview│ │
│  │  (Schedule Grid)  │  │  (DigitalTwin)   │  │  Live Monitor   │ │
│  └────────┬─────────┘  └────────┬─────────┘  └───────┬────────┘ │
│           │                     │                      │          │
└───────────┼─────────────────────┼──────────────────────┼──────────┘
            │                     │                      │
            ▼                     ▼                      ▼
     ┌──────────────┐    ┌────────────────┐    ┌────────────────┐
     │  REST API    │    │  REST API      │    │  REST API      │
     │  /schedule/* │    │  /forecast/*   │    │  /data/*       │
     └──────┬───────┘    └───────┬────────┘    └───────┬────────┘
            │                    │                      │
            ▼                    ▼                      ▼
     ┌──────────────────────────────────────────────────────┐
     │                   FastAPI Backend                     │
     │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │
     │  │ScheduleService│  │DecomEngine   │  │WeatherService│  │
     │  │(Excel parse) │  │(ML forecast) │  │(Open-Meteo) │  │
     │  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘  │
     │         │                │                  │         │
     │         ▼                ▼                  ▼         │
     │  ┌──────────────────────────────────────────────────┐  │
     │  │              PostgreSQL Database                  │  │
     │  │  ┌─────────────┐ ┌─────────────┐ ┌───────────┐  │  │
     │  │  │daily_dispatch│ │hourly_demand│ │ecg_hist-  │  │  │
     │  │  │_schedules   │ │hourly_supply│ │orical_dmd │  │  │
     │  │  └─────────────┘ └─────────────┘ └───────────┘  │  │
     │  └──────────────────────────────────────────────────┘  │
     └──────────────────────────────────────────────────────┘
```

## End-to-End Flow

### 1. Morning Upload (Manual)
```
Operator opens DispatchSchedule page
  → clicks upload area
  → selects ECG Daily Demand Data Sheet.xlsx
  → system parses Excel:
      - 24 hours × 5 demand entities (ECG, NEDCo, VALCO, Mines, Export)
      - 24 hours × N supply plants (Trojan I, Meienergy, BXC Solar)
      - Computes NITS_Total = sum of all demand entities
  → stores as DRAFT schedule in DB
  → displays editable grid
```

### 2. Auto-fill Forecast (Optional)
```
Operator clicks "Auto-fill Forecast" button
  → POST /schedule/{id}/auto-fill-forecast
  → DispatchForecastService.forecast_for_date(schedule.date)
      → WeatherService: fetch 48h forecast from Open-Meteo
      → DecomEngineHourly.predict():
          - Trend component (Holt-Winters from historical data)
          × Seasonal component (hour-of-day × day-of-week)
          × Temperature effect (deseasonalized anomaly model)
          × Holiday adjustment
          × Growth multiplier (YoY from DB history)
          + Kalman bias correction
  → updates all 24 ECG cells with forecast values
  → marks as is_forecasted = True
  → operator can still manually edit cells
```

### 3. Manual Edits (As Needed)
```
Operator clicks any cell → inline editor opens
  → enters new MW value
  → PATCH /schedule/{id}/cell updates the row
  → only editable on DRAFT schedules
```

### 4. Confirm (Lock)
```
Operator clicks "Confirm Schedule"
  → POST /schedule/{id}/confirm
  → status changes to "confirmed"
  → cells become read-only
  → schedule is final for dispatch day
```

### 5. Revise (Unlock + Edit)
```
If changes needed after confirmation:
  → "Revise" button → POST /schedule/{id}/revise (requires reason)
  → status back to DRAFT
  → operator edits cells
  → re-confirm when ready
```

### 6. Control Room (Real-time View)
```
DigitalTwinDashboard loads on page open:
  → GET /forecast/dispatch/tomorrow (DecomEngine forecast)
  → GET /schedule/latest (most recent uploaded schedule)
  → displays:
      - KPI cards (peak, total, average, entity count)
      - Stacked/Grouped/Total demand chart (all entities)
      - Demand Mix breakdown (avg MW + share %)
      - Supply sources (plant averages)
      - Forecast factors (seasonal range, temp sensitivity, growth)
```

## Pages & Their Purpose

| Page | Route | Component | Purpose |
|------|-------|-----------|---------|
| **System Overview** | `/?view=overview&mode=analytics` | `OverviewDashboard` | High-level STLF forecast, accuracy metrics, load regime, operating probabilities, alerts |
| **Control Room** | `/?view=overview&mode=control-room` | `DigitalTwinDashboard` | 24-hour dispatch forecast with full entity breakdown, demand mix, supply sources, decomposition factors |
| **Live Monitor** | `/?view=live-monitor` | `LiveMonitor` | Real-time data tracking, system load chart, countdown to next update cycle |
| **Planner** | `/?view=planner` | `PlannerDashboard` | Strategic planning tools and scenario modeling |
| **Data Management** | `/?view=data-upload` | `DataManagement` | Historical data uploads, SCADA ingestion, data health |
| **Model Performance** | `/?view=model-performance` | `ModelPerformance` | Forecast model accuracy, error metrics, model diagnostics |
| **Explainability** | `/?view=explainability` | `ExplainabilityView` | Physical decomposition breakdown of forecast into trend, seasonal, temp, holiday components |
| **Dispatch Schedule** | `/?view=dispatch` | `DispatchSchedule` | Daily dispatch schedule grid — upload, edit, confirm, revise, auto-fill forecast, audit trail |
| **Settings** | `/?view=settings` | `Settings` | System configuration, SCADA timeout, user preferences |

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/schedule/upload` | Upload Excel → parse → store schedule |
| GET | `/schedule/latest` | Get most recently uploaded schedule |
| GET | `/schedule/{id}` | Get full schedule by ID |
| PATCH | `/schedule/{id}/cell` | Update single cell value |
| POST | `/schedule/{id}/confirm` | Lock schedule as confirmed |
| POST | `/schedule/{id}/revise` | Unlock schedule for editing (requires reason) |
| POST | `/schedule/{id}/auto-fill-forecast` | Fill ECG row from DecomEngine forecast |
| POST | `/forecast/dispatch` | Run dispatch forecast for a specific date |
| GET | `/forecast/dispatch/tomorrow` | Run dispatch forecast for tomorrow |

## Database Tables

| Table | Contents |
|-------|----------|
| `daily_dispatch_schedules` | Schedule metadata (date, status, source file, notes) |
| `hourly_demand` | 24h × entity demand values per schedule (ECG, NEDCo, VALCO, Mines, Export, NITS_Total) |
| `hourly_supply` | 24h × plant supply values per schedule (Trojan I, Meienergy, BXC Solar) |
| `ecg_historical_demand` | Historical ECG hourly demand for model training |
| `users` | System users (admin, operator) |

## Key Design Decisions

- **Hourly resolution** — all data is hourly (24 values per day), not 15-minute. Matches ECG dispatch sheet format.
- **Excel-driven** — operator uploads the official ECG Daily Demand Data Sheet; the system parses it exactly.
- **Forecast as enhancement, not replacement** — the DecomEngine forecast populates ECG cells optionally; the Excel values remain the source of truth.
- **Draft → Confirmed → Revise cycle** — strict state machine prevents accidental edits after dispatch day starts.
- **Audit trail** — every action (upload, auto-correct, manual edit, confirm, revise, export) is tracked with timestamp + actor.
