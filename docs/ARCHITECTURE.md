# System Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Server                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   API Layer (v1)                       │  │
│  │  /forecast/*    /schedule/*   /explain/*               │  │
│  │  /alerts/*      /metrics/*    /data/*                  │  │
│  └──────────────────────┬────────────────────────────────┘  │
│                         │                                    │
│  ┌──────────────────────▼────────────────────────────────┐  │
│  │                  Service Layer                         │  │
│  │  DispatchForecastService   ForecastService             │  │
│  │  ScheduleService           MetricsService              │  │
│  └──────────────────────┬────────────────────────────────┘  │
│                         │                                    │
│  ┌──────────────────────▼────────────────────────────────┐  │
│  │                  ML Engine Layer                       │  │
│  │  ┌──────────────┐  ┌──────────────┐                   │  │
│  │  │ DLinear+H10  │  │  DecomEngine │  (interpret.)     │  │
│  │  │ (production) │  │  (explainer) │                   │  │
│  │  └──────┬───────┘  └──────────────┘                   │  │
│  │         │ circuit breaker                             │  │
│  │  ┌──────▼───────┐                                     │  │
│  │  │StatFallback  │                                     │  │
│  │  └──────────────┘                                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                         │                                    │
│  ┌──────────────────────▼────────────────────────────────┐  │
│  │                   Data Layer                           │  │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────────────┐   │  │
│  │  │ SQLite   │  │  H10       │  │  Forecast         │   │  │
│  │  │ (main DB)│  │  state.db  │  │  cache            │   │  │
│  │  └──────────┘  └───────────┘  └──────────────────┘   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

```
ECG CSV (2018-2026) ──► SQLite (ecg_historical_demand)
                              │
                    ┌─────────▼─────────┐
                    │   _fetch_history   │
                    │   (200h window)    │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  DLinearEngine     │
                    │  .predict()        │
                    │   ├─ Normalize     │
                    │   ├─ 6-fold ens.   │
                    │   ├─ H10 correct   │
                    │   └─ Denormalize   │
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
       ┌──────▼──────┐ ┌─────▼─────┐ ┌───────▼──────┐
       │  Forecast   │ │  H10      │ │  Metrics     │
       │  Cache      │ │  update   │ │  Service     │
       └─────────────┘ └───────────┘ └──────────────┘
```

## Engine Selection

```
DLinearEngine.predict()
    │
    ├── is_fitted? ─── No ──► StatisticalFallback
    │
    ├── predict() succeeds? ── No ──► StatisticalFallback
    │
    └── Return DLinear+H10 forecast
```
