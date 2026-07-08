# GridCo Load Forecasting System — Operator Runbook

## System Overview

Two forecasting engines run in parallel:

| Engine | Algorithm | Strength | Uncertainty |
|--------|-----------|----------|-------------|
| **DLinear + TIDE** | 6-fold DLinear ensemble + EMA bias correction | Best D+1 accuracy (75.9 MW MAE) | Per-hour P10/P90 from ensemble std + TIDE error std |
| **WT + DOW** | Weighted Trend + Day-of-Week profiles | Simpler, longer-horizon stability | Not yet implemented |

## Quick Start

```bash
docker compose up --build
```

This starts 4 services:
- `api` (FastAPI, port 8000)
- `db` (TimescaleDB, port 5432)
- `redis` (Redis 7, port 6379)
- `frontend` (Next.js, port 3000)

**First startup** may take 1-2 minutes (DB migrations, model loading).

## Health Check

```bash
# General health
curl http://localhost:8000/health

# DLinear engine health (checkpoints, MAE, TIDE state)
curl http://localhost:8000/api/v1/models/metrics

# Data freshness
curl http://localhost:8000/api/v1/forecast/baseline/freshness
```

Healthy response: `{"status": "healthy"}`

## Key API Endpoints (port 8000)

### DLinear + TIDE (Primary Engine)

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/forecast/dispatch/tomorrow` | 24h forecast with P10/P90 uncertainty |
| `GET /api/v1/forecast/dispatch/7day` | 7-day hourly forecast |
| `GET /api/v1/forecast/dispatch/30day` | 30-day daily aggregates |
| `GET /api/v1/forecast/dispatch/90day` | 90-day weekly aggregates |
| `GET /api/v1/forecast/dispatch/compare?date=YYYY-MM-DD` | Compare DLinear vs baseline |

All dispatch endpoints return `p10_mw` and `p90_mw` for uncertainty.

### WT + DOW (Baseline, Reference)

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/forecast/baseline/tomorrow` | 24h baseline forecast |
| `GET /api/v1/forecast/baseline/7day` | 7-day baseline |
| `GET /api/v1/forecast/baseline/freshness` | Data freshness info |

### Scheduling & Operations

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/schedule/upload` | Upload dispatch Excel |
| `POST /api/v1/schedule/{id}/auto-fill-forecast` | Fill schedule from forecast |
| `GET /api/v1/schedule/latest` | Latest confirmed schedule |
| `GET /api/v1/alerts/` | Active system alerts |

### Data

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/data/upload` | Upload SCADA CSV |
| `GET /api/v1/data/latest` | Latest validated data |

## Response Format (24h Forecast)

```json
{
  "forecast_date": "2026-06-09",
  "forecast_mw": [1850.3, 1820.1, ...],
  "p10_mw": [1792.5, 1765.3, ...],
  "p90_mw": [1908.1, 1874.9, ...],
  "uncertainty_mw": [45.2, 42.8, ...],
  "engine": "dlinear_tide",
  "inference_ms": 45.2
}
```

## Alerts

The system generates alerts for:
- **Capacity margin breach** (< 7% margin → warning, < 3% → critical)
- **DLinear engine elevated MAE** (> 150 MW rolling 24h)
- **DLinear engine offline** (not fitted, fallback active)

Alerts available at `GET /api/v1/alerts/`.

## Model Retraining

### DLinear + TIDE (every 6 months or when MAE degrades >10%)

```bash
cd Backend
python tools/retrain_dlinear.py
```

This retrains all 6 folds on the full dataset (2018-present). Checkpoints saved to `models/dlinear/`.

### WT + DOW (auto-retrains when >30 days old)

The baseline engine auto-retrains from DB history when its model file is >30 days old.
Manual retrain:

```bash
cd Backend
python tools/train_weighted_trend.py
```

## TIDE Corrector

TIDE (Temporal Integration of Drift Errors) is the online bias corrector:
- Tracks per-hour-of-day forecast error as an EMA (α=0.3)
- Computes per-hour error standard deviation for uncertainty
- Persists state to `models/dlinear/tide_state.db` (SQLite)
- Automatically primes from the last 48h of history on startup

To reset TIDE bias (e.g., after retraining):
```bash
curl -X POST http://localhost:8000/api/v1/forecast/dispatch/refresh
# OR directly:
python -c "
from app.ml.dlinear_engine import DLinearEngine
engine = DLinearEngine()
engine.reset_bias()
"
```

## Incident Response

### DLinear engine returns fallback forecast
1. Check engine health: `GET /api/v1/models/metrics`
2. Check logs: `docker compose logs api`
3. If checkpoints missing: run `python tools/retrain_dlinear.py`
4. Restart API: `docker compose restart api`

### Forecast cache needs clearing
```bash
curl -X POST http://localhost:8000/api/v1/forecast/dispatch/refresh
```

### Database issues
```bash
# Check DB connection
docker compose exec db pg_isready -U postgres

# Backup
docker compose exec db pg_dump -U postgres loadforecast > backup_$(date +%Y%m%d).sql

# Restore
cat backup.sql | docker compose exec -T db psql -U postgres loadforecast
```

### Complete restart
```bash
docker compose down
docker compose up --build -d
docker compose logs -f api
```

## Configuration

Key environment variables (in `.env` or docker-compose):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite+aiosqlite:///./loadforecast.db` | Database connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | CORS origins (comma-separated) |
| `SECRET_KEY` | `dev-secret-key...` | JWT signing key (CHANGE IN PRODUCTION) |
| `DEBUG` | `True` | Enable debug mode |

## File Locations (inside container)

| Path | Purpose |
|------|---------|
| `/app/app/` | FastAPI application code |
| `/app/models/dlinear/` | DLinear checkpoints + TIDE state |
| `/app/models/weighted_trend_engine.joblib` | WT+DOW model |
| `/app/data/` | Data files |
| `/app/migrations/` | Alembic DB migrations |

## Maintenance

- **DLinear retrain**: Every 6 months or when rolling MAE degrades >10%
- **WT+DOW retrain**: Auto-retrains when >30 days stale (from DB history)
- **DB backup**: Weekly pg_dump to external storage
- **Log rotation**: Docker logging driver handles this
- **Monitor**: Alert thresholds at 150 MW MAE, 7% capacity margin
