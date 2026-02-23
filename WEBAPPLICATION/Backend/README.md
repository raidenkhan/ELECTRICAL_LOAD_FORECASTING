# Load Forecasting Backend API

Production-ready FastAPI backend for electrical load forecasting with STLF (Short-Term Load Forecasting) and LTLF (Long-Term Load Forecasting) capabilities.

## Features

- **FastAPI** - High-performance async REST API
- **TimescaleDB** - Time-series optimized PostgreSQL database
- **Redis** - Caching and task queue
- **Docker Compose** - Easy deployment and development
- **ML Integration** - Autoformer, LightGBM, and ensemble models
- **Explainability** - SHAP-based model explanations

## Quick Start

### Local Development (with venv)

1. **Create and activate virtual environment:**
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your settings
```

4. **Run the API:**
```bash
uvicorn app.main:app --reload
```

5. **Access the API:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

### Docker Deployment

1. **Start all services:**
```bash
docker-compose up -d
```

2. **View logs:**
```bash
docker-compose logs -f api
```

3. **Stop services:**
```bash
docker-compose down
```

## API Endpoints

### Core Endpoints
- `GET /` - API information
- `GET /health` - Health check

### Data Management (Stage 2)
- `POST /api/v1/data/upload` - Upload SCADA CSV data
- `GET /api/v1/data/validation/{upload_id}` - Get validation report

### Forecasting (Stage 4 & 5)
- `POST /api/v1/forecast/stlf` - Short-term load forecast (0-24h)
- `POST /api/v1/forecast/ltlf` - Long-term load forecast (1-30d)
- `GET /api/v1/forecast/{forecast_id}` - Retrieve forecast

### Explainability (Stage 6)
- `GET /api/v1/explain/{forecast_id}` - SHAP values for forecast
- `GET /api/v1/models/status` - Model performance metrics

## Project Structure

```
Backend/
├── app/
│   ├── api/v1/          # API endpoints
│   ├── core/            # Configuration & logging
│   ├── db/              # Database models & session
│   ├── ml/              # ML model handlers
│   ├── schemas/         # Pydantic schemas
│   ├── services/        # Business logic
│   └── main.py          # FastAPI app
├── models/              # Trained ML models
├── data/                # Data storage
├── docker-compose.yml   # Container orchestration
├── Dockerfile           # API container
└── requirements.txt     # Python dependencies
```

## Development Stages

- ✅ **Stage 1**: System Foundation (Docker, FastAPI, DB)
- 🔄 **Stage 2**: Data Ingestion & Validation
- ⏳ **Stage 3**: Feature Engineering Service
- ⏳ **Stage 4**: Short-Term Forecast Engine (STLF)
- ⏳ **Stage 5**: Long-Term Forecast Engine (LTLF)
- ⏳ **Stage 6**: Explainability & Monitoring

## Technology Stack

- **FastAPI** 0.109.0 - Web framework
- **SQLAlchemy** 2.0 - Async ORM
- **TimescaleDB** - Time-series database
- **Redis** - Caching layer
- **LightGBM** - Gradient boosting models
- **PyTorch** - Deep learning models
- **SHAP** - Model explainability

## License

MIT
