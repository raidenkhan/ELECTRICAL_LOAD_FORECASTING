# Dispatch Schedule Backend Restructuring & Rollout Plan

This document serves as the architectural blueprint for restructuring the Load Forecasting application to handle complex, entity-based dispatch scheduling (24-hour periods) replacing the old flat SCADA architecture.

---

## The New System Architecture

The following diagram illustrates the flow of data from physical uploads through the extraction, storage, forecasting, and aggregation layers, culminating in the frontend UI.

```mermaid
flowchart TD
    %% Inputs
    Excel[7-Sheet Excel Data] --> UI_Upload[Upload Interface]
    Word[Word Documents] --> UI_Upload
    Weather_API[External Weather API] --> API_Ingest
    
    %% Ingestion Pipeline
    UI_Upload --> API_Ingest[Ingestion API]
    API_Ingest --> Parser_Excel[Pandas Extractor]
    API_Ingest --> Parser_Fallback[Gemini API Fallback]
    
    %% Database
    Parser_Excel --> DB[(Relational Database)]
    Parser_Fallback --> DB
    
    subgraph Database Schema
        DB --> Schedule[DailyDispatchSchedule]
        Schedule --> HDemand[HourlyDemand\n(ECG, VALCO, Mines...)]
        Schedule --> HSupply[HourlySupply\n(VRA, CENIT...)]
    end

    %% Forecasting Engine
    HDemand -.-> ECG_Filter[Filter: ECG Only]
    ECG_Filter --> DecomEngine[DecomEngine & SimDay\n(24h refactored)]
    Weather_API -.-> DecomEngine
    DecomEngine --> Forecasted_ECG[Forecasted ECG]

    HDemand -.-> Static_Filter[Filter: Mines, VALCO, etc.]
    Static_Filter --> Constant_Load[Constant/Deterministic Loads]

    %% Aggregation
    Forecasted_ECG --> Aggregator[Aggregation Engine]
    Constant_Load --> Aggregator
    HSupply --> Aggregator

    Aggregator --> Final_Schedule[Total Ghana Demand & Reserve]
    
    %% Frontend
    Final_Schedule --> UI_View[Dispatch Schedule UI]
```

---

## 1. Database Redesign (`app/db/models/schedule.py`)
We will introduce new relational models using SQLAlchemy:

*   **`DailyDispatchSchedule`**: The parent record for a specific day (`date`, `status: pending/confirmed`).
*   **`HourlyDemand`**: Links to the Schedule. 
    *   `hour` (1 to 24)
    *   `entity_name` (ECG, NEDCo, VALCO, Mines, Export)
    *   `demand_mw` (Float)
    *   `is_actual` (Boolean)
*   **`HourlySupply`**: Links to the Schedule.
    *   `hour` (1 to 24)
    *   `plant_name` (VRA, SEAP, CENIT, etc.)
    *   `supply_mw` (Float)

## 2. The Extraction & Ingestion Pipeline (`app/services/extraction_service.py`)
*   **Standard Parsing (Excel)**: A `pandas` extractor loops through the sheets, grabs the 24 columns, and maps them to the database records.
*   **Heuristic Fallback (Word/Malformed)**: Uses the Gemini 1.5 Flash API as a fallback parser to extract tables from tricky documents into a strict JSON schema.

## 3. Forecasting Architecture Refactor (The Fixes)
We are retaining the **DecomEngine** and the **SimDayEngine** (GRIDCo Bases), but refactoring them to fix critical flaws exposed by the new data structure:

*   **Fix 1: Granularity**: Refactor both engines from `steps_per_day = 96` (15-min intervals) to `steps_per_day = 24` to match the hourly dispatch schedule.
*   **Fix 2: The Component Shift**: Models will **only** train on and forecast **ECG Demand**, isolating the weather-dependent load from the flat static loads (Mines, VALCO). This prevents the temperature multiplier from erroneously inflating the constant mine loads.
*   **Fix 3: Dynamic Growth**: Replace hardcoded growth parameters (`ANNUAL_GROWTH = 0.08`, `cap_growth = 1.12`) with dynamic database queries calculating actual YoY growth as learned parameters.
*   **Fix 4: External Weather Integration**: Since the new Excel sheets do not contain weather data, we must integrate an **External Weather API** to fetch the 24-hour temperature forecast required by the models to accurately predict the ECG load.
*   **Fix 5: The Retraining Schedule**: We will implement a two-tiered MLOps retraining strategy to keep the models sharp without overfitting:
    *   **The Midnight Tuner (Daily Incremental Update):** Every midnight, a background job compares yesterday's forecast against the *actual* finalized dispatch data. It calculates the error margin and lightly adjusts short-term weights (like recent temperature sensitivity). This is fast and computationally cheap.
    *   **Deep Retrain (Weekly/Monthly):** A full historical retrain using the last several years of locked data. This updates the heavy, long-term learned parameters (like the `ANNUAL_GROWTH` cap and deep seasonal baselines) to ensure macro-economic shifts are captured.

## 4. The Aggregation Engine (`app/services/aggregator_service.py`)
Computes the final schedule:
*   **Total Ghana Demand Forecast** = `Forecasted(ECG) + Actual(NEDCo + VALCO + Mines + Export)`.
*   **Total Generation** = `SUM(All Supply Plants)`.
*   **Reserve** = `Total Generation - Total Demand`.

## 5. Frontend UI
1.  **Upload Interface**: Dropzone mapping to `/upload`.
2.  **Review Grid (Fail-Safe)**: Editable table to fix backend/LLM extraction errors manually before confirming.
3.  **Master Schedule View**: A rich data grid replacing the Excel sheet with sticky headers, conditional formatting, and real-time totals.
4.  **View Toggle**: The Master Schedule View must include a tab switcher allowing operators to toggle between the **Excel-style grid** (rows = entities, columns = hours) and a **Graph view** (line chart of ECG demand, supply stack, reserve margin over 24h). Both views share the same underlying data.
5.  **Reference Mockup**: See `RETHINK_PICX/dispatch_schedule_mockup.html` for the visual design reference of the Schedule Grid, summary cards, color-coded peak cells, and audit trail drawer. Open in any browser — this is the target UI for the AI builder.

## 6. Long-Term Storage & Export
*   **Data Archiving**: Once a Daily Dispatch Schedule is confirmed and executed, it is locked and stored in the TimescaleDB. This creates an immutable historical record that the `SimDayEngine` can look back upon for future training without data degradation.
*   **Export Engine**: We will implement an export module (using libraries like `openpyxl` or `WeasyPrint` for PDF) that allows operators to download the final schedule. 
*   **Better-Than-Image Formatting**: The exported file will perfectly replicate the layout required by their partners, but heavily enhanced with branded headers, clean typography, embedded graphs for the load curve, and color-coded heatmap highlights for peak hours.

---

## 7. Audit Trail & Change Management

### 7.1 Rationale
Dispatch schedules are operational decisions with financial and grid-stability consequences. Every modification — whether correcting a parser error, adjusting a forecast, or responding to a plant trip — must be recorded immutably. Without an audit trail, operators cannot reconstruct *what* happened, *who* approved the change, or *why* the final schedule differs from the original upload.

### 7.2 Database Model (`app/db/models/audit.py`)
Single append-only table:

```
ScheduleAuditLog
├── id: UUID (PK, auto)
├── schedule_id: UUID → FK(DailyDispatchSchedule.id)
├── action: Enum("upload","auto_correct","manual_edit","confirm","revise","archive","export")
├── actor_id: UUID → FK(User.id)
├── actor_role: Enum("operator","planner","admin","system")
├── field_path: str | NULL               # e.g. "hourly_demand.14.ECG"
├── old_value: JSON | NULL
├── new_value: JSON | NULL
├── change_reason: str | NULL            # Mandatory for manual_edit & revise
├── source: Enum("ui","api","system","llm_parser")
├── row_version: int                     # Optimistic lock counter
├── checksum_prev: str                   # SHA256 of previous log entry (hash chain)
├── created_at: TIMESTAMPTZ (default now(), immutable)
```

**Rules:**
- **Append-only.** No UPDATE, no DELETE. Not even for admins.
- `checksum_prev` chains entries per schedule — tampering breaks the chain.
- `row_version` enables optimistic concurrency: second save on same schedule returns 409.

### 7.3 State Machine

```
[Upload] ──action:upload──→ [Draft]
                   │
            auto_correct ──→ stays Draft
                   │
            manual_edit ──→ stays Draft
                   │
            confirm ──→ [Confirmed] ──revise──→ [Draft]
                                        │
                                  archive ──→ [Archived]
                                        │
                                  export ──→ stays Confirmed
```

| Transition | Allowed Roles | Reason Required |
|---|---|---|
| upload → Draft | Admin, System | No |
| auto_correct | System | No |
| manual_edit | Operator, Planner | Yes |
| Draft → Confirmed | Operator, Admin | Yes ("Ready for dispatch") |
| Confirmed → Draft (revise) | Admin | Yes ("VRA unit 3 trip") |
| Confirmed → Archived | System (automatic) | No |
| export | Operator, Planner, Admin | No |

### 7.4 Backend Implementation (`app/services/audit_service.py`)

```
AuditService
├── log()            → Writes entry, updates hash chain
├── get_history()    → Ordered timeline for a schedule
├── verify_chain()   → Recomputes hashes, returns false if tampered
└── assert_version() → Raises 409 if optimistic lock fails
```

**New endpoint:** `GET /api/v1/schedule/{id}/history` returns full audit trail.

### 7.5 Frontend Audit Trail UI

Located as a "History" drawer tab within the Master Schedule View:

```
┌─────────────────────────────────────────────────────────────┐
│  ⚡ Audit Trail — Dispatch Schedule 2026-05-22              │
├─────────────────────────────────────────────────────────────┤
│  🔵 Upload        08:15  admin@gridco.gh   "Morning upload" │
│  🟡 Auto-correct  08:17  system             "Corrected 6    │
│                                             cells (NITS)"    │
│  🔵 Manual Edit   08:23  operator@gridco   "Hour 10 Trojan  │
│                                             I: 0→15 MW"     │
│  🟢 Confirm       08:30  operator@gridco   "Ready for       │
│                                             dispatch"        │
│  🔴 Revise        14:02  admin@gridco      "VRA trip →      │
│                                             -50MW h12-18"   │
│  🟢 Confirm       14:05  admin@gridco      "Revised OK"     │
│  ⬜ Export         14:10  planner@gridco   "PDF for partners"│
├─────────────────────────────────────────────────────────────┤
│  🔗 Chain Integrity: ✅ Verified (7 entries, all hashes OK) │
└─────────────────────────────────────────────────────────────┘
```

### 7.6 Column-Change Resilience

Since the parser currently assumes fixed Excel column indices:

1. **Column fingerprinting** — On first upload, compute a schema fingerprint (column names, positions, types) and store in `FileFormatRegistry`.
2. **Drift detection** — If fingerprint doesn't match any registered format, **flag for review** instead of silently parsing wrong. Operator sees: *"Unrecognised layout. Map columns or register as new template."*
3. **Template learning** — Once confirmed, new fingerprint is registered; future matching files parse automatically.
4. **Gemini fallback** — Catch-all for completely novel formats.

### 7.7 Pre-Deployment Stress Tests

| Test | What to Verify |
|---|---|
| 100 concurrent uploads | No DB deadlock or data loss |
| LLM parser × 50 malformed docs | Cell-level accuracy ≥ 90% |
| Column-name drift scenario | Fingerprint mismatch → blocks parse, flags review |
| Conflicting edits (2 operators) | Second save → 409, forces refresh |
| Audit chain tampering | `verify_chain()` detects broken hash |
| Midnight auto-archive (30 schedules) | All ≤ yesterday archived within 60s |
| Weather API outage | Falls back to persistence gracefully |
| 6-month audit history query | Timeline renders in < 2s |

---

## 8. Deployment Note: Docker vs Direct Python

Two options for running the backend:

**Option A — Docker Engine (Recommended for production-like isolation)**
- Use existing `docker-compose.yml` in `Backend/`
- Spins up: FastAPI container + TimescaleDB + optional Redis
- Requires Docker Desktop on your PC
- Clean isolation, matches deployment target

**Option B — Direct Python (Lighter, faster iteration)**
- Already working: `venv/` has all dependencies
- Run with: `uvicorn app.main:app --reload`
- Uses SQLite by default (`.env` points to `loadforecast.db`)
- No Docker overhead, good for dev/cheap VPS

**Recommendation:** Develop with Option B, deploy with Option A. The codebase is identical either way — switching only changes the `DATABASE_URL` in `.env`.

---

## 9. Phased Build Plan (Test-First, Incremental)

Each phase is self-contained, testable, and gates the next phase. **Do not start Phase N+1 until Phase N passes its gate test.**

### Phase 0: Database Foundation + Raw Upload
**Goal:** Upload the sample Excel → data lands in the database correctly.

| Step | What to Build | Test |
|------|---------------|------|
| 0.1 | `DailyDispatchSchedule`, `HourlyDemand`, `HourlySupply` SQLAlchemy models | Migration runs, tables exist |
| 0.2 | Alembic migration | `alembic upgrade head` succeeds |
| 0.3 | `POST /api/v1/schedule/upload` — accepts Excel, uses pandas to extract 24 columns per row, stores in DB | Upload `RETHINK_PICX/ECG Daily Demand Data Sheet for Dispatch Day May 22, 2026.xlsx` → all 24 hours × 5 entity rows stored correctly |
| 0.4 | `GET /api/v1/schedule/{id}` — returns the parsed schedule as JSON | Response matches Excel values exactly |

**🧪 Gate Test (manual):** Upload the sample Excel. Run `GET /schedule/1` and compare every cell against the Excel row-by-row. All 120 cells (24h × 5 rows) must match within 0.5 MW. **If any cell mismatches, fix the parser before Phase 1.**

---

### Phase 1: Review Grid (Operator Trust)
**Goal:** Operator can see the parsed data, spot errors, and fix them before confirming.

| Step | What to Build | Test |
|------|---------------|------|
| 1.1 | Frontend: Add `?view=dispatch` route rendering a read-only version of `dispatch_schedule_mockup.html` (schedule grid + summary cards) | Grid renders all 24 columns and entity rows from API |
| 1.2 | Frontend: Make cells editable on click (inline edit, enter to save) | Click cell → edit value → enter → `PATCH` request sent |
| 1.3 | Backend: `PATCH /api/v1/schedule/{id}/cell` — update single cell value | Cell updates in DB, response confirms new value |
| 1.4 | Backend: `POST /api/v1/schedule/{id}/confirm` — flip status to `confirmed` | Status changes, further edits blocked (unless role=admin) |
| 1.5 | Frontend: Confirm button + draft/confirmed badge | Badge updates after confirm |

**🧪 Gate Test (manual):** Upload sample Excel. Spot-check hour 10 ECG demand (should be 1,808 MW). Edit it to 1,800. Confirm. Re-fetch — value is 1,800, status is `confirmed`. **Operator can trust the data before confirming.**

---

### Phase 2: DecomEngine Retrain (ECG-Only, Hourly)
**Goal:** The forecasting engine produces ECG-only hourly predictions. This is the hardest phase.

| Step | What to Build | Test |
|------|---------------|------|
| 2.1 | Collect 12+ months of ECG hourly demand data (historical uploads or GRIDCo archive) | Dataset exists with no gaps > 2h |
| 2.2 | Copy `app/ml/decom_engine.py` → `app/ml/decom_engine_hourly.py`. Refactor: `steps_per_day=24`, remove 15-min assumptions | Runs without error |
| 2.3 | Replace all temperature references: retrain temp multiplier on ECG data (not substation data) | Temperature coefficient is physically plausible (0.5–3%/°C) |
| 2.4 | Replace hardcoded growth params (`ANNUAL_GROWTH=0.08`) with DB queries of actual YoY growth | Growth rate computed from historical ECG data |
| 2.5 | Train the refactored engine on the ECG dataset | Training loss decreases monotonically |
| 2.6 | Integrate external weather API forecasts (Open-Meteo, 24h ahead) as model input | Forecast uses live weather, not persistence |
| 2.7 | `POST /api/v1/forecast/dispatch` — returns 24h ECG forecast | Response: array of 24 `{hour, forecast_mw}` values |

**🧪 Gate Test (automated):** Hold out last 7 days of ECG data. Run forecast for each day. Compute MAE. **MAE must be < 150 MW** (≈7% of mean ECG load ≈ 2,000 MW). If MAE > 150 MW, the retrain is not ready — iterate on features or hyperparameters.

---

### Phase 3: Aggregation Engine (Full Schedule)
**Goal:** Combine forecasted ECG + static loads + supply to produce the complete dispatch schedule.

| Step | What to Build | Test |
|------|---------------|------|
| 3.1 | `app/services/aggregator_service.py`: `aggregate(schedule_id)` — loads ECG forecast, adds NEDCo/VALCO/Mines/Export as static or DB-stored actuals | Total = ECG(forecast) + NEDCo(actual) + VALCO(constant) + Mines(constant) + Export(constant) |
| 3.2 | Add supply-side aggregation: sum all generation plants → total generation column | Generation total is correct per hour |
| 3.3 | Compute reserve = total generation − total demand per hour | Reserve ≥ 0 for all hours (or flagged if negative) |
| 3.4 | Store aggregated result in a new `AggregatedSchedule` table or computed view | Query returns 24 rows with demand/gen/reserve |

**🧪 Gate Test:** Take the sample Excel. Compare the "Scheduled Demand from NITS" row (Row 54 in the Excel). **The aggregator's output must match this row within 2%.** If it doesn't, the ECG forecast or static load assumptions are wrong.

---

### Phase 4: Master Schedule View + Audit Trail
**Goal:** Full UI matching the mockup + every change is recorded.

| Step | What to Build | Test |
|------|---------------|------|
| 4.1 | Frontend: Full Master Schedule View matching `dispatch_schedule_mockup.html` (summary cards, color-coded grid, demand/supply/combined toggle) | Visual match to mockup |
| 4.2 | Frontend: Audit trail drawer with timeline | Entries render with actor, action, timestamp, diff |
| 4.3 | Backend: `ScheduleAuditLog` model + `audit_service.py` | `verify_chain()` returns true for clean chain |
| 4.4 | Integrate audit logging into existing endpoints: upload, edit, confirm, revise, export | Every action creates an audit entry |
| 4.5 | Backend: `POST /api/v1/schedule/{id}/revise` — admin-only, reverts to draft | Status goes confirmed → draft, audit entry created |
| 4.6 | Frontend: "Edit Mode" toggle, confirm/revise buttons with reason dialog | Reason required for manual edits and revisions |
| 4.7 | End-to-end: upload → auto-correct → edit → confirm → revise → re-confirm | Full lifecycle traces in audit trail |

**🧪 Gate Test:** Walk through the full lifecycle: upload sample Excel → auto-correct fires → edit one cell → confirm → admin revises → edit supply → re-confirm. **Audit trail shows all 6+ entries with correct timestamps, actors, and diffs. Chain integrity ✅ verified.**

---

### Phase 5: Column Resilience + Stress Tests
**Goal:** The system survives real-world Excel variations and load.

| Step | What to Build | Test |
|------|---------------|------|
| 5.1 | `FileFormatRegistry` model + column fingerprinting on upload | Same-format files bypass; different-format files get flagged |
| 5.2 | Frontend: Column mismatch warning dialog with "Map columns" / "Register as new template" options | Dialog appears on format mismatch |
| 5.3 | Stress test: 100 concurrent uploads | No deadlocks, no data loss |
| 5.4 | Stress test: 50 malformed Word docs through Gemini fallback | Cell accuracy ≥ 90% |
| 5.5 | Stress test: 2 operators edit same schedule simultaneously | Second save → 409 error, refresh prompt |

**🧪 Gate Test:** Intentionally rename a column in the Excel. Upload. **System must reject with "Unrecognised layout" — not silently produce wrong data.**

---

### Phase 6: Export + Archive + Production Cutover
**Goal:** Operators can export the final schedule and the system auto-archives daily.

| Step | What to Build | Test |
|------|---------------|------|
| 6.1 | Export: PDF generation matching GRIDCo branded layout | Exported PDF matches partner format |
| 6.2 | Export: XLSX generation with load curve graph embedded | openpyxl output opens correctly |
| 6.3 | Archive: Midnight cron job archives confirmed schedules (status → archived) | Schedules with date ≤ yesterday auto-archive |
| 6.4 | Docker Compose for production: FastAPI + PostgreSQL + Redis | `docker-compose up` starts all services |
| 6.5 | Cutover: Replace old SCADA system with new dispatch-only backend | Old endpoints still respond; new dispatch endpoints are the primary UI |

**🧪 Gate Test:** Morning of go-live: operator uploads Excel, reviews grid, confirms. System forecasts ECG, aggregates full schedule, displays master view. Operator exports PDF. At midnight, schedule auto-archives. **This is the production workflow.**

---

## Dependency Graph

```
Phase 0 (DB + Upload)
    │
    ▼
Phase 1 (Review Grid) ──────────────────────────┐
    │                                             │
    ▼                                             ▼
Phase 2 (DecomEngine Retrain)          Phase 4 (Audit Trail)
    │                                             │
    ▼                                             │
Phase 3 (Aggregation) ◄──────────────────────────┘
    │
    ▼
Phase 5 (Column Resilience + Stress)
    │
    ▼
Phase 6 (Export + Archive + Cutover)
```

**Note:** Phase 4 (Audit Trail) can start in parallel with Phase 2 since the `ScheduleAuditLog` model has no dependency on the forecasting engine. However, Phase 1 (Review Grid) is a hard prerequisite for Phase 4's frontend audit timeline — operators can't audit changes they can't see.

---

## Quick Start

```bash
# Phase 0 — get the DB models up
cd Backend
venv\Scripts\python -m app.db.models.schedule  # verify models import
alembic revision --autogenerate -m "add dispatch schedule models"
alembic upgrade head

# Test: upload the sample Excel
venv\Scripts\python tools/test_upload.py ../RETHINK_PICX/ECG\ Daily\ Demand\ Data\ Sheet\ for\ Dispatch\ Day\ May\ 22,\ 2026.xlsx

# Phase 1 — start the frontend
cd ../frontend
npm run dev
# Open http://localhost:3000/?view=dispatch
```

Build one phase at a time. Test the gate before moving on. If a gate fails, fix that phase before touching the next one.
