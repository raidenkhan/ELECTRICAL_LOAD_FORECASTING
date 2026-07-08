# Multi-Substation Upgrade Design

**Date:** 2026-04-23  
**Status:** Draft (Pending Review)

## 1. Executive Summary
The current system is hardcoded for the Achimota-82 substation. This design upgrades the platform to a multi-tenant architecture where any GRIDCo substation can create an account, map their unique SCADA CSV format to a system-standard schema, and receive localized load forecasts and Digital Twin simulations.

## 2. Goals & Constraints
- **Goal:** Allow "Plug-and-Play" onboarding for new substations.
- **Goal:** Maintain ML model accuracy even when auxiliary data (voltage, temp) is missing.
- **Constraint:** Use account-based segregation (Operator A at Station X cannot see Station Y).
- **Constraint:** Minimize the manual effort required by station operators during upload.

## 3. Architecture Changes

### 3.1 Data Model (Database)
We will transition to an **Entity-Centric** relational model:

- **Substations Table:**
  - `id`: UUID (Primary Key)
  - `name`: String (e.g., "ACHIMOTA-82")
  - `region`: String (e.g., "Greater Accra")
  - `coordinates`: Point (Lat/Long for weather integration)
  - `schema_mapping`: JSONB (Stores the CSV column map)
  
- **Users Table (Updated):**
  - `substation_id`: UUID (Foreign Key to Substations)
  
- **ValidatedData Table (Updated):**
  - `substation_id`: UUID (Foreign Key to Substations)
  - *Standardized Columns:* The table remains fixed, but ingestion logic maps incoming data to these fields.

### 3.2 Ingestion Engine (Backend)
- **Mapping Middleware:** A service that intercepts CSV uploads and uses the `schema_mapping` from the user's substation to rename and cast columns.
- **Validation Tiers:**
  - **Tier 1 (Critical):** Check for `TIMESTAMP` and `TOTAL_MW`. Fail if missing.
  - **Tier 2 (Physical):** Check for `VOLTAGE`, `FREQUENCY`. If missing, log a warning and use fallback constants.

## 4. Component Designs

### 4.1 Frontend: Mapping Wizard
- **Trigger:** Launches when a user uploads a CSV with headers that don't match the current mapping.
- **Interaction:** A drag-and-drop interface where the user drags "Detected CSV Headers" onto "Required System Fields."
- **Persistence:** Saved to the `Substations.schema_mapping` field upon successful validation.

### 4.2 API Endpoints
- `POST /api/v1/substations/`: Register a new station (Admin only).
- `POST /api/v1/data/upload`: Automatically uses the `current_user.substation_id` to process the file.
- `GET /api/v1/forecast/latest`: Returns results scoped to the user's station.

## 5. Error Handling & Fallbacks
- **Missing Aux Data:** If a station does not provide `Temperature`, the system will automatically pull it via a weather service using the station's coordinates.
- **Inconsistent Frequency:** If data is not in 15-minute intervals, the system will perform linear interpolation and alert the user.

## 6. Testing Strategy
- **Integration Tests:** Mock uploads for three different substations with varying CSV headers.
- **Security Tests:** Verify that User A cannot access `/latest` data for User B's substation.
- **Validation Tests:** Ensure Tier 1 failures correctly stop the pipeline.

## 7. Meeting Talking Points (GRIDCo)
- Standardized templates for regions.
- Minimum data resolution requirements.
- Identity sync (Asset IDs).
- Weather data ownership.
