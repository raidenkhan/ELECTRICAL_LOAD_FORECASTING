# GRIDCo Dispatch Scheduling System — Operator Walkthrough

## Presentation Script for GRIDCo Operators

---

## Slide 0: Opening — Why This System Exists

**Say:**
> "Good morning. Today I'm going to walk you through the new GRIDCo Dispatch Scheduling System. This system replaces the old manual Excel-based process for managing Ghana's daily generation dispatch schedule.
>
> The problem we're solving is simple: until now, dispatch schedules were created by hand in spreadsheets — one operator copies data from the ECG Daily Demand Data Sheet, pastes into a template, calculates reserves in their head or on a calculator, and emails around an Excel file. There is no central record. There is no audit trail. If someone changes a number, nobody knows who, when, or why.
>
> This system gives us a single source of truth, cryptographic audit trails, optional ML-based forecast assistance, and real-time visibility — all in one web application."

---

## Slide 1: Login Page

**Say:**
> "You access the system from any browser on the GRIDCo network — just navigate to the URL. You'll see this login screen.
>
> On first use, an admin creates your account. You choose your role — Operator, Analyst, Planner, or Admin — and your region. For daily dispatch operations, you'll be set up as an Operator.
>
> The system authenticates you and manages your session. After you log in, you land in the Control Room."

**Show:** Login page with role and region selection.

---

## Slide 2: Control Room — The Overview Dashboard

**Say:**
> "This is the Control Room — your at-a-glance view of today's or tomorrow's dispatch picture.
>
> At the top, four KPI cards tell you the headline numbers: Peak Demand and what hour it occurs, Total Energy for the 24-hour window, Average Load across the day, and the total number of demand entities and supply sources we're tracking.
>
> On the left panel, you see the Demand Mix — each entity's average MW and percentage share of total demand — and the Supply Sources list showing average generation per plant. Below that, the Forecast Factors panel tells you what the ML engine is seeing: seasonal range, temperature sensitivity, and growth multiplier for today.
>
> The main chart has three viewing modes you can toggle across the top:
> - **Stacked Area** — shows each demand entity as a coloured layer stacking up to total demand. You can see how the mix changes hour by hour.
> - **Grouped Bars** — side-by-side comparison per entity per hour.
> - **Total** — a combined view with bars for individual entities and a line for NITS total demand, with the peak hour highlighted.
>
> On the right, you have the hourly values panel — a quick numeric reference showing each hour's demand."
>
> **The colour coding matters:** Each demand entity has a distinct colour — ECG is blue, NEDCo is amber, VALCO is violet, Mines is emerald, Export is cyan. These same colours carry through every page of the system, so once you learn them, you can find your entity instantly across any screen.

**Importance:**
> "The Control Room replaces the need to have multiple Excel sheets open, a calculator, and a wall clock. In one screen, you see exactly where the system stands — demand, supply, peak, energy, and the entity breakdown — all updated live. This is your situation awareness hub."

---

## Slide 3: ECG Forecast — The ML-Powered Lookahead

**Say:**
> "The ECG Forecast page gives you a scientifically computed 24-hour demand prediction for ECG — the single biggest demand entity on the grid.
>
> At the top, you see the headline summary: Peak Demand (with the hour it hits), Average Demand, Minimum Demand, and Total Energy — all from the forecast engine.
>
> The engine is called the Decomposition Engine v2.4 — it breaks down the historical load signal into components: an underlying trend, daily and weekly seasonality patterns, temperature sensitivity pulled from Open-Meteo weather data, a growth multiplier for year-over-year demand growth, and a Kalman filter that continuously corrects the bias based on recent forecast errors.
>
> You can toggle between two chart views:
> - **Forecast mode** — shows the predicted ECG demand as a filled area with a dashed trend baseline.
> - **Components mode** — shows how much of the forecast comes from each factor: temperature effect, growth effect, and Kalman bias.
>
> Below the chart, the Factors Table gives you the raw hourly multipliers: seasonal ratio, temperature ratio, and growth ratio for every hour of the day.
>
> **Most importantly:** This forecast is optional. It does not automatically override anything. The operator decides whether and when to use it."

**The "Apply to Schedule" button:**
> "When you're ready, click 'Apply to Schedule' — it navigates you directly to the Dispatch Schedule page where you can review and then auto-fill the forecast into the ECG demand row."

**Importance:**
> "This gives you a data-driven second opinion. Instead of relying purely on intuition or last week's numbers, the forecast engine ingests years of historical data, current weather, and calendar patterns to produce an objective prediction. You remain in control — you choose whether to accept it."

---

## Slide 4: Dispatch Schedule — The Core Operational Page

**Say:**
> "This is the heart of the system — the Dispatch Schedule page. This is where you will spend most of your operational time. Let me walk through the full workflow."

### 4a — Uploading a Schedule

**Say:**
> "If no schedule is loaded, you see an upload zone. You drag and drop — or click to browse — the ECG Daily Demand Data Sheet Excel file. The system parses it automatically: 24 hours × 5 demand entities plus all supply plants.
>
> Once uploaded, the schedule appears in DRAFT status. You can see the source filename, upload timestamp, and operator name at the top."

### 4b — The Grid View

**Say:**
> "The schedule grid shows 24 columns — one for each hour of the day, 01 through 24. You can view Demand only, Supply only, or Combined — pick your tab.
>
> **Demand side:** Six rows — ECG (which can be forecast-assisted), NEDCo, VALCO, Mines, Export, and NITS_Total. Each entity has its distinct colour as a left border and a subtle background tint so you can scan down a column and instantly identify which entity is which.
>
> **Supply side:** All generating plants, colour-coded by category: blue left border for hydro, red for thermal, violet for interconnection. Baseload plants have a 'B' badge — these are plants with a fixed contractual output. When you try to edit a baseload plant's value, the system asks you to confirm that you're deliberately overriding the contractual value — a safeguard against accidental changes.
>
> **Summary rows:** Total Demand and Reserve show you the net position for each hour. Cells with high values are highlighted — values over 2,000 MW in amber, over 2,500 MW in crimson. The peak demand cell across the day gets a special red highlight."

### 4c — Editing Cells

**Say:**
> "Editing is straightforward: click any white cell in the demand or supply section, type a new number, and press Enter or click away. The system saves immediately and logs the change to the audit trail.
>
> This replaces the old workflow of 'send someone an email asking them to update cell C14 and hope they get it right.' Every change is tracked."

### 4d — Auto-Fill Forecast

**Say:**
> "If you've reviewed the ECG Forecast page and you want to bring that prediction into your schedule, click 'Auto-fill Forecast.' The system populates the entire ECG demand row with the ML forecast values. You can then inspect, tweak, or accept them."

### 4e — Confirm and Revise

**Say:**
> "When the schedule is finalised — all values checked, reserve margins satisfactory — click 'Confirm Schedule.' This locks the schedule. No further edits are allowed. It enters CONFIRMED status and becomes the official dispatch plan.
>
> If something changes during the day — a plant trips, demand surges, a new request comes in — you can 'Revise Schedule.' You must provide a reason for the revision (this is mandatory and logged). The schedule goes back to DRAFT, you make your changes, and confirm again.
>
> This Draft → Confirmed → Revise → Draft cycle creates a clean, auditable record of every version of the dispatch plan."

### 4f — Audit Trail

**Say:**
> "Click 'Audit Trail' to open the panel on the right. Every action — upload, cell edit, forecast fill, confirm, revise — is logged with a timestamp, operator name, a description, and a SHA-256 cryptographic hash.
>
> The hash chain links each entry to the one before it. If any log entry is tampered with, the chain breaks, and the system flags it as invalid. This is the same technology used in blockchain — it makes the audit trail tamper-evident.
>
> This is a critical compliance feature. When GRIDCo needs to demonstrate to regulators or partners that a dispatch decision was made properly, you have a cryptographically verifiable record."

### 4g — Graph View

**Say:**
> "You can toggle from Grid view to Graph view — a visual chart of demand and supply for the day. Useful for spotting trends and patterns at a glance."

**Importance:**
> "This single page replaces your entire Excel-based dispatch workflow — upload, edit, confirm, revise, audit — all in one place with guardrails, colour coding, and a permanent record."

---

## Slide 5: Data Management — The Registry and Archive

**Say:**
> "The Data Management page has three tabs:

> **1. Schedule Archive** — Lists every dispatch schedule ever uploaded. You can see the date, source file, status, and creation time. This is your historical record of all dispatch plans.
>
> **2. Baseload Registry** — This is the list of 42 baseload plants with their contractual constant MW values. You can edit a plant's constant value directly in-line — useful when a power purchase agreement is updated or a new plant comes online. Changes here automatically apply to all future schedules.
>
> **3. Historical Data** — Shows summary statistics from the ECG historical demand database: total number of records, date range, and average demand. Useful for reference and analysis."

**Importance:**
> "This is your operational memory. The Schedule Archive means you never have to hunt through email attachments to find last week's dispatch plan. The Baseload Registry keeps your contractual constants in one authoritative place — no more version-confusion over what Akosombo's contracted output should be."

---

## Slide 6: Settings — Personal and System Configuration

**Say:**
> "The Settings page lets you manage your profile and some system-level options.
>
> - **User Profile** — update your name, email, or organisation.
> - **Alert Config** — toggle operational alerts: exceedance warnings, model drift detection, SCADA timeout, unit trip alerts.
> - **Model Registry** — if you have multiple forecast models active, you can select which kernel the forecast engine uses. Leave this on the default unless instructed otherwise by the analytics team.
> - **Security** — change your password or enable multi-factor authentication.
> - **Nodes & Sync** — infrastructure status (for system administrators)."

**Importance:**
> "This is where you personalise the system and manage credentials. Most operators will set up their profile and alerts once and rarely need to return."

---

## Slide 7: Tying It All Together — The Daily Workflow

**Say:**
> "Let me summarise how the pieces fit into your daily routine.
>
> **Morning:**
> 1. Log in — Control Room shows you the overall picture.
> 2. Open ECG Forecast — review the ML prediction for today.
> 3. Open Dispatch Schedule — upload the ECG Daily Demand Data Sheet.
> 4. Review and edit any cells that need adjustment.
> 5. Optionally auto-fill the ECG forecast if it looks good.
> 6. Confirm the schedule — it's now the official dispatch plan.
>
> **During the day:**
> - If conditions change, revise the schedule with a reason, make edits, confirm again.
> - Every change is logged in the audit trail with cryptographic verification.
>
> **At any time:**
> - Control Room gives you the live snapshot.
> - Data Management lets you reference past schedules or update baseload constants."

---

## Slide 8: Next Steps — What's Coming

**Say:**
> "The system is operational now, but we have several enhancements planned:
>
> **Near-term:**
> - **Column Resilience** — the system will automatically detect if the Excel upload format changes (e.g., someone adds or removes a column) and alert you rather than failing silently.
> - **Export to PDF/XLSX** — download confirmed schedules for distribution to stakeholders who aren't on the system yet.
> - **Supply Plant Category Legend** — a clear key showing which colours correspond to which supply categories.
>
> **Medium-term:**
> - **Real ECG historical data loading** — the forecast engine currently uses synthetic seed data. Loading several years of actual ECG demand will significantly improve forecast accuracy.
> - **P10/P90 confidence bands** — so you can see the range of possible outcomes, not just a single-point forecast.
> - **Ramp rate alerts** — the system flags hours where load is changing faster than normal.
>
> **Longer-term:**
> - **Multi-substation expansion** — extending the system to other GRIDCo substations.
> - **Reinforcement learning for dispatch optimisation** — the system learns to recommend the optimal generation mix based on cost, availability, and grid conditions.

---

## Closing

**Say:**
> "That's the system end-to-end. The goal is simple: give you better tools, a cleaner workflow, and a permanent record — so you can focus on the operational decisions that keep Ghana's grid stable, not on spreadsheet wrangling.
>
> Questions?"

---

*Document prepared for GRIDCo operator onboarding — May 2026*
