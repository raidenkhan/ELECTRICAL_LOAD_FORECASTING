# Peer Review: DLinear Paper

## Paper
**"Load Forecasting Under Extreme Demand Growth: A Case Study of a Doubling West African Grid"**

## Review Pipeline: Academic Paper Reviewer v1.9.0 (7-Agent)

### Phase 0: Field Analysis

| Attribute | Assessment |
|-----------|-----------|
| Primary discipline | Energy / Power Systems (load forecasting) |
| Secondary discipline | Applied ML (time series under distribution shift) |
| Research paradigm | Empirical case study with controlled experiments |
| Methodology type | Retrospective evaluation + simulation |
| Target journal | Applied Energy / IEEE Trans. on Power Systems |
| Paper maturity | Very early draft (no references, no related work, thin analysis, no code) |

### Reviewer Configuration

| Role | Identity | Expertise | Focus |
|------|----------|-----------|-------|
| **EIC** | Dr. Kenji Tanaka, University of Tokyo, AE at *Applied Energy* | Energy system modeling, forecasting | Journal fit, research paper vs technical report |
| **Reviewer 1 (Methodology)** | Dr. Priya Sharma, IIT Bombay | Time series under non-stationarity, distribution shift | Experimental design, statistical validity |
| **Reviewer 2 (Domain)** | Prof. Michael Adewale, University of Lagos | African power systems, electrification policy | African grid claims accuracy, domain depth |
| **Reviewer 3 (Perspective)** | Dr. Elena Vorontsova, Skoltech | Concept drift in ML, continual learning | Connections to drift adaptation literature |
| **Devil's Advocate** | — | Core argument challenge | Is the central claim defensible? |

---

### Phase 1: Review Reports

---

#### EIC Report — Dr. Kenji Tanaka

| Dimension | Score (0-100) | Comments |
|-----------|:---:|----------|
| Originality | 55 | DLinear handling 94% growth is moderately interesting, but no novel *method*. Originality is in the *setting*, not the *solution*. |
| Significance | 65 | Useful for practitioners. Limited academic significance — confirms linear models extrapolate linear trends. |
| Evidence | 45 | Too many claims not backed by sufficient data. Normalization compares only 2 strategies. Retraining simulation lacks standard errors. |
| Presentation | 40 | Reads as notes, not a paper. Section 5 lists bullet points as "Practical Lessons" — blog post structure, not journal paper. |
| Overall | 50 | Major Revision (or reformat as a letter/technical report) |

**Major Issues**

1. **No related work section**: Cannot evaluate contribution without situating in existing distribution shift, non-stationary time series, and African grid literature.
2. **Methodological scope too narrow**: DLinear compared only to Chronos, Seasonal Naive, ARIMA, LightGBM. Missing N-BEATS, Prophet, Theta, ETS, SSM-based forecasters (Mamba, S6). Chronos comparison questionable — zero-shot model shouldn't be expected to work without fine-tuning.
3. **"No degradation over 3 years" overstated**: Table 3.4 shows Fold_1 degrades 84.3 → 112.8 MW (+34%). Paper frames this as "no degradation" — misleading.
4. **Retraining recommendation exceeds evidence**: "Retrain annually, not more" based on simulation with 1-2 data points per frequency, no statistical testing.

**Minor Issues**

- "ERTE" in abstract is typo for "RTE"
- No figures required for load forecasting paper (load profiles, residuals, degradation curves)
- MAPE "decreases over time" (4.19% → 3.08%) is mathematically expected when denominator grows faster than absolute error. Not surprising.

---

#### Methodology Report — Dr. Priya Sharma

| Dimension | Score (0-100) | Comments |
|-----------|:---:|----------|
| Design rigor | 40 | Single architecture, single dataset. All findings could be dataset-specific. |
| Statistical validity | 35 | No confidence intervals. No statistical tests. Retraining simulation has no error bars. |
| Reproducibility | 30 | No hyperparameters (lr, batch size, kernel size, optimizer). No data description. No code. |
| Overall | 35 | Major Revision — or reject if unaddressed |

**Critical Issues**

1. **Single dataset is fatal for generalizability**: Cannot conclude "simple architectures outperform complex adaptive methods" from one grid's data. Need second dataset (Indian, Brazilian, or synthetic with controlled drift).
2. **No uncertainty quantification**: Point estimates only. Retraining frequency comparison most concerning — each row different experimental condition, no variance estimates.
3. **"No degradation" framing is misleading**: 28% degradation over 5 years ≠ "no degradation." Slow degradation, yes. Misrepresentation of data.
4. **Retraining simulation methodology**: How were quarterly and monthly simulations conducted? Expanding or rolling window? Warm-start or re-init? "Number of Models Needed" column (72 for monthly) unrealistic — models can be warm-started.
5. **Adaptive normalization confound**: Clarify whether Fold_6 statistics are computed from training data (2018-2025) only, or use test period information.
6. **MAPE decreasing is tautological**: Expected when growth is predictable and MAPE denominator grows. Not a surprising finding.

---

#### Domain Report — Prof. Michael Adewale

| Dimension | Score (0-100) | Comments |
|-----------|:---:|----------|
| Literature completeness | 30 | No related work. Missing African power systems literature, West African Power Pool studies. |
| Domain accuracy | 70 | Growth data plausible for rapidly electrifying African grid. But analysis lacks domain depth. |
| Practical relevance | 75 | Retraining recommendations relevant but paper lacks operational context. |
| Overall | 55 | Major Revision |

**Domain Issues**

1. **No discussion of grid characteristics**: Single-utility or wholesale market? Generation mix (hydro vs thermal)? Affects load patterns — hydro-dominated grids have seasonal patterns tied to water availability.
2. **Grid interconnection**: Many West African grids part of West African Power Pool (WAPP). Cross-border flows affect net load. Is forecast for gross or net consumption?
3. **Load composition matters**: Mining load flat and predictable; residential has high variance. Relative mix changes with electrification. Paper doesn't discuss this.
4. **Electrification dynamics**: Rural electrification shifts load from industrial-dominated to residential-dominated, changing diurnal pattern and variance. Structural break beyond trend growth.
5. **Practicality of 6-fold ensemble**: Operators struggle to maintain 1 model, let alone 6. What is operational burden? Hardware requirements?

---

#### Perspective Report — Dr. Elena Vorontsova

| Dimension | Score (0-100) | Comments |
|-----------|:---:|----------|
| Cross-disciplinary connections | 35 | No connection to drift adaptation, continual learning, or model monitoring literature. |
| Broader implications | 55 | Argues against frequent retraining but doesn't engage with continual learning literature addressing exactly this. |
| Overall | 45 | Major Revision |

**Perspective Issues**

1. **No engagement with drift literature**: Gama et al. (2014), Webb et al. (2016), Lu et al. (2018). DLinear handling drift through linear extrapolation is not adaptation — it's fortunate drift is approximately linear. For non-linear drift (accelerating growth), DLinear would fail.
2. **MAE alone ignores model risk**: Model never retrained may suddenly fail if trend changes (economic shock, policy change). Risk assessment incomplete.
3. **No monitoring discussion**: How does operator know model is failing? Paper doesn't discuss drift detection or alarm thresholds.
4. **No continual learning comparison**: ER, EWC, online GD with replay extensively studied for non-stationary time series. Paper claims "simple outperforms complex" without testing complex methods.
5. **Accelerating growth is a problem**: Growth accelerating 6.2% → 13.6% is not linear. At what point does DLinear break? Paper doesn't investigate failure boundary.

---

#### Devil's Advocate Report

**Strongest Counter-Argument**

Central claim — "DLinear with proper normalization self-adapts to the growth trend" — is misleading. DLinear does not "self-adapt." It has a linear projection that can extrapolate a linear trend. If growth were exponential, logarithmic, or had regime changes, DLinear would fail catastrophically. Paper reports on a fortuitous property of the dataset (approximately linear growth), not a property of the model.

"Retrain annually, not more" ignores model risk. Model trained annually may perform well on average but have periods of large error during regime shifts. Paper evaluates average, not worst-case.

Paper's contribution is descriptive but conclusions are written prescriptively. Fundamental tension.

**CRITICAL Issues**

1. **[CRITICAL] Single dataset**: All findings may be dataset-specific. Without second grid or synthetic validation, conclusions ungeneralizable.
2. **[CRITICAL] "Self-adaptation" framing**: DLinear linear extrapolation of linear trend is not adaptation. Damages credibility.
3. **[MAJOR] No continual learning comparison**: Claim "simple outperforms complex" without testing any complex adaptive methods.
4. **[MAJOR] Retraining recommendation ignores risk**: Needs failure-mode analysis (what if growth slows to 2%? what during recession?).

**Ignored Alternative Explanations**
- DLinear's success due to 168-hour lookback window, not linear projection. Simple AR(168) might perform similarly.
- "No degradation" of ensemble is just variance reduction from averaging 6 models.
- Adaptive normalization improvement (91.0 vs 141.2 MW) might be entirely explained by different z-score transformations.

---

### Phase 2: Editorial Synthesis

#### Consensus Issues

| Issue | Severity |
|-------|:--------:|
| No related work section | Critical |
| No confidence intervals or error bars | Critical |
| Single dataset limits generalizability | Critical |
| "Self-adaptation" / "no degradation" framing misleading | Critical |
| No comparison to drift adaptation or continual learning | Major |
| Retraining frequency simulation lacks methodological detail | Major |
| No failure-mode analysis | Major |
| Too thin for research paper (reads as technical report) | Major |

#### Decision: MAJOR REVISION (borderline Reject)

| Dimension | Weight | Score | Weighted |
|-----------|:-----:|:-----:|:--------:|
| Originality | 20% | 50 | 10.0 |
| Methodology | 25% | 35 | 8.8 |
| Evidence | 25% | 40 | 10.0 |
| Coherence | 15% | 50 | 7.5 |
| Writing | 15% | 40 | 6.0 |
| **Total** | **100%** | | **42.3** |

Score 42.3 falls in Reject range (< 50). Recommendation: Major Revision with high bar, or merge into companion paper.

#### Revision Roadmap

**Required as standalone paper:**
1. Add related work (distribution shift, concept drift, continual learning, African power systems)
2. Add second dataset (Indian, Brazilian, or synthetic)
3. Bootstrap CIs on all tables
4. Reframe "self-adaptation" — be honest about linear extrapolation limitations
5. Add failure-mode analysis
6. Add figures (load profiles, residuals, degradation curves)

**Recommended action:** Merge into TIDE paper as baseline context, rather than publishing as separate paper.
