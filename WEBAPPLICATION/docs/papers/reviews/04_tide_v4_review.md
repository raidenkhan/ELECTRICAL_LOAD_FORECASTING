# Verification Review Report (v4 — Re-Review)

## Manuscript
**"An Investigation into Online Bias Correction for Load Forecasting on a Rapidly Growing African Grid"**

## Review Round
v4 Re-Review (Verification Mode)

## Previous History
| Round | Score | Decision | Notes |
|-------|:-----:|----------|-------|
| v1 | 58/100 | Major Revision | Saved at `01_tide_review.md` |
| v2 | 62/100 | Major Revision | — |
| v3 | 74/100 | Minor Revision | — |
| v4 | **76/100** | **Minor Revision** | Current — verification of v3→v4 revisions |

---

## Decision
### Minor Revision

The authors have addressed the majority of previous review comments with thorough, well-executed revisions. Of the 14 original revision items (P1+P2), 9 are FULLY_ADDRESSED, 3 are PARTIALLY_ADDRESSED, 1 is NOT_ADDRESSED, and 1 is NOT_APPLICABLE. The paper is materially improved but three residual issues—a numerical inconsistency in architecture count, a missing data-access statement, and a partial gap in developing-economy literature coverage—prevent Acceptance at this stage.

---

## Revision Response Checklist

### Priority 1 — Required Revisions (from v1 Review Roadmap)

| # | Original Review Comment | Author's Claim | Response Status | Revision Location | Verified? | Quality Assessment |
|---|------------------------|---------------|-----------------|-------------------|-----------|-------------------|
| R1 | **[CRITICAL-DA]** Add comparison baselines: SMA (N=7, 14, 30), linear error trend extrapolation, no-correction baseline for all architectures | Added SMA (7/14/30d), Kalman (3 configurations), linear trend, TIDE comparisons | FULLY_ADDRESSED | §6.2, §6.3, Appendix C | ✅ Yes | Comprehensive. SMA, Kalman filter with 3 Q/R combos, linear trend, and TIDE at multiple alpha values all compared. All 11+ architectures tested. |
| R2 | **[CRITICAL-DA]** Bootstrap CIs: 95% confidence intervals for all MAE/MAPE values (1,000 iterations) | Added 95% CI using 10,000 bootstrap iterations | FULLY_ADDRESSED | §6.2 | ✅ Yes | CIs reported for DLinear baseline, TIDE, SMA-7d, linear trend, and Kalman. CI widths ~14-20 MW, demonstrating meaningful statistical separation. |
| R3 | **[MAJOR]** EMA sensitivity analysis: α ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 0.9} | Tested α ∈ {0.1, 0.3, 0.5, 0.7, 0.9} across all 6 folds | FULLY_ADDRESSED | Appendix C, Table C.1, Figure C.1 | ✅ Yes | Five alpha values tested. α≥0.3 cluster within 3%. α=0.1 measurably worse. Sensitivity clearly bounded and well-discussed. |
| R4 | **[MAJOR]** Expand literature: Kalman filter (Harvey 1989), online learning in energy forecasting, African energy access (IEA Africa Outlook), Indian/Brazilian grid forecasting, conformal prediction | Added Harvey (1989), Gama et al. (2014), IEA Africa Outlook (2023), Stankeviciute et al. (2021), Lu et al. (2018), Hyndman & Athanasopoulos (2021) | PARTIALLY_ADDRESSED | §8.1–8.4 | ⚠️ Partial | Kalman filter (§8.2), concept drift (§8.4), IEA Africa Outlook added. Conformal prediction (Stankeviciute 2021) cited. **Still missing**: developing-economy load forecasting studies (Indian, Brazilian, SE Asian grids) that share similar growth dynamics. These were flagged in both v1 and v2 reviews. |
| R5 | **[MAJOR]** Clarify EMA causality: explicitly state only past errors used; add pseudocode diagram | Added Mermaid flow diagram (Figure 6) showing update loop | PARTIALLY_ADDRESSED | §5.2, Figure 6 | ⚠️ Partial | The Mermaid diagram and text ("After each forecast, TIDE computes the normalized error, updates an EMA bias estimate, and subtracts it from all future predictions") imply causal ordering. However, no explicit "only errors observed before forecast timestamp" statement. Adding 1-2 sentences would eliminate any ambiguity. |
| R6 | **[MAJOR]** Address uniform 23-26% improvement: robustness or data-level correction? | Added dedicated §7.3 discussing why improvement is uniform | FULLY_ADDRESSED | §7.3 | ✅ Yes | Clearly explains that bias is a *data* property, not a *model* property. Argues uniformity confirms the hypothesis that the systematic bias is structural, not model-specific. Bounded correctly: "TIDE removes the systematic bias component, but does not reduce irreducible noise." |
| R7 | **[MAJOR]** Data/code: dataset granularity, hyperparameters, code repo link | Mentions "project repository" (Appendix A) and provides regenerate command for figures | PARTIALLY_ADDRESSED | Appendix A, Figure regeneration note (p.1) | ⚠️ Partial | No public repository URL provided. Dataset granularity (number of hours, missing data percentage, number of weather stations, holiday definitions) not explicitly stated. Figure regeneration command is useful but insufficient for full reproducibility. |
| R8 | **[MAJOR]** Document H12 crash mode (optimization divergence vs NaN vs memory) | Detailed H12 failure analysis in Appendix A.4 | FULLY_ADDRESSED | §A.4, H12 | ✅ Yes | Clearly documents: (1) online gradient: "training loss diverged within 3 days"; (2) EWC: "test MAE was 118 MW (26% worse)". Root cause analysis provided: "94% growth over 8 years exceeds convergence radius." |

### Priority 2 — Strongly Recommended Revisions (from v1 Review Roadmap)

| # | Original Review Comment | Response Status | Notes |
|---|------------------------|-----------------|-------|
| S1 | Test detrending alternative (predict Δ(load) instead of load) | NOT_ADDRESSED | No differencing or detrending baseline provided. This was a DA suggestion that remains unaddressed. |
| S2 | Discuss load shedding and non-technical losses | FULLY_ADDRESSED | §1.3 items 6-7 explicitly discuss non-technical losses (15-25%) and load shedding during 2023-2026. Accurate and well-contextualized. |
| S3 | Add single-station temperature limitation to discussion | PARTIALLY_ADDRESSED | §1.3 mentions "temperature insensitivity" (r = -0.25) and §A.2 discusses weaker temperature effect vs temperate climates. But paper never states how many weather stations were used or acknowledges this as a measurement limitation. |
| S4 | Replace "spreadsheet" claim with measured deployability assessment | FULLY_ADDRESSED | No "spreadsheet add-on" language remains. §7.7 now says "It can run as a lightweight service." Appropriate toning. |
| S5 | Fill all references and appendices | FULLY_ADDRESSED | Appendix A (4 sub-sections), Appendix B (Sobolev), Appendix C (alpha sensitivity + corrector comparison) all fully populated. 14 references. |
| S6 | Add Sobolev loss results to Appendix B | FULLY_ADDRESSED | Complete Sobolev trajectory loss ablation: fold-by-fold results, paired t-test, Wilcoxon, Cohen's d, ramp effects analysis. Well-executed. |

---

## New Issues (Discovered During Revision)

| # | Type | Location | Description | Severity |
|---|------|----------|-------------|----------|
| NEW-1 | Numerical inconsistency | Abstract + §6.3 | Abstract claims "generalizes across 12 architectures" and §6.3 states "all 12 architectures from the hypothesis study." The Section 6.3 table shows **11** models (DLinear, NLinear, LSTM, Transformer, GRU, MLP, CNN/WaveNet, SVR, LightGBM, Seasonal Naive, ARIMA). The 12th architecture (likely DeepAR, tested in H2) is not shown. Either add DeepAR to the table or correct the count to 11. | Major |
| NEW-2 | Missing artifact comparison | §6.3, §7.4 | TIDE improves Seasonal Naive by 25% (109.4 MW). This suggests the corrector is powerful enough to make a simple seasonal model competitive with a trained ML model. This finding—which has practical implications for resource-constrained operators—is not discussed. A seasonal model + TIDE at ~109 MW is still worse than DLinear alone at 91 MW, but the paper should comment on whether a simple model + TIDE could be a viable deployment strategy. | Minor |
| NEW-3 | Reproducibility gap | Throughout | No public repository URL, no data DOI, no dataset metadata table. Line 8 mentions "Regenerate PNGs: py -3.13 Backend/tools/generate_figures.py" but the code is not accessible to reviewers. The paper cannot be considered reproducible in its current form. | Major |
| NEW-4 | Experiment numbering inconsistency | §2.4 vs §6.1 | The anti-fabrication check (provided as context to this review) flags that Section 2.4's ensemble table (Fold_6 MAE: 94.6 MW for DLinear) and Section 2.5's degradation table use numbers from "original retrain" experiments, while Section 6.1's main result uses "tide_validation" experiments (Fold_6 DLinear: 120.7 MW). The discrepancy is explainable (different experiment tags, fold-level vs ensemble), but the paper should note this explicitly—otherwise an attentive reader will notice the numbers differ and wonder which to trust. | Major |
| NEW-5 | No explicit EMA causality statement | §5.2 | As noted in R5's partial assessment, the paper should add: "Importantly, TIDE only uses errors observed up to the current forecast timestamp—no future information leaks into the bias estimate." This 1-2 sentence addition would fully resolve the causality concern. | Minor |
| NEW-6 | Electricity market terminology | §1.1–1.2 | "Mean hourly demand" is used throughout but the paper uses "MW" as the unit. For electrical engineers, it would be clearer to specify whether this is mean hourly load (MWh/h), average MW over each hour, or instantaneous MW at each hour boundary. | Minor |

---

## Quality Score Breakdown

| Dimension | Weight | Score (v4) | Score (v1) | Delta | Rationale |
|-----------|:------:|:----------:|:----------:|:-----:|-----------|
| Originality | 20% | 72 | 65 | +7 | Core finding (EMA bias correction) unchanged, but warm-start analysis (§2.6) and Sobolev ablation (Appendix B) add methodological novelty. Negative results documentation (§4) remains a distinctive strength. |
| Methodological Rigor | 25% | 78 | 45 | +33 | **Largest improvement.** Bootstrap CIs, alpha sensitivity grid, comparison against 3 alternative correctors, full statistical testing (t-test, Wilcoxon, Cohen's d in Appendix B). 12-hypothesis ablation framework is strong. |
| Evidence Sufficiency | 25% | 78 | 55 | +23 | Full Appendix A (one of the strongest negative-results documentation pools in recent load forecasting literature), Appendix B (Sobolev), Appendix C (sensitivity + baselines). 26 experimental conditions tested. References expanded from ~4 to 14. |
| Argument Coherence | 15% | 75 | 70 | +5 | "Hypothesis→failure→TIDE→validation→discussion" narrative is intact and compelling. Minor blemish: "12 architectures" claim not matching table (NEW-1). |
| Writing Quality | 15% | 76 | 65 | +11 | Readable, direct, and engaging. The investigative/narrative tone works well. No grammatical or structural issues noted. Minor terminology precision issues (NEW-6). |
| **Weighted Total** | **100%** | **76.05** | **58.3** | **+17.7** | **Minor Revision** |

### Decision Band
| Score | Band |
|:-----:|------|
| ≥ 80 | Accept |
| 65–79 | Minor Revision |
| 50–64 | Major Revision |
| < 50 | Reject |

The v4 score of **76/100** places the paper in **Minor Revision** territory. The uplift from v1 (+18 points) is driven almost entirely by methodological rigor (+33) and evidence (+23), reflecting the successful implementation of bootstrap CIs, comparison baselines, sensitivity analysis, and the full appendix content.

---

## Decision Rationale

This v4 manuscript represents substantial revision progress. The three original CRITICAL gaps from the v1 Devil's Advocate—(a) no comparison baselines, (b) no statistical testing, (c) uniform improvement unexplained—are all now addressed satisfactorily. The 6-fold DLinear ensemble, 12-hypothesis ablation design, and comprehensive corrector comparison form a methodologically sound empirical investigation.

The paper's core contribution—that a zero-parameter EMA corrector operating in normalized space reduces systematic bias by ~19% across all folds and ~23-26% across all architectures—is now well-supported by the evidence. The empirical finding is practically significant for developing-economy grid operators and is presented with appropriate caveats (bounded improvement, residual noise, limitation to additive bias drift).

**What prevents Acceptance (v4 → Accept requires ≥ 80):**

1. **NEW-1 (12 vs 11 architectures):** A factual inconsistency that must be corrected. Either DeepAR results belong in the table or the count should be corrected to 11.
2. **NEW-3 (Reproducibility gap):** No data access, no repository URL. For any applied ML paper, this is a standard expectation. Provide a GitHub/anonymous repo link or a data availability statement.
3. **NEW-4 (Experiment numbering inconsistency):** The discrepancy between §2.4/§2.5 numbers and §6.1 numbers needs an explicit note. Readers who cross-reference will notice.
4. **R4 (Developing-economy literature):** Still a gap. Indian, Brazilian, and SE Asian grid forecasting studies share the growth dynamics central to this paper's argument. At minimum, acknowledge the gap as a limitation.
5. **S1 (Detrending baseline):** Recommended but not required for acceptance. Mark as a future work direction.

---

## Residual Issues (Unresolved from v3→v4)

| # | Original Item | Status | Recommended Action |
|---|---------------|--------|-------------------|
| R4 | Literature: developing-economy forecasting | PARTIALLY_ADDRESSED | Add 2-3 references (Indian grid, Brazilian grid) OR add a "Literature coverage caveat" sentence in Limitations |
| R5 | EMA causality explicit statement | PARTIALLY_ADDRESSED | Add 1-2 sentences to §5.2 stating temporal ordering explicitly |
| R7 | Data/code reproducibility | PARTIALLY_ADDRESSED | Provide GitHub repo URL or Data Availability statement. Add dataset metadata (hours, stations, missing data %) |
| S1 | Detrending (Δ(load)) baseline | NOT_ADDRESSED | Add to §7.6 Open Questions or mark as future work |
| S3 | Single-station temperature limitation | PARTIALLY_ADDRESSED | State number of weather stations used; acknowledge as limitation |
| NEW-1 | 12 vs 11 architectures | — | Correct table or abstract; add DeepAR results or adjust count |
| NEW-3 | Reproducibility gap | — | Provide repository URL |
| NEW-4 | Experiment numbering inconsistency | — | Add explanatory footnote reconciling §2.4/§2.5 numbers with §6.1 |

---

## Required Revisions for v5

| # | Item | Severity | Section | Effort |
|---|------|----------|---------|--------|
| R1 | Correct "12 architectures" to "11 architectures" (or add DeepAR to table) | Major | Abstract + §6.3 | < 1 hr |
| R2 | Add public repository URL or data availability statement | Major | New section after References | < 1 hr |
| R3 | Add footnote reconciling §2.4/§2.5 (original retrain) vs §6.1 (tide_validation) experiment numbers | Major | §2.4 or §6.1 | < 30 min |
| R4 | Add explicit causality statement in §5.2 | Minor | §5.2 | < 10 min |
| R5 | Acknowledge developing-economy literature gap or add 2-3 references | Minor | §7.6 or §8 | 1-2 hr |

### Suggested Revisions

| # | Item | Effort |
|---|------|--------|
| S1 | Add detrending/differencing to Open Questions (§7.6) | < 30 min |
| S2 | Note number of weather stations; acknowledge as limitation | < 15 min |
| S3 | Discuss Seasonal Naive + TIDE as viable low-cost deployment option | < 1 hr |
| S4 | Clarify "MW" vs "MWh/h" terminology | < 15 min |

### Total Estimated Effort for v5
- **Required**: 2-4 hours
- **+ Suggested**: 3-5 hours

---

## Summary

The v4 manuscript has transformed substantially from the v1 draft. The three most critical methodological gaps—no baselines, no CIs, no sensitivity analysis—are fully resolved. The paper now has strong empirical evidence supporting its central claim, and the investigative narrative is well-constructed and engaging.

The remaining issues are primarily about precision and completeness: a numerical discrepancy (12 vs 11), missing data access, and an experiment numbering inconsistency. None require new experiments or analysis. A 2-4 hour revision cycle addressing these items should bring the paper to Acceptable quality for submission to a venue such as *Applied Energy* or *Energy & AI*.

**Assessment**: The authors should be encouraged that the core contribution is solid and well-evidenced. The revision history (58 → 62 → 74 → 76) shows consistent improvement. With the required fixes, this paper will be a strong, distinctive contribution to the load forecasting literature for developing economies.
