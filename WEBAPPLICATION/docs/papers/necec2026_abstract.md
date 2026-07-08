---
title: "Online Bias Correction for Short-Term Load Forecasting on a Rapidly Growing Grid: The TIDE Mechanism"
author:
- "[Name]"
- "[Co-author]"
affiliation: "[Affiliation]"
email: "[email]"
keywords: "load forecasting; TIDE; online bias correction; DLinear; African grid"
---

# Online Bias Correction for Short-Term Load Forecasting on a Rapidly Growing Grid: The TIDE Mechanism

[Name], [Co-author]  
[Affiliation]  
[email]

**Keywords:** load forecasting; DLinear; TIDE; online correction; non-stationary time series

## Extended Abstract

Short-term load forecasting (STLF) on rapidly growing grids presents a challenge absent from the mature-grid literature: distribution shift is not a marginal concern but the dominant feature. The West African national grid in this study grew 94% in eight years --- from 1,692 MW mean hourly demand in 2018 to 3,275 MW in 2026. Peak demand reached 4,031 MW. Annual growth accelerated from 6.2% to 13.6%.

We built a DLinear baseline (168-hour input, 24-hour forecast horizon, ~40,000 parameters, trained in under 30 CPU-seconds per fold) and evaluated it via 6-fold expanding-window cross-validation spanning 2018--2026. Despite capturing diurnal and weekly patterns well (mean fold MAE: 93.9 MW, MAPE: 3.65%), the model accumulated substantial systematic under-forecasting bias between retraining cycles. The bias grew monotonically with test-year demand. An autocorrelation analysis of the forecast residuals revealed a highly persistent, slowly varying signal ($\rho = 0.6$--$0.8$ at lags 1--48 hours) concentrated at low frequencies --- a "bias hum" driven by the grid's growth trajectory.

We propose **Temporal Integration of Drift Errors (TIDE)**, a zero-parameter online bias corrector. TIDE operates in normalized $z$-score space for scale invariance: it tracks the running mean of recent forecast errors and updates an exponentially weighted moving average (EMA, $\alpha = 0.3$) of the bias, which is then subtracted from future predictions before denormalization. The mechanism has zero trainable parameters, one minimally sensitive hyperparameter, and approximately 20 lines of implementation.

TIDE improves the DLinear mean fold MAE from 93.9 MW to 75.9 MW (19.2% relative reduction, $p < 0.001$) and reduces systematic under-forecasting bias by over 80%. The improvement is consistent across all six folds (16.8--20.9%). TIDE generalizes uniformly across eleven diverse forecasting architectures --- including LSTMs, GRUs, Transformers, LightGBM, SVR, ARIMA, and seasonal naïve --- yielding a consistent 23--26% error reduction on ensemble predictions. It outperforms alternative online correctors (simple moving averages, linear trend extrapolation, and Kalman filters). Varying the smoothing parameter $\alpha$ across 0.3--0.9 produces a narrow 3% error band, confirming the mechanism requires no continuous tuning.

For production deployment, we establish that a single DLinear model trained on the most recent 2--4 years of history combined with TIDE provides an optimal, lightweight forecasting solution. Training on 4 years yields identical accuracy to the full 8-year history (118.4 MW baseline), reducing data storage and training costs. Annual retraining suffices because TIDE handles the inter-cycle drift: increasing retraining to monthly yields only 3.1% additional improvement. This configuration requires no GPU, no gradient updates at inference, and no manual parameter adjustments --- making it suitable for resource-constrained utility environments where infrastructure and engineering teams are limited.

## References

[1] A. Zeng et al., "Are transformers effective for time series forecasting?" *Proc. AAAI*, vol. 37, no. 9, pp. 11121--11129, 2023.

[2] N. H. Mvungi et al., "Load forecasting for Tanzania's power system," *J. Energy Southern Africa*, vol. 32, no. 2, pp. 48--59, 2021.

[3] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. OTexts, 2021.
