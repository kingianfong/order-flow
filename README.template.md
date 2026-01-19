# Order Flow
This project models the arrival process of Binance UM BTCUSDT trade messages with Poisson and Hawkes point processes, focusing on intraday seasonality and self-exciting clustering (including a heavy-tailed excitation kernel), with exact handling of timestamp collisions.

**Potential use case**: real-time intensity forecasts for queueing/liquidity regime detection or adaptive quoting.

**Concepts**: Maximum Likelihood Estimation (MLE), Hawkes Process, Numerical Stability, Identifiability, Robust Statistics, Automatic Differentiation (AD, JAX), Market Microstructure

## Findings
* Power-law Hawkes has higher validation log likelihood than exponential Hawkes
* QQ residuals show remaining underestimation during bursts
* Gap between robust and Hessian standard errors indicates model misspecification

## Key Contributions
* Built end-to-end pipeline point-process modelling BTCUSDT trade arrivals (JAX, Polars, 68m trades, 17m unique timestamps)
* Derived closed-form solution for exact collision-aware likelihood during timestamp collisions while avoiding jittering,
reduced effective data size by 74% without bias or information loss (unique timestamp aggregation)
* Performed inference and diagnostics using exact gradients/Hessians (autodiff);
assessed identifiability with inverse-Hessian structure;
quantified misspecification with Godambe (sandwich) robust SEs and robust/Hessian SE ratios
* Optimised full likelihood and gradient evaluation to run under 1 second (10 CPU M1 Max laptop, excluding IO, 17m timestamps), enabled iterative MLE (L-BFGS)

## Highlights
{{ intro }}

{{ rbf_hawkes_bases }}

{{ rbf_hawkes_inv_hess }}

### Iterative Model Development
1. `Homogeneous Poisson process` with constant intensity
1. `Daily seasonality` with periodic radial basis functions
1. `Self-excitation` with Hawkes process
1. `Long-memory` with power-law decay kernel

### Statistical Diagnostics
1. `Inverse Hessian Matrix` helps detect weak identifiability and local collinearity between parameters
1. `Godambe Information Matrix` provides standard errors which are robust to model misspecification

### Efficiency
1. `Closed form solution` for log likelihood when there are [multiple events per timestamp](#multiple-events-per-timestamp)
1. `Computational complexity reduction`
    1. $O(n^2) \to O(n)$ time for power-law decay calculation using sum-of-exponentials approximation
    1. $O(n) \to O(\log n)$ parallel span for exponential decay calculation, $h_i = e^{-\omega \Delta t} h_{i-1} + 1$, using parallel prefix scan on a linear recurrence  (`jax.lax.associative_scan` in `decayed_counts.py`)
1. `Gradient-based optimization`
    1. Used Automatic Differentiation (AD) to eliminate manual derivation of log-likelihood gradients for kernels
    1. Compute the exact Hessian for diagnostics, avoiding the numerical instability of finite-difference methods
1. `Property-based testing` ensures power-law approximation remains accurate over valid inputs (`hypothesis` library in `power_law_approx.py`)
1. `Automated report generation` reduces room for human error when producing reports such as this file (`Jinja2` library in `generate_reports.py`)

## Scope
The scope is restricted to conditional intensity estimation for point processes.

Although the implementation handles tens of millions of events, the emphasis is on statistical fit and model diagnostics rather than system performance.

The analysis does not cover price dynamics, alpha generation, or backtesting.


## Data
This project uses [Binance UM futures trade data](https://data.binance.vision/data/futures/um/daily).
* instrument: `BTCUSDT` USDM futures
* train dates: `2025-09-01` to `2025-09-30`
* validation dates: `2025-10-01` to `2025-10-15`
* timestamp resolution: 1 millisecond; multiple trades may share the same timestamp

{{ data_head }}

{{ data_stats }}

It is unclear whether each record corresponds to a single aggressive order or a single passive fill. From the documentation of a [similar dataset](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams#trade-streams) and visualisations, it seems that (2) is more likely. This implies that we could be modelling `arrival of distinct passive order fills`, rather than `arrival of distinct decisions to cross the spread`.

## Modelling Assumptions
To be addressed in [Known Limitations and Future Work](#known-limitations-and-future-work) section.

### Stationarity
The estimated parameters are assumed to remain constant.

### Multiple Events per Timestamp
This project assumes order arrivals follow a point process with the following log likelihood:

$$
\log L(\theta) = \sum_i^n \log \lambda_\theta( t_i ) - \int_0^T \lambda_\theta(s) ds
$$

where
* $\lambda_\theta(t)$ is the conditional intensity given the event history up to time $t$.

Timestamps have `1ms` resolution and each timestamp may have multiple events. For Hawkes models, these events are assumed to arrive instantaneously ($\Delta t = 0$) and sequentially, so each event's likelihood includes the previous event with no decay.

The log likelihood for sequential instantaneous arrivals at the same timestamp uses the Rising Factorial (Pochhammer) polynomial:

$$
\begin{align}
\ell_t(\theta)
    &= \sum_{j = 0}^{c_t - 1} \log \lambda_j(t) \\
    &= \sum_{j = 0}^{c_t - 1} \log( \phi_0 + D_t J + j J ) \\
    &= \sum_{j = 0}^{c_t - 1} \log( a + j d ) \\
    &= c_t \log d  + \log\Gamma(\frac a d + c_t) - \log\Gamma(\frac a d)
\end{align}
$$

where
* $c_t$ is the number of events with timestamp $t$
* $\phi_0$ is the base intensity
* $D_t$ is the decayed sum of events right before $t$
* $J$ is the jump size
* $\lambda_0(t) = \phi_0 + D_t J$
* $\lambda_1(t) = \phi_0 + D_t J + J$
* $\lambda_k(t) = \phi_0 + D_t J + k J$
* $a := \phi_0 + D_t J$ is the pre-computed intensity contribution from history
* $d := J$

This formulation allows the likelihood to be calculated exactly even when multiple events arrive at the same timestamp, and avoids discarding information or artificially jittering timestamps to increase the data size.

## Results

### Log Likelihood
These tables report final log likelihood divided by total number of unique timestamps, $m$:

$$
\frac 1 m \left( \sum_i^m \log \lambda_\theta( t_i ) - \int_0^T \lambda_\theta(s) ds \right)
$$

We report log-likelihood per unique timestamp (not per event) because timestamps are aggregated.

{{ loglik_mean }}

{{ loglik_diff }}

Because market activity differs across days, absolute log likelihood values are not directly comparable across train/validation and comparisons are only meaningful for within each segment.

### Validation QQ Plots
{{ qq_val }}

All the models have heavy right tails, meaning they underestimate the rate of trades. The self-exciting models perform better.

### Validation Time Series Plots
{{ ts_val2 }}

{{ ts_val3 }}

The residuals seem to follow the same pattern as the expected counts at higher aggregation levels, but that behaviour disappears after zooming in.

### Estimated Parameters (Click to Expand)
Timings are taken on an M1 Max MacBook Pro.

{{ all_model_results }}

## Known Limitations and Future Work
1. `Trading Applications`
    * By integrating the models into a trading system and observing their commercial impact, it becomes clear which parts of the model to prioritise
    * This would also give us a wider range of metrics for comparing models (e.g., PnL, Sharpe, max drawdown)
1. [`Modelling Assumptions`](#modelling-assumptions)
    * Stationarity
        * Regimes can change very quickly as this is crypto
        * The model could be refitted frequently on recent data, or make some adjustment so that the parameters update dynamically
    * Multiple Events per Timestamp
        * Because multiple events may correspond to the aggressive order, the conditional iid assumption is suspect
        * This can be addressed by using a marked Hawkes process, where the mark could be the count, volume or notional value of orders filled at a timestamp, or by using a nonlinear impact kernel
        * Care must be taken to apply impact kernels before aggregation to ensure consistency with production, where trades arrive sequentially and we cannot easily guarantee that there are no more events with the same timestamp
1. `Time-rescaling Theorem QQ Plot`
    * The plot was not used as, in its default form, it assumes separate events
    * Possible adjustments include inserting (zero or jittered) timestamps which introduces bias, or dropping timestamps where more than one event happens, which would correspond to the most important times during trading
1. `Regularisation`
    * Penalties were chosen heuristically by observing the convergence, identifiability and misspecification diagnostics
    * Consequently, inverse Hessian matrices may be overly optimistic about identifiability
    * Regularisation parameters can instead be systematically tuned using cross-validation
1. `Bayesian Statistics`
    * The aforementioned regularisation parameters can be replaced with prior distributions to allow a Bayesian interpretation
    * Given that the parameters are continuous and the models are differentiable, efficient gradient-aware MCMC samplers such as HMC and NUTS can be used
1. `Computational Efficiency`
    * The L-BFGS training loop forces device-host synchronisation and may be problematic if the number of parameters increase
    * Some steps of the pipeline create $O(n)$ data (e.g., gradient outer products for Godambe Information Matrix, caches for rbf and power-law), and may require refactoring or batching when working with larger datasets


## Conclusion
The relative performance of the power-law Hawkes kernel is consistent with apparent long-term memory.

However, the high SE-ratio (Robust SE / Hessian SE) suggests remaining misspecification, likely from exogenous events, non-linear feedback loops (e.g., stop losses, liquidation, automatic deleveraging) or regime shifts.

This work provides a baseline for future research into multivariate (buy, sell) or marked (count, volume) Hawkes models or other non-linear or covariate-driven intensities.


## Appendix

### Project Structure
* `environment.yml` Dependencies
* `point_process.py` Data loading, model definitions, optimisation, diagnostic plots.
* `download_trades.py` Multithreaded logic for downloading Binance UM Futures tick data and saving them to compressed parquet files with delta encoding
* `decayed_counts.py` Implementation and tests for exponential Hawkes state recursion
* `power_law_approx.py` Implementation and tests for power-law kernel approximation
* `generate_reports.py` Generates `README.md`

### Reproduction
```
conda env create -f environment.yml
conda activate order-flow
python src/download_trades.py
python src/point_process.py

# optional: regenerate README.md
python src/generate_reports.py
```

### Notable Libraries
JAX, NumPy, Pandas, Polars, PyArrow, Hypothesis
