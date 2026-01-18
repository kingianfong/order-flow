# Order Flow
This project models high-frequency BTCUSDT trade arrivals using parametric point-processes (Poisson and Hawkes), with intraday seasonality and long-memory kernels, and evaluates them via statistical diagnostics.

**Languages/Libaries**: Python, JAX, NumPy, Pandas, Polars, PyArrow, Hypothesis

**Concepts**: Vectorization, Automatic Differentiation (AD), Maximum Likelihood Estimation (MLE), Robust Statistics, Identifiability, Numerical Stability, Market Microstructure

## Highlights
{{ intro }}

{{ rbf_hawkes_bases }}

{{ rbf_hawkes_inv_hess }}

### Iterative Model Development
1. `Constant intensity` with Poisson baseline
1. `Daily seasonality` with periodic radial basis functions
1. `Self-excitation` with Hawkes process
1. `Long-memory` with power-law decay kernel

### Statistical Diagnostics
1. `Inverse Hessian Matrix` helps detect weak identifiability and local collinearity between parameters
1. `Godambe Information Matrix` provides standard errors which are robust to model misspecification

### Engineering
1. `Computational complexity reduction`
    1. $O(n^2) \to O(n)$ time for power-law decay calculation using sum-of-exponentials approximation
    1. $O(n) \to O(\log n)$ parallel span for exponential decay calculation, $h_i = e^{-\lambda \Delta t} h_{i-1} + 1$, using parallel prefix scan on a linear recurrence  (`jax.lax.associative_scan` in `decayed_counts.py`)
1. `JAX Automatic Differentiation (AD)`
    1. Eliminated manual derivation of log-likelihood gradients for kernels
    1. Compute the exact Hessian for diagnostics, avoiding the numerical instability of finite-difference methods
1. `Property-based testing` ensures power-law approximation remains accurate over valid inputs (`hypothesis` library in `power_law_approx.py`)
1. `Automated report generation` reduces room for human error when producing reports such as this file (`Jinja2` library in `generate_reports.py`)

## Scope
This project is a statistical modelling and engineering exercise for the arrival of trades. This is limited to parameter estimation and diagnostic testing and does not include trading applications such as alpha generation, backtesting, or execution.

## Project Structure
* `environment.yml` Dependencies
* `point_process.py` Data loading, model definitions, optimisation, diagnostic plots.
* `download_trades.py` Multithreaded logic for downloading Binance UM Futures tick data and saving them to compressed parquet files with delta encoding
* `decayed_counts.py` Implementation and tests for exponential Hawkes state recursion
* `power_law_approx.py` Implementation and tests for power-law kernel approximation
* `generate_reports.py` Generates `README.md`

## Usage
```
conda env create -f environment.yml
conda activate order-flow
python src/download_trades.py
python src/point_process.py
```

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
\log L(\theta) = \sum_i \log \lambda_\theta( t_i ) - \int_0^T \lambda_\theta(s) ds
$$

where
* $\lambda_\theta(t)$ is the conditional intensity given the event history up to time $t$.

Timestamps have `1ms` resolution and each timestamp may have multiple events. For Hawkes models, these events are assumed to be conditionally iid and therefore arrive sequentially and self-excite. This is included in the log likelihood using the Rising Factorial (Pochhammer) polynomial:

$$
\begin{align}
\log \lambda (t) &= \sum_{j=0}^{n-1} \log \lambda_j(t) \\
    &= \sum_{j=0}^{n-1} \log( \phi_0 + D_t J + j J) \\
    &= \sum_{j=0}^{n-1} \log( a + j d) \\
    &= n \log d  + \log\Gamma(\frac a d + n) - \log\Gamma(\frac a d)
\end{align}
$$

where
* $\phi_0$ is the base intensity
* $D_t$ is the decayed sum of events right before $t$
* $J$ is the jump size
* $\lambda_0(t) = \phi_0 + D_t J$
* $\lambda_1(t) = \phi_0 + D_t J + J$
* $\lambda_k(t) = \phi_0 + D_t J + k J$
* $a := \phi_0 + D_t J$ is the intensity right before the events
* $d := J$

This formulation allows the likelihood to be calculated exactly even when multiple events arrive at the same timestamp, and avoids discarding information or artificially jittering timestamps to increase the data size.

## Results

### Log Likelihood
{{ loglik_mean }}

{{ loglik_relative }}

Validation log likelihood is consistently higher than training log likelihood. This is expected as the validation dataset includes `2025-10-10`, when the market was particularly active.

### Validation QQ Plots
{{ qq_val }}

All the models have heavy right tails, meaning they underestimate the rate of trades. The self-exciting models perform better.

### Validation Time Series Plots
{{ ts_val2 }}

{{ ts_val3 }}

The residuals seem to follow the same pattern as the expected counts at higher aggregation levels, but that behaviour disappears after zooming in.

### Estimated Parameters (Click to Expand)
{{ all_model_results }}

## Known Limitations and Future Work
1. `Trading Applications`
    * By integrating the models into a trading system and observing their commercial impact, it becomes clear which parts of the model to prioritise
1. [`Modelling Assumptions`](#modelling-assumptions)
    * Stationarity
        * Regimes can change very quickly as this is crypto
        * The model could be refitted frequently on recent data, or make some adjustment so that the parameters update dynamically
    * Multiple Events per Timestamp
        * Because multiple events may correspond to the aggressive order, the conditional iid assumption is suspect
        * This can be addressed by using a marked Hawkes process, where the mark could be the count, volume or notional value of orders filled at a timestamp, or by using a nonlinear impact kernel
1. `Regularisation`
    * Penalties were chosen heuristically by observing the convergence, identifiability and misspecification diagnostics
    * They can instead be systematically tuned using cross-validation
1. `Bayesian Statistics`
    * The aforementioned regularisation parameters can be replaced with prior distributions to allow a Bayesian interpretation
    * Given that the parameters are continuous and the models are differentiable, efficient gradient-aware MCMC samplers such as HMC and NUTS can be used
1. `Computational Efficiency`
    * The L-BFGS training loop forces device-host synchronisation and may be problematic if the number of parameters increase
    * Some steps of the pipeline create $O(n)$ data (e.g., gradient outer products for Godambe Information Matrix, caches for rbf and power-law), and may require refactoring or batching when working with larger datasets


## Conclusion
TODO
