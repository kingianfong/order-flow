# Order Flow
<!-- TODO: include these charts:
1. rbf hawkes hessian
1. rbf hawkes basis plots
1. model comparison -->

This project models high-frequency BTCUSDT trade arrivals using parametric point-processes (Poisson and Hawkes), with intraday seasonality and long-memory kernels, and evaluates them via statistical diagnostics.

## Highlights
### Iterative Model Development
1. `Constant intensity` with Poisson baseline
1. `Daily seasonality` with periodic radial basis functions
1. `Self-excitation` with Hawkes process
1. `Long memory` with power-law decay kernel

### Statistical Diagnostics
1. `Inverse Hessian Matrix` helps detect identifiability and colinearity between parameters
1. `Godambe Information Matrix` provides standard errors which are robust to model misspecification

### Engineering
1. `Computational complexity reduction`
    1. of power-law (Lomax) decay calculation from $O(n^2)$ time to $O(n)$ using sum-of-exponentials approximation
    1. of exponential decay calculation from $O(n)$ span to $O(\log n)$ using parallel prefix scan on a linear recurrence (`jax.lax.associatve_scan`)
1. `Property-based testing` ensures power-law approximation remains accurate over valid inputs (`hypothesis` library)

## Non-Objectives
1. `Alpha generation`: This is a statistical modelling exercise, not a production trading strategy.
1. `Price prediction`: This project only models the occurrence of trades, not price direction.


## Results
TODO
<!-- 1. table of logliks
1. a diagnostic plot
1. an intraday intensity plot -->


## Data
This project uses [Binance UM futures trade data](https://data.binance.vision/data/futures/um/daily).
* instrument: `BTCUSDT` USDM futures
* train dates: `2025-09-01` to `2025-09-30`
* validation dates: `2025-10-01` to `2025-10-15`
* timestamp resolution: 1 millisecond; multiple trades may share the same timestamp

## Assumptions
Order arrival follows a point process with the following log likelihood:

$$
\log L(\theta) = \sum_i \log \lambda_\theta( t_i ) - \int_0^T \lambda_\theta(s) ds
$$

where
* $\lambda_\theta(t)$ is the conditional intensity given the event history up to time $t$.

`Multiple events per timestamp` are assumed to self-excite for Hawkes process models. This is included in the log likelihood using the Rising Factorial (Pochhammer) polynomial.

If $n$ events occur in the same timestamp, and the intensity jumps by $J$ after each event, their total contribution is

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

This formulation allows the likelihood to be calculated exactly even when multiple events arrive at the same timestamp, and avoids discarding information or artificially jittering timestamps to increase the data size.

## Project Structure
* `point_process.py`: Data loading, model defintions, optimisation, diagnostic plots.
* `download_trades.py`: Multithreaded logic for downloading Binance UM Futures tick data and saving them to compressed parquet files with delta encoding.
* `decayed_counts.py`: Parallel prefix-scan implementation of exponential Hawkes state recursion.
* `power_law_approx.py`: Approximates Power-Law (Lomax) kernel decay using sum of exponential decays to reduce computation from $O(n^2)$ time to $O(n)$. Includes property-based testing with `hypothesis`

## Known Limitations
1. `Data Granularity`: It is unclear whether each record corresponds to a single aggressive trade or a passive fill. The dataset is not explicitly documented. From the documentation of a [similar dataset](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams#trade-streams) and visualisations, it seems that each sample in the dataset corresponds to a single passive order getting filled, rather than a single aggressive order.
1. `Regularisation Tuning`: Penalties were chosen heuristically by observing the convergence, identifiability and misspecficiation diagnostics.
1. `Optimisation`: The L-BFGS training loop forces device-host synchronisation and may be problematic if number of parameters increase.

## Future Work
1. Apply model to a trading use case
1. Tune regularisation via cross-validation or empirical Bayes
1. Extend to
    1. multivariate Hawkes (buy, sell)
    1. marked processes (volume, notional)
    1. nonlinear impact kernels
