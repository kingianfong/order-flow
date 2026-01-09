# %%


from pathlib import Path
from typing import NamedTuple
import datetime

from IPython.display import display
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.scipy.optimize import minimize
from jax.scipy.special import logit
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


DATA_DIR = Path(__file__).parent.parent / 'data/preprocessed'


# %%


# raw data only has sometimes has multiple trades per timestamp
# this input dataset is grouped by millisecond
def load_data(sym: str) -> pl.DataFrame:
    df = (
        pl.scan_parquet(DATA_DIR)
        .filter(
            pl.col('sym') == pl.lit(sym),
            pl.col('date') >= datetime.date(2025, 12, 11),
            pl.col('date') <= datetime.date(2025, 12, 15),
        )
        .select(
            pl.col('time'),
            curr_count=pl.col('total_count'),
            elapsed_precise=pl.col('time').diff().cast(pl.Duration('ns')),
        )
        .with_columns(
            elapsed=(
                pl.col('elapsed_precise')
                .cast(pl.Float32) * 1e-6  # milliseconds
            ),
            hour=(
                pl.col('time')
                - pl.col('time').dt.truncate('1d')
            ).dt.total_hours(fractional=True),
        )
        .filter(pl.col('elapsed').is_not_null())
        .collect()
    )
    assert (df['curr_count'] > 0).all()
    assert (df['elapsed'] > 0).all()
    assert (df['hour'] >= 0).all()
    assert df['time'].is_unique().all()
    assert df['time'].is_sorted()
    return df


INPUT_DF = load_data('BTCUSDT')
display(INPUT_DF)


# %%


# for multipliers based on time-of-day
class RbfConstants:
    n_centers: int = 24
    n_hours: int = 24
    centers: Array = jnp.linspace(0, n_centers, n_hours, endpoint=False)

    width_factor: float = 0.5
    sigma = width_factor * n_hours / n_centers
    inv_sigma_sq = 1.0 / (sigma**2)

    l2_reg: float = 0.01


def calc_rbf_basis(time_of_day: Array) -> Array:
    dist = jnp.abs(time_of_day[:, None] - RbfConstants.centers[None, :])
    dist = jnp.where(dist > RbfConstants.n_hours / 2, 24 - dist, dist)
    exponent = -0.5 * (dist**2) * RbfConstants.inv_sigma_sq
    basis = jnp.exp(exponent)
    return basis


class Dataset(NamedTuple):
    curr_count: Array
    elapsed: Array
    rbf_basis: Array


DATASET = Dataset(
    curr_count=INPUT_DF['curr_count'].cast(pl.Float32).to_jax(),
    elapsed=INPUT_DF['elapsed'].to_jax(),
    rbf_basis=calc_rbf_basis(INPUT_DF['hour'].to_jax()),
)


# %%


closed_form_rate = DATASET.curr_count.sum() / DATASET.elapsed.sum()
display(closed_form_rate, jnp.log(closed_form_rate).item())


# %%


# wrapper over jax.scipy.minimize
# to facilitate replacement with other optimisers
def run_optim(init, loss_fn, args):
    flat_init, unflatten = ravel_pytree(init)

    def flat_wrapper(flat_params, args):
        params = unflatten(flat_params)
        return loss_fn(params, args)

    optim_result = minimize(
        flat_wrapper,
        flat_init,
        args=args,
        method='bfgs',  # jax 0.7.2 uses zoom line search
        options=dict(
            maxiter=20,
        )
    )
    if not optim_result.success:
        print('warning: optimisation failed')
        result_dict = optim_result._asdict()
        scalars = {k: v for k, v in result_dict.items() if jnp.isscalar(v)}
        df = pl.DataFrame(
            data=dict(
                key=scalars.keys(),
                value=scalars.values(),
            )
        )
        display(df)

    return unflatten(optim_result.x)


def constant_loglik(rate: ArrayLike, dataset: Dataset) -> Array:
    # assume constant rate throughout interval
    return dataset.curr_count * jnp.log(rate) - rate * dataset.elapsed


@jax.jit
def constant_rate_loss(params: Array, dataset: Dataset):
    log_rate, = params
    rate = jnp.exp(log_rate)
    return -constant_loglik(rate, dataset).mean()


log_constant_rate_result = run_optim(
    init=(jnp.log(closed_form_rate + 1e-1), ),
    loss_fn=constant_rate_loss,
    args=(DATASET,),
)


constant_rate = jnp.exp(log_constant_rate_result[0]).item()

print(f'{closed_form_rate=:.8f}, {constant_rate=:.8f}')


# %%


class ModelOutput(NamedTuple):
    loglik: Array  # loglik of (no event since prev t) + (events at t)
    rate: Array  # used for predictions after observing events at t


class RbfRateParams(NamedTuple):
    log_base_rate: float
    # weights: Array = 0.05 * jnp.sin(jnp.arange(RbfConstants.n_centers))
    weights: Array = 0.1 * jax.random.normal(jax.random.PRNGKey(0),
                                             (RbfConstants.n_centers,),)


def calc_rbf(params: RbfRateParams, dataset: Dataset) -> ModelOutput:
    log_rate_factor = dataset.rbf_basis @ params.weights
    rate = jnp.exp(params.log_base_rate + log_rate_factor)
    loglik = \
        dataset.curr_count * jnp.log(rate) \
        - rate * dataset.elapsed  # assume constant rate throughout interval
    return ModelOutput(loglik=loglik, rate=rate)


def plot_rbf(log_base_rate: float, weights: Array) -> None:
    time_of_day = jnp.linspace(-2, 26, 500, endpoint=False)
    log_factor = calc_rbf_basis(time_of_day) @ weights
    base_rate = jnp.exp(log_base_rate).item()
    rate = jnp.exp(log_base_rate + log_factor)

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(time_of_day, rate)
    ax1.axhline(base_rate, label=f'{base_rate=:.6f}',
                c='g', alpha=0.4, linestyle='-')
    ax1.axvline(0, c='k', alpha=0.2, linestyle='--')
    ax1.axvline(24, c='k', alpha=0.2, linestyle='--')
    ax1.legend(loc='upper left')

    for ctr, weight in zip(RbfConstants.centers, weights, strict=True):
        dist = jnp.abs(time_of_day[:, None] - ctr)
        dist = jnp.where(dist > RbfConstants.n_hours / 2, 24 - dist, dist)
        exponent = -0.5 * (dist**2) * RbfConstants.inv_sigma_sq
        basis = jnp.exp(exponent)
        ax2.plot(time_of_day, weight * basis)

    plt.show()


init_rbf_params = RbfRateParams(
    log_base_rate=float(jnp.log(constant_rate)),
)


plot_rbf(init_rbf_params.log_base_rate, init_rbf_params.weights)


# %%


@jax.jit
def rbf_loss(params: RbfRateParams, dataset: Dataset):
    output = calc_rbf(params, dataset)
    reg_penalty = jnp.sum(jnp.square(params.weights)) * RbfConstants.l2_reg
    return -output.loglik.mean() + reg_penalty


rbf_optim_params = run_optim(
    init_rbf_params,
    rbf_loss,
    args=(DATASET,),
)
rbf_outputs = calc_rbf(rbf_optim_params, DATASET)
plot_rbf(rbf_optim_params.log_base_rate, rbf_optim_params.weights)


# %%


# exponential decay kernel for now
class HawkesParams(NamedTuple):
    log_base_rate: float
    logit_norm: float = logit(0.9).item()
    log_omega: float = jnp.log(1).item()  # log(1 / avg_life_ms)


def calc_hawkes_baseline(params: HawkesParams, dataset: Dataset) -> ModelOutput:
    assert jnp.all(dataset.curr_count > 0.0)
    assert jnp.all(dataset.elapsed > 0.0)

    base_rate = jnp.exp(params.log_base_rate)
    norm = jax.nn.sigmoid(params.logit_norm)
    omega = jnp.exp(params.log_omega)

    def step(carry, x):
        decayed_count = carry
        count, elapsed = x

        # rate(t) = base_rate + norm * omega * decayed_count

        # loglik =
        #   sum(log(rate)) at each event
        #   - integral(rate) over duration

        # loglik of interval that just passed with no events
        integral_over_interval = -jnp.expm1(-omega * elapsed)
        interval_term = \
            base_rate * elapsed  \
            + norm * decayed_count * integral_over_interval

        # loglik of events at current timestamp
        decay_factor = jnp.exp(-omega * elapsed)
        decayed_count *= decay_factor

        # assume events happen instantaneously without self-excitation
        # 1. jittering is unacceptable as it increases data size too much
        # 2. an attempt using logarithm of rising factorial aka Pochhammer
        # symbol has resulted in omega blowing up due to

        event_rate = base_rate + norm * omega * decayed_count
        event_term = count * jnp.log(event_rate)
        loglik = event_term - interval_term

        decayed_count += count
        forecast_rate = base_rate + norm * omega * decayed_count
        return decayed_count, (loglik, forecast_rate)

    xs = dataset.curr_count, dataset.elapsed
    _, (loglik, rate) = jax.lax.scan(step, 0, xs)

    return ModelOutput(
        loglik=loglik,
        rate=rate,
    )


@jax.jit
def calculate_decayed_counts(decay_factors: Array, counts: Array) -> Array:
    def combine(prefix, step):
        # f_left(x)  = a_left * x + b_left
        # f_right(x) = a_right * x + b_right
        # f_right(f_left(x)) = a_right * (a_left * x + b_left) + b_right
        #                    = (a_right * a_left) * x + (a_right * b_left + b_right)
        decay_left, count_left = prefix
        decay_right, count_right = step
        combined_decay = decay_right * decay_left
        combined_count = decay_right * count_left + count_right
        return combined_decay, combined_count

    if counts.ndim == 1 and decay_factors.ndim > 1:
        counts = counts[:, jnp.newaxis]
        counts = jnp.broadcast_to(counts, decay_factors.shape)

    elems = decay_factors, counts
    _, decayed_counts = jax.lax.associative_scan(combine, elems)
    return decayed_counts


def test_calculate_decayed_counts():
    def linear_scan_step(carry, xs):
        decayed_count = carry
        decay_factor, count = xs
        decayed_count *= decay_factor
        decayed_count += count
        return decayed_count, decayed_count

    n = 1_000
    for i in range(10):
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(i), 4)

        # 1d
        normal = 10 * jax.random.normal(k1, (n, ))
        decay_factors = jax.nn.sigmoid(normal)
        counts = 1.0 + jax.random.randint(k2, (n, ), 0, 5)

        init = 0
        xs = decay_factors, counts
        _, expected = jax.lax.scan(linear_scan_step, init, xs)

        actual = calculate_decayed_counts(decay_factors, counts)
        assert jnp.allclose(actual, expected)

        # multidim
        m = 5
        normal = 10 * jax.random.normal(k3, (n, m))
        decay_factors = jax.nn.sigmoid(normal)
        counts = 1.0 + jax.random.randint(k4, (n, ), 0, 5)

        init = jnp.zeros((m,))
        xs = decay_factors, counts
        _, expected = jax.lax.scan(linear_scan_step, init, xs)

        actual = calculate_decayed_counts(decay_factors, counts)
        assert jnp.allclose(actual, expected)


test_calculate_decayed_counts()


# %%


@jax.jit
def calc_hawkes(params: HawkesParams, dataset: Dataset) -> ModelOutput:
    base_rate = jnp.exp(params.log_base_rate)
    norm = jax.nn.sigmoid(params.logit_norm)
    omega = jnp.exp(params.log_omega)
    decay_factors = jnp.exp(-omega * dataset.elapsed)
    decayed_count = calculate_decayed_counts(decay_factors, dataset.curr_count)

    # loglik of interval that just passed with no events
    prev_decayed_count = jnp.roll(decayed_count, 1).at[0].set(0.0)
    integral_over_interval = -jnp.expm1(-omega * dataset.elapsed)
    interval_term = \
        dataset.elapsed * base_rate \
        + norm * prev_decayed_count * integral_over_interval

    # loglik of event(s) at current timestamp
    curr_minus_count = prev_decayed_count * decay_factors
    event_rate = base_rate + norm * omega * curr_minus_count
    event_term = dataset.curr_count * jnp.log(event_rate)

    forecast_rate = base_rate + norm * omega * decayed_count
    return ModelOutput(
        loglik=event_term - interval_term,
        rate=forecast_rate,
    )


def plot_hawkes(params: HawkesParams,
                dataset: Dataset,
                input_df: pl.DataFrame) -> None:
    baseline_outputs = calc_hawkes_baseline(params, dataset)

    outputs = calc_hawkes(params, dataset)
    assert jnp.allclose(outputs.loglik, baseline_outputs.loglik, atol=1e-4)
    assert jnp.allclose(outputs.rate, baseline_outputs.rate, rtol=1e-4)

    display(params)

    base_rate = jnp.exp(params.log_base_rate).item()
    norm = jax.nn.sigmoid(params.logit_norm).item()
    omega = jnp.exp(params.log_omega).item()
    avg_life_seconds = 1000 / omega

    print(f'{base_rate=}, {norm=}, {omega=}')
    print(f'{avg_life_seconds=}')

    df = (
        input_df
        .with_columns(
            loglik=np.asarray(outputs.loglik),
            rate=np.asarray(outputs.rate),
            baseline_loglik=np.asarray(baseline_outputs.loglik),
            baseline_rate=np.asarray(baseline_outputs.rate),
        )
    )
    display(df)
    display(
        df
        .with_columns(
            baseline_minus_loglik=pl.col('baseline_loglik') - pl.col('loglik'),
        )
        .with_columns(
            abs_diff=pl.col('baseline_minus_loglik').abs(),
            rel_diff=(
                pl.col('baseline_minus_loglik') /
                pl.col('baseline_loglik').abs()
            ),
        )
        .filter(
            pl.col('rel_diff').abs() > 0.01
        )
        .sort('rel_diff')
    )
    subset = (
        df
        .filter(
            pl.col('time') >= datetime.datetime(2025, 12, 12, hour=12),
            pl.col('time') <= datetime.datetime(2025, 12, 13),
        )
    )
    _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.scatter(subset['time'], subset[['curr_count']], alpha=0.05, marker='.')
    ax2.plot(subset['time'], subset[['rate']])
    ax3.plot(subset['time'], subset[['loglik']])
    plt.tight_layout()
    plt.show()


init_hawkes_params = HawkesParams(
    log_base_rate=jnp.log(constant_rate).item(),
)


plot_hawkes(init_hawkes_params, DATASET, INPUT_DF)


# %%


@jax.jit
def hawkes_loss(params: HawkesParams, dataset: Dataset):
    output = calc_hawkes(params, dataset)
    return -output.loglik.mean()


hawkes_optim_params = run_optim(
    init_hawkes_params,
    hawkes_loss,
    args=(DATASET,),
)

hawkes_outputs = calc_hawkes(hawkes_optim_params, DATASET)
plot_hawkes(hawkes_optim_params, DATASET, INPUT_DF)


# %%


# exponential decay kernel for now
class RbfHawkesParams(NamedTuple):
    log_base_rate: float
    logit_norm: float
    log_omega: float
    weights: Array


@jax.jit
def calc_rbf_hawkes(params: RbfHawkesParams, dataset: Dataset) -> ModelOutput:
    log_factor = dataset.rbf_basis @ params.weights
    base_rate = jnp.exp(params.log_base_rate + log_factor)
    norm = jax.nn.sigmoid(params.logit_norm)
    omega = jnp.exp(params.log_omega)
    decay_factors = jnp.exp(-omega * dataset.elapsed)
    decayed_count = calculate_decayed_counts(decay_factors, dataset.curr_count)

    # loglik of interval that just passed with no events
    prev_decayed_count = jnp.roll(decayed_count, 1).at[0].set(0.0)
    integral_over_interval = -jnp.expm1(-omega * dataset.elapsed)
    interval_term = \
        dataset.elapsed * base_rate \
        + norm * prev_decayed_count * integral_over_interval

    # loglik of event(s) at current timestamp
    curr_minus_count = prev_decayed_count * decay_factors
    event_rate = base_rate + norm * omega * curr_minus_count
    event_term = dataset.curr_count * jnp.log(event_rate)

    forecast_rate = base_rate + norm * omega * decayed_count
    return ModelOutput(
        loglik=event_term - interval_term,
        rate=forecast_rate,
    )


def plot_rbf_hawkes(params: RbfHawkesParams,
                    dataset: Dataset,
                    input_df: pl.DataFrame) -> None:
    plot_hawkes(params=params,   # type: ignore
                dataset=dataset,
                input_df=input_df)
    plot_rbf(log_base_rate=params.log_base_rate,
             weights=params.weights)


init_rbf_hawkes = RbfHawkesParams(
    log_base_rate=hawkes_optim_params.log_base_rate,
    logit_norm=hawkes_optim_params.logit_norm,
    log_omega=hawkes_optim_params.log_omega,
    weights=rbf_optim_params.weights,
)


plot_rbf_hawkes(init_rbf_hawkes, DATASET, INPUT_DF)


# %%


@jax.jit
def rbf_hawkes_loss(params: RbfHawkesParams, dataset: Dataset):
    output = calc_rbf_hawkes(params, dataset)
    reg_penalty = (
        jnp.sum(jnp.square(params.weights)) * RbfConstants.l2_reg
    )
    return -output.loglik.mean() + reg_penalty


rbf_hawkes_optim_params = run_optim(
    init_rbf_hawkes,
    rbf_hawkes_loss,
    args=(DATASET,),
)
rbf_hawkes_outputs = calc_rbf_hawkes(rbf_hawkes_optim_params, DATASET)
plot_rbf_hawkes(rbf_hawkes_optim_params, DATASET, INPUT_DF)


# %%


class PowerLawApproxParams(NamedTuple):
    weights: Array
    rates: Array


def power_law_decay_approx_params(omega: ArrayLike, beta: ArrayLike,
                                  max_history_duration_ms: ArrayLike,
                                  n_exponentials: int) -> PowerLawApproxParams:
    # to approximate lomax decay
    #   g(t) = (1 + omega * t) ^ -(1 + beta)
    #        = E[ h(X; t) ]
    # where
    #   X ~ Gamma(1+beta, scale=omega)
    #       -> f(x) = k * x^{beta} * exp{-x / omega}
    #   h(x; t) = exp{-x * t}

    min_history_duration_ms = 1e-3  # one microsecond

    # decay rate for each exponential
    rates = jnp.geomspace(
        1 / max_history_duration_ms,
        1 / min_history_duration_ms,
        n_exponentials,
    )
    # quadrature weights:
    # approximate integral[ f(x)   * exp(-x   * t)   dx        ]
    # as               sum[ f(x_i) * exp(-x_i * t) * delta_x_i ]
    # Since x_i is geomspaced, delta_x_i is proportional to x_i.
    log_pdf = jax.scipy.stats.gamma.logpdf(rates, a=1 + beta, scale=omega)
    unnorm_log_weights = log_pdf + jnp.log(rates)
    weights = jax.nn.softmax(unnorm_log_weights)

    return PowerLawApproxParams(
        weights=jnp.asarray(weights),
        rates=jnp.asarray(rates),
    )


def plot_power_law_decay():
    def calc_exact(t: Array, omega: float, beta: float):
        return (1.0 + omega * t) ** -(1.0 + beta)

    # done sequentially to ensure markovian nature holds
    def calc_approx(t: Array, params: PowerLawApproxParams) -> Array:
        elapsed = jnp.concat([jnp.zeros(1), jnp.diff(t)])
        counts = jnp.zeros_like(elapsed).at[0].set(1)
        outer = jnp.outer(elapsed, params.rates)
        decay_rates = jnp.exp(-outer)
        decayed_counts = calculate_decayed_counts(decay_rates, counts)
        return decayed_counts @ params.weights

    n = 10_000
    one_micro = 1e-3
    one_hour = 60 * 60 * 1e3  # 8 orders of magnitude
    min_t_ms = one_micro
    max_t_ms = one_hour * 10
    t_geom = jnp.geomspace(min_t_ms, max_t_ms, n)
    max_history_duration = one_hour
    n_exponentials = 16  # 2x orders of magnitude

    omegas = 100, 1.0, 0.01
    betas = 1.0, 2.0, 3.0  # The 'beta' in the kernel definition

    f1, axes1 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(9, 9))
    f1.suptitle('Lomax Kernel')
    f2, axes2 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(9, 9))
    f2.suptitle('linear y')
    f3, axes3 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(9, 9))
    f3.suptitle('absolute error')
    f4, axes4 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(9, 9))
    f4.suptitle('relative error')

    for r, omega in enumerate(omegas):
        for c, beta in enumerate(betas):

            kernel_params = power_law_decay_approx_params(
                omega=omega,
                beta=beta,
                max_history_duration_ms=max_history_duration,
                n_exponentials=n_exponentials,
            )

            exact = calc_exact(t_geom, omega=omega, beta=beta)
            approx = calc_approx(t_geom, kernel_params)

            ax1 = axes1[r, c]
            ax1.loglog(t_geom, exact, 'k-', label='exact', lw=2)
            ax1.loglog(t_geom, approx, 'r--', label='approx', lw=2)
            ax1.set_title(f"$\\omega={omega}, \\beta={beta}$")
            ax1.grid(True, which="both", ls="-", alpha=0.2)
            ax1.legend(loc='lower left')

            ax2 = axes2[r, c]
            ax2.plot(t_geom, exact, 'k-', label='exact', lw=2)
            ax2.plot(t_geom, approx, 'r--', label='approx', lw=2)
            ax2.set_xscale('log')
            ax2.set_title(f"$\\omega={omega}, \\beta={beta}$")
            ax2.grid(True, which="both", ls="-", alpha=0.2)
            ax2.legend(loc='upper right')

            ax3 = axes3[r, c]
            ax3.loglog(t_geom, abs(approx - exact), 'k-', lw=2)
            ax3.set_title(f"$\\omega={omega}, \\beta={beta}$")
            ax3.grid(True, which="both", ls="-", alpha=0.2)

            ax4 = axes4[r, c]
            ax4.loglog(t_geom, abs(approx - exact) / exact, 'k-', lw=2)
            ax4.set_title(f"$\\omega={omega}, \\beta={beta}$")
            ax4.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_power_law_decay()


# %%


class PowerLawHawkesParams(NamedTuple):
    log_base_rate: float
    logit_norm: float
    log_omega: float
    log_beta: float = jnp.log(0.5).item()
    log_max_history_duration: float = jnp.log(60 * 60 * 1_000).item()


@jax.jit
def calc_power_law_hawkes(params: PowerLawHawkesParams, dataset: Dataset) -> ModelOutput:
    n_exponentials = 10  # not differentiable

    base_rate = jnp.exp(params.log_base_rate)
    norm = jax.nn.sigmoid(params.logit_norm)
    omega = jnp.exp(params.log_omega)
    beta = jnp.exp(params.log_beta)
    max_history_duration = jnp.exp(params.log_max_history_duration)

    kernel_factor = norm * omega * beta

    approx_params = power_law_decay_approx_params(
        omega=omega,
        beta=beta,
        max_history_duration_ms=max_history_duration,
        n_exponentials=n_exponentials,
    )
    weights, rates = approx_params.weights, approx_params.rates

    decay_factor_exponents = -jnp.outer(dataset.elapsed, rates)
    decay_factors = jnp.exp(decay_factor_exponents)
    decayed_count = calculate_decayed_counts(decay_factors, dataset.curr_count)

    # loglik of interval that just passed with no events
    prev_decayed_count = jnp.roll(decayed_count, 1, axis=0).at[0, :].set(0.0)
    term_per_exp = -jnp.expm1(decay_factor_exponents) / rates
    integral_history = (prev_decayed_count * term_per_exp) @ weights

    interval_term = \
        (dataset.elapsed * base_rate) \
        + (kernel_factor * integral_history)

    # loglik of event(s) at current timestamp
    curr_minus_count = prev_decayed_count * decay_factors
    event_rate = base_rate + kernel_factor * (curr_minus_count @ weights)
    event_term = dataset.curr_count * jnp.log(event_rate)

    forecast_rate = base_rate + kernel_factor * (decayed_count @ weights)

    return ModelOutput(
        loglik=event_term - interval_term,
        rate=forecast_rate,
    )


def plot_power_law_hawkes_rate(params: PowerLawHawkesParams,
                               dataset: Dataset,
                               input_df: pl.DataFrame) -> None:
    outputs = calc_power_law_hawkes(params, dataset)
    display(params)

    base_rate = jnp.exp(params.log_base_rate).item()
    norm = jax.nn.sigmoid(params.logit_norm).item()
    omega = jnp.exp(params.log_omega).item()

    print(f'{base_rate=}, {norm=}, {omega=}')
    df = (
        input_df
        .with_columns(
            loglik=np.asarray(outputs.loglik),
            rate=np.asarray(outputs.rate),
        )
    )
    display(df)
    subset = (
        df
        .filter(
            pl.col('time') >= datetime.datetime(2025, 12, 12, hour=12),
            pl.col('time') <= datetime.datetime(2025, 12, 13),
        )
    )
    _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.scatter(subset['time'], subset[['curr_count']], alpha=0.05, marker='.')
    ax2.plot(subset['time'], subset[['rate']])
    ax3.plot(subset['time'], subset[['loglik']])
    plt.tight_layout()
    plt.show()


init_pl_hawkes_params = PowerLawHawkesParams(
    log_base_rate=hawkes_optim_params.log_base_rate,
    logit_norm=hawkes_optim_params.logit_norm,
    log_omega=hawkes_optim_params.log_omega,
)


plot_power_law_hawkes_rate(init_pl_hawkes_params, DATASET, INPUT_DF)

# %%


@jax.jit
def power_law_hawkes_loss(params: PowerLawHawkesParams, dataset: Dataset):
    output = calc_power_law_hawkes(params, dataset)
    return -output.loglik.mean()


power_law_hawkes_optim_params = run_optim(
    init_pl_hawkes_params,
    power_law_hawkes_loss,
    args=(DATASET,),
)
power_law_hawkes_outputs = calc_power_law_hawkes(
    power_law_hawkes_optim_params, DATASET)
plot_power_law_hawkes_rate(power_law_hawkes_optim_params, DATASET, INPUT_DF)


# %%
with_logliks = (
    INPUT_DF
    .with_columns(
        constant_loglik=np.asarray(constant_loglik(constant_rate, DATASET)),
        rbf_loglik=np.asarray(rbf_outputs.loglik),
        hawkes_loglik=np.asarray(hawkes_outputs.loglik),
        rbf_hawkes_loglik=np.asarray(rbf_hawkes_outputs.loglik),
        pl_hawkes_loglik=np.asarray(power_law_hawkes_outputs.loglik),
    )
)


result_df = (
    with_logliks
    .group_by(pl.col('time').dt.truncate('1h'), maintain_order=True)
    .sum()
    .with_columns(
        rbf_improvement=(
            pl.col('rbf_loglik')
            - pl.col('constant_loglik')
        ),
        hawkes_improvement=(
            pl.col('hawkes_loglik')
            - pl.col('rbf_loglik')
        ),
    )
)


display(result_df)


# %%


display(
    with_logliks.select(pl.selectors.ends_with('_loglik')).sum(),
    with_logliks.select(pl.selectors.ends_with('_loglik')).mean(),
)


# %%


(
    result_df
    .group_by(pl.col('time').dt.time().alias('time')).sum()
    .sort('hawkes_improvement')
)


# %%


(
    result_df
    .to_pandas()
    .set_index('time')
    [[
        'constant_loglik',
        'rbf_loglik',
        'hawkes_loglik',
        'rbf_hawkes_loglik',
        'pl_hawkes_loglik',
    ]]
    .plot(alpha=0.6)
)
plt.show()


# %%
