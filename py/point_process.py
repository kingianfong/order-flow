# %%


from pathlib import Path
from typing import NamedTuple
import datetime

from jax import Array
from jax.flatten_util import ravel_pytree
from jax.scipy.optimize import minimize
from jax.scipy.stats import poisson
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import plot_power_law_approx


DATA_DIR = Path(__file__).parent.parent / 'data/preprocessed'


# %%


# raw data only has sometimes has multiple trades per timestamp
# this input dataset is grouped by millisecond
def load_data(sym: str) -> pl.DataFrame:
    df = (
        pl.scan_parquet(DATA_DIR)
        .filter(
            pl.col('sym') == pl.lit(sym),
            pl.col('date') >= datetime.date(2025, 12, 1),
        )
        .select(
            pl.col('time'),
            curr_count=pl.col('total_count'),
            elapsed_precise=pl.col('time').diff().cast(pl.Duration('ns')),
        )
        .with_columns(
            elapsed=(
                pl.col('elapsed_precise')
                .cast(float) * 1e-6  # milliseconds
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
INPUT_DF


# %%


class Dataset(NamedTuple):
    curr_count: Array
    elapsed: Array
    time_of_day: Array


DATASET = Dataset(
    curr_count=INPUT_DF['curr_count'].to_jax(),
    elapsed=INPUT_DF['elapsed'].to_jax(),
    time_of_day=INPUT_DF['hour'].to_jax(),
)


# %%


closed_form_rate = DATASET.curr_count.sum() / DATASET.elapsed.sum()
closed_form_rate


# %%


# assumes rate is effectively constant
def calc_loglik(rate: ArrayLike, dataset: Dataset) -> Array:
    return poisson.logpmf(k=dataset.curr_count, mu=rate*dataset.elapsed)


def constant_rate_loss(params: Array, dataset: Dataset):
    log_rate, = params
    rate = jnp.exp(log_rate)
    return -calc_loglik(rate, dataset).mean()


init_constant_rate_guess = jnp.array([jnp.log(closed_form_rate + 1e-1)])


constant_optim_result = minimize(
    constant_rate_loss,
    init_constant_rate_guess,
    args=(DATASET,),
    method='BFGS',
)


constant_rate = jnp.exp(constant_optim_result.x[0])
print(f'{closed_form_rate=:.8f}, {constant_rate=:.8f}')


# %%


class RbfConstants:
    n_centers: int = 24
    n_hours: int = 24
    centers: Array = jnp.linspace(0, n_centers, n_hours, endpoint=False)

    width_factor: float = 0.5
    sigma = width_factor * n_hours / n_centers
    inv_sigma_sq = 1.0 / (sigma**2)

    l2_reg: float = 0.01


class RbfRateParams(NamedTuple):
    log_base_rate: float
    # weights: Array = 0.05 * jnp.sin(jnp.arange(RbfConstants.n_centers))
    weights: Array = 0.1 * jax.random.normal(jax.random.PRNGKey(0),
                                             (RbfConstants.n_centers,),)


def calc_rbf_rate_log_factor(weights: Array, time_of_day: Array) -> Array:
    dist = jnp.abs(time_of_day[:, None] - RbfConstants.centers[None, :])
    dist = jnp.where(dist > RbfConstants.n_hours / 2, 24 - dist, dist)
    exponent = -0.5 * (dist**2) * RbfConstants.inv_sigma_sq
    basis = jnp.exp(exponent)
    log_factor = basis @ weights
    return log_factor


def plot_rbf_rate(log_base_rate: float, weights: Array) -> None:
    time_of_day = jnp.linspace(-2, 26, 500, endpoint=False)
    log_factor = calc_rbf_rate_log_factor(weights, time_of_day)
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


plot_rbf_rate(init_rbf_params.log_base_rate, init_rbf_params.weights)


# %%


flat_rbf_params, unflatten_rbf_params = ravel_pytree(init_rbf_params)


@jax.jit
def rbf_rate_loss(flat_params: Array,
                  dataset: Dataset):
    params: RbfRateParams = unflatten_rbf_params(flat_params)
    log_rate_factor = calc_rbf_rate_log_factor(
        params.weights,
        dataset.time_of_day
    )
    rate = jnp.exp(params.log_base_rate + log_rate_factor)
    nll = -calc_loglik(rate, dataset).mean()
    reg_penalty = jnp.sum(jnp.square(params.weights)) * RbfConstants.l2_reg
    return nll + reg_penalty


rbf_optim_result = minimize(
    rbf_rate_loss,
    flat_rbf_params,
    args=(DATASET,),
    method='BFGS',
)


rbf_optim_params = unflatten_rbf_params(rbf_optim_result.x)
plot_rbf_rate(rbf_optim_params.log_base_rate, rbf_optim_params.weights)


rbf_log_rate_factor = calc_rbf_rate_log_factor(
    rbf_optim_params.weights, DATASET.time_of_day)
rbf_rate = jnp.exp(rbf_optim_params.log_base_rate + rbf_log_rate_factor)


# %%


def logit(y) -> Array:
    return jnp.log(y) - jnp.log1p(-y)


# exponential decay kernel for now
class HawkesParams(NamedTuple):
    log_base_rate: float
    logit_norm: float = logit(jnp.array(0.85)).item()
    log_omega: float = -jnp.log(30 * 1_000).item()


class HawkesOutputs(NamedTuple):
    decay_factor: Array
    decayed_count: Array
    loglik: Array  # excludes events[t]
    rate: Array  # includes events[t]


@jax.jit
def calc_hawkes(params: HawkesParams, dataset: Dataset) -> HawkesOutputs:
    base_rate = jnp.exp(params.log_base_rate)
    norm = jax.nn.sigmoid(params.logit_norm)
    omega = jnp.exp(params.log_omega)

    decay_factors = jnp.exp(-omega * dataset.elapsed)

    def step(carry, x):
        decayed_count = carry
        count, elapsed, decay_factor = x

        decayed_count *= decay_factor
        loglik_rate = base_rate + norm * omega * decayed_count
        loglik = poisson.logpmf(k=count, mu=loglik_rate * elapsed)

        decayed_count += count
        forecast_rate = base_rate + norm * omega * decayed_count
        return decayed_count, (decayed_count, loglik, forecast_rate)

    xs = dataset.curr_count, dataset.elapsed, decay_factors
    _, y = jax.lax.scan(step, 0, xs)
    decayed_count, loglik, rate = y

    return HawkesOutputs(
        decay_factor=decay_factors,
        decayed_count=decayed_count,
        loglik=loglik,
        rate=rate,
    )


def plot_hawkes_rate(params: HawkesParams,
                     dataset: Dataset,
                     input_df: pl.DataFrame) -> None:
    outputs = calc_hawkes(params, dataset)
    display(params)

    base_rate = jnp.exp(params.log_base_rate).item()
    norm = jax.nn.sigmoid(params.logit_norm).item()
    omega = jnp.exp(params.log_omega).item()

    print(f'{base_rate=}, {norm=}, {omega=}')
    df = (
        input_df
        .with_columns(
            decay_factor=np.asarray(outputs.decay_factor),
            decayed_count=np.asarray(outputs.decayed_count),
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


init_hawkes_params = HawkesParams(
    log_base_rate=float(jnp.log(constant_rate)),
)


plot_hawkes_rate(init_hawkes_params, DATASET, INPUT_DF)


# %%


flat_hawkes_params, unflatten_hawkes_params = ravel_pytree(init_hawkes_params)


@jax.jit
def hawkes_loss(flat_params: Array, dataset: Dataset):
    hawkes_params = unflatten_hawkes_params(flat_params)
    output = calc_hawkes(hawkes_params, dataset)
    return -output.loglik.mean()


hawkes_optim_result = minimize(
    hawkes_loss,
    flat_hawkes_params,
    args=(DATASET,),
    method='BFGS',
)


hawkes_optim_params = unflatten_hawkes_params(hawkes_optim_result.x)
hawkes_outputs = calc_hawkes(hawkes_optim_params, DATASET)
plot_hawkes_rate(hawkes_optim_params, DATASET, INPUT_DF)


# %%


# exponential decay kernel for now
class RbfHawkesParams(NamedTuple):
    log_base_rate: float
    logit_norm: float = logit(jnp.array(0.85)).item()
    log_omega: float = -jnp.log(30 * 1_000).item()
    rbf_weights: Array = 0.1 * jax.random.normal(jax.random.PRNGKey(0),
                                                 (RbfConstants.n_centers,),)


class RbfHawkesOutputs(NamedTuple):
    base_rate: Array  # from rbf
    decay_factor: Array
    decayed_count: Array
    loglik: Array  # excludes events[t]
    rate: Array  # includes events[t]


@jax.jit
def calc_rbf_hawkes(params: RbfHawkesParams,
                    dataset: Dataset) -> RbfHawkesOutputs:
    log_factor = calc_rbf_rate_log_factor(
        params.rbf_weights, dataset.time_of_day)
    base_rate = jnp.exp(params.log_base_rate + log_factor)

    norm = jax.nn.sigmoid(params.logit_norm)
    omega = jnp.exp(params.log_omega)
    decay_factors = jnp.exp(-omega * dataset.elapsed)

    def step(carry, x):
        decayed_count = carry
        base_rate, count, elapsed, decay_factor = x

        decayed_count *= decay_factor
        loglik_rate = base_rate + norm * omega * decayed_count
        loglik = poisson.logpmf(k=count, mu=loglik_rate * elapsed)

        decayed_count += count
        forecast_rate = base_rate + norm * omega * decayed_count
        return decayed_count, (decayed_count, loglik, forecast_rate)

    xs = base_rate, dataset.curr_count, dataset.elapsed, decay_factors
    _, y = jax.lax.scan(step, 0, xs)
    decayed_count, loglik, rate = y

    return RbfHawkesOutputs(
        base_rate=base_rate,
        decay_factor=decay_factors,
        decayed_count=decayed_count,
        loglik=loglik,
        rate=rate,
    )


def plot_rbf_hawkes(params: RbfHawkesParams,
                    dataset: Dataset,
                    input_df: pl.DataFrame) -> None:
    outputs = calc_rbf_hawkes(params, dataset)
    display(params)

    norm = jax.nn.sigmoid(params.logit_norm).item()
    omega = jnp.exp(params.log_omega).item()

    print(f'{norm=}, {omega=}')
    df = (
        input_df
        .with_columns(
            decay_factor=np.asarray(outputs.decay_factor),
            decayed_count=np.asarray(outputs.decayed_count),
            loglik=np.asarray(outputs.loglik),
            rate=np.asarray(outputs.rate),
        )
    )
    display(df)

    plot_rbf_rate(log_base_rate=params.log_base_rate,
                  weights=params.rbf_weights)
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


init_rbf_hawkes = RbfHawkesParams(
    log_base_rate=hawkes_optim_params.log_base_rate,
    logit_norm=hawkes_optim_params.logit_norm,
    log_omega=hawkes_optim_params.log_omega,
    rbf_weights=jnp.zeros(RbfConstants.n_centers)
)


plot_rbf_hawkes(init_rbf_hawkes, DATASET, INPUT_DF)


# %%


flat_rbf_hawkes_params, unflatten_rbf_hawkes_params = ravel_pytree(
    init_rbf_hawkes)


@jax.jit
def rbf_hawkes_loss(flat_params: Array, dataset: Dataset):
    prams: RbfHawkesParams = unflatten_rbf_hawkes_params(flat_params)
    output = calc_rbf_hawkes(prams, dataset)

    nll = -output.loglik.mean()
    reg_penalty = jnp.sum(jnp.square(prams.rbf_weights)) * RbfConstants.l2_reg
    return nll + reg_penalty


hawkes_optim_result = minimize(
    rbf_hawkes_loss,
    flat_rbf_hawkes_params,
    args=(DATASET,),
    method='BFGS',
)


rbf_hawkes_optim_params = unflatten_rbf_hawkes_params(hawkes_optim_result.x)
rbf_hawkes_outputs = calc_rbf_hawkes(rbf_hawkes_optim_params, DATASET)
plot_rbf_hawkes(rbf_hawkes_optim_params, DATASET, INPUT_DF)


# %%


result_df = (
    INPUT_DF
    .with_columns(
        pl.col('time').dt.truncate('1h'),
        constant_loglik=np.asarray(calc_loglik(constant_rate, DATASET)),
        rbf_loglik=np.asarray(calc_loglik(rbf_rate, DATASET)),
        hawkes_loglik=np.asarray(hawkes_outputs.loglik),
        rbf_hawkes_loglik=np.asarray(rbf_hawkes_outputs.loglik),
    )
    .group_by('time', maintain_order=True)
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


result_df


# %%


result_df.select(pl.selectors.ends_with('_loglik').sum())


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
        # 'constant_loglik',
        # 'rbf_loglik',
        'hawkes_loglik',
        'rbf_hawkes_loglik',
    ]]
    .plot(alpha=0.6)
)
plt.show()


# %%
