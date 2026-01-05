# %%


from pathlib import Path
from typing import Callable, NamedTuple
import datetime

from jax import Array
from jax.flatten_util import ravel_pytree
from jax.scipy.optimize import minimize
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


DATA_DIR = Path(__file__).parent.parent / 'data/preprocessed'


# %%


def load_data(sym: str) -> pl.DataFrame:
    df = (
        pl.scan_parquet(DATA_DIR)
        .filter(
            pl.col('sym') == pl.lit(sym),
            pl.col('date') >= datetime.date(2025, 12, 1),
        )
        .select('time', count=pl.col('total_count'))
        .with_columns(
            duration_precise=pl.col('time').diff().cast(pl.Duration('ns')),
        )
        .with_columns(
            duration=(
                pl.col('duration_precise')
                .cast(float) * 1e-6  # milliseconds
            ),
            hour=(
                pl.col('time')
                - pl.col('time').dt.truncate('1d')
            ).dt.total_hours(fractional=True),
        )
        .filter(pl.col('duration').is_not_null())
        .collect()
    )
    assert (df['count'] > 0).all()
    assert (df['duration'] > 0).all()
    assert (df['hour'] >= 0).all()
    assert df['time'].is_sorted()
    return df


INPUT_DF = load_data('BTCUSDT')
INPUT_DF


# %%


class Dataset(NamedTuple):
    counts: Array
    durations: Array
    times_of_day: Array


DATASET = Dataset(
    counts=INPUT_DF['count'].to_jax(),
    durations=INPUT_DF['duration'].to_jax(),
    times_of_day=INPUT_DF['hour'].to_jax(),
)


# %%


closed_form_rate = DATASET.counts.sum() / DATASET.durations.sum()
closed_form_rate


# %%


def calc_loglik(rate: ArrayLike, dataset: Dataset) -> Array:
    return jax.scipy.stats.poisson.logpmf(
        k=dataset.counts,
        mu=rate*dataset.durations,
    )


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


N_CENTERS = 24  # arbitrary


class RbfRateParams(NamedTuple):
    base_log_rate: float
    rbf_weights: Array


@jax.jit
def calc_log_rate_rbf(params: RbfRateParams, time_of_day: Array) -> Array:
    n_hours = 24
    centers = jnp.linspace(0, n_hours, N_CENTERS, endpoint=False)

    sigma = n_hours / N_CENTERS
    inv_sigma_sq = 1.0 / (sigma**2)

    dist = jnp.abs(time_of_day[:, None] - centers[None, :])
    dist = jnp.where(dist > n_hours / 2, 24 - dist, dist)

    exponent = -0.5 * (dist**2) * inv_sigma_sq
    basis = jnp.exp(exponent)

    log_rate = basis @ params.rbf_weights + params.base_log_rate
    return log_rate


def plot_rbf_rate(params: RbfRateParams) -> None:
    time_of_day = jnp.linspace(-2, 26, 500, endpoint=False)
    log_rates = calc_log_rate_rbf(params, time_of_day=time_of_day)

    baseline_rate = jnp.exp(params.base_log_rate).item()
    rates = jnp.exp(log_rates)

    plt.plot(time_of_day, rates)
    plt.axhline(baseline_rate, label=f'{baseline_rate=:.6f}',
                c='g', alpha=0.4, linestyle='-')
    plt.axvline(0, c='k', alpha=0.2, linestyle='--')
    plt.axvline(24, c='k', alpha=0.2, linestyle='--')
    plt.legend(loc='upper left')
    plt.show()


init_rbf_params = RbfRateParams(
    base_log_rate=float(jnp.log(constant_rate)),
    rbf_weights=jnp.sin(jnp.arange(N_CENTERS)),
)


plot_rbf_rate(init_rbf_params)


# %%


flat_rbf_params, unflatten_rbf_params = ravel_pytree(init_rbf_params)


def rbf_rate_loss(flat_params: Array,
                  unflatten_fn: Callable,
                  dataset: Dataset):
    rbf_params = unflatten_fn(flat_params)
    log_rate = calc_log_rate_rbf(rbf_params, dataset.times_of_day)
    rate = jnp.exp(log_rate)
    return -calc_loglik(rate, dataset).mean()


rbf_optim_result = minimize(
    rbf_rate_loss,
    flat_rbf_params,
    args=(unflatten_rbf_params, DATASET),
    method='BFGS',
)


rbf_optim_params = unflatten_rbf_params(rbf_optim_result.x)
plot_rbf_rate(rbf_optim_params)


# %%


rbf_log_rate = calc_log_rate_rbf(rbf_optim_params, DATASET.times_of_day)
rbf_rate = jnp.exp(rbf_log_rate)


# %%


def logit(y) -> Array:
    return jnp.log(y) - jnp.log1p(-y)


# exponential decay kernel for now
class HawkesParams(NamedTuple):
    base_log_rate: float
    logit_norm: float = logit(jnp.array(0.85)).item()
    log_omega: float = -jnp.log(30 * 1_000).item()


class HawkesOutputs(NamedTuple):
    decay_factors: Array
    decayed_counts: Array
    rates: Array


@jax.jit
def calc_hawkes(params: HawkesParams, dataset: Dataset) -> HawkesOutputs:
    base_rate = jnp.exp(params.base_log_rate)
    norm = jax.nn.sigmoid(params.logit_norm)
    omega = jnp.exp(params.log_omega)
    decay_factors = jnp.exp(-omega * dataset.durations)

    def step(carry, x):
        decayed_counts = carry
        count, decay_factor = x
        decayed_counts *= decay_factor
        decayed_counts += count
        return decayed_counts, decayed_counts

    _, decayed_counts = jax.lax.scan(step, 0, (dataset.counts, decay_factors))
    rates = base_rate + norm * omega * decayed_counts

    return HawkesOutputs(
        decay_factors=decay_factors,
        decayed_counts=decayed_counts,
        rates=rates,
    )


def plot_hawkes_rate(params: HawkesParams,
                     dataset: Dataset,
                     input_df: pl.DataFrame) -> None:
    outputs = calc_hawkes(params, dataset)
    display(params)

    base_rate = jnp.exp(params.base_log_rate).item()
    norm = jax.nn.sigmoid(params.logit_norm).item()
    omega = jnp.exp(params.log_omega).item()

    print(f'{base_rate=}, {norm=}, {omega=}')
    df = (
        input_df
        .with_columns(
            decay_factors=np.asarray(outputs.decay_factors),
            decayed_counts=np.asarray(outputs.decayed_counts),
            rates=np.asarray(outputs.rates),

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
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.scatter(subset['time'], subset[['count']], alpha=0.05, marker='.')
    ax2.plot(subset['time'], subset[['rates']])
    plt.tight_layout()
    plt.show()


init_hawkes_params = HawkesParams(
    base_log_rate=float(jnp.log(constant_rate)),
)


plot_hawkes_rate(init_hawkes_params, DATASET, INPUT_DF)


# %%


flat_hawkes_params, unflatten_hawkes_params = ravel_pytree(init_hawkes_params)


def hawkes_loss(flat_params: Array,
                unflatten_fn: Callable,
                dataset: Dataset):
    hawkes_params = unflatten_fn(flat_params)
    output = calc_hawkes(hawkes_params, dataset)
    return -calc_loglik(output.rates, dataset).mean()


hawkes_optim_result = minimize(
    hawkes_loss,
    flat_hawkes_params,
    args=(unflatten_hawkes_params, DATASET),
    method='BFGS',
)


hawkes_optim_params = unflatten_hawkes_params(hawkes_optim_result.x)
hawkes_outputs = calc_hawkes(hawkes_optim_params, DATASET)
plot_hawkes_rate(hawkes_optim_params, DATASET, INPUT_DF)


# %%


result_df = (
    INPUT_DF
    .with_columns(
        pl.col('time').dt.truncate('30m'),
        constant_loglik=np.asarray(
            calc_loglik(constant_rate, DATASET),
        ),
        rbf_loglik=np.asarray(
            calc_loglik(rbf_rate, DATASET),
        ),
        hawkes_loglik=np.asarray(
            calc_loglik(hawkes_outputs.rates, DATASET),
        ),
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
    [['constant_loglik', 'rbf_loglik', 'hawkes_loglik']]
    .plot(alpha=0.6)
)
plt.show()


# %%
