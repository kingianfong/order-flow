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
        .with_columns(elapsed=pl.col('time').diff().cast(pl.Duration('ns')))
        .filter(pl.col('elapsed').is_not_null())
        .with_columns(
            hour=(
                pl.col('time')
                - pl.col('time').dt.truncate('1d')
            ).dt.total_hours(fractional=True),
            duration=pl.col('elapsed').cast(float) * 1e-6,  # milliseconds
        )
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
                c='k', alpha=0.6, linestyle='--')
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
    )
    .group_by('time', maintain_order=True)
    .sum()
    .with_columns(
        rbf_improvement=(
            pl.col('rbf_loglik')
            - pl.col('constant_loglik')
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
    .sort('rbf_improvement')
)


# %%


(
    result_df
    .to_pandas()
    .set_index('time')
    [['constant_loglik', 'rbf_loglik']]
    .plot(alpha=0.6)
)
plt.show()


# %%
