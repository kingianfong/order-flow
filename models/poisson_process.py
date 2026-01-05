# %%


from pathlib import Path
from typing import NamedTuple
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


SYM = 'BTCUSDT'

df = (
    pl.scan_parquet(DATA_DIR)
    .filter(
        pl.col('sym') == pl.lit(SYM),
        pl.col('date') >= datetime.date(2025, 12, 1),
    )
    .select('time', count=pl.col('total_count'))
    .with_columns(elapsed=pl.col('time').diff().cast(pl.Duration('ns')))
    .filter(pl.col('elapsed').is_not_null())
    .with_columns(
        hour=(pl.col('time')
              - pl.col('time').dt.truncate('1d')
              ).dt.total_hours(fractional=True),
        duration=pl.col('elapsed').cast(float) * 1e-6,
    )
    .collect()
)

assert (df['count'] > 0).all()
assert (df['duration'] > 0).all()
assert (df['hour'] >= 0).all()
assert df['time'].is_sorted()

df


# %%


count = df['count'].to_jax()
duration = df['duration'].to_jax()
hour = df['hour'].to_jax()
closed_form_rate = count.sum() / duration.sum()
closed_form_rate


# %%


def calc_loglik(rate: ArrayLike, duration: Array, count: Array) -> Array:
    return jax.scipy.stats.poisson.logpmf(k=count, mu=rate*duration)


def obj_fn(params, duration: Array, count: Array):
    log_rate, = params
    rate = jnp.exp(log_rate)
    return -calc_loglik(rate, duration, count).mean()


initial_guess = jnp.array([jnp.log(closed_form_rate + 1e-1)])


rbf_optim_result = minimize(
    obj_fn,
    initial_guess,
    args=(duration, count),
    method='BFGS',
    tol=1e-7,
)

rbf_optim_result


# %%


optimised_rate = float(jnp.exp(rbf_optim_result.x[0]))
optim_diff = float(optimised_rate - closed_form_rate)

print(f'{closed_form_rate=:.8f}, {optimised_rate=:.8f}, {optim_diff=}')


# %%


N_CENTERS = 12  # arbitrary


class RbfRateParams(NamedTuple):
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

    log_rate = basis @ params.rbf_weights
    return log_rate


def plot_rbf_rate(params: RbfRateParams) -> None:
    time_of_day = jnp.linspace(0, 24, 200, endpoint=False)
    log_rates = calc_log_rate_rbf(params, time_of_day=time_of_day)
    rates = jnp.exp(log_rates)
    plt.plot(time_of_day, rates)
    plt.show()


initial_rbf_params = RbfRateParams(
    rbf_weights=jnp.sin(jnp.arange(N_CENTERS)),
)


plot_rbf_rate(initial_rbf_params)


# %%


flat_params, unflatten_fn = ravel_pytree(initial_rbf_params)


def rbf_obj_fn(flat_params: Array, unflatten_fn, duration: Array, count: Array, time_of_day: Array):
    rbf_params = unflatten_fn(flat_params)
    log_rate = calc_log_rate_rbf(rbf_params, time_of_day)
    rate = jnp.exp(log_rate)
    return -calc_loglik(rate, duration, count).mean()


rbf_optim_result = minimize(
    rbf_obj_fn,
    flat_params,
    args=(unflatten_fn, duration, count, hour),
    method='BFGS',
    tol=1e-7,
)


rbf_optim_params = unflatten_fn(rbf_optim_result.x)
plot_rbf_rate(rbf_optim_params)


# %%


log_rate_rbf = calc_log_rate_rbf(rbf_optim_params, hour)
rate_rbf = jnp.exp(log_rate_rbf)


# %%


result_df = (
    df
    .with_columns(
        pl.col('time').dt.truncate('30m'),
        constant_rate_loglik=np.asarray(
            calc_loglik(optimised_rate, duration, count),
        ),
        rbf_rate_loglik=np.asarray(
            calc_loglik(rate_rbf, duration, count),
        )
    )
    .group_by('time', maintain_order=True)
    .sum()
    .with_columns(
        rbf_improvement=pl.col('rbf_rate_loglik') -
        pl.col('constant_rate_loglik'),
    )
)


result_df


# %%


(
    result_df
    .group_by(pl.col('time').dt.time().alias('time')).sum()
    .sort('constant_rate_loglik')
)


# %%


(
    result_df
    .to_pandas()
    .set_index('time')
    [['constant_rate_loglik', 'rbf_rate_loglik']]
    .plot(alpha=0.6)
)
plt.show()

# %%
