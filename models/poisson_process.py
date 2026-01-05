# %%

from pathlib import Path
import datetime

import jax
from jax.typing import ArrayLike
from jax import Array
import jax.numpy as jnp
import numpy as np
import polars as pl
from jax.scipy.optimize import minimize


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
        duration=pl.col('elapsed').cast(float) * 1e-6,
    )
    .collect()
)

assert (df['count'] > 0).all()
assert (df['duration'] > 0).all()
assert df['time'].is_sorted()

df


# %%


count = df['count'].to_jax()
duration = df['duration'].to_jax()
closed_form_rate = count.sum() / duration.sum()
closed_form_rate


# %%


def calc_loglik(rate: ArrayLike, duration: Array, count: Array) -> Array:
    return jax.scipy.stats.poisson.logpmf(k=count, mu=rate*duration)


def obj_fn(params, duration: Array, count: Array):
    log_rate, = params
    rate = jnp.exp(log_rate)
    return -calc_loglik(rate, duration, count).mean()


inigial_guess = jnp.array([jnp.log(closed_form_rate + 1e-1)])


optim_result = minimize(
    obj_fn,
    inigial_guess,
    args=(duration, count),
    method='BFGS',
    tol=1e-7,
)

optim_result


# %%


optimised_rate = float(jnp.exp(optim_result.x[0]))
diff = float(optimised_rate - closed_form_rate)

print(f'{closed_form_rate=:.8f}, {optimised_rate=:.8f}, {diff=}')

# %%
