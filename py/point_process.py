# %%


from pathlib import Path
from typing import Any, NamedTuple
import datetime

from IPython.display import display
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.scipy.special import logit
from jax.typing import ArrayLike
import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import polars as pl
import seaborn as sns


DATA_DIR = Path(__file__).parent.parent / 'data/preprocessed'


# TODO: train test split


# %%


# raw data only has sometimes has multiple trades per timestamp
# this input dataset is grouped by millisecond
def load_data(sym: str) -> pl.DataFrame:
    df = (
        pl.scan_parquet(DATA_DIR)
        .filter(
            pl.col('sym') == pl.lit(sym),
            # pl.col('date') < datetime.date(2025, 11, 1),
            pl.col('date') >= datetime.date(2025, 12, 1),
            pl.col('date') <= datetime.date(2025, 12, 5),
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
    centers: Array = jnp.linspace(0, n_hours, n_centers, endpoint=False)

    width_factor: float = 0.5
    sigma = width_factor * n_hours / n_centers
    inv_sigma_sq = 1.0 / (sigma**2)

    @staticmethod
    def reg_penalty(weights: Array) -> Array:
        # TODO: consider penalising differences between adjacent elements
        l2_reg: float = 0.1
        return jnp.sum(jnp.square(weights)) * l2_reg


def calc_rbf_basis(time_of_day: Array) -> Array:
    dist = jnp.abs(time_of_day[:, None] - RbfConstants.centers[None, :])
    dist = jnp.where(dist > RbfConstants.n_hours / 2,  - dist, dist)
    exponent = -0.5 * (dist**2) * RbfConstants.inv_sigma_sq
    basis = jnp.exp(exponent)
    return basis


class Dataset(NamedTuple):
    curr_count: Array
    elapsed: Array
    rbf_basis: Array

    @property
    def n_samples(self):
        chex.assert_equal_shape_prefix(
            (self.curr_count, self.elapsed, self.rbf_basis),
            prefix_len=1,
        )
        return len(self.curr_count)


DATASET = Dataset(
    curr_count=INPUT_DF['curr_count'].cast(pl.Float32).to_jax(),
    elapsed=INPUT_DF['elapsed'].to_jax(),
    rbf_basis=calc_rbf_basis(INPUT_DF['hour'].to_jax()),
)

assert DATASET.n_samples == INPUT_DF.height


# %%


closed_form_rate = (DATASET.curr_count.sum() / DATASET.elapsed.sum()).item()
print(f'{closed_form_rate=}, around {closed_form_rate*1000:.2f}/second')
print(f'{jnp.log(closed_form_rate)=:.4f}')


# %%


def get_pytree_labels(params):
    """Generates a list of strings representing each scalar in the PyTree."""
    # We need the paths (keys) to each leaf
    # This requires jax.tree_util.tree_leaves_with_path (available in modern JAX)
    leaves_with_path = jax.tree_util.tree_leaves_with_path(params)

    labels = []
    for path, leaf in leaves_with_path:
        # Convert path tuple (e.g., (DictKey(key='w'),)) to a string "w"
        path_str = ".".join(
            [str(p.key if hasattr(p, 'key') else p) for p in path])

        if leaf.size == 1:
            labels.append(path_str)
        else:
            # For arrays, add indices: w[0], w[1], etc.
            # Using row-major (C-style) ordering to match JAX
            indices = jnp.unravel_index(jnp.arange(leaf.size), leaf.shape)
            for i in range(leaf.size):
                idx_tuple = [int(axis[i]) for axis in indices]
                labels.append(f"{path_str}{idx_tuple}")

    return labels


def run_optim(init, loss_fn, loss_kwargs,
              nll_samples: int | None = None,
              verbose=False) -> Any:
    opt = optax.lbfgs(
        memory_size=20,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=20,
            verbose=verbose,
            initial_guess_strategy='one'
        ),
    )
    max_iter = 20
    tol = 1e-5

    value_and_grad_fun = optax.value_and_grad_from_state(loss_fn)

    @jax.jit
    def step(carry):
        params, state, loss_kwargs = carry
        value, grad = value_and_grad_fun(params, state=state, **loss_kwargs)
        updates, state = opt.update(
            grad, state, params,
            value=value,
            grad=grad,
            value_fn=loss_fn,
            **loss_kwargs,
        )
        params = optax.apply_updates(params, updates)
        return params, state, loss_kwargs

    @jax.jit
    def continuing_criterion(carry):
        _, state, _ = carry
        iter_num = optax.tree.get(state, 'count')
        grad = optax.tree.get(state, 'grad')
        err = optax.tree.norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = init, opt.init(init), loss_kwargs
    final_params, final_state, _ = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )

    assert isinstance(final_state, tuple), \
        f'{type(final_state)=}'
    lbfgs_final_state = final_state[0]
    assert isinstance(lbfgs_final_state, optax.ScaleByLBFGSState), \
        f'{type(lbfgs_final_state)=}'

    n_iter = optax.tree.get(final_state, 'count').item()
    if n_iter == max_iter:
        grad = optax.tree.get(final_state, 'grad')
        err = optax.tree.norm(grad).item()
        print(f'warning: did not converge after {n_iter=}, {err=}')
    else:
        print(f'converged: {n_iter=}')

    # assume -mean(loglik) insetad of -sum(loglik)
    if nll_samples is not None:
        print('calculating hessian')
        start_time = datetime.datetime.now()

        flat_params, unravel = ravel_pytree(final_params)

        def flat_loss_fn(flat_p, dataset):
            return loss_fn(unravel(flat_p), dataset)
        calc_hess = jax.jit(jax.hessian(flat_loss_fn))
        inv_hess_mat = jnp.linalg.pinv(calc_hess(flat_params, dataset=DATASET))
        cond = jnp.linalg.cond(inv_hess_mat).item()

        labels = get_pytree_labels(final_params)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pd.DataFrame(inv_hess_mat, index=labels, columns=labels),
            center=0.0, robust=True,
        )
        plt.title(f"Inverse Hessian Matrix {cond=:.2f}")
        plt.show()

        mean = flat_params
        var = jnp.diag(inv_hess_mat) / nll_samples
        std_err = jnp.sqrt(var)
        display(
            pd.DataFrame(
                dict(
                    mean=mean,
                    std_err=std_err,
                    z=mean / std_err,
                ),
                index=labels,
            )
        )
        stop_time = datetime.datetime.now()
        elapsed = stop_time - start_time
        print(f'hessian took {elapsed}')

    return final_params


def constant_loglik(rate: ArrayLike, dataset: Dataset) -> Array:
    # assume constant rate throughout interval
    return dataset.curr_count * jnp.log(rate) - rate * dataset.elapsed


@jax.jit
def constant_rate_loss(params: Array, dataset: Dataset):
    log_rate, = params
    rate = jnp.exp(log_rate)
    return -constant_loglik(rate, dataset).mean()


log_constant_rate_result = run_optim(
    init=(jnp.log(closed_form_rate + 1e-4), ),
    loss_fn=constant_rate_loss,
    loss_kwargs=dict(dataset=DATASET),
    # nll_samples=DATASET.n_samples,
)


constant_rate = jnp.exp(log_constant_rate_result[0]).item()

print(f'{closed_form_rate=:.8f}, {constant_rate=:.8f}')


# %%


class ModelOutput(NamedTuple):
    loglik: Array  # loglik of (no event since prev t) + (events at t)
    rate: Array  # used for predictions after observing events at t


class RbfRateParams(NamedTuple):
    log_base_rate: Array
    weights: Array = jnp.zeros((RbfConstants.n_centers,),)


def calc_rbf(params: RbfRateParams, dataset: Dataset) -> ModelOutput:
    log_rate_factor = dataset.rbf_basis @ params.weights
    log_rate = params.log_base_rate + log_rate_factor
    rate = jnp.exp(log_rate)
    loglik = \
        dataset.curr_count * log_rate \
        - rate * dataset.elapsed  # assume constant rate throughout interval
    return ModelOutput(loglik=loglik, rate=rate)


def plot_rbf(log_base_rate: Array, weights: Array) -> None:
    time_of_day = jnp.linspace(-2, 26, 500, endpoint=False)
    log_factor = calc_rbf_basis(time_of_day) @ weights
    base_rate = jnp.exp(log_base_rate).item()
    rate = jnp.exp(log_base_rate + log_factor)
    per_second = base_rate * 1000

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(time_of_day, rate)
    ax1.axhline(base_rate, label=f'base rate$\\approx${per_second:.2f}/s',
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
    log_base_rate=jnp.log(constant_rate),
)


plot_rbf(init_rbf_params.log_base_rate, init_rbf_params.weights)


# %%


@jax.jit
def rbf_loss(params: RbfRateParams, dataset: Dataset):
    output = calc_rbf(params, dataset)
    return -output.loglik.mean() + RbfConstants.reg_penalty(params.weights)


rbf_optim_params = run_optim(
    init_rbf_params,
    rbf_loss,
    loss_kwargs=dict(dataset=DATASET,),
    nll_samples=DATASET.n_samples,
)
rbf_outputs = calc_rbf(rbf_optim_params, DATASET)
plot_rbf(rbf_optim_params.log_base_rate, rbf_optim_params.weights)


# %%


# exponential decay kernel for now
class HawkesParams(NamedTuple):
    log_base_rate: Array
    logit_norm: Array = logit(0.9)
    log_omega: Array = jnp.log(1)  # log(1 / avg_life_ms)


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


def plot_model_output(outputs: ModelOutput, input_df: pl.DataFrame):
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
            pl.col('time').dt.date() == pl.col('time').dt.date().min(),
        )
    )
    _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.scatter(subset['time'], subset[['curr_count']], alpha=0.05, marker='.')
    ax2.plot(subset['time'], subset[['rate']])
    ax3.plot(subset['time'], subset[['loglik']])
    plt.tight_layout()
    plt.show()


def print_params(params):
    to_print = {}
    if hasattr(params, 'log_base_rate'):
        to_print['base_rate'] = jnp.exp(params.log_base_rate).item()
        to_print['per_second'] = to_print['base_rate'] * 1_000
    if hasattr(params, 'logit_norm'):
        to_print['norm'] = jax.nn.sigmoid(params.logit_norm).item()
    if hasattr(params, 'log_omega'):
        to_print['omega'] = jnp.exp(params.log_omega).item()
        to_print['decay_avg_life_seconds'] = 1_000 / to_print['omega']
    if hasattr(params, 'log_beta'):
        to_print['beta'] = jnp.exp(params.log_beta).item()
    display(pd.Series(to_print, name='param').to_frame())


def show_hawkes(params, dataset: Dataset, input_df: pl.DataFrame) -> None:
    baseline_outputs = calc_hawkes_baseline(params, dataset)

    outputs = calc_hawkes(params, dataset)
    assert jnp.allclose(outputs.loglik, baseline_outputs.loglik, atol=1e-4)
    assert jnp.allclose(outputs.rate, baseline_outputs.rate, rtol=1e-4)

    print_params(params)
    plot_model_output(outputs, input_df)


init_hawkes_params = HawkesParams(
    log_base_rate=jnp.log(constant_rate),
)


show_hawkes(init_hawkes_params, DATASET, INPUT_DF)


# %%


@jax.jit
def hawkes_loss(params: HawkesParams, dataset: Dataset):
    output = calc_hawkes(params, dataset)
    return -output.loglik.mean()


hawkes_optim_params = run_optim(
    init_hawkes_params,
    hawkes_loss,
    loss_kwargs=dict(dataset=DATASET),
    nll_samples=DATASET.n_samples,
)

hawkes_outputs = calc_hawkes(hawkes_optim_params, DATASET)
show_hawkes(hawkes_optim_params, DATASET, INPUT_DF)


# %%


class RbfHawkesParams(NamedTuple):
    log_base_rate: Array
    logit_norm: Array
    log_omega: Array
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


def show_rbf_hawkes(params: RbfHawkesParams, dataset: Dataset, input_df: pl.DataFrame):
    outputs = calc_rbf_hawkes(params, dataset)
    print_params(params)
    plot_model_output(outputs, input_df)
    plot_rbf(log_base_rate=params.log_base_rate, weights=params.weights)


init_rbf_hawkes = RbfHawkesParams(
    log_base_rate=hawkes_optim_params.log_base_rate,
    logit_norm=hawkes_optim_params.logit_norm,
    log_omega=hawkes_optim_params.log_omega,
    weights=rbf_optim_params.weights,
)


show_rbf_hawkes(init_rbf_hawkes, DATASET, INPUT_DF)


# %%


@jax.jit
def rbf_hawkes_loss(params: RbfHawkesParams, dataset: Dataset):
    output = calc_rbf_hawkes(params, dataset)
    return -output.loglik.mean() + RbfConstants.reg_penalty(params.weights)


rbf_hawkes_optim_params = run_optim(
    init_rbf_hawkes,
    rbf_hawkes_loss,
    loss_kwargs=dict(dataset=DATASET),
    nll_samples=DATASET.n_samples,
)
rbf_hawkes_outputs = calc_rbf_hawkes(rbf_hawkes_optim_params, DATASET)
show_rbf_hawkes(rbf_hawkes_optim_params, DATASET, INPUT_DF)


# %%


class PowerLawApproxParams(NamedTuple):
    weights: Array
    rates: Array


_one_millisecond = 1.0  # timestamp resolution
_one_minute = _one_millisecond * 60e3
_one_hour = _one_minute * 60


def power_law_decay_approx_params(omega: ArrayLike, beta: ArrayLike,
                                  min_history_duration_ms: ArrayLike,
                                  max_history_duration_ms: ArrayLike,
                                  n_exponentials: int) -> PowerLawApproxParams:
    # to approximate lomax decay
    #   g(t) = (1 + omega * t) ^ -(1 + beta)
    #        = E[ h(X; t) ]
    # where
    #   X ~ Gamma(1+beta, scale=omega)
    #       -> f(x) = k * x^{beta} * exp{-x / omega}
    #   h(x; t) = exp{-x * t}

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
    min_t_ms = _one_millisecond
    max_t_ms = _one_hour
    t_geom = jnp.geomspace(min_t_ms, max_t_ms * 10, n)
    orders_of_magnitude = jnp.log10(max_t_ms / min_t_ms).item()
    print(f'{orders_of_magnitude=:.2f}')
    n_exponentials = 14  # 2x orders of magnitude

    omegas = 1 / _one_millisecond, 1 / _one_minute, 1 / _one_hour
    betas = 0.15, 0.3, 0.5

    f1, axes1 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(9, 9))
    f1.suptitle('Lomax Kernel')
    f2, axes2 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(9, 9))
    f2.suptitle('linear y')
    f3, axes3 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(9, 9))
    f3.suptitle('linear x, linear y')
    f4, axes4 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(9, 9))
    f4.suptitle('absolute error')
    f5, axes5 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(9, 9))
    f5.suptitle('relative error')

    for r, omega in enumerate(omegas):
        inv_omega = 1.0 / omega
        for c, beta in enumerate(betas):

            kernel_params = power_law_decay_approx_params(
                omega=omega,
                beta=beta,
                min_history_duration_ms=min_t_ms,
                max_history_duration_ms=max_t_ms,
                n_exponentials=n_exponentials,
            )
            exact_integral = 1 / (omega * beta)
            approx_integral = jnp.sum(kernel_params.weights
                                      / kernel_params.rates).item()
            assert jnp.allclose(approx_integral, exact_integral,
                                rtol=1e-4, atol=jnp.inf), \
                f'{approx_integral=}, {exact_integral=}'

            exact = calc_exact(t_geom, omega=omega, beta=beta)
            approx = calc_approx(t_geom, kernel_params)

            ax1 = axes1[r, c]
            ax1.loglog(t_geom, exact, 'k-', label='exact', lw=2)
            ax1.loglog(t_geom, approx, 'r--', label='approx', lw=2)
            ax1.set_title(f"$\\omega$={omega:1.1e},$\\beta$={beta}")
            ax1.grid(True, which="both", ls="--", alpha=0.2)
            ax1.legend(loc='lower left')

            ax2 = axes2[r, c]
            ax2.plot(t_geom, exact, 'k-', label='exact', lw=2)
            ax2.plot(t_geom, approx, 'r--', label='approx', lw=2)
            ax2.axvline(inv_omega, c='g', linestyle='--', alpha=0.6,
                        label=r'$\omega^{-1}$')
            ax2.axvline(max_t_ms, c='k', linestyle='--', alpha=0.2)
            ax2.set_xscale('log')
            ax2.set_title(f"$\\omega$={omega:1.1e},$\\beta$={beta}")
            ax2.grid(True, which="both", ls="--", alpha=0.2)
            ax2.legend(loc='upper right')

            ax3 = axes3[r, c]
            keep = t_geom < (2 * _one_hour)
            ax3.plot(t_geom[keep], exact[keep], 'k-', label='exact', lw=2)
            ax3.plot(t_geom[keep], approx[keep], 'r--', label='approx', lw=2)
            ax3.axvline(inv_omega, c='g', linestyle='--', alpha=0.6,
                        label=r'$\omega^{-1}$')
            ax3.set_title(f"$\\omega$={omega:1.1e},$\\beta$={beta}")
            ax3.grid(True, which="both", ls="--", alpha=0.2)
            ax3.legend(loc='upper right')

            ax4 = axes4[r, c]
            ax4.loglog(t_geom, abs(approx - exact), 'k-', lw=2)
            ax4.axvline(inv_omega, c='g', linestyle='--', alpha=0.6,
                        label=r'$\omega^{-1}$')
            ax4.set_title(f"$\\omega$={omega:1.1e},$\\beta$={beta}")
            ax4.grid(True, which="both", ls="--", alpha=0.2)

            ax5 = axes5[r, c]
            ax5.loglog(t_geom, abs(approx - exact) / exact, 'k-', lw=2)
            ax5.axvline(inv_omega, c='g', linestyle='--', alpha=0.6,
                        label=r'$\omega^{-1}$')
            ax5.set_title(f"$\\omega$={omega:1.1e},$\\beta$={beta}")
            ax5.grid(True, which="both", ls="--", alpha=0.2)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_power_law_decay()


# %%


class PowerLawHawkesParams(NamedTuple):
    log_base_rate: Array
    logit_norm: Array
    log_beta: Array = jnp.log(0.15)


@jax.jit
def calc_power_law_hawkes(params: PowerLawHawkesParams,
                          dataset: Dataset,
                          # fix omega to min resolution (1ms) to avoid
                          # identifiability with beta. roughly corresponds to
                          # where the plateau crosses over to power-law tail
                          omega: float = 1.0 / _one_millisecond,
                          ) -> ModelOutput:
    n_exponentials = 10  # not differentiable
    min_history_duration_ms: Array = jnp.array(_one_millisecond)
    max_history_duration_ms: Array = jnp.array(_one_hour * 2)

    base_rate = jnp.exp(params.log_base_rate)
    norm = jax.nn.sigmoid(params.logit_norm)
    omega = 1.0 / _one_millisecond
    beta = jnp.exp(params.log_beta)

    approx_params = power_law_decay_approx_params(
        omega=omega,
        beta=beta,
        min_history_duration_ms=min_history_duration_ms,
        max_history_duration_ms=max_history_duration_ms,
        n_exponentials=n_exponentials,
    )
    weights, rates = approx_params.weights, approx_params.rates
    approx_integral = jnp.sum(weights / rates)
    kernel_factor = norm / approx_integral

    # TODO: switch to scan if run out of memory
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


def show_power_law_hawkes(params: PowerLawHawkesParams,
                          dataset: Dataset,
                          input_df: pl.DataFrame) -> None:
    outputs = calc_power_law_hawkes(params, dataset)
    print_params(params)
    plot_model_output(outputs=outputs, input_df=input_df)


init_pl_hawkes_params = PowerLawHawkesParams(
    log_base_rate=hawkes_optim_params.log_base_rate,
    logit_norm=hawkes_optim_params.logit_norm,
)


show_power_law_hawkes(init_pl_hawkes_params, DATASET, INPUT_DF)

# %%


@jax.jit
def power_law_hawkes_loss(params: PowerLawHawkesParams, dataset: Dataset):
    output = calc_power_law_hawkes(params, dataset)
    return -output.loglik.mean()


power_law_hawkes_optim_params = run_optim(
    init_pl_hawkes_params,
    power_law_hawkes_loss,
    loss_kwargs=dict(dataset=DATASET),
    nll_samples=DATASET.n_samples,
)
power_law_hawkes_outputs = calc_power_law_hawkes(
    power_law_hawkes_optim_params, DATASET)
show_power_law_hawkes(power_law_hawkes_optim_params, DATASET, INPUT_DF)


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


print('note: not a fair comparison because number of parameters differ')
display(
    with_logliks.select(pl.selectors.ends_with('_loglik')).sum(),
    with_logliks.select(pl.selectors.ends_with('_loglik')).mean(),
)


(
    with_logliks
    .group_by(pl.col('time').dt.truncate('1h'), maintain_order=True)
    .agg(
        pl.selectors.ends_with('_loglik').sum()
    )
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
