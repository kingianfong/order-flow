# %%


from pathlib import Path
from typing import NamedTuple, Callable
import datetime
import re

from IPython.display import display
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.scipy.special import logit, gammaln, erfc
from jax.typing import ArrayLike
import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import polars as pl
import scipy
import seaborn as sns

from decayed_counts import calculate_decayed_counts
import power_law_approx


# %%


RAW_DATA_DIR = Path.cwd().parent / 'data/raw'
SYM = 'BTCUSDT'
DATA_START = datetime.datetime(2025, 9, 1)
TRAIN_END = datetime.datetime(2025, 9, 30)
VAL_END = datetime.datetime(2025, 10, 15)


# %%


# timestamps only hav millisecond resolution
# there can be multiple trades per timestamp
def load_data(raw_data_dir: Path,
              sym: str,
              start: datetime.datetime,
              train_end: datetime.datetime,
              val_end: datetime.datetime) -> pl.DataFrame:
    assert start <= train_end <= val_end, \
        f'{start=}, {train_end=}, {val_end=}'
    df = (
        pl.scan_parquet(raw_data_dir)
        .filter(
            pl.col('sym') == sym,
            pl.col('date') >= start,
            pl.col('date') <= val_end,
        )
        .with_columns(
            pl.col('time').cast(pl.Datetime('ms')),
            is_train=pl.col('date') <= train_end,
        )
        .drop('sym', 'date')
        .sort('time')
        .group_by('time', maintain_order=True)
        .agg(
            pl.col('is_train').first(),
            curr_count=pl.len(),
            hi_price=pl.col('price').max(),
            lo_price=pl.col('price').min(),
        )
        .with_columns(
            time_since_prev=pl.col('time').diff(),
            hour=((pl.col('time') - pl.col('time').dt.truncate('1d'))
                  .dt.total_hours(fractional=True)),
        )
        .filter(pl.col('time_since_prev').is_not_null())
        .collect()
    )

    assert df['time'].is_unique().all()
    assert df['time'].is_sorted()
    assert (df['curr_count'] > 0).all()
    assert (df['time_since_prev'] > datetime.timedelta(0)).all()
    assert (df['hour'] >= 0).all()
    assert (df['hour'] <= 24).all()
    return df


INPUT_DF = load_data(
    raw_data_dir=RAW_DATA_DIR,
    sym=SYM,
    start=DATA_START,
    train_end=TRAIN_END,
    val_end=VAL_END,
)
display(INPUT_DF)


# %%


N_RBF_CENTERS: int = 24


def calc_rbf_basis(time_of_day: Array) -> Array:
    n_hours_in_a_day = 24
    centers = jnp.linspace(0, n_hours_in_a_day, N_RBF_CENTERS, endpoint=False)
    width_factor = 0.5
    sigma = width_factor * n_hours_in_a_day / N_RBF_CENTERS
    inv_sigma_sq = 1.0 / (sigma**2)

    half = n_hours_in_a_day / 2
    dist_from_ctrs = time_of_day[:, None] - centers[None, :]
    dist_from_ctrs = (dist_from_ctrs + half) % n_hours_in_a_day - half
    exponent = -0.5 * (dist_from_ctrs**2) * inv_sigma_sq
    basis = jnp.exp(exponent)
    return basis


class PowerLawCache(NamedTuple):
    decay_rates: Array
    decayed_count: Array
    curr_minus_count: Array
    decay_integral: Array  # for compensator


def calc_power_law_cache(curr_count: Array,
                         time_since_prev: Array) -> PowerLawCache:
    decay_rates = power_law_approx.calc_decay_rates(
        min_history=ONE_MILLISECOND,
        max_history=ONE_HOUR,
        n_exponentials=14,  # 3.6e6 ms in an hour, 2x orders of magnitude
    )
    decay_factor_exponents = (
        # KNOWN_LIMITATION: this may run out of memory for large datasets
        # switch to jax.lax.scan if necessary
        -jnp.outer(time_since_prev, decay_rates)
    )
    decay_factors = jnp.exp(decay_factor_exponents)
    decayed_count = calculate_decayed_counts(decay_factors, curr_count)
    prev_decayed_count = jnp.roll(decayed_count, 1, axis=0).at[0, :].set(0.0)

    return PowerLawCache(
        decay_rates=decay_rates,
        decayed_count=decayed_count,
        curr_minus_count=prev_decayed_count * decay_factors,
        decay_integral=(prev_decayed_count
                        * -jnp.expm1(decay_factor_exponents)
                        / decay_rates),
    )


class Dataset(NamedTuple):
    curr_count: Array
    time_since_prev: Array
    rbf_basis: Array
    power_law_cache: PowerLawCache
    n_samples: int


ONE_MILLISECOND: float = 1.0 * 1e-3  # timestamp resolution
ONE_SECOND: float = 1.0 * 1e0
ONE_MIN: float = 60.0 * 1e0
ONE_HOUR: float = 3600.0 * 1e0


def create_dataset(df: pl.DataFrame) -> Dataset:
    assert ONE_SECOND == 1.0, f'{ONE_SECOND=}'

    curr_count = df['curr_count'].cast(float).to_jax()
    time_since_prev = (df['time_since_prev']
                       .dt.total_seconds(fractional=True)
                       .to_jax())
    return Dataset(
        curr_count=curr_count,
        time_since_prev=time_since_prev,
        rbf_basis=calc_rbf_basis(df['hour'].to_jax()),
        power_law_cache=calc_power_law_cache(
            curr_count=curr_count,
            time_since_prev=time_since_prev,
        ),
        n_samples=df.height,
    )


DATASET = create_dataset(INPUT_DF.filter(pl.col('is_train')))


# %%


closed_form_intensity = (DATASET.curr_count.sum() /
                         DATASET.time_since_prev.sum()).item()
print(f'{closed_form_intensity=}, around {closed_form_intensity*ONE_SECOND:.2f}/second')
print(f'{jnp.log(closed_form_intensity)=:.4f}')


# %%


def get_pytree_labels(params: chex.ArrayTree) -> list[str]:
    """Generates a list of strings representing each scalar in the PyTree."""

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


class ModelOutput(NamedTuple):
    point_term: Array  # count * log(intensity) at t_i
    compensator: Array  # integral(intensity) from t_{i-1} to t_i
    forecast_intensity: Array  # for predictions after observing events at t
    reg_penalty: Array = jnp.array(0.0)

    @property
    def loglik(self):
        # loglik =
        #   sum(log(intensity)) at each event
        #   - integral(intensity) over duration
        return self.point_term - self.compensator


type ModelFn[Params: chex.ArrayTree] = Callable[[Params, Dataset], ModelOutput]


def calc_robust_var[Params: chex.ArrayTree](params: Params,
                                            dataset: Dataset,
                                            model_fn: ModelFn[Params]):
    """Sandwich estimator: Var = B^-1 @ A @ B^-1"""

    flat_params, unravel = ravel_pytree(params)

    def flat_loss_fn_no_reg(p):
        out = model_fn(unravel(p), dataset)
        return -jnp.mean(out.loglik)

    hess = chex.chexify(jax.hessian(flat_loss_fn_no_reg))(flat_params)
    bread = jnp.linalg.pinv(hess)

    def per_obs_grad(p):
        out = model_fn(unravel(p), dataset)
        return out.loglik

    # (N_samples, N_params)
    jacobian = chex.chexify(jax.jacfwd(per_obs_grad))(flat_params)
    # Outer product of gradients: (N_params, N_params)
    meat = (jacobian.T @ jacobian) / dataset.n_samples
    # Scale by 1/N because B is based on the mean loss
    robust_var = (bread @ meat @ bread) / dataset.n_samples
    return robust_var


def run_optim[Params: chex.ArrayTree](init_params: Params,
                                      model_fn: ModelFn[Params],
                                      dataset: Dataset,
                                      show_hessian: bool,
                                      force_plot: bool = False,
                                      verbose: bool = False) -> Params:
    timeout = datetime.timedelta(seconds=150)
    max_iter = 50
    tol = 1e-3

    opt = optax.lbfgs(
        memory_size=100,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=20,
            verbose=verbose,
            initial_guess_strategy='one'
        ),
    )

    def loss_fn(params: Params, dataset: Dataset) -> Array:
        output = model_fn(params, dataset)
        return -output.loglik.mean() + output.reg_penalty / dataset.n_samples

    value_and_grad_fun = optax.value_and_grad_from_state(loss_fn)

    @chex.chexify
    @jax.jit
    def update(carry, dataset):
        params, state, *_ = carry
        value, grad = value_and_grad_fun(params, state=state, dataset=dataset)
        updates, state = opt.update(
            grad, state, params,
            value=value,
            grad=grad,
            value_fn=loss_fn,
            dataset=dataset,
        )
        params = optax.apply_updates(params, updates)
        return params, state, value, grad

    params = init_params
    state = opt.init(init_params)

    start_time = datetime.datetime.now()
    elapsed = datetime.timedelta()
    n_iter = 0
    n_linesearch = 0
    params_hist = []
    losses_hist = []
    grads_hist = []
    grad_norms = []
    converged = False

    # this loop may force syncs between GPU and CPU
    while True:
        params, state, loss, grad = update((params, state), dataset=dataset)
        _, _, linesearch_state = state

        assert isinstance(linesearch_state, optax.ScaleByZoomLinesearchState), \
            f'{type(linesearch_state)=}'
        n_linesearch += int(linesearch_state.info.num_linesearch_steps)
        grad_norm = optax.tree.norm(grad)
        params_hist.append(params)
        losses_hist.append(loss)
        grads_hist.append(grad)
        grad_norms.append(grad_norm)

        if grad_norm <= tol:
            print(f'converged: {n_iter=}, {n_linesearch=}')
            converged = True
            break

        elapsed = datetime.datetime.now() - start_time
        n_iter += 1
        if n_iter == max_iter or elapsed > timeout:
            print('did not converge')
            break

    labels = get_pytree_labels(params)
    if force_plot or not converged:
        last_grad_norm = grad_norms[-1]
        print(f'{converged=}, {n_iter=}, {n_linesearch=}, {elapsed=}')
        print(f'{last_grad_norm=:.4f}')

        def _get_key(label):
            return re.sub(r'\[\d+\]', '[]', label)

        metrics_df = pd.DataFrame(
            dict(
                loss=losses_hist,
                grad_norm=grad_norms,
            ),
        )
        params_df = (
            pd.DataFrame(
                list(ravel_pytree(p)[0] for p in params_hist),
                columns=labels,
            )
            .rename(columns=lambda x: f'{x} values')
        )
        grads_df = (
            pd.DataFrame(
                list(ravel_pytree(g)[0] for g in grads_hist),
                columns=labels,
            )
            .rename(columns=lambda x: f'{x} grads')
        )
        optim_df = (pd.concat([metrics_df, params_df, grads_df], axis=1)
                    .astype(float))

        col_keys = {col: _get_key(col) for col in optim_df.columns}
        unique_keys = sorted(set(col_keys.values()),
                             key=lambda x: (x.startswith('.'), x.endswith('grads')))
        n_rows = len(unique_keys)
        f, axes = plt.subplots(n_rows, 1, figsize=(8, n_rows*2))
        key_to_ax = dict(zip(unique_keys, axes))

        f.suptitle('optimisation outputs')
        assert n_rows > 1
        for col in optim_df.columns:
            if col.endswith('grads'):
                color = 'C1'
            elif col.endswith('values'):
                color = 'C2'
            else:
                color = 'C0'
            key = _get_key(col)
            ax = key_to_ax[key]
            ax.set_title(key, color=color)
            ax.plot(optim_df[col], drawstyle='steps-post')
            ax.grid(True, which="both", ls="--", alpha=0.2)

        plt.tight_layout()
        plt.show()
        display(optim_df)

    # assume -mean(loglik) insetad of -sum(loglik)
    if show_hessian:
        flat_params, unravel = ravel_pytree(params)

        def flat_loss_fn(flat_p, dataset):
            return loss_fn(unravel(flat_p), dataset)

        print('calculating hessian')
        start_time = datetime.datetime.now()
        calc_hess = chex.chexify(jax.jit(jax.hessian(flat_loss_fn)))
        hess_mat = calc_hess(flat_params, dataset=dataset)
        print(f'hessian took {datetime.datetime.now() - start_time}')

        eigvals = jnp.linalg.eigvalsh(hess_mat)
        if jnp.any(eigvals <= 0):
            print("Warning: Hessian not positive definite. Standard errors invalid.")

        cond = jnp.linalg.cond(hess_mat).item()
        inv_hess_mat = jnp.linalg.pinv(hess_mat)
        diag_sqrt = jnp.sqrt(jnp.diag(inv_hess_mat))
        hess_se = diag_sqrt / jnp.sqrt(dataset.n_samples)
        corr_mat = inv_hess_mat / jnp.outer(diag_sqrt, diag_sqrt)
        mask = np.triu(np.ones_like(corr_mat, dtype=bool))

        print('plotting hessian')
        start_time = datetime.datetime.now()
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            data=pd.DataFrame(corr_mat, index=labels, columns=labels),
            mask=mask, center=0.0, robust=False,
        )
        plt.title(f"Inverse Hessian Matrix {cond=:.2f}")
        plt.show()
        print(f'plotting took {datetime.datetime.now() - start_time}')

        print('calculating errors')
        start_time = datetime.datetime.now()
        robust_var = calc_robust_var(params, dataset, model_fn)
        robust_se = jnp.sqrt(jnp.diag(robust_var))
        print(f'errors took {datetime.datetime.now() - start_time}')

        mean = flat_params
        z_score = mean / robust_se
        p_value = erfc(jnp.abs(z_score) / jnp.sqrt(2))
        meff = robust_se / hess_se
        display(
            pd.DataFrame(
                dict(
                    mean=mean,
                    hess_se=hess_se,
                    robust_se=robust_se,
                    z_score=z_score,
                    p_value=map(lambda x: f'{x:.4%}', p_value),
                    meff=meff,
                ),
                index=labels,
            )
            .style
            .background_gradient(
                subset=['hess_se', 'robust_se', 'meff'],
            )
        )

    return params


def softplus_inverse(x: ArrayLike) -> Array:
    if not isinstance(x, Array):
        x = jnp.array(x)
    # based on https://www.tensorflow.org/probability/api_docs/python/tfp/math/softplus_inverse
    threshold = jnp.log(jnp.finfo(x.dtype).eps) + 2.0
    is_too_small = x < jnp.exp(threshold)
    is_too_large = x > -threshold
    too_small_value = jnp.log(x)
    too_large_value = x
    x = jnp.where(is_too_small | is_too_large, jnp.ones_like(x), x)
    y = x + jnp.log(-jnp.expm1(-x))
    return jnp.where(is_too_small,
                     too_small_value,
                     jnp.where(is_too_large,
                               too_large_value,
                               y))


assert jnp.allclose(
    softplus_inverse(jax.nn.softplus(jnp.arange(10))),
    jnp.arange(10),
)


class ConstantIntensityParams(NamedTuple):
    sp_inv_base_intensity: Array


def calc_const(params: ConstantIntensityParams, dataset: Dataset) -> ModelOutput:
    intensity = jax.nn.softplus(params.sp_inv_base_intensity)
    log_intensity = jnp.log(intensity)
    return ModelOutput(
        forecast_intensity=intensity * jnp.ones_like(dataset.time_since_prev),
        point_term=dataset.curr_count * log_intensity,
        compensator=intensity * dataset.time_since_prev,
    )


fitted_constant_intensity_params = run_optim(
    init_params=ConstantIntensityParams(
        sp_inv_base_intensity=softplus_inverse(
            closed_form_intensity + 1e-4),
    ),
    model_fn=calc_const,
    dataset=DATASET,
    show_hessian=False,
)
assert jnp.allclose(fitted_constant_intensity_params.sp_inv_base_intensity,
                    softplus_inverse(closed_form_intensity))


# %%


class RbfParams(NamedTuple):
    sp_inv_base_rate: Array
    weights: Array = jnp.zeros((N_RBF_CENTERS,),) + 0.1


def calc_rbf(params: RbfParams, dataset: Dataset) -> ModelOutput:
    intensity = jax.nn.softplus(params.sp_inv_base_rate
                                + dataset.rbf_basis @ params.weights)
    return ModelOutput(
        point_term=dataset.curr_count * jnp.log(intensity),
        compensator=intensity * dataset.time_since_prev,
        forecast_intensity=intensity,
        reg_penalty=(
            jnp.zeros(())
            + jnp.square(params.sp_inv_base_rate) * 1e2
            + jnp.sum(jnp.square(params.weights)) * 1e2
            + jnp.square(jnp.sum(params.weights)) * 1e2
        ),
    )


def plot_rbf(sp_inv_base_intensity: Array, weights: Array) -> None:
    # exceed [0, 24] to verify periodic boundary conditions
    time_of_day = jnp.linspace(-4, 28, 500, endpoint=False)
    bases = calc_rbf_basis(time_of_day)
    intensity = jax.nn.softplus(sp_inv_base_intensity + bases @ weights)
    per_second = jax.nn.softplus(sp_inv_base_intensity) * ONE_SECOND
    per_basis = bases * weights

    f, axes = plt.subplots(2, 1, sharex=True)
    f.suptitle('exogenous intensity')
    for ax in axes:
        ax.axvline(0, c='k', alpha=0.2, linestyle='--')
        ax.axvline(24, c='k', alpha=0.2, linestyle='--')

    ax1, ax2 = axes
    ax1.plot(time_of_day, intensity)
    ax1.axhline(jax.nn.softplus(sp_inv_base_intensity),
                label=f'baseline $\\approx${per_second:.2f}/s',
                c='g', alpha=0.4, linestyle='-')
    ax1.set_ylabel('intensity')
    ax1.legend(loc='upper left')

    ax2.plot(time_of_day, per_basis, alpha=0.6)
    ax2.set_ylabel('weighted bases')
    ax2.set_xlabel('hour')
    plt.tight_layout()
    plt.show()


init_rbf_params = RbfParams(
    sp_inv_base_rate=fitted_constant_intensity_params.sp_inv_base_intensity,
)
plot_rbf(init_rbf_params.sp_inv_base_rate,
         init_rbf_params.weights)


# %%


fitted_rbf_params = run_optim(
    init_params=init_rbf_params,
    model_fn=calc_rbf,
    dataset=DATASET,
    show_hessian=True,
    force_plot=True,
)
plot_rbf(fitted_rbf_params.sp_inv_base_rate,
         fitted_rbf_params.weights)


# %%


# exponential decay kernel
class HawkesParams(NamedTuple):
    sp_inv_base_intensity: Array
    logit_branching_ratio: Array = logit(0.9)
    # 1 / avg_life
    sp_inv_decay_rate: Array = softplus_inverse(1.0 / ONE_MILLISECOND)


def calc_hawkes_baseline(params: HawkesParams, dataset: Dataset) -> ModelOutput:
    """Calculate hawkes process outputs sequentially using scan.

    This is a reference implementation which prioritises readability over
    performance, intended for comparison against optimised implementations.
    """
    assert jnp.all(dataset.curr_count > 0.0)
    assert jnp.all(dataset.time_since_prev > 0.0)

    base_intensity = jax.nn.softplus(params.sp_inv_base_intensity)
    branching_ratio = jax.nn.sigmoid(params.logit_branching_ratio)
    decay_rate = jax.nn.softplus(params.sp_inv_decay_rate)

    def step(carry, x):
        decayed_count = carry
        count, time_since_prev = x

        integral_over_interval = -jnp.expm1(-decay_rate * time_since_prev)
        compensator = \
            base_intensity * time_since_prev  \
            + branching_ratio * decayed_count * integral_over_interval

        decay_factor = jnp.exp(-decay_rate * time_since_prev)
        decayed_count *= decay_factor
        point_intensity = base_intensity \
            + branching_ratio * decay_rate * decayed_count

        # log rising factorial / Pochhammer to handle excitation across
        # events within the same timestamp
        a = point_intensity
        d = branching_ratio * decay_rate
        point_term = \
            count * jnp.log(d) \
            + gammaln(a / d + count) \
            - gammaln(a / d)

        decayed_count += count
        forecast_intensity = base_intensity \
            + branching_ratio * decay_rate * decayed_count
        return decayed_count, (point_term, compensator, forecast_intensity)

    xs = dataset.curr_count, dataset.time_since_prev
    _, (point_term, compensator, intensity) = jax.lax.scan(step, 0, xs)

    return ModelOutput(
        point_term=point_term,
        compensator=compensator,
        forecast_intensity=intensity,
    )

# %%


def calc_hawkes(params: HawkesParams, dataset: Dataset) -> ModelOutput:
    base_intensity = jax.nn.softplus(params.sp_inv_base_intensity)
    branching_ratio = jax.nn.sigmoid(params.logit_branching_ratio)
    decay_rate = jax.nn.softplus(params.sp_inv_decay_rate)

    chex.assert_tree_all_finite(base_intensity)
    chex.assert_tree_all_finite(branching_ratio)
    chex.assert_tree_all_finite(decay_rate)

    decay_factors = jnp.exp(-decay_rate * dataset.time_since_prev)
    chex.assert_tree_all_finite(decay_factors)
    decayed_count = calculate_decayed_counts(decay_factors, dataset.curr_count)
    chex.assert_tree_all_finite(decayed_count)

    prev_decayed_count = jnp.roll(decayed_count, 1).at[0].set(0.0)
    integral_over_interval = -jnp.expm1(-decay_rate * dataset.time_since_prev)
    compensator = \
        dataset.time_since_prev * base_intensity \
        + branching_ratio * prev_decayed_count * integral_over_interval
    chex.assert_tree_all_finite(compensator)

    # rising factorial / pochhammer
    curr_minus_count = prev_decayed_count * decay_factors
    d = branching_ratio * decay_rate
    a_over_d = (base_intensity / d) + curr_minus_count
    p1 = dataset.curr_count * jnp.log(d)
    p2 = gammaln(a_over_d + dataset.curr_count)
    p3 = -gammaln(a_over_d)
    point_term = p1 + p2 + p3
    chex.assert_tree_all_finite(p1)
    chex.assert_tree_all_finite(p2)
    chex.assert_tree_all_finite(p3)
    chex.assert_tree_all_finite(point_term)

    forecast_intensity = base_intensity \
        + branching_ratio * decay_rate * decayed_count
    return ModelOutput(
        point_term=point_term,
        compensator=compensator,
        forecast_intensity=forecast_intensity,
        reg_penalty=(
            jnp.zeros(())
            + jnp.square(params.sp_inv_base_intensity) * 1e5
            + jnp.square(params.logit_branching_ratio) * 1e5
            + jnp.square(params.sp_inv_decay_rate) * 1e5
        ),
    )


def plot_model_output(outputs: ModelOutput, input_df: pl.DataFrame) -> None:
    df = (
        input_df
        .with_columns(
            loglik=np.asarray(outputs.loglik),
            intensity=np.asarray(outputs.forecast_intensity),
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
    ax2.plot(subset['time'], subset[['intensity']])
    ax3.plot(subset['time'], subset[['loglik']])
    plt.tight_layout()
    plt.show()


def print_params(params) -> None:
    prev = params._asdict()
    series = pd.Series(prev, name='param')

    prefix_transforms = dict(
        log_=jnp.exp,
        logit_=jax.nn.sigmoid,
        sp_inv_=jax.nn.softplus,
    )
    for k, v in prev.items():
        for prefix, transform in prefix_transforms.items():
            if k.startswith(prefix):
                series[k.replace(prefix, '')] = transform(v)
    display(series.to_frame())


def show_hawkes(params: HawkesParams,
                dataset: Dataset,
                input_df: pl.DataFrame) -> None:
    baseline_outputs = calc_hawkes_baseline(params, dataset)

    outputs = chex.chexify(calc_hawkes)(params, dataset)
    assert jnp.allclose(outputs.forecast_intensity,
                        baseline_outputs.forecast_intensity)
    assert jnp.allclose(outputs.compensator,
                        baseline_outputs.compensator)
    assert jnp.allclose(outputs.point_term,
                        baseline_outputs.point_term,
                        rtol=1e-3, atol=1e-2)

    print_params(params)
    plot_model_output(outputs, input_df)


init_hawkes_params = HawkesParams(
    sp_inv_base_intensity=fitted_constant_intensity_params.sp_inv_base_intensity,
)
show_hawkes(init_hawkes_params, DATASET, INPUT_DF.filter('is_train'))


# %%


fitted_hawkes_params = run_optim(
    init_params=init_hawkes_params,
    model_fn=calc_hawkes,
    dataset=DATASET,
    show_hessian=True,
)
hawkes_outputs = calc_hawkes(fitted_hawkes_params, DATASET)
show_hawkes(fitted_hawkes_params, DATASET, INPUT_DF.filter('is_train'))


# %%


class RbfHawkesParams(NamedTuple):
    sp_inv_base_intensity: Array
    logit_branching_ratio: Array
    sp_inv_decay_rate: Array
    weights: Array


def calc_rbf_hawkes(params: RbfHawkesParams, dataset: Dataset) -> ModelOutput:
    base_intensity = jax.nn.softplus(params.sp_inv_base_intensity
                                     + dataset.rbf_basis @ params.weights)
    branching_ratio = jax.nn.sigmoid(params.logit_branching_ratio)
    decay_rate = jax.nn.softplus(params.sp_inv_decay_rate)
    decay_factors = jnp.exp(-decay_rate * dataset.time_since_prev)
    decayed_count = calculate_decayed_counts(decay_factors, dataset.curr_count)

    prev_decayed_count = jnp.roll(decayed_count, 1).at[0].set(0.0)
    integral_over_interval = -jnp.expm1(-decay_rate * dataset.time_since_prev)
    interval_term = \
        dataset.time_since_prev * base_intensity \
        + branching_ratio * prev_decayed_count * integral_over_interval
    chex.assert_tree_all_finite(interval_term)

    # rising factorial / pochhammer
    curr_minus_count = prev_decayed_count * decay_factors
    d = branching_ratio * decay_rate
    a_over_d = (base_intensity / d) + curr_minus_count
    point_term = dataset.curr_count * jnp.log(d) \
        + gammaln(a_over_d + dataset.curr_count) \
        - gammaln(a_over_d)
    chex.assert_tree_all_finite(point_term)

    forecast_intensity = base_intensity \
        + branching_ratio * decay_rate * decayed_count
    return ModelOutput(
        point_term=point_term,
        compensator=interval_term,
        forecast_intensity=forecast_intensity,
        reg_penalty=(
            jnp.square(params.sp_inv_base_intensity) * 1e3
            + jnp.square(params.logit_branching_ratio) * 1e3
            + jnp.square(params.sp_inv_decay_rate) * 1e3
            + jnp.sum(jnp.square(params.weights)) * 1e2
            + jnp.square(jnp.sum(params.weights)) * 1e2
        ),
    )


def show_rbf_hawkes(params: RbfHawkesParams, dataset: Dataset,
                    input_df: pl.DataFrame) -> None:
    outputs = chex.chexify(calc_rbf_hawkes)(params, dataset)
    print_params(params)
    plot_model_output(outputs, input_df)
    plot_rbf(sp_inv_base_intensity=params.sp_inv_base_intensity,
             weights=params.weights)


init_rbf_hawkes = RbfHawkesParams(
    sp_inv_base_intensity=fitted_hawkes_params.sp_inv_base_intensity,
    logit_branching_ratio=fitted_hawkes_params.logit_branching_ratio,
    sp_inv_decay_rate=fitted_hawkes_params.sp_inv_decay_rate,
    weights=fitted_rbf_params.weights,
)
show_rbf_hawkes(init_rbf_hawkes, DATASET, INPUT_DF.filter('is_train'))


# %%


fitted_rbf_hawkes_params = run_optim(
    init_params=init_rbf_hawkes,
    model_fn=calc_rbf_hawkes,
    dataset=DATASET,
    show_hessian=True,
)
show_rbf_hawkes(fitted_rbf_hawkes_params, DATASET, INPUT_DF.filter('is_train'))


# %%


class PowerLawHawkesParams(NamedTuple):
    sp_inv_base_intensity: Array
    logit_branching_ratio: Array
    sp_inv_beta_raw: Array = softplus_inverse(0.2)


def calc_power_law_hawkes(params: PowerLawHawkesParams, dataset: Dataset) -> ModelOutput:
    # KNOWN_LIMITATION: omega is fixed to avoid identifiability with beta
    # omega^{-1} roughly corresponds to where the plateau crosses over to
    # power-law tail
    omega = 1.0 / ONE_MILLISECOND
    base_intensity = jax.nn.softplus(params.sp_inv_base_intensity)
    branching_ratio = jax.nn.sigmoid(params.logit_branching_ratio)
    beta = jax.nn.softplus(params.sp_inv_beta_raw)
    chex.assert_tree_all_finite(base_intensity)
    chex.assert_tree_all_finite(branching_ratio)
    chex.assert_tree_all_finite(beta)

    cache = dataset.power_law_cache
    weights = power_law_approx.calc_weights(
        omega=omega,
        beta=beta,
        rates=cache.decay_rates,
    )
    chex.assert_trees_all_close(jnp.sum(weights), jnp.array(1.0))

    kernel_integral = jnp.sum(weights / cache.decay_rates)
    kernel_factor = branching_ratio / kernel_integral
    compensator_integral = cache.decay_integral @ weights
    compensator = \
        dataset.time_since_prev * base_intensity \
        + kernel_factor * compensator_integral
    chex.assert_tree_all_finite(kernel_integral)
    chex.assert_tree_all_finite(kernel_factor)
    chex.assert_tree_all_finite(compensator)

    # rising factorial / pochhammer
    total_a = base_intensity \
        + kernel_factor * (cache.curr_minus_count @ weights)
    total_d = kernel_factor  # weights sum to 1
    a_over_d = total_a / total_d
    point_term = \
        dataset.curr_count * jnp.log(total_d) \
        + gammaln(a_over_d + dataset.curr_count) \
        - gammaln(a_over_d)
    chex.assert_tree_all_finite(total_a)
    chex.assert_tree_all_finite(total_d)
    chex.assert_tree_all_finite(a_over_d)
    chex.assert_tree_all_finite(point_term)

    forecast_intensity = base_intensity \
        + kernel_factor * cache.decayed_count @ weights

    return ModelOutput(
        point_term=point_term,
        compensator=compensator,
        forecast_intensity=forecast_intensity,
        reg_penalty=(
            jnp.zeros(())
            # + jnp.square(params.sp_inv_base_intensity) * 1e5
            # + jnp.square(params.logit_branching_ratio) * 1e5
            # + jnp.square(params.sp_inv_beta_raw) * 1e5
        ),
    )


def show_power_law_hawkes(params: PowerLawHawkesParams,
                          dataset: Dataset,
                          input_df: pl.DataFrame) -> None:
    outputs = calc_power_law_hawkes(params, dataset)
    print_params(params)
    plot_model_output(outputs=outputs, input_df=input_df)


init_pl_hawkes_params = PowerLawHawkesParams(
    sp_inv_base_intensity=fitted_hawkes_params.sp_inv_base_intensity,
    logit_branching_ratio=fitted_hawkes_params.logit_branching_ratio,
)
show_power_law_hawkes(init_pl_hawkes_params, DATASET,
                      INPUT_DF.filter('is_train'))


# %%


fitted_power_law_hawkes_params = run_optim(
    init_params=init_pl_hawkes_params,
    model_fn=calc_power_law_hawkes,
    dataset=DATASET,
    show_hessian=True,
)
show_power_law_hawkes(fitted_power_law_hawkes_params,
                      DATASET, INPUT_DF.filter('is_train'))


# %%


DATASET_INCLUDING_VALIDATION_DATA = create_dataset(INPUT_DF)


def _calc_model_outputs[Params: chex.ArrayTree](prefix: str,
                                                params: Params,
                                                model_fn: ModelFn[Params]) -> dict[str, np.ndarray]:
    # full dataset is used to avoid bias from a separate "warm up" step
    # without having to pass state across datasets
    outputs = model_fn(params, DATASET_INCLUDING_VALIDATION_DATA)
    return {
        f'{prefix}_compensator': np.asarray(outputs.compensator),
        f'{prefix}_loglik': np.asarray(outputs.loglik),
    }


WITH_MODEL_OUTPUTS = (
    INPUT_DF
    .with_columns(
        **_calc_model_outputs('constant', fitted_constant_intensity_params, calc_const),
        **_calc_model_outputs('rbf', fitted_rbf_params, calc_rbf),
        **_calc_model_outputs('hawkes', fitted_hawkes_params, calc_hawkes),
        **_calc_model_outputs('rbf_hawkes', fitted_rbf_hawkes_params, calc_rbf_hawkes),
        **_calc_model_outputs('pl_hawkes', fitted_power_law_hawkes_params, calc_power_law_hawkes),
    )
)


display(WITH_MODEL_OUTPUTS)

# %%


display(
    WITH_MODEL_OUTPUTS
    .group_by('is_train')
    .agg(pl.selectors.ends_with('_loglik').mean())
    .sort('is_train', descending=True)
    .to_pandas()
    .rename(columns=lambda x: x.replace('_loglik', ''))
    .style.background_gradient(axis=1)
)


# %%


def plot_point_process_qq(start: datetime.datetime,
                          end: datetime.datetime,
                          title: str):
    subset = WITH_MODEL_OUTPUTS.filter(
        pl.col('time') >= start,
        pl.col('time') <= end,
    )
    assert (subset['curr_count'] > 0).all(), \
        'need to sum compensators by event'

    # KNOWN_LIMITATION: There are many timestamps with more than one event.
    # This is handled by introducing zero-valued compensators and should result
    # in a poor fit to exponential distribution
    extra_counts = (subset  # number of zero-compensators
                    .select((pl.col('curr_count') - 1).clip(lower_bound=0).sum())
                    .item())
    n_events = subset.height + extra_counts
    empties = jnp.zeros(extra_counts)

    max_subsamples = 100_000
    if n_events > max_subsamples:
        indices = jnp.linspace(0, n_events-1, max_subsamples, dtype=int)
    else:
        indices = jnp.arange(n_events, dtype=int)

    prefixes = [
        'constant',
        'rbf',
        'hawkes',
        'rbf_hawkes',
        'pl_hawkes',
    ]
    f, axes = plt.subplots(3, 2, sharex=True, sharey=False, figsize=(6, 8))
    f.suptitle(f'rescaled time plots for {title}\n[ {start} , {end}]')
    for i, prefix in enumerate(prefixes):
        compensator = (
            subset[f'{prefix}_compensator']
            .sort()  # needed for subsample indices to get the right quantiles
                     # without missing tails
            .to_jax()
        )
        rescaled_times = jnp.concat([empties, compensator])
        assert jnp.all(jnp.diff(rescaled_times) >= 0)
        assert len(rescaled_times) == n_events, \
            f'{len(rescaled_times)=}, {n_events=}'
        subsamples = rescaled_times[indices]

        ax = axes[i // 2, i % 2]
        scipy.stats.probplot(subsamples, dist='expon',
                             sparams=(0, 1), plot=ax)
        ax.set_title(f'{prefix}')
        ax.set_ylabel('Predicted Quantiles')
        ax.set_xlabel('Exponential(1) Quantiles')
        ax.grid(True, alpha=0.3)

    for j in range(len(prefixes), len(axes.flatten())):
        axes.flatten()[j].axis('off')

    plt.tight_layout()
    plt.show()


plot_point_process_qq(DATA_START, TRAIN_END, 'train')
plot_point_process_qq(TRAIN_END + datetime.timedelta(days=1), VAL_END,
                      'validation')


# %%


def plot_results(start: datetime.datetime,
                 end: datetime.datetime,
                 *,
                 duration: str,
                 prefixes: list[str] | None = None) -> None:
    df = (
        WITH_MODEL_OUTPUTS
        .filter(
            pl.col('time') >= start,
            pl.col('time') <= end,
        )
        .with_columns(
            pl.col('curr_count').alias('actual_count'),
            pl.col('time_since_prev').cast(float),
            (
                pl.selectors.ends_with('_compensator')
                .name.replace('_compensator', '_expected_count')
            ),
        )
        .group_by(pl.col('time').dt.truncate(duration), maintain_order=True)
        .agg(
            pl.col('hi_price').max(),
            pl.col('lo_price').min(),
            pl.col('actual_count').sum(),
            pl.selectors.ends_with('_expected_count').sum(),
        )
        .with_columns(
            (
                -pl.selectors.ends_with('_expected_count')
                .name.replace('_expected_count', '_err')
                + pl.col('actual_count')
            )
            / pl.selectors.ends_with('_expected_count').sqrt(),
        )
        .to_pandas()
        .set_index('time')
    )

    prefixes = prefixes or [
        'constant',
        'rbf',
        'hawkes',
        'rbf_hawkes',
        'pl_hawkes',
    ]
    f, axes = plt.subplots(1 + len(prefixes), 1,
                           sharex=True, sharey=False, figsize=(8, 12))
    title = '\n'.join(
        (
            f'counts over {duration} buckets',
            f'[ {start} , {end} ]',
        )
    )
    f.suptitle(title)
    count_ax = axes[0]
    count_ax.set_title('actual')
    count_ax.set_yscale('log')
    count_ax.scatter(df.index, df['actual_count'],
                     alpha=0.4, marker='+', c='C0')
    count_ax.set_ylabel('actual count $y$', c='C0')

    price_ax = count_ax.twinx()
    price_ax.plot(df.index, df[['hi_price', 'lo_price']],
                  alpha=0.6, drawstyle='steps-post', c='C1')
    price_ax.set_ylabel('price (hi, lo)', c='C1')

    for i, prefix in enumerate(prefixes):
        ax1 = axes[1 + i]
        ax1.set_title(prefix)
        ax1.set_yscale('log')
        ax1.scatter(df.index, df[f'{prefix}_expected_count'],
                    alpha=0.6, marker='+', c='C0')
        ax1.set_ylabel('expected count $\\hat y$', c='C0')

        ax2 = ax1.twinx()
        ax2.scatter(df.index, df[f'{prefix}_err'],
                    alpha=0.4, marker='x', c='C2')
        ax2.set_ylabel(r'$(y - \hat y) / \sqrt{\hat y}$', c='C2')
        ax2.axhline(0.0, linestyle='--', c='C2', alpha=0.6)

    for ax in axes:
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.tick_params(axis='x', labelrotation=30)
        if start <= TRAIN_END and TRAIN_END <= end:
            ax.axvline(TRAIN_END, linestyle='--', c='k', alpha=0.4)

    plt.tight_layout()
    plt.show()


plot_results(DATA_START, VAL_END, duration='2h')


# %%
# arbitrary training date
plot_results(datetime.datetime(2025, 9, 17, hour=9),
             datetime.datetime(2025, 9, 18, hour=6),
             duration='2m')
plot_results(datetime.datetime(2025, 9, 17, hour=17),
             datetime.datetime(2025, 9, 17, hour=20),
             duration='20s', prefixes=['hawkes', 'rbf_hawkes', 'pl_hawkes'])
# %%
plot_results(datetime.datetime(2025, 9, 17, hour=17, minute=59, second=45,
                               microsecond=0),
             datetime.datetime(2025, 9, 17, hour=18, minute=0, second=15,
                               microsecond=0),
             duration='100ms', prefixes=['hawkes', 'rbf_hawkes', 'pl_hawkes'])


# %%
# 2025/10/10 volatility event
plot_results(datetime.datetime(2025, 10, 10, hour=9),
             datetime.datetime(2025, 10, 11, hour=6),
             duration='90s')
plot_results(datetime.datetime(2025, 10, 10, hour=19),
             datetime.datetime(2025, 10, 11, hour=1),
             duration='30s', prefixes=['hawkes', 'rbf_hawkes', 'pl_hawkes'])
# %%
plot_results(datetime.datetime(2025, 10, 10, hour=21, minute=12, second=45),
             datetime.datetime(2025, 10, 10, hour=21, minute=13, second=45),
             duration='100ms', prefixes=['hawkes', 'rbf_hawkes', 'pl_hawkes'])


# %%
