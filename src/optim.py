from pathlib import Path
from typing import NamedTuple, Callable
import datetime

from IPython.display import display
from jax.flatten_util import ravel_pytree
import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import seaborn as sns


def get_pytree_labels(params: chex.ArrayTree) -> list[str]:
    """Generates a list of strings representing each scalar in the PyTree."""

    leaves_with_path = jax.tree_util.tree_leaves_with_path(params)
    labels = []
    for path, leaf in leaves_with_path:
        # Convert path tuple (e.g., (DictKey(key='w'),)) to a string "w"
        path_str = ".".join([str(p.key if hasattr(p, "key") else p) for p in path])

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


class FitDiagnostics(NamedTuple):
    results_df: pd.DataFrame
    hess_corr_mat: np.ndarray
    mask: np.ndarray
    cond: float
    labels: list[str]


def compute_fit_diagnostics[Params: chex.ArrayTree](
    params: Params,
    per_obs_loss_fn: Callable,
    n_samples: int,
    data: chex.ArrayTree,
) -> FitDiagnostics:
    flat_params, unravel = ravel_pytree(params)
    n_params = len(flat_params)
    assert n_params <= 1_000, f"{n_params=}, hessian is memory intensive"
    labels = get_pytree_labels(params)

    def flat_per_obs_loss(flat_p, data):
        return per_obs_loss_fn(unravel(flat_p), data=data)

    def flat_loss(flat_p, data):
        return jnp.mean(flat_per_obs_loss(flat_p, data))

    calc_hess = chex.chexify(jax.jit(jax.hessian(flat_loss)))
    calc_jacobian = chex.chexify(jax.jit(jax.jacfwd(flat_per_obs_loss)))

    print("calculating hessian")
    start_time = datetime.datetime.now()
    hess_mat = calc_hess(flat_params, data)
    hess_mat.block_until_ready()
    print(f"hessian took {datetime.datetime.now() - start_time}")

    eigvals = jnp.linalg.eigvalsh(hess_mat)
    if jnp.any(eigvals <= 0):
        print("Warning: Hessian not positive definite. Standard errors invalid.")

    cond = jnp.linalg.cond(hess_mat).item()
    inv_hess_mat = jnp.linalg.pinv(hess_mat)
    diag_sqrt = jnp.sqrt(jnp.diag(inv_hess_mat))
    hess_se = diag_sqrt / jnp.sqrt(n_samples)
    hess_corr_mat = inv_hess_mat / jnp.outer(diag_sqrt, diag_sqrt)
    mask = np.triu(np.ones_like(hess_corr_mat, dtype=bool))

    print("calculating sandwich estimator errors")
    start_time = datetime.datetime.now()
    jacobian = calc_jacobian(flat_params, data)
    assert jacobian.shape == (n_samples, n_params), (
        f"{jacobian.shape=}, {n_samples=}, {n_params=}"
    )
    meat = (jacobian.T @ jacobian) / n_samples
    bread = inv_hess_mat
    robust_var = (bread @ meat @ bread) / n_samples
    robust_se = jnp.sqrt(jnp.diag(robust_var))
    robust_se.block_until_ready()
    print(f"errors took {datetime.datetime.now() - start_time}")

    mean = flat_params
    z_score = mean / robust_se
    p_value = 2 * jax.scipy.stats.norm.sf(jnp.abs(z_score))
    se_ratio = robust_se / hess_se

    results_df = pd.DataFrame(
        dict(
            mean=mean,
            hess_se=hess_se,
            robust_se=robust_se,
            z_score=z_score,
            p_value=p_value,
            se_ratio=se_ratio,
        ),
        index=labels,
    )

    return FitDiagnostics(
        results_df=results_df,
        hess_corr_mat=np.asarray(hess_corr_mat),
        mask=mask,
        cond=cond,
        labels=labels,
    )


def plot_fit_diagnostics(diag: FitDiagnostics, *, out_dir: Path) -> None:
    diag.results_df.to_csv(out_dir / "diagnostics.csv")

    print("plotting hessian")
    start_time = datetime.datetime.now()
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        data=pd.DataFrame(diag.hess_corr_mat, index=diag.labels, columns=diag.labels),
        mask=diag.mask,
        center=0.0,
        robust=False,
    )
    plt.title(f"Inverse Hessian Matrix cond={diag.cond:.2f}")
    plt.tight_layout()
    plt.savefig(out_dir / "inv_hessian.png")
    plt.show()
    print(f"plotting took {datetime.datetime.now() - start_time}")

    display(
        diag.results_df.style.background_gradient(
            subset=["hess_se", "robust_se", "se_ratio"],
        ).format(
            dict(
                mean="{:.4f}",
                hess_se="{:.4f}",
                robust_se="{:.4f}",
                z_score="{:.2f}",
                p_value="{:.2%}",
                se_ratio="{:.4f}",
            )
        )
    )


class OptimResult[Params: chex.ArrayTree](NamedTuple):
    params: Params
    convergence_stats: pd.Series
    diagnostics: FitDiagnostics | None = None


def run_optim[Params: chex.ArrayTree](
    init_params: Params,
    loss_fn: Callable,
    data: chex.ArrayTree,
    *,
    per_obs_loss_fn: Callable,
    n_samples: int,
    verbose: bool = False,
) -> OptimResult[Params]:
    timeout = datetime.timedelta(seconds=150)
    max_iter = 50
    tol = 1e-3

    opt = optax.lbfgs(
        memory_size=100,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=20, verbose=verbose, initial_guess_strategy="one"
        ),
    )

    value_and_grad_fun = optax.value_and_grad_from_state(loss_fn)

    @chex.chexify
    @jax.jit
    def update(carry, data):
        params, state, *_ = carry
        value, grad = value_and_grad_fun(params, state=state, data=data)
        updates, state = opt.update(
            grad,
            state,
            params,
            value=value,
            grad=grad,
            value_fn=loss_fn,
            data=data,
        )
        params = optax.apply_updates(params, updates)
        return params, state, value, grad

    params = init_params
    state = opt.init(init_params)

    start_time = datetime.datetime.now()
    n_iter = 0
    n_linesearch = 0
    converged = False

    while True:
        params, state, loss, grad = update((params, state), data=data)
        _, _, linesearch_state = state

        assert isinstance(linesearch_state, optax.ScaleByZoomLinesearchState), (
            f"{type(linesearch_state)=}"
        )
        n_linesearch += int(linesearch_state.info.num_linesearch_steps)
        grad_norm = optax.tree.norm(grad)

        elapsed = datetime.datetime.now() - start_time
        if grad_norm <= tol:
            converged = True
            break
        n_iter += 1
        if n_iter == max_iter or elapsed > timeout:
            print("did not converge")
            break

    jax.block_until_ready(grad_norm)
    elapsed = datetime.datetime.now() - start_time
    n_params = len(ravel_pytree(params)[0])

    convergence_stats = pd.Series(
        dict(
            converged=converged,
            n_iter=n_iter,
            n_linesearch=n_linesearch,
            n_params=n_params,
            elapsed_seconds=elapsed.total_seconds(),
        ),
    )

    diagnostics = None
    if per_obs_loss_fn is not None and n_samples is not None:
        diagnostics = compute_fit_diagnostics(
            params, per_obs_loss_fn, n_samples, data=data
        )

    return OptimResult(
        params=params,
        convergence_stats=convergence_stats,
        diagnostics=diagnostics,
    )


def plot_optim(result: OptimResult, *, out_dir: Path) -> None:
    display(result.convergence_stats.to_frame())
    result.convergence_stats.to_json(out_dir / "convergence_stats.json", indent=2)

    if result.diagnostics is not None:
        plot_fit_diagnostics(result.diagnostics, out_dir=out_dir)
