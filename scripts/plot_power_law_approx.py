# %%


from typing import NamedTuple

from jax import Array
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


# %%


def exact_power_law(t: Array, alpha: float) -> Array:
    return jnp.pow(t, -alpha)


# https://arxiv.org/abs/physics/0605149
class PowerLawApproxParams(NamedTuple):
    weights: Array
    rates: Array


def uniform_approx_params(alpha: float,
                          max_history_duration: float,
                          n_exponentials: int) -> PowerLawApproxParams:
    k = jnp.log10(max_history_duration)
    beta = 10 ** (k / n_exponentials)
    indices = jnp.arange(n_exponentials + 1)

    weight_denom = (
        beta**(-indices * alpha)
        @ jnp.exp(-alpha / (beta**indices))
    )
    weight_numer = beta**(-indices*alpha)
    weights = weight_numer / weight_denom
    rates = alpha / (beta**indices)
    return PowerLawApproxParams(weights=weights, rates=rates)


def higher_order_approx_params(alpha: float,
                               max_history_duration: float,
                               n_exponentials: int,
                               n_order: int) -> PowerLawApproxParams:
    k = jnp.log10(max_history_duration)
    beta = jnp.pow(10, k / n_exponentials)
    indices = jnp.arange(n_exponentials + 1)

    # mu = (
    #     jax.scipy.special.gamma(alpha + n_order)
    #     / jax.scipy.special.gamma(alpha)
    # ) ** (1 / n_order)

    ln_mu = (
        jax.scipy.special.gammaln(alpha + n_order)
        - jax.scipy.special.gammaln(alpha)
    ) / n_order
    mu = jnp.exp(ln_mu)

    weight_denom = (
        beta**(-indices * alpha)
        @ jnp.exp(-mu / (beta**indices))
    )
    weight_numer = beta**(-indices*alpha)
    weights = weight_numer / weight_denom
    rates = mu / (beta**indices)
    return PowerLawApproxParams(weights=weights, rates=rates)


def approx_iterative(t: Array, params: PowerLawApproxParams) -> tuple[Array, Array]:

    def step(carry, xs):
        prev_t, decayed = carry
        curr_t = xs
        elapsed = curr_t - prev_t

        decay_rates = jnp.exp(-params.rates * elapsed)
        decayed *= decay_rates
        curr_val = params.weights @ decayed

        return (curr_t, decayed), (decayed, curr_val)

    init_carry = 0.0, jnp.ones_like(params.rates)
    _, (components, final) = jax.lax.scan(step, init_carry, xs=t)

    return components, final


def plot_power_law_kernel(*, apply_jitter: bool):
    print(f'{apply_jitter=}')

    n = 1_000
    max_t = 10**5
    t_geom = jnp.geomspace(1, max_t, n)

    if apply_jitter:
        key = jax.random.PRNGKey(n)
        normal = jax.random.normal(key, (n, ))
        jitter = 10.0 * jnp.exp(normal).cumsum()
        t_geom += jitter

    alpha = 2.0
    max_history_duration = 10**4
    n_exponentials = 6
    n_order = 4

    exact = exact_power_law(t_geom, alpha)

    uniform_components, uniform = approx_iterative(
        t_geom,
        uniform_approx_params(
            alpha=alpha,
            max_history_duration=max_history_duration,
            n_exponentials=n_exponentials,
        ),
    )
    higher_order_components, higher_order = approx_iterative(
        t_geom,
        higher_order_approx_params(
            alpha=alpha,
            max_history_duration=max_history_duration,
            n_exponentials=n_exponentials,
            n_order=n_order,
        ),
    )

    kwargs = dict(drawstyle='steps-post', alpha=0.6)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(5, 10))
    ax1.loglog(t_geom, exact, label='exact', **kwargs)
    ax1.loglog(t_geom, uniform, label='uniform', **kwargs)
    ax1.loglog(t_geom, higher_order, label='higher_order', **kwargs)
    ax1.legend()

    ax2.loglog(t_geom, abs(uniform-exact), label='uniform', **kwargs)
    ax2.loglog(t_geom, abs(higher_order-exact), label='higher_order', **kwargs)
    ax2.set_title('abs(approx-exact)')
    ax2.legend()

    ax3.loglog(t_geom, uniform_components, **kwargs)
    ax3.set_title('uniform_components')

    ax4.loglog(t_geom, higher_order_components, **kwargs)
    ax4.set_title('higher_order_components')

    plt.tight_layout()
    plt.show()


plot_power_law_kernel(apply_jitter=False)
plot_power_law_kernel(apply_jitter=True)


# %%
