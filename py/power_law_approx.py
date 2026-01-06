# %%


from typing import NamedTuple

from jax import Array
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# %%


class PowerLawApproxParams(NamedTuple):
    weights: Array
    rates: Array


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


def kernel_power_law(t: Array, omega: float, beta: float):
    numerator = omega * beta
    denominator = (1 + omega * t) ** (1 + beta)
    return numerator / denominator


def kernel_power_law_params(omega: ArrayLike,
                            beta: ArrayLike,
                            max_history_duration: ArrayLike,
                            n_exponentials: int):

    alpha = 1.0 + beta
    indices = jnp.arange(n_exponentials)
    logb = jnp.log1p(omega * max_history_duration) / n_exponentials
    inv_bi = jnp.exp(-indices * logb)          # b^{-i}
    r = alpha * inv_bi                   # r_i = α / b^i

    # a_i = b^{-iα}
    a = jnp.exp(-alpha * indices * logb)

    # Normalize so that Σ a_i e^{-r_i} = 1  (=> Σ kernel_weights = ωβ)
    er = jnp.exp(-r)
    Z = jnp.sum(a * er)
    a = a / Z

    kernel_rates = omega * r
    kernel_weights = omega * beta * a * er

    return PowerLawApproxParams(
        weights=jnp.asarray(kernel_weights),
        rates=jnp.asarray(kernel_rates),
    )


def plot_kernel_power_law():
    n = 10_000
    min_t = 1e-3
    max_t = 1e5
    t_geom = jnp.geomspace(min_t, max_t, n)

    omega = 0.1
    beta = 0.2  # The 'beta' in the kernel definition
    max_history_duration = 1e5
    n_exponentials = 8

    kernel_params = kernel_power_law_params(
        omega=omega,
        beta=beta,
        max_history_duration=max_history_duration,
        n_exponentials=n_exponentials,
    )

    exact = kernel_power_law(t_geom, omega=omega, beta=beta)
    _, approx = approx_iterative(t_geom, kernel_params)

    plt.figure(figsize=(8, 5))
    plt.loglog(t_geom, exact, 'k-', label='Exact Lomax Kernel', lw=2)
    plt.loglog(t_geom, approx, 'r--', label='SOE Approximation', lw=2)
    plt.title(
        f"Hawkes Power-Law Kernel (Lomax) Approximation\n$\\omega={omega}, \\beta={beta}$")
    plt.xlabel("Time (t)")
    plt.ylabel("Kernel Value $k(t)$")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()


if __name__ == '__main__':
    plot_kernel_power_law()


# %%
