from jax import Array
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

from decayed_counts import calculate_decayed_counts


# fix the decay rates for each exponential
def calc_decay_rates(min_history: float,
                     max_history: float,
                     n_exponentials: int) -> Array:
    return jnp.geomspace(1.0 / max_history, 1.0 / min_history, n_exponentials)


def calc_weights(omega: ArrayLike, beta: ArrayLike, rates: Array) -> Array:
    # to approximate lomax decay as a sum of exponentials by using the
    # Laplace transform of Gamma(1+beta, scale=omega):
    #   (1 + omega * t) ^ -(1 + beta) = E[ exp{-t * X} ]
    # where
    #   f(x) = k * x^{beta} * exp{-x / omega}

    # quadrature weights:
    # approximate integral[ f(x)   * exp(-x   * t)   dx        ]
    # as               sum[ f(x_i) * exp(-x_i * t) * delta_x_i ]
    # delta_x_i is proportional to x_i because x_i is geomspaced.
    # This can be derived using change of variable x = exp{u} for the integral
    log_pdf = jax.scipy.stats.gamma.logpdf(rates, a=1 + beta, scale=omega)
    unnorm_log_weights = log_pdf + jnp.log(rates)
    weights = jax.nn.softmax(unnorm_log_weights)
    return weights


@pytest.mark.parametrize("omega_pow", range(-5, 2 + 1))
@pytest.mark.parametrize("beta_step", range(1, 20 + 1))
@pytest.mark.parametrize("t", [0.01, 1.0, 100.0])
def test_calc_weights(omega_pow, beta_step, t):
    rates = calc_decay_rates(1e-3, 1e6, n_exponentials=32)
    omega = jnp.power(10.0, omega_pow)
    beta = 0.05 * beta_step
    weights = calc_weights(omega, beta, rates)

    assert jnp.all(jnp.isfinite(weights))
    assert jnp.all(weights >= 0)
    assert jnp.sum(weights) == pytest.approx(1.0)

    approx_decay = weights @ jnp.exp(-rates * t)
    exact_decay = jnp.power(1.0 + omega * t, -(1.0 + beta))
    assert approx_decay == pytest.approx(exact_decay, rel=0.01)

    actual_mean = weights @ rates
    theo_mean = omega * (1.0 + beta)
    assert actual_mean == pytest.approx(theo_mean, rel=0.1)


def plot_power_law_decay() -> None:
    def calc_exact(t: Array, omega: ArrayLike, beta: ArrayLike):
        return (1.0 + omega * t) ** -(1.0 + beta)

    # done sequentially to ensure markovian nature holds
    def calc_approx(t: Array, rates: Array, weights: Array) -> Array:
        elapsed = jnp.concat([jnp.zeros(1), jnp.diff(t)])
        counts = jnp.zeros_like(elapsed).at[0].set(1)
        outer = jnp.outer(elapsed, rates)
        decay_rates = jnp.exp(-outer)
        decayed_counts = calculate_decayed_counts(decay_rates, counts)
        return decayed_counts @ weights

    one_second = 1.0
    one_ms = one_second * 1e-3
    one_minute = one_second * 60.0
    one_hour = one_minute * 60

    n = 10_000
    min_t = one_ms
    max_t = one_hour
    t_geom = jnp.geomspace(min_t, max_t * 10, n)
    orders_of_magnitude = jnp.log10(max_t / min_t).item()
    print(f'{orders_of_magnitude=:.2f}')
    n_exponentials = 14  # 2x orders of magnitude
    rates = calc_decay_rates(
        min_history=min_t,
        max_history=max_t,
        n_exponentials=n_exponentials,
    )

    inv_omegas = one_ms,  one_minute,  one_hour
    betas = 0.15, 0.3, 0.5

    f1, axes1 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(6, 6))
    f1.suptitle('Lomax Kernel $(1 + \\omega \\beta) ^{-(1 + \\beta)}$')
    f2, axes2 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(6, 6))
    f2.suptitle('linear y')
    f3, axes3 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(6, 6))
    f3.suptitle('linear x, linear y')
    f4, axes4 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(6, 6))
    f4.suptitle('absolute error')
    f5, axes5 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(6, 6))
    f5.suptitle('relative error')

    for r, inv_omega in enumerate(inv_omegas):
        omega = 1.0 / inv_omega
        for c, beta in enumerate(betas):

            weights = calc_weights(
                omega=omega,
                beta=beta,
                rates=rates,
            )
            exact_integral = 1 / (omega * beta)
            approx_integral = jnp.sum(weights / rates).item()
            assert jnp.allclose(approx_integral, exact_integral,
                                rtol=1e-4, atol=jnp.inf), \
                f'{approx_integral=}, {exact_integral=}'

            exact = calc_exact(t_geom, omega=omega, beta=beta)
            approx = calc_approx(t_geom, rates=rates, weights=weights)

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
            ax2.axvline(max_t, c='k', linestyle='--', alpha=0.2)
            ax2.set_xscale('log')
            ax2.set_title(f"$\\omega$={omega:1.1e},$\\beta$={beta}")
            ax2.grid(True, which="both", ls="--", alpha=0.2)
            ax2.legend(loc='upper right')

            ax3 = axes3[r, c]
            keep = t_geom < (2 * one_hour)
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
