import jax
import jax.numpy as jnp
import pytest
from jax import Array


@jax.jit
def calculate_decayed_counts(decay_factors: Array, counts: Array) -> Array:
    def combine(prefix, step):
        decay_left, count_left = prefix
        decay_right, count_right = step
        combined_decay = decay_right * decay_left
        combined_count = decay_right * count_left + count_right
        return combined_decay, combined_count

    elems = decay_factors, counts
    _, decayed_counts = jax.lax.associative_scan(combine, elems)
    return decayed_counts


def get_scan_baseline(decay_factors: Array, counts: Array) -> Array:
    """Reference implementation using sequential jax.lax.scan."""

    @jax.jit
    def linear_scan_step(carry, xs):
        prev_decayed = carry
        decay_factor, curr_count = xs
        next_decayed = prev_decayed * decay_factor + curr_count
        return next_decayed, next_decayed

    init = jnp.zeros_like(decay_factors[0])
    _, expected = jax.lax.scan(linear_scan_step, init, (decay_factors, counts))
    return expected


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_decayed_counts_correctness(seed):
    """Verifies that associative scan matches the sequential scan baseline."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    n = 1_000

    decay_factors = jax.nn.sigmoid(10 * jax.random.normal(k1, (n,)))
    counts = 1.0 + jax.random.randint(k2, (n,), 0, 5)

    actual = calculate_decayed_counts(decay_factors, counts)
    expected = get_scan_baseline(decay_factors, counts)

    assert jnp.allclose(actual, expected)


@pytest.mark.parametrize("n", [1_000_000, 10_000_000, 100_000_000])
@pytest.mark.parametrize(
    "func",
    [
        calculate_decayed_counts,
        get_scan_baseline,
    ],
)
def test_performance(benchmark, n, func):
    """
    Optional: Performance regression check.
    Note: Real-world benchmarks usually belong in a separate suite (e.g., pytest-benchmark).
    """

    benchmark.group = f"size-{n:,}"

    key = jax.random.PRNGKey(0)

    decay_factors = jax.nn.sigmoid(jax.random.normal(key, (n,)))
    counts = jnp.ones((n,))

    def blocking_func(decay_factors, counts):
        return func(decay_factors, counts).block_until_ready()

    blocking_func(decay_factors, counts)  # warm up

    benchmark(blocking_func, decay_factors, counts)
