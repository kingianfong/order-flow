
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

    # Handle broadcasting if counts is 1D but decay_factors is 2D
    if counts.ndim == 1 and decay_factors.ndim > 1:
        counts = counts[:, jnp.newaxis]
        counts = jnp.broadcast_to(counts, decay_factors.shape)

    elems = decay_factors, counts
    _, decayed_counts = jax.lax.associative_scan(combine, elems)
    return decayed_counts


def get_scan_baseline(decay_factors: Array, counts: Array) -> Array:
    """Reference implementation using sequential jax.lax.scan."""
    @jax.jit
    def linear_scan_step(carry, xs):
        decay_factor, count = xs
        next_count = carry * decay_factor + count
        return next_count, next_count

    # Handle broadcasting for baseline to match logic
    if counts.ndim == 1 and decay_factors.ndim > 1:
        counts = jnp.broadcast_to(counts[:, jnp.newaxis], decay_factors.shape)
        init = jnp.zeros(decay_factors.shape[1])
    else:
        init = 0.0

    _, expected = jax.lax.scan(linear_scan_step, init, (decay_factors, counts))
    return expected


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("shape_type", ["1d", "multidimensional"])
def test_decayed_counts_correctness(seed, shape_type):
    """Verifies that associative scan matches the sequential scan baseline."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    n = 1_000

    if shape_type == "1d":
        decay_factors = jax.nn.sigmoid(10 * jax.random.normal(k1, (n,)))
        counts = 1.0 + jax.random.randint(k2, (n,), 0, 5)
    else:
        m = 8
        decay_factors = jax.nn.sigmoid(10 * jax.random.normal(k1, (n, m)))
        counts = 1.0 + jax.random.randint(k2, (n,), 0, 5)

    actual = calculate_decayed_counts(decay_factors, counts)
    expected = get_scan_baseline(decay_factors, counts)

    # Use higher tolerance for float32 accumulation if necessary,
    # though allclose is usually fine.
    assert jnp.allclose(actual, expected)


@pytest.mark.parametrize("n", [1_000_000, 10_000_000, 100_000_000])
@pytest.mark.parametrize("func", [
    calculate_decayed_counts,
    get_scan_baseline,
])
def test_performance(benchmark, n, func):
    """
    Optional: Performance regression check.
    Note: Real-world benchmarks usually belong in a separate suite (e.g., pytest-benchmark).
    """

    benchmark.group = f'size-{n:,}'

    key = jax.random.PRNGKey(0)

    decay_factors = jax.nn.sigmoid(jax.random.normal(key, (n,)))
    counts = jnp.ones((n,))

    def blocking_func(decay_factors, counts):
        return func(decay_factors, counts).block_until_ready()

    blocking_func(decay_factors, counts)  # warm up

    benchmark(blocking_func, decay_factors, counts)
