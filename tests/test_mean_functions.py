import jax
import jax.numpy as jnp
import pytest
# from gpjax.mean_functions import Constant, Linear, Zero
from gpjax.mean_functions import Additive, Constant, Product, Zero

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(0)


class Datum:
    input_dim, output_dim = 3, 2
    N, Ntest, M = 20, 30, 10


# Constant(c=jax.random.normal(key, shape=(Datum.output_dim,))),


class Data:
    x1 = jnp.linspace(0, 10, 20).reshape(10, 2)


_mean_functions = [
    Zero(),
    # Linear(
    #     A=rng.randn(Datum.input_dim, Datum.output_dim),
    #     b=rng.randn(Datum.output_dim, 1).reshape(-1),
    # ),
    Constant(c=jax.random.normal(key, shape=(Datum.output_dim,))),
]


@pytest.mark.parametrize("mean_function_1", _mean_functions)
@pytest.mark.parametrize("mean_function_2", _mean_functions)
@pytest.mark.parametrize("operation", ["+", "*"])
def test_mean_functions_output_shape(
    mean_function_1, mean_function_2, operation
):
    """
    Test the output shape for basic and compositional mean functions, also
    check that the combination of mean functions returns the correct class
    """
    X = jax.random.normal(key, shape=(Datum.N, Datum.input_dim))
    Y = mean_function_1(X)
    # basic output shape check
    assert Y.shape in [(Datum.N, Datum.output_dim), (Datum.N, 1)]

    # composed mean function output shape check
    if operation == "+":
        mean_composed = mean_function_1 + mean_function_2
    elif operation == "*":
        mean_composed = mean_function_1 * mean_function_2
    else:
        raise (NotImplementedError)

    Y_composed = mean_composed(X)
    assert Y_composed.shape in [(Datum.N, Datum.output_dim), (Datum.N, 1)]


@pytest.mark.parametrize("mean_function_1", _mean_functions)
@pytest.mark.parametrize("mean_function_2", _mean_functions)
@pytest.mark.parametrize("operation", ["+", "*"])
def test_mean_functions_composite_type(
    mean_function_1, mean_function_2, operation
):
    if operation == "+":
        mean_composed = mean_function_1 + mean_function_2
        assert isinstance(mean_composed, Additive)
    elif operation == "*":
        mean_composed = mean_function_1 * mean_function_2
        assert isinstance(mean_composed, Product)
    else:
        raise (NotImplementedError)
