import chex
import jax
import pytest
from absl.testing import parameterized
from gpjax.utilities.ops import leading_transpose
from jax import numpy as jnp

from gpjax.mean_functions import Zero, Constant

jax.config.update("jax_enable_x64", True)


@chex.dataclass
class TestData:
    X: chex.ArrayDevice
    Y: chex.ArrayDevice


data = TestData(
    X=jnp.linspace(0, 10, 20).reshape(10, 2), Y=jnp.linspace(0, 10, 30).reshape(30, 1)
)
batched_data = TestData(
    X=jnp.linspace(0, 10, 80 * 5).reshape(5, 4, 10, 2),
    Y=jnp.linspace(0, 10, 80 * 5).reshape(5, 4, 10, 2),
    # X2=jnp.linspace(0, 10, 240).reshape(5, 4, 6, 2),
)


class TestMeanFunctions(chex.TestCase):
    # @chex.assert_max_traces(n=1)
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(X=[data.X, batched_data.X], output_dim=[1, 2, 5])
    def test_Zero(self, X, output_dim):
        mean_function = Zero(output_dim=output_dim)
        params = mean_function.get_params()
        var_mean_function = self.variant(mean_function)

        mean = var_mean_function(params, X)
        assert mean.shape[-1] == output_dim
        assert mean.shape[:-1] == X.shape[:-1]

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(X=[data.X, batched_data.X], output_dim=[1, 2, 5])
    def test_Constant(self, X, output_dim):
        constant = 4.0
        mean_function = Constant(c=constant, output_dim=output_dim)
        params = mean_function.get_params()
        var_mean_function = self.variant(mean_function)

        mean = var_mean_function(params, X)
        assert mean.shape[-1] == output_dim
        assert mean.shape[:-1] == X.shape[:-1]


# import jax
# import jax.numpy as jnp
# import pytest
# # from gpjax.mean_functions import Constant, Linear, Zero
# from gpjax.mean_functions import Additive, Constant, Product, Zero

# jax.config.update("jax_enable_x64", True)
# key = jax.random.PRNGKey(0)


# class Datum:
#     input_dim, output_dim = 3, 2
#     N, Ntest, M = 20, 30, 10


# # Constant(c=jax.random.normal(key, shape=(Datum.output_dim,))),


# class Data:
#     x1 = jnp.linspace(0, 10, 20).reshape(10, 2)


# _mean_functions = [
#     Zero(),
#     # Linear(
#     #     A=rng.randn(Datum.input_dim, Datum.output_dim),
#     #     b=rng.randn(Datum.output_dim, 1).reshape(-1),
#     # ),
#     Constant(c=jax.random.normal(key, shape=(Datum.output_dim,))),
# ]


# @pytest.mark.parametrize("mean_function_1", _mean_functions)
# @pytest.mark.parametrize("mean_function_2", _mean_functions)
# @pytest.mark.parametrize("operation", ["+", "*"])
# def test_mean_functions_output_shape(
#     mean_function_1, mean_function_2, operation
# ):
#     """
#     Test the output shape for basic and compositional mean functions, also
#     check that the combination of mean functions returns the correct class
#     """
#     X = jax.random.normal(key, shape=(Datum.N, Datum.input_dim))
#     Y = mean_function_1(X)
#     # basic output shape check
#     assert Y.shape in [(Datum.N, Datum.output_dim), (Datum.N, 1)]

#     # composed mean function output shape check
#     if operation == "+":
#         mean_composed = mean_function_1 + mean_function_2
#     elif operation == "*":
#         mean_composed = mean_function_1 * mean_function_2
#     else:
#         raise (NotImplementedError)

#     Y_composed = mean_composed(X)
#     assert Y_composed.shape in [(Datum.N, Datum.output_dim), (Datum.N, 1)]


# @pytest.mark.parametrize("mean_function_1", _mean_functions)
# @pytest.mark.parametrize("mean_function_2", _mean_functions)
# @pytest.mark.parametrize("operation", ["+", "*"])
# def test_mean_functions_composite_type(
#     mean_function_1, mean_function_2, operation
# ):
#     if operation == "+":
#         mean_composed = mean_function_1 + mean_function_2
#         assert isinstance(mean_composed, Additive)
#     elif operation == "*":
#         mean_composed = mean_function_1 * mean_function_2
#         assert isinstance(mean_composed, Product)
#     else:
#         raise (NotImplementedError)
