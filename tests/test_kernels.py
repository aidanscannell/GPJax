import chex
import haiku as hk
import jax
import pytest
from absl.testing import parameterized
from gpjax.utilities.ops import leading_transpose
from jax import numpy as jnp

from gpjax.kernels import SquaredExponential

jax.config.update("jax_enable_x64", True)


@chex.dataclass
class Data:
    X1: chex.ArrayDevice
    X2: chex.ArrayDevice


@chex.dataclass
class StationaryKernelParams:
    lengthscales: chex.ArrayDevice
    variance: chex.ArrayDevice


data = Data(
    X1=jnp.linspace(0, 10, 20).reshape(10, 2), X2=jnp.linspace(0, 10, 60).reshape(30, 2)
)
batched_data = Data(
    X1=jnp.linspace(0, 10, 80 * 5).reshape(5, 4, 10, 2),
    X2=jnp.linspace(0, 10, 240).reshape(5, 4, 6, 2),
)
stationary_kern_params = StationaryKernelParams(
    lengthscales=[jnp.array([1, 0.1], dtype=jnp.float64)], variance=2.0
)


class TestSquaredExponential(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        (
            "normal",
            data.X1,
            data.X2,
            stationary_kern_params.lengthscales,
            stationary_kern_params.variance,
        ),
        (
            "batched",
            batched_data.X1,
            batched_data.X2,
            stationary_kern_params.lengthscales,
            stationary_kern_params.variance,
        ),
    )
    @chex.assert_max_traces(n=1)
    def test_SquaredExponential(self, X1, X2, lengthscales, variance):
        kernel = SquaredExponential(lengthscales=lengthscales, variance=variance)
        params = kernel.init_params()
        var_kernel = self.variant(kernel)

        cov_x1x2 = var_kernel(params, X1, X2)
        cov_x2x1 = var_kernel(params, X2, X1)
        assert cov_x1x2.shape[-2] == X1.shape[-2]
        assert cov_x1x2.shape[-1] == X2.shape[-2]
        cov_x2x1T = leading_transpose(cov_x2x1, perm=[..., -1, -2])
        chex.assert_equal_shape([cov_x2x1T, cov_x1x2])

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        (
            "normal",
            data.X1,
            stationary_kern_params.lengthscales,
            stationary_kern_params.variance,
        ),
        (
            "batched",
            batched_data.X1,
            stationary_kern_params.lengthscales,
            stationary_kern_params.variance,
        ),
    )
    @chex.assert_max_traces(n=1)
    def test_K_diag(self, X1, lengthscales, variance):
        kernel = SquaredExponential(lengthscales=lengthscales, variance=variance)
        params = kernel.init_params()
        var_kernel_diag = self.variant(kernel.K_diag)

        cov_x1 = var_kernel_diag(params, X1)
        batch_dims = X1.shape[:-1]
        self.assertEqual(batch_dims, cov_x1.shape)


# @chex.dataclass
# class Data:
#     x1: chex.ArrayDevice
#     x2: chex.ArrayDevice


# @chex.dataclass
# class SEKernParams:
#     lengthscales: chex.ArrayDevice
#     variance: chex.ArrayDevice


# data = Data(
#     x1=jnp.linspace(0, 10, 20).reshape(10, 2), x2=jnp.linspace(0, 10, 60).reshape(30, 2)
# )
# batched_data = Data(
#     x1=jnp.linspace(0, 10, 80 * 5).reshape(5, 4, 10, 2),
#     x2=jnp.linspace(0, 10, 240).reshape(5, 4, 6, 2),
# )
# se_kern_params = SEKernParams(
#     lengthscales=[jnp.array([1, 0.1], dtype=jnp.float64)], variance=2.0
# )


# class TestSquaredExponential(chex.TestCase):
#     @chex.variants(with_jit=True, without_jit=True)
#     @parameterized.named_parameters(
#         (
#             "normal",
#             data.x1,
#             data.x2,
#             se_kern_params.lengthscales,
#             se_kern_params.variance,
#         ),
#         (
#             "batched",
#             batched_data.x1,
#             batched_data.x2,
#             se_kern_params.lengthscales,
#             se_kern_params.variance,
#         ),
#     )
#     def test_SquaredExponential(self, x1, x2, lengthscales, variance):
#         print("lengthscales")
#         print(lengthscales)

#         def kern_fn(x1, x2):
#             return SquaredExponential(lengthscales=lengthscales, variance=variance).K(
#                 x1, x2
#             )

#         key = next(hk.PRNGSequence(42))
#         kern_x1x2 = hk.transform(kern_fn)
#         params = kern_x1x2.init(key, x1, x2)

#         var_kern_x1x2 = self.variant(kern_x1x2.apply)
#         cov_x1x2 = var_kern_x1x2(params, key, x1, x2)
#         assert cov_x1x2.shape[-2] == x1.shape[-2]
#         assert cov_x1x2.shape[-1] == x2.shape[-2]
#         cov_x2x1 = var_kern_x1x2(params, key, x2, x1)
#         cov_x2x1T = leading_transpose(cov_x2x1, perm=[..., -1, -2])
#         chex.assert_equal_shape([cov_x2x1T, cov_x1x2])
#         # assert cov12 == cov21
#         # self.assertEqual(cov21T, cov12)
