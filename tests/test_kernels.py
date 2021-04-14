import chex
import jax
from absl.testing import parameterized
from gpjax.config import default_float
from gpjax.utilities.ops import leading_transpose
from jax import numpy as jnp

from gpjax.kernels import SquaredExponential

jax.config.update("jax_enable_x64", True)

input_dim = 2
full_covs = [True, False]


@chex.dataclass
class Data:
    X1 = [
        jnp.linspace(0, 10, 20).reshape(10, input_dim),
        jnp.linspace(0, 10, 80 * 5).reshape(5, 4, 10, 2),
    ]
    X2 = [
        jnp.linspace(0, 10, 60).reshape(30, input_dim),
        jnp.linspace(0, 10, 240).reshape(5, 4, 6, 2),
    ]


@chex.dataclass
class StationaryKernelParams:
    lengthscales = [
        jnp.ones(input_dim, dtype=jnp.float64),
        jnp.array([1.0], dtype=default_float()),
    ]
    variance = [jnp.array([2.0], dtype=default_float())]


class TestSquaredExponential(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        X1=Data.X1,
        X2=Data.X2,
        lengthscales=StationaryKernelParams.lengthscales,
        variance=StationaryKernelParams.variance,
        full_cov=full_covs,
    )
    @chex.assert_max_traces(n=1)
    def test_SquaredExponential(self, X1, X2, lengthscales, variance, full_cov):
        if not full_cov:
            X2 = None
        kernel = SquaredExponential(lengthscales=lengthscales, variance=variance)
        params = kernel.get_params()

        def kernel_(params, X1, X2):
            return kernel(params, X1, X2, full_cov=full_cov)

        var_kernel = self.variant(kernel_)
        cov_x1x2 = var_kernel(params, X1, X2)

        # full_cov=True tests K and full_cov=False tests K_diag
        if full_cov:
            assert cov_x1x2.shape[-2] == X1.shape[-2]
            cov_x2x1 = var_kernel(params, X2, X1)
            cov_x2x1T = leading_transpose(cov_x2x1, perm=[..., -1, -2])
            chex.assert_equal_shape([cov_x2x1T, cov_x1x2])
        else:
            batch_dims = X1.shape[:-1]
            self.assertEqual(batch_dims, cov_x1x2.shape)
