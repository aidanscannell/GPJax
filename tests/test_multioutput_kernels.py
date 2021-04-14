#!/usr/bin/env python3
import chex
import jax
from absl.testing import parameterized
from gpjax.config import default_float
from gpjax.kernels import SeparateIndependent, SquaredExponential
from gpjax.utilities.ops import leading_transpose
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)

input_dim = 2
full_output_covs = [True, False]
full_covs = [True, False]
num_kernels = 3


@chex.dataclass
class Data:
    X1 = [
        jnp.linspace(0, 10, 20).reshape(10, input_dim),
        jnp.linspace(0, 10, 80 * 5).reshape(5, 4, 10, input_dim),
    ]
    X2 = [
        jnp.linspace(0, 10, 60).reshape(30, input_dim),
        jnp.linspace(0, 10, 240).reshape(5, 4, 6, input_dim),
    ]


@chex.dataclass
class StationaryKernelParams:
    lengthscales = [
        jnp.ones(input_dim, dtype=jnp.float64),
        jnp.array([1.0], dtype=default_float()),
    ]
    variance = [jnp.array([2.0], dtype=default_float())]


kernels = [
    SquaredExponential(
        lengthscales=jnp.ones(input_dim, dtype=jnp.float64), variance=2.0
    )
    for _ in range(num_kernels)
]


class TestSeparateIndependent(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        X1=Data.X1,
        X2=Data.X2,
        full_cov=full_covs,
        full_output_cov=full_output_covs,
    )
    @chex.assert_max_traces(n=1)
    def test_SeparateIndependent(self, X1, X2, full_cov, full_output_cov):
        if not full_cov:
            X2 = None
        kernel = SeparateIndependent(kernels)
        params = kernel.get_params()

        def kernel_(params, X1, X2):
            return kernel(params, X1, X2, full_cov, full_output_cov)

        var_kernel = self.variant(kernel_)
        cov_x1x2 = var_kernel(params, X1, X2)

        if full_cov:
            cov_x2x1 = var_kernel(params, X2, X1)
            # subclasses of Mok should implement `K` which returns:
            # - [N, P, N, P] if full_output_cov = True
            # - [P, N, N] if full_output_cov = False
            if full_output_cov:
                assert cov_x1x2.shape[-1] == cov_x1x2.shape[-3] == num_kernels
                assert cov_x1x2.shape[-2] == X2.shape[-2]
                assert cov_x1x2.shape[-4] == X1.shape[-2]

                assert cov_x2x1.shape[-2] == cov_x1x2.shape[-4]
                assert cov_x2x1.shape[-4] == cov_x1x2.shape[-2]
            else:
                assert cov_x1x2.shape[-2] == X1.shape[-2]
                assert cov_x1x2.shape[-1] == X2.shape[-2]
                cov_x2x1T = leading_transpose(cov_x2x1, perm=[..., -1, -2])
                chex.assert_equal_shape([cov_x2x1T, cov_x1x2])
        else:
            # `K_diag` returns:
            # - [N, P, P] if full_output_cov = True
            # - [N, P] if full_output_cov = False
            if full_output_cov:
                assert cov_x1x2.shape[-1] == cov_x1x2.shape[-2] == num_kernels
                assert cov_x1x2.shape[-3] == X1.shape[-2]
            else:
                assert cov_x1x2.shape[-2] == X1.shape[-2]
                assert cov_x1x2.shape[-1] == num_kernels
