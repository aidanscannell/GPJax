#!/usr/bin/env python3
import chex
import jax
from absl.testing import parameterized
from gpjax.mean_functions import Constant
from gpjax.models import SVGP
from gpjax.likelihoods import Gaussian
from jax import numpy as jnp
from gpjax.kernels import SquaredExponential, SeparateIndependent

key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)

# Data variants
input_dims = [1, 5]
# output_dims = [4]
output_dims = [1, 5]
# num_datas = [300, (2, 3, 100)]
num_datas = [300]


# SVGP variants
num_inducings = [30]
whitens = [True, False]
q_diags = [True, False]

# SVGP.predict_f variants
full_covs = [True, False]
# full_covs = [True]
# TODO test full_output_covs=True
# full_output_covs = [True, False]
full_output_covs = [False]


class TestSVGP(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        input_dim=input_dims,
        output_dim=output_dims,
        num_data=num_datas,
        num_inducing=num_inducings,
        q_diag=q_diags,
        whiten=whitens,
        full_cov=full_covs,
        full_output_cov=full_output_covs,
    )
    def test_predict_f(
        self,
        input_dim,
        output_dim,
        num_data,
        num_inducing,
        q_diag,
        whiten,
        full_cov,
        full_output_cov,
    ):
        # data = TestData(input_dim=input_dim, output_dim=output_dim, num_data=num_data)
        # data = TestData(input_dim, output_dim, num_data)
        Xnew = jax.random.uniform(key, shape=(num_data, input_dim))
        # Y = jax.random.uniform(key, shape=(num_data, output_dim))

        mean_function = Constant(output_dim=output_dim)
        likelihood = Gaussian()
        kernels = [
            SquaredExponential(
                lengthscales=jnp.ones(input_dim, dtype=jnp.float64), variance=2.0
            )
            for _ in range(output_dim)
        ]
        kernel = SeparateIndependent(kernels)
        # kernel = SquaredExponential(
        #     lengthscales=jnp.ones(input_dim, dtype=jnp.float64), variance=2.0
        # )
        inducing_variable = jax.random.uniform(key=key, shape=(num_inducing, input_dim))

        svgp = SVGP(
            kernel,
            likelihood,
            inducing_variable,
            mean_function,
            num_latent_gps=output_dim,
            q_diag=q_diag,
            whiten=whiten,
        )

        params = svgp.init_params()

        def predict_f(params, Xnew):
            return svgp.predict_f(params, Xnew, full_cov, full_output_cov)

        var_predict_f = self.variant(predict_f)
        mean, cov = var_predict_f(params, Xnew)

        if not full_output_cov:
            assert mean.ndim == 2
            assert mean.shape[0] == num_data
            assert mean.shape[1] == output_dim
            if full_cov:
                assert cov.ndim == 3
                assert cov.shape[0] == output_dim
                assert cov.shape[1] == cov.shape[2] == num_data
            else:
                assert cov.ndim == 2
                assert cov.ndim == 2
                assert cov.shape[0] == num_data
                assert cov.shape[1] == output_dim
        else:
            raise NotImplementedError("Need to add tests for full_output_cov=True")
