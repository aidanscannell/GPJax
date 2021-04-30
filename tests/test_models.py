#!/usr/bin/env python3
import chex
import jax
import tensorflow_probability.substrates.jax.distributions as tfd
from absl.testing import parameterized
from gpjax.config import default_float, default_jitter
from gpjax.kernels import SeparateIndependent, SquaredExponential
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Constant
from gpjax.models import SVGP
from jax import numpy as jnp

key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)

# Data variants
input_dims = [1, 5]
# input_dims = [5]
# output_dims = [4]
output_dims = [1, 4]
# output_dims = [1]
# num_datas = [300, (2, 3, 100)]
num_datas = [300]


# SVGP variants
num_inducings = [30]
whitens = [True, False]
# whitens = [False]
q_diags = [True, False]
# q_diags = [False]

# SVGP.predict_f variants
full_covs = [True, False]
# full_covs = [True]
# TODO test full_output_covs=True
# full_output_covs = [True, False]
full_output_covs = [False]


# SVGP.predict_f_samples variants
num_samples = 2


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
        """Check shapes of output"""
        Xnew = jax.random.uniform(key, shape=(num_data, input_dim))
        # Y = jax.random.uniform(key, shape=(num_data, output_dim))

        mean_function = Constant(output_dim=output_dim)
        likelihood = Gaussian()
        if output_dim > 1:
            kernels = [
                SquaredExponential(
                    lengthscales=jnp.ones(input_dim, dtype=jnp.float64), variance=2.0
                )
                for _ in range(output_dim)
            ]
            kernel = SeparateIndependent(kernels)
        else:
            kernel = SquaredExponential(
                lengthscales=jnp.ones(input_dim, dtype=jnp.float64), variance=2.0
            )
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

        params = svgp.get_params()

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
                assert cov.shape[0] == num_data
                assert cov.shape[1] == output_dim
        else:
            raise NotImplementedError("Need to add tests for full_output_cov=True")

        def predict_f_samples(params, Xnew):
            return svgp.predict_f_samples(params, key, Xnew, num_samples, full_cov)

        var_predict_f_samples = self.variant(predict_f_samples)
        samples = var_predict_f_samples(params, Xnew)
        if not full_output_cov:
            assert samples.ndim == 3
            assert samples.shape[0] == num_samples
            assert samples.shape[1] == num_data
            assert samples.shape[2] == output_dim
        else:
            raise NotImplementedError("Need to add tests for full_output_cov=True")

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        input_dim=input_dims,
        output_dim=output_dims,
        num_inducing=num_inducings,
        q_diag=q_diags,
        whiten=whitens,
    )
    def test_kl(
        self,
        input_dim,
        output_dim,
        num_inducing,
        q_diag,
        whiten,
    ):
        """Check KL returns same as TensorFlow to 4 decimal places."""
        inducing_variable = jax.random.uniform(key, shape=(num_inducing, input_dim))

        if output_dim > 1:
            kernels = [
                SquaredExponential(
                    lengthscales=jnp.ones(input_dim, dtype=jnp.float64), variance=2.0
                )
                for _ in range(output_dim)
            ]
            kernel = SeparateIndependent(kernels)
        else:
            kernel = SquaredExponential(
                lengthscales=jnp.ones(input_dim, dtype=jnp.float64), variance=2.0
            )
        likelihood = Gaussian()
        mean_function = Constant()

        svgp = SVGP(
            kernel,
            likelihood,
            inducing_variable,
            mean_function,
            num_latent_gps=output_dim,
            q_diag=q_diag,
            whiten=whiten,
        )

        params = svgp.get_params()

        var_prior_kl = self.variant(svgp.prior_kl)
        kl = var_prior_kl(params)

        if whiten:
            K = jnp.eye(num_inducing)
        else:
            K = svgp.kernel(params["kernel"], inducing_variable, inducing_variable)
            K += jnp.eye(K.shape[-2]) * default_jitter()

        p = tfd.MultivariateNormalFullCovariance(
            jnp.zeros(svgp.q_mu.T.shape, dtype=default_float()), K
        )
        q_mu = svgp.q_mu.T
        if q_diag:
            q_cov = svgp.q_sqrt.T
            q = tfd.MultivariateNormalDiag(q_mu, q_cov)
        else:
            q_cov = svgp.q_sqrt
            q = tfd.MultivariateNormalFullCovariance(q_mu, q_cov)
        kl_tf = tfd.kl_divergence(q, p)
        kl_tf_sum = jnp.sum(kl_tf)
        assert jnp.round(kl, 4) == jnp.round(kl_tf_sum, 4)
