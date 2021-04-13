#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from typing import Optional
from gpjax.config import default_float

from gpjax.conditionals import conditional
from gpjax.custom_types import InputData, MeanAndCovariance, NumInducing, OutputDim
from gpjax.kernels import Kernel
from gpjax.mean_functions import MeanFunction
from gpjax.models import GPModel
from gpjax.likelihoods import Likelihood
from gpjax.prediction import gp_predict_f
from gpjax.utilities.bijectors import positive, triangular

jax.config.update("jax_enable_x64", True)

# create types
InducingVariable = None


def init_svgp_variational_parameters(
    num_inducing: int,
    num_latent_gps: int,
    q_mu: Optional[jnp.ndarray] = None,
    q_sqrt: Optional[jnp.ndarray] = None,
    q_diag: Optional[bool] = False,
):
    """Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.

    :param num_inducing: int
        Number of inducing variables, typically refered to as M.
    :param num_latent_gps: int
        Number of latent GPs
    :param q_mu:  np.array or None
        Mean of the variational Gaussian posterior. If None initialise
        the mean with zeros.
    :param q_sqrt: np.array or None
        Cholesky of the covariance of the variational Gaussian posterior.
        If None the function will initialise `q_sqrt` with identity matrix.
        If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
    :param q_diag: bool
        Used to check if `q_mu` and `q_sqrt` have the correct shape or to
        construct them with the correct shape. If `q_diag` is true,
        `q_sqrt` is two dimensional and only holds the square root of the
        covariance diagonal elements. If False, `q_sqrt` is three dimensional.
    """
    if q_mu is None:
        q_mu = jnp.zeros((num_inducing, num_latent_gps))
    q_mu = jnp.array(q_mu, dtype=default_float())
    if q_sqrt is None:
        if q_diag:
            q_sqrt = jnp.ones((num_inducing, num_latent_gps), dtype=default_float())
            # q_sqrt = Parameter(ones, transform=positive())  # [M, P]
            # q_sqrt = tfp.util.TransformedVariable(
            #     q_sqrt, bijector=positive()
            # )  # [M, P]
        else:
            q_sqrt = [
                jnp.eye(num_inducing, dtype=default_float())
                for _ in range(num_latent_gps)
            ]
            q_sqrt = jnp.array(q_sqrt)
            # q_sqrt = Parameter(q_sqrt, transform=triangular())  # [P, M, M]
            # q_sqrt = tfp.util.TransformedVariable(
            #     q_sqrt, bijector=triangular()
            # )  # [P, M, M]
    else:
        if q_diag:
            assert q_sqrt.ndim == 2
            num_latent_gps = q_sqrt.shape[1]
            # q_sqrt = Parameter(q_sqrt, transform=positive())  # [M, L|P]
            # q_sqrt = tfp.util.TransformedVariable(
            #     q_sqrt, bijector=positive()
            # )  # [M, L|P]
        else:
            assert q_sqrt.ndim == 3
            num_latent_gps = q_sqrt.shape[0]
            num_inducing = q_sqrt.shape[1]
            # q_sqrt = Parameter(q_sqrt, transform=triangular())  # [L|P, M, M]
            # q_sqrt = tfp.util.TransformedVariable(
            #     q_sqrt, bijector=triangular()
            # )  # [L|P, M, M]
    return q_mu, q_sqrt


class SVGP(GPModel):
    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        inducing_variable: InducingVariable,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: Optional[int] = 1,
        q_diag: Optional[bool] = False,
        q_mu: Optional[tjax.Array2[NumInducing, OutputDim]] = None,
        q_sqrt: Optional[
            Union[
                tjax.Array2[NumInducing, OutputDim],
                tjax.Array2[OutputDim, NumInducing, NumInducing],
            ]
        ] = None,
        whiten: Optional[bool] = True,
        # num_data=None,
        # jitter: Optional[jnp.float64] = 1e-6,
    ):
        super().__init__(
            kernel=kernel,
            likelihood=likelihood,
            mean_function=mean_function,
            num_latent_gps=num_latent_gps,
            # jitter=jitter,
        )
        self.whiten = whiten
        self.q_diag = q_diag

        # init variational parameters
        self.inducing_variable = jnp.array(inducing_variable, dtype=default_float())
        self.num_inducing = self.inducing_variable.shape[-2]
        self.q_mu, self.q_sqrt = init_svgp_variational_parameters(
            self.num_inducing, num_latent_gps, q_mu, q_sqrt, q_diag
        )

    def init_params(self):
        kernel_params = self.kernel.init_params()
        likelihood_params = self.likelihood.init_params()
        mean_function_params = self.mean_function.init_params()
        return {
            "kernel": kernel_params,
            "likelihood": likelihood_params,
            "mean_function": mean_function_params,
            "inducing_variable": self.inducing_variable,
            "q_mu": self.q_mu,
            "q_sqrt": self.q_sqrt,
        }

    # def prior_kl(self) -> tf.Tensor:
    #     return kullback_leiblers.prior_kl(
    #         self.inducing_variable, self.kernel, self.q_mu, self.q_sqrt, whiten=self.whiten
    #     )

    # def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
    #     return self.elbo(data)

    # def elbo(self, data: RegressionData) -> tf.Tensor:
    #     """
    #     This gives a variational bound (the evidence lower bound or ELBO) on
    #     the log marginal likelihood of the model.
    #     """
    #     X, Y = data
    #     kl = self.prior_kl()
    #     f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
    #     var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
    #     if self.num_data is not None:
    #         num_data = tf.cast(self.num_data, kl.dtype)
    #         minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
    #         scale = num_data / minibatch_size
    #     else:
    #         scale = tf.cast(1.0, kl.dtype)
    #     return tf.reduce_sum(var_exp) * scale - kl

    @jax.partial(jax.jit, static_argnums=(0, 3, 4))
    def predict_f(
        self,
        params: dict,
        Xnew: InputData,
        full_cov: Optional[bool] = False,
        full_output_cov: Optional[bool] = False,
    ) -> MeanAndCovariance:
        return gp_predict_f(
            params,
            Xnew,
            self.kernel,
            self.mean_function,
            full_cov,
            full_output_cov,
            self.whiten,
        )
        # mean, cov = conditional(
        #     params["kernel"],
        #     Xnew,
        #     params["inducing_variable"],
        #     self.kernel,
        #     # kernel=self.kernel,
        #     f=params["q_mu"],
        #     full_cov=full_cov,
        #     full_output_cov=full_output_cov,
        #     q_sqrt=params["q_sqrt"],
        #     white=self.whiten,
        #     # jitter=self.jitter,
        # )
        # return mean + self.mean_function(params["mean_function"], Xnew), cov
