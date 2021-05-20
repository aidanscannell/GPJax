#!/usr/bin/env python3
from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import tensor_annotations.jax as tjax
import tensorflow_probability.substrates.jax as tfp
from gpjax import kullback_leiblers
from gpjax.config import default_float
from gpjax.custom_types import (
    InducingVariable,
    InputData,
    InputDim,
    MeanAndCovariance,
    NumInducing,
    OutputDim,
)
from gpjax.kernels import Kernel
from gpjax.likelihoods import Likelihood
from gpjax.mean_functions import MeanFunction
from gpjax.models import GPModel
from gpjax.prediction import gp_predict_f
from tensor_annotations.axes import Batch

jax.config.update("jax_enable_x64", True)


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
            q_sqrt = jnp.ones(
                (num_inducing, num_latent_gps), dtype=default_float()
            )  # [M, P]
        else:
            # q_sqrt = jnp.zeros(
            #     (num_latent_gps, int(num_inducing ** 2 / 2 + num_inducing / 2)),
            #     dtype=default_float(),
            # )
            # q_sqrt = jnp.array(q_sqrt)  # [P, M, M] after with fill_triangular transform
            q_sqrt = [
                jnp.eye(num_inducing, dtype=default_float())
                for _ in range(num_latent_gps)
            ]
            q_sqrt = jnp.array(q_sqrt)  # [P, M, M]
    else:
        if q_diag:
            assert q_sqrt.ndim == 2
            num_latent_gps = q_sqrt.shape[1]  # [M, P]
        else:
            assert q_sqrt.ndim == 3
            num_latent_gps = q_sqrt.shape[0]
            num_inducing = q_sqrt.shape[1]
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
                tjax.Array3[OutputDim, NumInducing, NumInducing],
            ]
        ] = None,
        whiten: Optional[bool] = True,
    ):
        super().__init__(
            kernel=kernel,
            likelihood=likelihood,
            mean_function=mean_function,
            num_latent_gps=num_latent_gps,
        )
        self.whiten = whiten
        self.q_diag = q_diag

        # init variational parameters
        self.inducing_variable = jnp.array(inducing_variable, dtype=default_float())
        self.num_inducing = self.inducing_variable.shape[-2]
        self.q_mu, self.q_sqrt = init_svgp_variational_parameters(
            self.num_inducing, num_latent_gps, q_mu, q_sqrt, q_diag
        )

    def get_params(self):
        kernel_params = self.kernel.get_params()
        if self.likelihood is not None:
            likelihood_params = self.likelihood.get_params()
        else:
            likelihood_params = {}
        mean_function_params = self.mean_function.get_params()
        return {
            "kernel": kernel_params,
            "likelihood": likelihood_params,
            "mean_function": mean_function_params,
            "inducing_variable": self.inducing_variable,
            "q_mu": self.q_mu,
            "q_sqrt": self.q_sqrt,
        }

    def get_transforms(self) -> dict:
        kernel_transforms = self.kernel.get_transforms()
        if self.likelihood is not None:
            likelihood_transforms = self.likelihood.get_transforms()
        else:
            likelihood_transforms = {}
        mean_function_transforms = self.mean_function.get_transforms()

        if self.q_diag:
            q_sqrt_transform = tfp.bijectors.Softplus()
        else:
            # q_sqrt_transform = tfp.bijectors.Softplus()
            q_sqrt_transform = tfp.bijectors.Identity()

        return {
            "kernel": kernel_transforms,
            "likelihood": likelihood_transforms,
            "mean_function": mean_function_transforms,
            "inducing_variable": tfp.bijectors.Identity(),
            "q_mu": tfp.bijectors.Identity(),
            "q_sqrt": q_sqrt_transform,
        }

    def prior_kl(self, params: dict) -> jnp.float64:
        return kullback_leiblers.prior_kl(
            params["kernel"],
            params["inducing_variable"],
            self.kernel,
            params["q_mu"],
            params["q_sqrt"],
            whiten=self.whiten,
        )

    def build_elbo(
        self,
        constrain_params: Callable,
        num_data: Optional[int] = None,
    ) -> jnp.float64:
        """Evidence Lower BOund

        Variational lower bound (the evidence lower bound or ELBO) on the log
        marginal likelihood of the model.
        """

        def elbo(
            params: dict,
            data: Tuple[tjax.Array2[Batch, InputDim], tjax.Array2[Batch, OutputDim]],
        ):
            X, Y = data
            params = constrain_params(params)
            kl = self.prior_kl(params)
            f_mean, f_var = self.predict_f(
                params, X, full_cov=False, full_output_cov=False
            )
            var_exp = self.likelihood.variational_expectations(
                params["likelihood"], f_mean, f_var, Y
            )
            # print("var_exp")
            # print(var_exp.shape)
            var_exp_sum = jnp.sum(var_exp)
            if num_data is not None:
                minibatch_size = X.shape[0]
                scale = num_data / minibatch_size
            else:
                scale = 1.0
            # print(var_exp_sum.shape)
            return jnp.sum(var_exp) * scale - kl

        return elbo

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
            params["inducing_variable"],
            self.kernel,
            self.mean_function,
            f=params["q_mu"],
            full_cov=full_cov,
            full_output_cov=full_output_cov,
            q_sqrt=params["q_sqrt"],
            whiten=self.whiten,
        )
        # return gp_predict_f(
        #     params,
        #     Xnew,
        #     self.kernel,
        #     self.mean_function,
        #     full_cov,
        #     full_output_cov,
        #     self.whiten,
        # )
        )
