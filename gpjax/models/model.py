#!/usr/bin/env python3
import abc
from typing import Optional

import jax
import jax.numpy as jnp

from gpjax.conditionals import base_conditional
from gpjax.covariances import (
    hessian_cov_fn_wrt_X1X1,
    hessian_rbf_cov_fn_wrt_X1X1,
    jacobian_cov_fn_wrt_X1,
)
from gpjax.custom_types import InputData, MeanAndCovariance
from gpjax.kernels import Kernel
from gpjax.mean_functions import MeanFunction, Zero
from gpjax.base import Module
from gpjax.likelihoods import Likelihood

jax.config.update("jax_enable_x64", True)

# create types
InducingVariable = None


class GPModel(Module, abc.ABC):
    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: int = None,
        jitter=1e-6,
    ):
        assert num_latent_gps is not None, "GP requires specification of num_latent_gps"
        self.num_latent_gps = num_latent_gps
        self.kernel = kernel
        self.likelihood = likelihood
        if mean_function is None:
            mean_function = Zero()
        self.mean_function = mean_function
        self.jitter = jitter

    @abc.abstractmethod
    def predict_f(
        self,
        params: dict,
        Xnew: InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndCovariance:
        """Compute mean and (co)variance of latent function at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :param full_cov:
            If True, draw correlated samples over Xnew. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.
        :param full_output_cov:
            TODO Not implemented
        :returns: tuple of Tensors (mean, variance),
            means.shape == [num_test, output_dim],
            If full_cov=True and full_output_cov=False,
                var.shape == [output_dim, num_test, num_test]
            If full_cov=False,
                var.shape == [num_test, output_dim]
        """
        raise NotImplementedError

    def predict_y(
        self,
        params: dict,
        Xnew: InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndCovariance:
        """Compute the mean and (co)variance of function at Xnew."""
        if full_cov or full_output_cov:
            raise NotImplementedError(
                "The predict_y method currently supports only the argument values full_cov=False and full_output_cov=False"
            )

        f_mean, f_var = self.predict_f(
            params, Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return self.likelihood.predict_mean_and_var(params, f_mean, f_var)


class GPR(GPModel):
    def init_params(
        self,
    ) -> dict:
        kernel_params = self.kernel.init_params()
        likelihood_params = self.likelihood.init_params()
        # return {'kernel':kernel, 'likelihood':.likelihood,}
