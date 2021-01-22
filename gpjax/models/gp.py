#!/usr/bin/env python3
import abc
from typing import List, Tuple

from gpjax.conditionals import base_conditional
from gpjax.custom_types import InputData, MeanAndVariance, MeanFunc, OutputData
from gpjax.kernels import Kernel
from jax import jacfwd, jacrev, jit
from jax import numpy as np
from jax import partial
from jax import scipy as sp
from jax import vmap

# create types
Likelihood = None
MeanFunction = None
InducingVariable = None


class GPBase(objax.Module, abc.ABC):
    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: int = None,
    ):
        assert (
            num_latent_gps is not None
        ), "GP requires specification of num_latent_gps"
        self.num_latent_gps = num_latent_gps
        self.kernel = Kernel
        self.likelihood = likelihood
        if mean_function is None:
            mean_function = Zero()
        self.mean_function = mean_function

    @abc.abstractmethod
    def predict_f(
        self,
        Xnew: InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
        """ "Compute mean and (co)variance of latent function at Xnew.

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

    def predict_jacobian_f_at_Xnew(
        self,
        Xnew: InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
        # TODO implement this method here
        return NotImplemented

    def predict_y(
        self,
        Xnew: InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
        """Compute the mean and (co)variance of function at Xnew."""
        if full_cov or full_output_cov:
            raise NotImplementedError(
                "The predict_y method currently supports only the argument values full_cov=False and full_output_cov=False"
            )

        f_mean, f_var = self.predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return self.likelihood.predict_mean_and_var(f_mean, f_var)


class SVGP(GPBase):
    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        inducing_variable: InducingVariable,
        mean_function: MeanFunction = None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
        whiten: bool = True,
        # num_data=None,
    ):
        super().__init__(
            kernel=kernel,
            likelihood=likelihood,
            mean_function=mean_function,
            num_latent_gps=num_latent_gps,
        )
        self.whiten = whiten
        self.q_diag = q_diag

        # TODO set q_mu/q_sqrt if = None
        self.inducing_variable = inducing_variable
        self.q_mu = q_mu
        self.q_sqrt = q_sqrt

    def predict_f(
        self, Xnew: InputData, full_cov=False, full_output_cov=False
    ) -> MeanAndVariance:
        # mean, cov = conditional(
        #     Xnew,
        #     self.inducing_variable,
        #     self.kernel,
        #     self.q_mu,
        #     q_sqrt=self.q_sqrt,
        #     full_cov=full_cov,
        #     white=self.whiten,
        #     full_output_cov=full_output_cov,
        # )

        # TODO map over output dimension of Y??
        mean, cov = conditional(
            Xnew,
            inducing_variable,
            kernel=self.kernel,
            f=self.q_mu,
            full_cov=self.full_cov,
            full_output_cov=self.full_output_cov,
            q_sqrt=self.q_sqrt,
            white=self.whiten,
            jitter=self.jitter)
        return mean + self.mean_function(Xnew), cov




