#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from gpjax.conditionals import conditional
from gpjax.custom_types import InputData, MeanAndVariance
from gpjax.kernels import Kernel
from gpjax.mean_functions import MeanFunction
from gpjax.models import GPBase

jax.config.update("jax_enable_x64", True)

# create types
Likelihood = None
InducingVariable = None


class SVGP(GPBase):
    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        inducing_variable: InducingVariable,
        mean_function: MeanFunction = None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu: jnp.ndarray = None,
        q_sqrt: jnp.ndarray = None,
        whiten: bool = True,
        # num_data=None,
        jitter: jnp.float64 = 1e-6,
    ):
        super().__init__(
            kernel=kernel,
            likelihood=likelihood,
            mean_function=mean_function,
            num_latent_gps=num_latent_gps,
            jitter=jitter,
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
        # TODO map over output dimension of Y??
        mean, cov = conditional(
            Xnew,
            self.inducing_variable,
            kernel=self.kernel,
            f=self.q_mu,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
            q_sqrt=self.q_sqrt,
            white=self.whiten,
            jitter=self.jitter,
        )
        return mean + self.mean_function(Xnew), cov
