#!/usr/bin/env python3
import jax.numpy as jnp
import jax.scipy as jsp
from gpjax.likelihoods import ScalarLikelihood
from gpjax.quadrature import gauss_hermite_quadrature
from gpjax import logdensities


def inv_probit(x):
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + jsp.special.erf(x / jnp.sqrt(2.0))) * (1 - 2 * jitter) + jitter


class Bernoulli(ScalarLikelihood):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_params(self):
        return {}

    def get_transforms(self):
        return {}

    def inv_link(self, x):
        return inv_probit(x)

    def _scalar_log_prob(self, F, Y):
        return logdensities.bernoulli(Y, self.inv_link(F))

    def predict_mean_and_var(self, params: dict, Fmu, Fvar):
        p = self.inv_link(Fmu / jnp.sqrt(1 + Fvar))
        return p, p - jnp.square(p)

    def _predict_log_density(self, Fmu, Fvar, Y):
        p = self.predict_mean_and_var(Fmu, Fvar)[0]
        return jnp.sum(logdensities.bernoulli(Y, p), axis=-1)

    def conditional_mean(self, params: dict, F):
        return self.inv_link(F)

    def _conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - (p ** 2)

    def variational_expectations(self, params: dict, Fmu, Fvar, Y):
        return gauss_hermite_quadrature(self._scalar_log_prob, Fmu, Fvar, Y=Y)
