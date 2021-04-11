#!/usr/bin/env python3
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from gpjax import logdensities
from gpjax.utilities.bijectors import positive, softplus
from gpjax.likelihoods.base import ScalarLikelihood
from gpjax import logdensities
from gpjax.config import default_float

# from .utils import inv_probit


class Gaussian(ScalarLikelihood):
    """
    The Gaussian likelihood is appropriate where uncertainties associated with
    the data are believed to follow a normal distribution, with constant
    variance.

    Very small uncertainties can lead to numerical instability during the
    optimization process. A lower bound of 1e-6 is therefore imposed on the
    likelihood variance by default.
    """

    DEFAULT_VARIANCE_LOWER_BOUND = 1e-6

    def __init__(
        self, variance=1.0, variance_lower_bound=DEFAULT_VARIANCE_LOWER_BOUND, **kwargs
    ):
        """
        :param variance: The noise variance; must be greater than
            ``variance_lower_bound``.
        :param variance_lower_bound: The lower (exclusive) bound of ``variance``.
        :param kwargs: Keyword arguments forwarded to :class:`ScalarLikelihood`.
        """
        super().__init__(**kwargs)

        if variance <= variance_lower_bound:
            raise ValueError(
                f"The variance of the Gaussian likelihood must be strictly greater than {variance_lower_bound}"
            )
        variance = jnp.array([variance], dtype=default_float())
        self._variance = variance
        # self.bijector= tfp.util.TransformedVariable(
        #     variance, bijector=positive(lower=variance_lower_bound)
        # )

    @property
    def variance(self):
        return softplus(self._variance)

    def init_params(
        self,
    ) -> dict:
        return {"variance": self.variance}

    def _scalar_log_prob(self, params: dict, F, Y):
        return logdensities.gaussian(Y, F, params["variance"])

    # def conditional_mean(self, F):  # pylint: disable=R0201
    #     return tf.identity(F)

    # def conditional_variance(self, F):
    #     return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    # def predict_mean_and_var(self, Fmu, Fvar):
    #     return tf.identity(Fmu), Fvar + self.variance

    # def predict_log_density(self, Fmu, Fvar, Y):
    #     return tf.reduce_sum(
    #         logdensities.gaussian(Y, Fmu, Fvar + self.variance), axis=-1
    #     )

    def variational_expectations(self, params, Fmu, Fvar, Y):
        variance = params["variance"]
        return jnp.reduce_sum(
            -0.5 * jnp.log(2 * jnp.pi)
            - 0.5 * jnp.log(variance)
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / variance,
            axis=-1,
        )
