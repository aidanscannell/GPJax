#!/usr/bin/env python3
from gpjax.base import Module
import abc


class Likelihood(Module, metaclass=abc.ABCMeta):
    def __init__(self, latent_dim: int, observation_dim: int):
        """
        A base class for likelihoods, which specifies an observation model
        connecting the latent functions ('F') to the data ('Y').

        All of the members of this class are expected to obey some shape conventions, as specified
        by latent_dim and observation_dim.

        If we're operating on an array of function values 'F', then the last dimension represents
        multiple functions (preceding dimensions could represent different data points, or
        different random samples, for example). Similarly, the last dimension of Y represents a
        single data point. We check that the dimensions are as this object expects.

        The return shapes of all functions in this class is the broadcasted shape of the arguments,
        excluding the last dimension of each argument.

        :param latent_dim: the dimension of the vector F of latent functions for a single data point
        :param observation_dim: the dimension of the observation vector Y for a single data point
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim

    # @abc.abstractmethod
    # def log_prob(self, F, Y):
    #     """
    #     The log probability density log p(Y|F)

    #     :param F: function evaluation Tensor, with shape [..., latent_dim]
    #     :param Y: observation Tensor, with shape [..., observation_dim]:
    #     :returns: log pdf, with shape [...]
    #     """
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def conditional_mean(self, F):
    #     """
    #     The conditional mean of Y|F: [E[Y₁|F], ..., E[Yₖ|F]]
    #     where K = observation_dim

    #     :param F: function evaluation Tensor, with shape [..., latent_dim]
    #     :returns: mean [..., observation_dim]
    #     """
    #     raise NotImplementedError

    # def conditional_variance(self, F):
    #     """
    #     The conditional marginal variance of Y|F: [var(Y₁|F), ..., var(Yₖ|F)]
    #     where K = observation_dim

    #     :param F: function evaluation Tensor, with shape [..., latent_dim]
    #     :returns: variance [..., observation_dim]
    #     """
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def predict_mean_and_var(self, Fmu, Fvar):
    #     """
    #     Given a Normal distribution for the latent function,
    #     return the mean and marginal variance of Y,

    #     i.e. if
    #         q(f) = N(Fmu, Fvar)

    #     and this object represents

    #         p(y|f)

    #     then this method computes the predictive mean

    #        ∫∫ y p(y|f)q(f) df dy

    #     and the predictive variance

    #        ∫∫ y² p(y|f)q(f) df dy  - [ ∫∫ y p(y|f)q(f) df dy ]²

    #     :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
    #     :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
    #     :returns: mean and variance, both with shape [..., observation_dim]
    #     """
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def predict_log_density(self, Fmu, Fvar, Y):
    #     r"""
    #     Given a Normal distribution for the latent function, and a datum Y,
    #     compute the log predictive density of Y,

    #     i.e. if
    #         q(F) = N(Fmu, Fvar)

    #     and this object represents

    #         p(y|F)

    #     then this method computes the predictive density

    #         log ∫ p(y=Y|F)q(F) df

    #     :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
    #     :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
    #     :param Y: observation Tensor, with shape [..., observation_dim]:
    #     :returns: log predictive density, with shape [...]
    #     """
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def variational_expectations(self, Fmu, Fvar, Y):
    #     r"""
    #     Compute the expected log density of the data, given a Gaussian
    #     distribution for the function values,

    #     i.e. if
    #         q(f) = N(Fmu, Fvar)

    #     and this object represents

    #         p(y|f)

    #     then this method computes

    #        ∫ log(p(y=Y|f)) q(f) df.

    #     This only works if the broadcasting dimension of the statistics of q(f) (mean and variance)
    #     are broadcastable with that of the data Y.

    #     :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
    #     :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
    #     :param Y: observation Tensor, with shape [..., observation_dim]:
    #     :returns: expected log density of the data given q(F), with shape [...]
    #     """
    #     raise NotImplementedError


# class ScalarLikelihood(QuadratureLikelihood):
class ScalarLikelihood(Likelihood):
    """
    A likelihood class that helps with scalar likelihood functions: likelihoods where
    each scalar latent function is associated with a single scalar observation variable.

    The `Likelihood` class contains methods to compute marginal statistics of functions
    of the latents and the data ϕ(y,f):
     * variational_expectations:  ϕ(y,f) = log p(y|f)
     * predict_log_density: ϕ(y,f) = p(y|f)
    Those statistics are computed after having first marginalized the latent processes f
    under a multivariate normal distribution q(f) that is fully factorized.

    Some univariate integrals can be done by quadrature: we implement quadrature routines for 1D
    integrals in this class, though they may be overwritten by inheriting classes where those
    integrals are available in closed form.
    """

    def __init__(self, **kwargs):
        super().__init__(latent_dim=None, observation_dim=None, **kwargs)

    def log_prob(self, F, Y):
        r"""
        Compute log p(Y|F), where by convention we sum out the last axis as it represented
        independent latent functions and observations.
        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., latent_dim]
        """
        return tf.reduce_sum(self._scalar_log_prob(F, Y), axis=-1)

    @abc.abstractmethod
    def _scalar_log_prob(self, F, Y):
        raise NotImplementedError
