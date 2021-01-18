#!/usr/bin/env python3
import abc
from typing import List, Optional, Union

import haiku as hk
import jax
import numpy as np
from gpjax.kernels.base import Kernel
from gpjax.utilities.ops import difference_matrix, square_distance
from jax import numpy as jnp
from jax import vmap

PRNGKey = jnp.ndarray
ActiveDims = Union[slice, list]


class Stationary(Kernel):
    """
    Base class for kernels that are stationary, that is, they only depend on
        d = x - x'
    This class handles 'ard' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(
        self, variance_init: int = 1.0, lengthscales_init: int = 1.0, **kwargs
    ):
        """
        :param variance: the (initial) value for the variance parameter.
        :param lengthscales: the (initial) value for the lengthscale
            parameter(s), to induce ARD behaviour this must be initialised as
            an array the same length as the the number of active dimensions
            e.g. [1., 1., 1.]. If only a single value is passed, this value
            is used as the lengthscale of each dimension.
        :param kwargs: accepts `name` and `active_dims`, which is a list or
            slice of indices which controls which columns of X are used (by
            default, all columns are used).
        """
        for kwarg in kwargs:
            if kwarg not in {"name", "active_dims"}:
                raise TypeError(f"Unknown keyword argument: {kwarg}")

        super().__init__(**kwargs)
        if not isinstance(variance_init, jnp.ndarray):
            self.variance_init = jnp.array(variance_init)
        else:
            self.variance_init = variance_init
        if not isinstance(variance_init, jnp.ndarray):
            self.lengthscales_init = lengthscales_init
        else:
            self.lengthscales_init = jnp.array(lengthscales_init)
        self._validate_ard_active_dims(lengthscales_init)

    @property
    def ard(self) -> bool:
        """
        Whether ARD behaviour is active.
        """
        return self.lengthscales_init.ndims > 0

    def scale(self, X: jnp.ndarray) -> jnp.ndarray:
        if X is not None:
            lengthscales = hk.get_parameter(
                "lengthscales",
                shape=self.lengthscales_init.shape,
                dtype=X.dtype,
                init=hk.initializers.Constant(self.lengthscales_init),
            )
            X_scaled = X / lengthscales
            return X_scaled
        # X_scaled = X / lengthscales if X is not None else X
        else:
            return X

    def K_diag(self, X):
        # TODO not tested yet
        variance = hk.get_parameter(
            "variance",
            shape=self.variance_init.shape,
            dtype=X.dtype,
            init=hk.initializers.Constant(self.variance_init),
        )
        return jnp.ones(jnp.shape(X)[:-1]) * jnp.squeeze(variance)
        # return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))


class IsotropicStationary(Stationary):
    """
    Base class for isotropic stationary kernels, i.e. kernels that only
    depend on
        r = ‖x - x'‖
    Derived classes should implement one of:
        K_r2(self, r2): Returns the kernel evaluated on r² (r2), which is the
        squared scaled Euclidean distance Should operate element-wise on r2.
        K_r(self, r): Returns the kernel evaluated on r, which is the scaled
        Euclidean distance. Should operate element-wise on r.
    """

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        r2 = self.scaled_squared_euclid_dist(X, X2)
        return self.K_r2(r2)

    def K_r2(self, r2: jnp.ndarray) -> jnp.ndarray:
        if hasattr(self, "K_r"):
            # Clipping around the (single) float precision which is ~1e-45.
            r = jnp.sqrt(jnp.maximum(r2, 1e-36))
            return self.K_r(r)  # pylint: disable=no-member
        raise NotImplementedError

    def scaled_squared_euclid_dist(
        self, X: jnp.ndarray, X2: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Returns ‖(X - X2ᵀ) / ℓ‖², i.e. the squared L₂-norm.
        """
        return square_distance(self.scale(X), self.scale(X2))


class AnisotropicStationary(Stationary):
    """
    Base class for anisotropic stationary kernels, i.e. kernels that only
    depend on
        d = x - x'
    Derived classes should implement K_d(self, d): Returns the kernel evaluated
    on d, which is the pairwise difference matrix, scaled by the lengthscale
    parameter ℓ (i.e. [(X - X2ᵀ) / ℓ]). The last axis corresponds to the
    input dimension.
    """

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None):
        return self.K_d(self.scaled_difference_matrix(X, X2))

    def scaled_difference_matrix(self, X: jnp.ndarray, X2: jnp.ndarray = None):
        """
        Returns [(X - X2ᵀ) / ℓ]. If X has shape [..., N, D] and
        X2 has shape [..., M, D], the output will have shape [..., N, M, D].
        """
        return difference_matrix(self.scale(X), self.scale(X2))

    def K_d(self, d):
        raise NotImplementedError


class SquaredExponential(IsotropicStationary):
    """
    The radial basis function (RBF) or squared exponential kernel. The kernel equation is
        k(r) = σ² exp{-½ r²}
    where:
    r   is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
    σ²  is the variance parameter
    """

    def K_r2(self, r2: jnp.ndarray) -> jnp.ndarray:
        variance = hk.get_parameter(
            "variance",
            shape=self.variance_init.shape,
            dtype=r2.dtype,
            init=hk.initializers.Constant(self.variance_init),
        )
        return variance * jnp.exp(-0.5 * r2)


class Cosine(AnisotropicStationary):
    """
    The Cosine kernel. Functions drawn from a GP with this kernel are sinusoids
    (with a random phase).  The kernel equation is
        k(r) = σ² cos{2πd}
    where:
    d  is the sum of the per-dimension differences between the input points, scaled by the
    lengthscale parameter ℓ (i.e. Σᵢ [(X - X2ᵀ) / ℓ]ᵢ),
    σ² is the variance parameter.
    """

    def K_d(self, d: jnp.ndarray) -> jnp.ndarray:
        d = jnp.sum(d, axis=-1)
        # d = tf.reduce_sum(d, axis=-1)
        variance = hk.get_parameter(
            "variance",
            shape=self.variance_init.shape,
            dtype=d.dtype,
            init=hk.initializers.Constant(self.variance_init),
        )
        return variance * jnp.cos(2 * jnp.pi * d)


if __name__ == "__main__":
    x1 = jnp.linspace(0, 10, 20).reshape(10, 2)
    x2 = jnp.linspace(0, 10, 60).reshape(30, 2)
    # x1 = jnp.linspace(0, 10, 80 * 5).reshape(5, 4, 10, 2)
    # x2 = jnp.linspace(0, 10, 36).reshape(1, 3, 6, 2)
    print(x1.shape)
    print(x2.shape)
    # x2 = None

    lengthscales = jnp.array([1, 0.1], dtype=jnp.float64)
    variance = 2.0
    kernel = hk.transform(
        lambda x1, x2: SquaredExponential(
            lengthscales_init=lengthscales, variance_init=variance
        )(x1, x2)
    )
    # kernel = hk.transform(lambda x1, x2: Cosine(lengthscales_init=lengthscales, variance_init=variance)(x1, x2))
    # kernel = SquaredExponential(
    #     lengthscales_init=lengthscales, variance_init=variance_init
    # )

    @jax.jit
    def cov_fn(
        # kernel: Kernel,
        params: hk.Params,
        rng_key: PRNGKey,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        """Covariance function"""
        K = kernel.apply(params, rng_key, x1, x2)
        return K
        # outputs: VAEOutput = model.apply(params, rng_key, batch["image"])

        # log_likelihood = -binary_cross_entropy(batch["image"], K)
        # kl = kl_gaussian(outputs.mean, outputs.stddev**2)
        # elbo = log_likelihood - kl

        # return -jnp.mean(elbo)

    print("kerne")
    print(kernel)
    rng_seq = hk.PRNGSequence(42)
    params = kernel.init(next(rng_seq), x1, x2)
    # params = kernel.init(next(rng_seq), x1)
    print("params")
    print(params)
    cov = cov_fn(params, next(rng_seq), x1, x2)
    # cov = cov_fn(params, next(rng_seq), x1, x2)
    # cov = cov_fn(params, next(rng_seq), x1)
    print("cov")
    print(cov.shape)
