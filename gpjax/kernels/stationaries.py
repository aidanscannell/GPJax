#!/usr/bin/env python3

import objax
from gpjax.custom_types import InputData, Lengthscales, Variance
from gpjax.kernels.base import Kernel
from gpjax.utilities.ops import difference_matrix, square_distance
from jax import numpy as jnp


class Stationary(Kernel):
    """
    Base class for kernels that are stationary, that is, they only depend on
        d = x - x'
    This class handles 'ard' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(
        self,
        variance: Variance = jnp.array([1.0]),
        lengthscales: Lengthscales = jnp.array([1.0]),
        **kwargs,
    ):
        """
        :param variance: the (initial) value for the variance parameter.
        :param lengthscales: the (initial) value for the lengthscale.
            Initialising as an array induces ARD behaviour.
            Array must be the same length as the the number of active dimensions.
            If a single value is passed, then it is used for each dimension.
        :param kwargs: accepts `name` and `active_dims`, which is a list or
            slice of indices which controls which columns of X are used (by
            default, all columns are used).
        """
        for kwarg in kwargs:
            if kwarg not in {"name", "active_dims"}:
                raise TypeError(f"Unknown keyword argument: {kwarg}")

        super().__init__(**kwargs)
        if not isinstance(variance, jnp.ndarray):
            self.variance = jnp.array(variance)
        else:
            self.variance = variance
        if not isinstance(lengthscales, jnp.ndarray):
            self.lengthscales = lengthscales
        else:
            self.lengthscales = jnp.array(lengthscales)
        self.lengthscales = objax.TrainVar(self.lengthscales)
        self.variance = objax.TrainVar(self.variance)
        self._validate_ard_active_dims(self.lengthscales)

    @property
    def ard(self) -> bool:
        """
        Whether ARD behaviour is active.
        """
        return self.lengthscales.ndims > 0

    def scale(self, X: InputData) -> jnp.ndarray:
        if X is not None:
            X_scaled = X / self.lengthscales.value
            return X_scaled
        # X_scaled = X / lengthscales if X is not None else X
        else:
            return X

    def K_diag(self, X: InputData):
        # TODO not tested yet
        return jnp.ones(jnp.shape(X)[:-1]) * jnp.squeeze(self.variance.value)
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

    # @jax.partial(jax.jit, static_argnums=(0, 1))
    # @jax.partial(jax.jit, static_argnums=(0,))
    def K(self, X: InputData, X2: InputData = None) -> jnp.ndarray:
        r2 = self.scaled_squared_euclid_dist(X, X2)
        return self.K_r2(r2)

    def __call__(self, X: InputData, X2: InputData = None) -> jnp.ndarray:
        return self.K(X, X2)

    def K_r2(self, r2: jnp.ndarray) -> jnp.ndarray:
        if hasattr(self, "K_r"):
            # Clipping around the (single) float precision which is ~1e-45.
            r = jnp.sqrt(jnp.maximum(r2, 1e-36))
            # r = jnp.sqrt(r2)
            return self.K_r(r)  # pylint: disable=no-member
        raise NotImplementedError

    def scaled_squared_euclid_dist(
        self, X: InputData, X2: InputData = None
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

    # @jax.partial(jax.jit, static_argnums=(0, 1))
    def K(self, X: InputData, X2: InputData = None):
        return self.K_d(self.scaled_difference_matrix(X, X2))

    def scaled_difference_matrix(self, X: InputData, X2: InputData = None):
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
        # print('inside K_r2')
        # print(r2)
        return self.variance.value * jnp.exp(-0.5 * r2)


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
        return self.variance.value * jnp.cos(2 * jnp.pi * d)
