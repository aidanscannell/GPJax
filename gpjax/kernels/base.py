#!/usr/bin/env python3
from typing import List, Optional, Union
import abc
import jax
from jax import numpy as jnp
from jax import vmap
import haiku as hk

PRNGKey = jnp.ndarray
ActiveDims = Union[slice, list]


class Kernel(hk.Module, metaclass=abc.ABCMeta):
    """
    The basic kernel class. Handles active dims.
    """

    def __init__(
        self, active_dims: Optional[ActiveDims] = None, name: Optional[str] = None
    ):
        """
        :param active_dims: active dimensions, either a slice or list of
            indices into the columns of X.
        :param name: optional kernel name.
        """
        super().__init__(name=name)
        self._active_dims = self._normalize_active_dims(active_dims)

    @staticmethod
    def _normalize_active_dims(value):
        if value is None:
            value = slice(None, None, None)
        if not isinstance(value, slice):
            value = jnp.array(value, dtype=int)
        return value

    @property
    def active_dims(self):
        return self._active_dims

    @active_dims.setter
    def active_dims(self, value):
        self._active_dims = self._normalize_active_dims(value)

    def _validate_ard_active_dims(self, ard_parameter):
        """
        Validate that ARD parameter matches the number of active_dims (provided active_dims
        has been specified as an array).
        """
        if self.active_dims is None or isinstance(self.active_dims, slice):
            # Can only validate parameter if active_dims is an array
            return

        if ard_parameter.shape.rank > 0 and ard_parameter.shape[0] != len(self.active_dims):
            raise ValueError(
                f"Size of `active_dims` {self.active_dims} does not match "
                f"size of ard parameter ({ard_parameter.shape[0]})"
            )


class Stationary(Kernel):
    """
    Base class for kernels that are stationary, that is, they only depend on
        d = x - x'
    This class handles 'ard' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(self, variance_init=1.0, lengthscales_init=1.0, **kwargs):
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
        if not isinstance(variance_init,jnp.ndarray):
            self.variance_init = jnp.array(variance_init)
        else:
            self.variance_init = variance_init
        self.lengthscales_init = lengthscales_init
        self._validate_ard_active_dims(lengthscales_init)

    @property
    def ard(self) -> bool:
        """
        Whether ARD behaviour is active.
        """
        return self.lengthscales_init.shape.ndims > 0

    def scale(self, X):
        lengthscales = hk.get_parameter(
            "lengthscales",
            shape=self.lengthscales_init.shape,
            dtype=X.dtype,
            init=hk.initializers.Constant(self.lengthscales_init)
        )
        X_scaled = X / lengthscales if X is not None else X
        return X_scaled

    # def K_diag(self, X):
    #     return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))


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

    def __call__(self, X, X2=None):
        r2 = self.scaled_squared_euclid_dist(X, X2)
        return self.K_r2(r2)

    def K_r2(self, r2):
        if hasattr(self, "K_r"):
            # Clipping around the (single) float precision which is ~1e-45.
            r = jnp.sqrt(jnp.maximum(r2, 1e-36))
            return self.K_r(r)  # pylint: disable=no-member
        raise NotImplementedError

    def scaled_squared_euclid_dist(self, X, X2=None):
        """
        Returns ‖(X - X2ᵀ) / ℓ‖², i.e. the squared L₂-norm.
        """
        return square_distance(self.scale(X), self.scale(X2))


class SquaredExponential(IsotropicStationary):
    """
    The radial basis function (RBF) or squared exponential kernel. The kernel equation is
        k(r) = σ² exp{-½ r²}
    where:
    r   is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
    σ²  is the variance parameter
    """

    def K_r2(self, r2):
        variance = hk.get_parameter(
            "variance",
            shape=self.variance_init.shape,
            dtype=r2.dtype,
            init=hk.initializers.Constant(self.variance_init)
        )
        return variance * jnp.exp(-0.5 * r2)


def broadcasting_elementwise(op, a, b):
    """
    Apply binary operation `op` to every pair in tensors `a` and `b`.
    :param op: binary operator on tensors, e.g. tf.add, tf.substract
    :param a: tf.Tensor, shape [n_1, ..., n_a]
    :param b: tf.Tensor, shape [m_1, ..., m_b]
    :return: tf.Tensor, shape [n_1, ..., n_a, m_1, ..., m_b]
    """
    flatres = op(jnp.reshape(a, [-1, 1]), jnp.reshape(b, [1, -1]))
    shape_a = jnp.array(jnp.shape(a))
    shape_b = jnp.array(jnp.shape(b))
    return jnp.reshape(flatres, [shape_a, shape_b])
    # return jnp.reshape(flatres, jnp.concatenate((10,30), 0))
    # return jnp.reshape(flatres, jnp.concatenate((jnp.array(jnp.shape(a)), jnp.array(jnp.shape(b))), 0))


def square_distance(X, X2):
    """
    Returns ||X - X2ᵀ||²
    Due to the implementation and floating-point imprecision, the
    result may actually be very slightly negative for entries very
    close to each other.
    This function can deal with leading dimensions in X and X2.
    In the sample case, where X and X2 are both 2 dimensional,
    for example, X is [N, D] and X2 is [M, D], then a tensor of shape
    [N, M] is returned. If X is [N1, S1, D] and X2 is [N2, S2, D]
    then the output will be [N1, S1, N2, S2].
    """
    # if X2 is None:
    #     Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
    #     dist = -2 * tf.matmul(X, X, transpose_b=True)
    #     dist += Xs + tf.linalg.adjoint(Xs)
    #     return dist
    Xs = jnp.sum(jnp.square(X), axis=-1)
    X2s = jnp.sum(jnp.square(X2), axis=-1)
    dist = -2 * jnp.tensordot(X, X2, [[-1], [-1]])
    dist += broadcasting_elementwise(jnp.add, Xs, X2s)
    return dist


if __name__ == "__main__":
    x1 = jnp.linspace(0, 10, 20).reshape(10, 2)
    x2 = jnp.linspace(0, 10, 60).reshape(30,2)
    # x1 = jnp.linspace(0, 10, 800*5).reshape(5,4,10, 20)
    # x2 = jnp.linspace(0, 10, 120).reshape(1,3,2,20)
    print(x1.shape)
    print(x2.shape)
    # cov = square_distance(x1, x2)
    # print(cov.shape)

    lengthscales = jnp.array([1, 0.1], dtype=jnp.float64)
    variance_init = 2.
    kernel = hk.transform(lambda x1, x2: SquaredExponential(lengthscales_init=lengthscales, variance_init=variance_init)(x1, x2))
    # kernel = hk.transform(SquaredExponential(lengthscales_init=lengthscales, variance_init=variance_init))

    @jax.jit
    def loss_fn(params: hk.Params, rng_key: PRNGKey, x1, x2) -> jnp.ndarray:
        """ELBO loss: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1)."""
        K = kernel.apply(params, rng_key, x1, x2)
        return K
        # outputs: VAEOutput = model.apply(params, rng_key, batch["image"])

        # log_likelihood = -binary_cross_entropy(batch["image"], K)
        # kl = kl_gaussian(outputs.mean, outputs.stddev**2)
        # elbo = log_likelihood - kl

        # return -jnp.mean(elbo)

    print('kerne')
    print(kernel)
    rng_seq = hk.PRNGSequence(42)
    params = kernel.init(next(rng_seq), x1, x2)
    print('params')
    print(params)
    # params=(lengthscales, variance_init)
    val_loss = loss_fn(params, next(rng_seq), x1, x2)
    print('val loss')
    print(val_loss)
