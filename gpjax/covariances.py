#!/usr/bin/env python3
from jax import jacfwd, jacrev
from jax import numpy as jnp

from gpjax.custom_types import InputData
from gpjax.kernels import Kernel


def Kuu(inducing_inputs, kernel: Kernel, jitter: jnp.float64 = 1e-4):
    Kzz = kernel.K(inducing_inputs, inducing_inputs)
    Kzz += jitter * jnp.eye(len(inducing_inputs), dtype=Kzz.dtype)
    return Kzz


def Kuf(inducing_inputs, kernel: Kernel, Xnew: InputData):
    return kernel.K(inducing_inputs, Xnew)


def jacobian_cov_fn_wrt_x1(cov_fn, x1, x2):
    """Calculate derivative of cov_fn wrt to x1

    :param cov_fn: covariance function with signature cov_fn(x1, x2)
    :param x1: [1, input_dim]
    :param x2: [num_x2, input_dim]
    """
    dk = jacfwd(cov_fn, (0))(x1, x2)
    # TODO replace squeeze with correct dimensions
    dk = jnp.squeeze(dk)
    return dk


def hessian_cov_fn_wrt_x1x1(cov_fn, x1):
    """Calculate derivative of cov_fn(x1, x1) wrt to x1

    :param cov_fn: covariance function with signature cov_fn(x1, x1)
    :param x1: [1, input_dim]
    """

    def cov_fn_(x1):
        x1 = x1.reshape([1, -1])
        return cov_fn(x1, x1)

    print("inside hessian cov_fn")
    print(x1.shape)
    x1 = x1.reshape(-1)
    d2k = jacrev(jacfwd(cov_fn_))(x1)
    print(d2k.shape)
    # TODO replace squeeze with correct dimensions
    d2k = jnp.squeeze(d2k)

    return d2k


def hessian_cov_fn_wrt_x1x1_hard_coded(cov_fn, lengthscale, x1):
    """Calculate derivative of cov_fn(x1, x1) wrt to x1

    :param cov_fn: covariance function with signature cov_fn(x1, x1)
    :param x1: [1, input_dim]
    """
    # lengthscale = jnp.array([0.4, 0.4])
    # l2 = lengthscale ** 2
    l2 = lengthscale.value ** 2
    # l2 = kernel.lengthscale**2
    l2 = jnp.diag(l2)
    d2k = l2 * cov_fn(x1, x1)
    return d2k


# def cov_map(cov_func, xs, xs2=None, cov_args=None):
#     """Compute a covariance matrix from a covariance function and data points.
#     Args:
#       cov_func: callable function, maps pairs of data points to scalars.
#       xs: array of data points, stacked along the leading dimension.
#     Returns:
#       A 2d array `a` such that `a[i, j] = cov_func(xs[i], xs[j])`.
#     """
#     if xs2 is None:
#         return vmap(lambda x: vmap(lambda y: cov_func(x, y, cov_args))(xs))(xs)
#     else:
#         return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs2).T


# def exp_quadratic(x1, x2):
#     return jnp.exp(-jnp.sum((x1 - x2) ** 2))
