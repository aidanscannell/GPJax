#!/usr/bin/env python3
import jax
from jax import jacfwd
from jax import numpy as jnp
from jax.config import config

from gpjax.custom_types import InputData
from gpjax.kernels import Kernel

config.update("jax_enable_x64", True)


def Kuu(inducing_inputs, kernel: Kernel, jitter: jnp.float64 = 1e-4):
    Kzz = kernel.K(inducing_inputs, inducing_inputs)
    Kzz += jitter * jnp.eye(len(inducing_inputs), dtype=Kzz.dtype)
    return Kzz


def Kuf(inducing_inputs, kernel: Kernel, Xnew: InputData):
    return kernel.K(inducing_inputs, Xnew)


def jacobian_cov_fn_wrt_X1(
    cov_fn, X1: InputData, X2: InputData
) -> jnp.ndarray:
    """Calculate Jacobian of cov_fn(X1, X2) wrt to X1

    :param cov_fn: covariance function with signature cov_fn(x1, x2)
    :param x1: tensor [num_X1, input_dim] or [input_dim,]
    :param x2: tensor [num_x2, input_dim]
    :returns: jacobian of covariance funciton wrt X1
              if X1.shape == [num_X1, input_dim]
                hessian.shape = [num_X1, num_X2, input_dim]
              if X1.shape == [input_dim,]
                hessian.shape = [num_X2, input_dim]
    """
    input_dim = X2.shape[1]
    num_X2 = X2.shape[0]

    def jacobian_cov_fn_wrt_single_x1(x1):
        jac = jacfwd(cov_fn, (0))(x1, X2)
        jac = jac.reshape(num_X2, input_dim)
        return jac

    if len(X1.shape) == 1:
        jac = jacobian_cov_fn_wrt_single_x1(X1)
        assert jac.shape == (num_X2, input_dim)
    else:
        num_X1 = X1.shape[0]
        jac = jax.vmap(jacobian_cov_fn_wrt_single_x1)(X1)
        assert jac.shape == (num_X1, num_X2, input_dim)
    return jac


def hessian_cov_fn_wrt_X1X1(cov_fn, X1: InputData) -> jnp.ndarray:
    """Calculate Hessian of cov_fn(X1, X1) wrt to X1

    :param cov_fn: covariance function with signature cov_fn(X1, X1)
    :param x1: [num_X1, input_dim] or [input_dim,]
    :returns: hessian of covarinace function wrt to X1
              if X1.shape == [num_X1, input_dim]
                hessian.shape = [num_X1, input_dim, input_dim]
              if X1.shape == [input_dim,]
                hessian.shape = [input_dim, input_dim]
    """

    def hessian_cov_fn_wrt_single_x1x1(x1: InputData):
        def cov_fn_single_input(x):
            x = x.reshape(1, -1)
            return cov_fn(x)

        hessian = jax.hessian(cov_fn_single_input)(x1)
        hessian = hessian.reshape([input_dim, input_dim])
        return hessian

    if len(X1.shape) == 1:
        input_dim = X1.shape[0]
        hessian = hessian_cov_fn_wrt_single_x1x1(X1)
        assert hessian.shape == (input_dim, input_dim)
    else:
        input_dim = X1.shape[1]
        num_X1 = X1.shape[0]
        hessian = jax.vmap(hessian_cov_fn_wrt_single_x1x1)(X1)
        assert hessian.shape == (num_X1, input_dim, input_dim)
    return hessian


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
