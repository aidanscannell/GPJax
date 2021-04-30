#!/usr/bin/env python3
from typing import Callable
import numpy as np
import jax.numpy as jnp

"""The number of Gauss-Hermite points to use for quadrature"""
DEFAULT_NUM_GAUSS_HERMITE_POINTS = 20


def gauss_hermite_quadrature(
    fun: Callable,
    mean,
    var,
    deg: int = DEFAULT_NUM_GAUSS_HERMITE_POINTS,
    *args,
    **kwargs
):
    gh_points, gh_weights = np.polynomial.hermite.hermgauss(deg)
    stdev = jnp.sqrt(var)
    X = mean + stdev * gh_points
    W = gh_weights
    return jnp.sum(fun(X, *args, **kwargs) * W, axis=0)
