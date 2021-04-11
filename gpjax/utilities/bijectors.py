#!/usr/bin/env python3
from typing import Optional
from tensorflow_probability.substrates import jax as tfp
from gpjax.config import Config, to_default_float
import jax.numpy as jnp


def softplus(x):
    # return jnp.log(1 + jnp.exp(x))
    return jnp.log(1 + jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)


def positive(
    lower: Optional[float] = None,
    base=None,
    # base: Optional[tfp.bijectors.Bijector] = None,
):
    # ) -> tfp.bijectors.Bijector:
    """
    Returns a positive bijector (a reversible transformation from real to positive numbers).

    :param lower: overrides default lower bound
    :param base: overrides base positive bijector
    :returns: a bijector instance
    """
    # default_positive_bijector = tfp.bijectors.Softplus
    # default_lower = 0.0
    default_positive_bijector = Config.positive_bijector
    default_lower = Config.positive_minimum

    bijector = base if base is not None else default_positive_bijector
    lower_bound = lower if lower is not None else default_lower

    if lower_bound != 0.0:
        shift = tfp.bijectors.Shift(to_default_float(lower_bound))
        bijector = tfp.bijectors.Chain(
            [shift, bijector()]
        )  # from unconstrained to constrained
        print("bij")
        print(bijector)
    return bijector


# def triangular() -> tfp.bijectors.Bijector:
def triangular():
    """
    Returns instance of a triangular bijector.
    """
    return tfp.bijectors.FillTriangular()
