#!/usr/bin/env python3
from dataclasses import dataclass

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


def default_float():
    return jnp.float64


def to_default_float(value):
    return default_float()(value)


@dataclass(frozen=True)
class Config:
    """Immutable object for storing global GPJax settings

    :param int: Integer data type, int32 or int64.
    :param float: Float data type, float32 or float64
    :param jitter: Used to improve stability of badly conditioned matrices.
            Default value is `1e-6`.
    :param positive_bijector: Method for positive bijector, either "softplus" or "exp".
            Default is "softplus".
    :param positive_minimum: Lower bound for the positive transformation.
    """

    int: type = jnp.int64
    float: type = jnp.float64
    jitter: float = 1e-6
    positive_bijector: tfp.bijectors.Bijector = tfp.bijectors.Softplus
    positive_minimum: float = 0.0
