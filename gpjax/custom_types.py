#!/usr/bin/env python3
from typing import Tuple, Union

from jax import numpy as jnp

MeanAndVariance = Tuple[jnp.ndarray, jnp.ndarray]
InputData = jnp.ndarray
OutputData = jnp.ndarray
MeanFunc = jnp.float64

Variance = Union[jnp.float64, jnp.ndarray]
Lengthscales = Union[jnp.float64, jnp.ndarray]
