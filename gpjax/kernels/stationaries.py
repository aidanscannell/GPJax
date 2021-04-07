#!/usr/bin/env python3
import abc
from typing import Optional, Union

from gpjax.custom_types import InputData
from gpjax.kernels import Kernel
from jax import numpy as jnp


def scaled_squared_euclidean_distance(
    x1: InputData,
    x2: InputData,
    lengthscales: Optional[jnp.DeviceArray] = jnp.array([1.0]),
) -> jnp.DeviceArray:
    """Returns ‖(X1 - X2ᵀ) / ℓ‖², i.e. the squared L₂-norm.

    :param x1: Single input array [input_dim]
    :param x2: Optional single input array [input_dim]
    :param lengthscales: lengthscale parameter, either [input_dim] or [1]
    :returns: covariance matrix [1]
    """
    scaled_diff = (x1 - x2) / lengthscales
    squared_euclidean_distance = jnp.dot(scaled_diff, scaled_diff.T)
    return jnp.squeeze(squared_euclidean_distance)


def squared_exponential_cov_fn(
    x1: InputData,
    x2: InputData,
    lengthscales: jnp.DeviceArray,
    variance: jnp.DeviceArray,
) -> jnp.DeviceArray:
    """Evaluate SE kernel between two single inputs.

    :param x1: Single input array [input_dim]
    :param x2: Optional single input array [input_dim]
    :param lengthscales: lengthscale parameter(s), either [input_dim] or [1]
              ARD behaviour governed by shape
    :param variance: signal variance parameter, [1]
    :returns: covariance matrix [1]
    """
    if variance.ndim > 0:
        assert variance.shape[0] == 1 and variance.ndim == 1
        variance = variance.squeeze()
    scaled_dist = scaled_squared_euclidean_distance(x1, x2, lengthscales)
    return variance * jnp.exp(-0.5 * scaled_dist)


class Stationary(Kernel, abc.ABC):
    def __init__(
        self,
        lengthscales: Optional[Union[float, jnp.DeviceArray]] = jnp.array([1.0]),
        variance: Optional[Union[float, jnp.DeviceArray]] = jnp.array([1.0]),
        name: Optional[str] = "Stationary kernel",
    ):
        super().__init__(name=name)
        self.lengthscales = jnp.array(lengthscales)
        self.variance = jnp.array(variance)

    def init_params(self) -> dict:
        return {"lengthscales": self.lengthscales, "variance": self.variance}


class SquaredExponential(Stationary):
    def __init__(
        self,
        lengthscales: Optional[Union[float, jnp.DeviceArray]] = jnp.array([1.0]),
        variance: Optional[Union[float, jnp.DeviceArray]] = jnp.array([1.0]),
        name: Optional[str] = "Squared exponential kernel",
    ):
        super().__init__(lengthscales=lengthscales, variance=variance, name=name)

    @staticmethod
    def k(
        params: dict, x1: InputData, x2: Optional[InputData] = None
    ) -> jnp.DeviceArray:
        """Evaluate SE kernel between two single inputs.

        :param params: dictionary of required parameters for kernel
        :param x1: Single input array [input_dim]
        :param x2: Optional single input array [input_dim]
        :returns: covariance matrix [1]
        """
        return squared_exponential_cov_fn(
            x1, x2, lengthscales=params["lengthscales"], variance=params["variance"]
        )
