#!/usr/bin/env python3
import abc
from typing import Optional, Union

from gpjax.config import default_float
from gpjax.custom_types import Covariance, Input1, Input2
from gpjax.kernels import Kernel, kernel_decorator
from gpjax.kernels.distances import scaled_squared_euclidean_distance
from jax import numpy as jnp


@kernel_decorator
def squared_exponential_cov_fn(
    params: dict, X1: Input1, X2: Input2 = None
) -> Covariance:
    """Evaluate squared exponential kernel between two input vectors.

    Without the decorator this function handles vector inputs, i.e.
            X1.shape == X2.shape == [input_dim]
    The @kernel_decorator handles matrix/batched inputs.

    :param params: Dictionary of parameters
        {"lengthscales": lengthscales, "variance": variance}
        - lengthscales: lengthscales with shape [input_dim] or [1] or []
              ARD behaviour governed by shape
        - variance: signal variance with shape [1] or []
        :param X1: Array of inputs [..., N1, input_dim]
        :param X2: Optional array of inputs [..., N2, input_dim]
    :returns: covariance matrix
    """
    variance = params["variance"]
    lengthscales = params["lengthscales"]
    if variance.ndim > 0:
        assert variance.shape[0] == 1 and variance.ndim == 1
        variance = variance.squeeze()
    scaled_dist = scaled_squared_euclidean_distance(X1, X2, lengthscales)
    return variance * jnp.exp(-0.5 * scaled_dist)


class Stationary(Kernel, abc.ABC):
    def __init__(
        self,
        lengthscales: Optional[Union[jnp.float64, jnp.DeviceArray]] = jnp.array(
            [1.0], dtype=default_float()
        ),
        variance: Optional[Union[jnp.float64, jnp.DeviceArray]] = jnp.array(
            [1.0], dtype=default_float()
        ),
        name: Optional[str] = "Stationary kernel",
    ):
        super().__init__(name=name)
        self.lengthscales = jnp.array(lengthscales)
        self.variance = jnp.array(variance)

    def get_params(self) -> dict:
        return {"lengthscales": self.lengthscales, "variance": self.variance}


class SquaredExponential(Stationary):
    def __init__(
        self,
        lengthscales: Optional[Union[jnp.float64, jnp.DeviceArray]] = jnp.array(
            [1.0], dtype=default_float()
        ),
        variance: Optional[Union[jnp.float64, jnp.DeviceArray]] = jnp.array(
            [1.0], dtype=default_float()
        ),
        name: Optional[str] = "Squared exponential kernel",
    ):
        super().__init__(lengthscales=lengthscales, variance=variance, name=name)

    @staticmethod
    def K(params: dict, X1: Input1, X2: Input2 = None) -> Covariance:
        """Evaluate squared exponential kernel between two inputs.

        :param params: dictionary of required parameters for kernel
        :param X1: Array of inputs [..., N1, input_dim]
        :param X2: Optional array of inputs [..., N2, input_dim]
        :returns: covariance matrix
        """
        return squared_exponential_cov_fn(params, X1, X2)
