#!/usr/bin/env python3
import abc
from typing import Optional, Union

import tensor_annotations.jax as tjax
from gpjax.config import Config, default_float
from gpjax.custom_types import Covariance, Input1, Input2, InputDim
from gpjax.kernels import Kernel, covariance_decorator
from gpjax.kernels.distances import scaled_squared_euclidean_distance
from jax import numpy as jnp


def squared_exponential_kern_fn(
    params: dict, x1: tjax.Array1[InputDim], x2: tjax.Array1[InputDim] = None
) -> jnp.float64:
    """Evaluate squared exponential kernel between two input vectors.

        R^D x R^D -> R

    :param params: Dictionary of parameters
        {"lengthscales": lengthscales, "variance": variance}
        - lengthscales: lengthscales with shape [input_dim] or [1] or []
              ARD behaviour governed by shape
        - variance: signal variance with shape [1] or []
    :param x1: single input [input_dim]
    :param x2: Optional single input [input_dim]
    """
    variance = params["variance"]
    lengthscales = params["lengthscales"]
    if variance.ndim > 0:
        assert variance.shape[0] == 1 and variance.ndim == 1
        variance = variance.squeeze()
    scaled_dist = scaled_squared_euclidean_distance(x1, x2, lengthscales)
    return variance * jnp.exp(-0.5 * scaled_dist)


@covariance_decorator
def squared_exponential_cov_fn(
    params: dict, X1: Input1, X2: Input2 = None
) -> Covariance:
    """Return covariance matrix between X1 and X2 using squared exponential kernel

    The @kernel_decorator handles matrix/batched inputs.

    :param params: Dictionary of parameters
        {"lengthscales": lengthscales, "variance": variance}
        - lengthscales: lengthscales with shape [input_dim] or [1] or []
              ARD behaviour governed by shape
        - variance: signal variance with shape [1] or []
        :param X1: Array of inputs [..., N1, input_dim] or [... input_dim]
        :param X2: Optional array of inputs [..., N2, input_dim] or [..., input_dim]
    :returns: covariance matrix
    """
    return squared_exponential_kern_fn(params, X1, X2)


@covariance_decorator
def rectangle_cov_fn(params: dict, X1: Input1, X2: Input2) -> Covariance:
    width = params["lengthscales"]
    lengthscales = params["lengthscales"]
    variance = params["variance"]
    print("inside rect")
    print(X1)
    print(X2)
    print(width)

    # def sigmoid(x1, x2):
    #     return 1.0 / (
    #         (1 + jnp.exp(-variance * ((x2 - x1) + width)))
    #         * (1 + jnp.exp(variance * ((x2 - x1) - width)))
    #     )
    def sigmoid(x1, x2):
        return 1.0 / (
            (1 + jnp.exp(-variance * ((x2 - x1) + width)))
            * (1 + jnp.exp(variance * ((x2 - x1) - width)))
        )

    sig = sigmoid(X1, X2)
    print("sig")
    print(sig.shape)
    print(sig)
    prod = jnp.prod(sig)
    print(prod)
    # return prod
    scaled_dist = scaled_squared_euclidean_distance(X1, X2, lengthscales)
    return variance * jnp.exp(-0.5 * prod)
    # return sig[1]
    # return sig.flatten()
    # return jnp.prod(sig, axis=-1)
    # if X1.ndim == 2:
    #     print("2d")
    #     print(sig.shape)
    #     return jnp.prod(sig, axis=-1)
    # elif X1.ndim == 1:
    #     return sig.flatten()


class Stationary(Kernel, abc.ABC):
    def __init__(
        self,
        lengthscales: Optional[Union[jnp.float64, tjax.Array1[InputDim]]] = jnp.array(
            [1.0], dtype=default_float()
        ),
        variance: Optional[jnp.float64] = 1.0,
        # variance: Optional[Union[jnp.float64, jnp.DeviceArray]] = jnp.array(
        #     [1.0], dtype=default_float()
        # ),
        name: Optional[str] = "Stationary kernel",
    ):
        super().__init__(name=name)
        self.lengthscales = jnp.array(lengthscales)
        self.variance = jnp.array(variance)

    def get_params(self) -> dict:
        return {"lengthscales": self.lengthscales, "variance": self.variance}

    def get_transforms(self) -> dict:
        return {
            "lengthscales": Config.positive_bijector,
            "variance": Config.positive_bijector,
        }


class SquaredExponential(Stationary):
    def __init__(
        self,
        lengthscales: Optional[Union[jnp.float64, tjax.Array1[InputDim]]] = jnp.array(
            [1.0], dtype=default_float()
        ),
        variance: Optional[jnp.float64] = 1.0,
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
