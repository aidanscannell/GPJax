#!/usr/bin/env python3
import abc
from typing import Optional, Union

import tensor_annotations.jax as tjax

import tensorflow_probability.substrates.jax as tfp
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
    # if variance.ndim > 0:
    #     assert variance.shape[0] == 1 and variance.ndim == 1
    #     variance = variance.squeeze()
    scaled_dist = scaled_squared_euclidean_distance(x1, x2, lengthscales)
    return variance * jnp.exp(-0.5 * scaled_dist)
    # x1 = x1 / lengthscales
    # x2 = x2 / lengthscales
    # x1 = x1.reshape(1, -1)
    # x2 = x2.reshape(1, -1)
    # print("sxq")
    # print(jnp.square(x1).shape)
    # print(jnp.square(x1))
    # jnp.dot(x1, x2.T)[0, 0]
    # dot = (
    #     jnp.sum(x1 * x2, 1)[0]
    #     + jnp.sum(jnp.square(x1), 1)[0]
    #     + jnp.sum(jnp.square(x2), 1)[0]
    # )
    # dot = jnp.clip(dot, 0, jnp.inf)
    # print("dot")
    # print(dot)
    # print(dot.shape)
    # # return variance * jnp.exp(-0.5 * jnp.dot(x1, x2.T))[0, 0]
    # K = variance * jnp.exp(-0.5 * jnp.sqrt(dot))
    # return K

    # x1 = x1 / lengthscales
    # x2 = x2 / lengthscales
    # x1 = x1.reshape(1, -1)
    # x2 = x2.reshape(1, -1)
    # X1sq = jnp.sum(jnp.square(x1), 1)
    # X2sq = jnp.sum(jnp.square(x2), 1)
    # r2 = -2.0 * jnp.dot(x1, x2.T) + (X1sq[:, None] + X2sq[None, :])
    # r2 = jnp.clip(r2, 0, jnp.inf)
    # scaled_dist = jnp.sqrt(r2)
    # K = variance * jnp.exp(-scaled_dist)
    # print("K")
    # print(K.shape)
    # return K[0, 0]


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


def rectangle_kern_fn(
    params: dict, x1: tjax.Array1[InputDim], x2: tjax.Array1[InputDim]
):
    center = params["center"]
    width = params["width"]
    variance = params["variance"]

    def feature_map(x):
        return 1 - jnp.prod(
            1.0
            / (
                (1 + jnp.exp(variance * (x - (center + width / 2))))
                * (1 + jnp.exp(-variance * (x - (center - width / 2))))
            )
        )

    transformed_x1 = feature_map(x1).reshape(1, -1)
    transformed_x2 = feature_map(x2).reshape(1, -1)
    prod = jnp.dot(transformed_x1, transformed_x2.T)
    return jnp.squeeze(prod)


@covariance_decorator
def rectangle_cov_fn(params: dict, X1: Input1, X2: Input2) -> Covariance:
    return rectangle_kern_fn(params, X1, X2)


class SigmoidRectangle(Kernel):
    def __init__(
        self,
        width: tjax.Array1[InputDim] = jnp.array([1.0, 1.0], dtype=default_float()),
        center: tjax.Array1[InputDim] = jnp.array([0.0, 0.0], dtype=default_float()),
        variance: Optional[jnp.float64] = jnp.array(3.0, dtype=default_float()),
        name: Optional[str] = "Rectangle kernel",
    ):
        super().__init__(name=name)
        self.width = width
        self.center = center
        self.variance = variance

    def get_params(self) -> dict:
        return {"width": self.width, "center": self.center, "variance": self.variance}

    def get_transforms(self) -> dict:
        # return {"width": None, "center": None, "variance": None}
        return {
            "width": Config.positive_bijector,
            "center": tfp.bijectors.Identity(),
            # "center": Config.positive_bijector,
            "variance": Config.positive_bijector,
        }

    @staticmethod
    def K(params: dict, X1: Input1, X2: Input2 = None) -> Covariance:
        """Evaluate squared exponential kernel between two inputs.

        :param params: dictionary of required parameters for kernel
        :param X1: Array of inputs [..., N1, input_dim]
        :param X2: Optional array of inputs [..., N2, input_dim]
        :returns: covariance matrix
        """
        return rectangle_cov_fn(params, X1, X2)

    @staticmethod
    def K_diag(params: dict, X: Input1) -> jnp.DeviceArray:
        """Evaluate kernel between X, i.e. K(X, X)

        :param params: dictionary of required parameters for kernel
        :param X: Array of inputs [..., N, input_dim]
        :returns: covariance matrix [..., N]
        """
        diag = jnp.diag(rectangle_cov_fn(params, X, X))
        print("diaggg")
        print(diag.shape)
        return diag
        # return jnp.ones(jnp.shape(X)[:-1]) * jnp.squeeze(params["variance"])
