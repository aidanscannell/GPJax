#!/usr/bin/env python3
import abc
from typing import Callable, List, Optional, Union

import jax
import jax.numpy as jnp
from gpjax.custom_types import InputData


class Kernel(abc.ABC):
    def __init__(self, name: Optional[str] = "Kernel"):
        self.name = name

    def __call__(
        self,
        params: dict,
        X1: InputData,
        X2: InputData = None,
        full_cov: Optional[bool] = True,
    ) -> jnp.DeviceArray:
        """Evaluate kernel between two input vectors.

        This method handles leading batch dimensions.

        :param params: dictionary of required parameters for kernel
        :param X1: Array of input points [..., N1, input_dim]
        :param X2: Optional array of input points [..., N2, input_dim]
        :param full_cov: if true K_diag(params, X1, X2) else K(params, X1, X2)
        :returns: covariance matrix [..., N1, N2]
        """
        # TODO implement slicing
        if (not full_cov) and (X2 is not None):
            raise ValueError(
                "Ambiguous inputs: `not full_cov` and `X2` are not compatible."
            )

        if not full_cov:
            assert X2 is None
            # return covariance_map(params, self.k_diag, X1)
            return self.K_diag(params, X1)

        else:
            return covariance_map(params, self.k, X1, X2)

    @abc.abstractmethod
    def init_params(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def k(params: dict, x1: InputData, x2: InputData = None) -> jnp.DeviceArray:
        """Evaluate kernel between two single inputs.

        :param params: dictionary of required parameters for kernel
        :param x1: Single input array [input_dim]
        :param x2: Optional single input array [input_dim]
        :returns: covariance matrix [1]
        """
        raise NotImplementedError

    @staticmethod
    def K_diag(
        params: dict,
        X: InputData,
    ) -> jnp.DeviceArray:
        """Evaluate kernel between X, i.e. K(X, X)

        :param params: dictionary of required parameters for kernel
        :param X: Array of inputs [..., N, input_dim]
        :returns: covariance matrix [..., N]
        """
        return jnp.ones(jnp.shape(X)[:-1]) * jnp.squeeze(params["variance"])

    @property
    def ard(self) -> bool:
        """Whether ARD behaviour is active."""
        return self.lengthscales.ndim > 0


class Combination(Kernel):
    """ Combine a list of kernels"""

    def __init__(self, kernels: List[Kernel], name: Optional[str] = None):
        super().__init__(name=name)

        if not all(isinstance(k, Kernel) for k in kernels):
            raise TypeError("can only combine Kernel instances")

        self.kernels = kernels
        self.num_kernels = len(kernels)

    def init_params(self) -> dict:
        return jax.tree_map(lambda kern: kern.init_params(), self.kernels)


@jax.partial(jnp.vectorize, excluded=(0, 1), signature="(n,d),(m,d)->(n,m)")
def covariance_map(
    params: dict,
    kernel: Union[Kernel, Callable[[dict, InputData, InputData], jnp.DeviceArray]],
    X1: InputData,
    X2: Optional[InputData] = None,
) -> jnp.DeviceArray:
    """Compute covariance matrix from a covariance function and two input vectors.

    This method handles leading batch dimensions.

    :param params: dictionary of required parameters for kernel
    :param kernel: callable covariance function that maps pairs of data points to scalars.
                    e.g. Kernel(params, x1, x2)
    :param X1: Array of inputs [..., N1, input_dim]
    :param X2: Optional array of inputs [..., N2, input_dim]
    :returns: covariance matrix [..., N1, N2]
    """
    if X2 is not None:
        assert X1.shape[-1] == X2.shape[-1] and X1.ndim == X2.ndim == 2
    return jax.vmap(lambda xi: jax.vmap(lambda xj: kernel(params, xi, xj))(X2))(X1)
