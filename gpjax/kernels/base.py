#!/usr/bin/env python3
import abc
from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
import tensor_annotations.jax as tjax
from gpjax.custom_types import Covariance, Input1, Input2, InputDim, SingleInput


def covariance_decorator(
    kern_fn: Callable[[dict, SingleInput, Optional[SingleInput]], tjax.Array1]
) -> Callable[[dict, Input1, Input2], Covariance]:
    """Decorator that converts a kernel function to handle all input shapes.

    Given two vectors (in input space) a kernel function maps them to a real
    number. Intuitively, it defines the similarity between the inputs.

    This decorator converts a kernel function which accepts vector inputs,
    i.e. with shape [input_dim], to handle sets of vectors, i.e. with shape
    [N, input_dim] and returns the associated covariance matrix.
    It can also handle batch dimensions, i.e. with shape [batch, N, input_dim].

    It converts a kernel function that evaluates pairs of single inputs, i.e.
       kern_fn : [input_dim] X [input_dim] -> []
    to handle matrix inputs, i.e.
       kern_fn : [N1, input_dim] X [N2, input_dim] -> [N1, N2]
    and batched matrix inputs, i.e.
       kern_fn : [batch, N1, input_dim] X [batch, N2, input_dim] -> [batch, N1, N2]

    :param kern_fn: Callable that returns a single value given two vector inputs
    :returns: Callable kernel function that returns a (batched) covariance matrix
    """

    def wrapper(params: dict, X1: Input1, X2: Input2 = None) -> Covariance:
        if X2 is None:
            X2 = X1
        if X1.ndim > 1 and X2.ndim > 1:
            return batched_covariance_map(params, kern_fn, X1, X2)
        elif X1.ndim > 1:
            return single_batched_covariance_map(params, kern_fn, X1, X2)
        elif X2.ndim > 1:
            return single_batched_covariance_map(params, kern_fn, X2, X1)
        else:
            return kern_fn(params, X1, X2)

    return wrapper


@jax.partial(jnp.vectorize, excluded=(0, 1), signature="(n,d),(m,d)->(n,m)")
def batched_covariance_map(
    params: dict,
    kernel: Callable[[dict, Input1, Input2], Covariance],
    X1: Input1,
    X2: Input2 = None,
) -> Covariance:
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


@jax.partial(jnp.vectorize, excluded=(0, 1), signature="(n,d),(d)->(n)")
def single_batched_covariance_map(
    params: dict,
    kernel: Callable[[dict, Input1, Input2], Covariance],
    X1: Input1,
    x2: Input2 = None,
) -> Covariance:
    """Compute covariance matrix from a covariance function and two input vectors.

    This method handles leading batch dimensions.

    :param params: dictionary of required parameters for kernel
    :param kernel: callable covariance function that maps pairs of data points to scalars.
                    e.g. Kernel(params, x1, x2)
    :param X1: Array of inputs [..., N1, input_dim]
    :param x2: single input array [..., input_dim]
    :returns: covariance matrix [..., N1]
    """
    return jax.vmap(lambda xi: kernel(params, xi, x2))(X1)


class Kernel(abc.ABC):
    def __init__(self, name: Optional[str] = "Kernel"):
        self.name = name

    def __call__(
        self,
        params: dict,
        X1: Input1,
        X2: Input2 = None,
        full_cov: Optional[bool] = True,
    ) -> Covariance:
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
            return self.K_diag(params, X1)

        else:
            return self.K(params, X1, X2)

    @abc.abstractmethod
    def get_params(self) -> dict:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def K(params: dict, X1: Input1, X2: Input2 = None) -> Covariance:
        """Evaluate kernel between two single inputs.

        :param params: dictionary of required parameters for kernel
        :param X1: Array of inputs [..., N1, input_dim]
        :param X2: Optional array of inputs [..., N2, input_dim]
        :returns: covariance matrix [N1, N2]
        """
        raise NotImplementedError

    @staticmethod
    def K_diag(params: dict, X: Input1) -> jnp.DeviceArray:
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

    def get_params(self) -> dict:
        return jax.tree_map(lambda kern: kern.get_params(), self.kernels)

    def get_transforms(self) -> dict:
        return jax.tree_map(lambda kern: kern.get_transforms(), self.kernels)
