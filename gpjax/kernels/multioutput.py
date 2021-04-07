#!/usr/bin/env python3
import abc
import jax
from gpjax.kernels import Kernel, Combination
from gpjax.utilities.ops import leading_transpose, batched_diag
from gpjax.custom_types import InputData
from typing import List, Optional, Union
from jax import lax
import jax.numpy as jnp


class MultioutputKernel(Kernel):
    """Base class for multioutput kernels.

    This kernel can represent correlation between outputs of different datapoints.
    Therefore, subclasses of Mok should implement `K` which returns:
    - [N, P, N, P] if full_output_cov = True
    - [P, N, N] if full_output_cov = False
    and `K_diag` returns:
    - [N, P, P] if full_output_cov = True
    - [N, P] if full_output_cov = False
    The `full_output_cov` argument holds whether the kernel should calculate
    the covariance between the outputs. In case there is no correlation but
    `full_output_cov` is set to True the covariance matrix will be filled with zeros
    until the appropriate size is reached.
    """

    @staticmethod
    @abc.abstractmethod
    def K(
        params: dict,
        kernels: Union[List[Kernel], Kernel],
        X1: InputData,
        X2: InputData = None,
        full_output_cov: Optional[bool] = True,
    ) -> jnp.DeviceArray:
        """
        Returns the correlation of f(X1) and f(X2), where f(.) can be multi-dimensional.
        :param X1: data matrix, [N1, input_dim]
        :param X2: data matrix, [N2, input_dim]
        :param full_output_cov: calculate correlation between outputs.
        :return: cov[f(X1), f(X2)] with shape
        - [N1, P, N2, P] if `full_output_cov` = True
        - [P, N1, N2] if `full_output_cov` = False
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def K_diag(
        params: dict,
        kernels: Union[List[Kernel], Kernel],
        X: InputData,
        full_output_cov: Optional[bool] = True,
    ) -> jnp.DeviceArray:
        """
        Returns the correlation of f(X) and f(X), where f(.) can be multi-dimensional.
        :param X: data matrix, [N, input_dim]
        :param full_output_cov: calculate correlation between outputs.
        :return: var[f(X)] with shape
        - [N, P, N, P] if `full_output_cov` = True
        - [N, P] if `full_output_cov` = False
        """
        raise NotImplementedError

    def __call__(
        self,
        params: dict,
        X1: InputData,
        X2: Optional[InputData] = None,
        full_cov: Optional[bool] = False,
        full_output_cov: Optional[bool] = True,
    ) -> jnp.DeviceArray:
        # TODO implement sliced
        if not full_cov and X2 is not None:
            raise ValueError(
                "Ambiguous inputs: passing in `X2` is not compatible with `full_cov=False`."
            )
        if not full_cov:
            return self.K_diag(
                params, self.kernels, X1, full_output_cov=full_output_cov
            )
        return self.K(params, self.kernels, X1, X2, full_output_cov=full_output_cov)


class SeparateIndependent(MultioutputKernel, Combination):
    """Separate independent kernels for each output dimension"""

    def __init__(self, kernels: List[Kernel], name: Optional[str] = None):
        super().__init__(kernels=kernels, name=name)

    def k(
        self,
        params: dict,
        x1: InputData,
        x2: Optional[InputData] = None,
        full_output_cov: Optional[bool] = True,
    ) -> jnp.DeviceArray:
        raise NotImplementedError

    @staticmethod
    def K(
        params: dict,
        kernels: Union[List[Kernel], Kernel],
        X1: InputData,
        X2: Optional[InputData] = None,
        full_output_cov: Optional[bool] = True,
    ) -> jnp.DeviceArray:
        Kxxs = jax.tree_multimap(
            lambda kern, params_: kern(params_, X1, X2), kernels, params
        )
        if full_output_cov:
            Kxxs = jnp.stack(Kxxs, axis=-1)  # [N1, N2, P]
            diag = batched_diag(Kxxs)
            return leading_transpose(diag, [..., -4, -2, -3, -1])  # [..., N1, P, N2, P]
        else:
            return jnp.stack(Kxxs, axis=0)  # [..., P, N1, N2]

    @staticmethod
    def K_diag(
        params: dict,
        kernels: Union[List[Kernel], Kernel],
        X: InputData,
        full_output_cov: Optional[bool] = False,
    ) -> jnp.DeviceArray:
        stacked = jnp.stack(
            jax.tree_multimap(
                lambda kern, params_: kern.K_diag(params_, X), kernels, params
            ),
            axis=-1,
        )  # [N, P]
        diag = batched_diag(stacked)
        if full_output_cov:
            return diag  # [..., N, P]
        else:
            return stacked  # [..., N, P, P]
