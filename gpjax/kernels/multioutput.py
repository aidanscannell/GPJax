#!/usr/bin/env python3
import abc
from typing import List, Optional, Union

import jax
import jax.numpy as jnp
from gpjax.custom_types import Input1, Input2, MultiOutputCovariance
from gpjax.kernels import Combination, Kernel
from gpjax.kernels.base import batched_covariance_map
from gpjax.utilities.ops import batched_diag, leading_transpose


# def separate_independent_cov_fn(
#     params: dict,
#     kernels: List[Kernel],
#     X1: Input1,
#     X2: Input2,
#     full_output_cov: Optional[bool] = True,
# ) -> MultiOutputCovariance:
#     Kxxs = jax.tree_multimap(
#         lambda kern, params_: kern(params_, X1, X2), kernels, params
#     )
#     if full_output_cov:
#         Kxxs = jnp.stack(Kxxs, axis=-1)  # [..., N1, N2, P]
#         diag = batched_diag(Kxxs)  # [..., N1, N2, P, P]
#         return leading_transpose(diag, [..., -4, -2, -3, -1])  # [..., N1, P, N2, P]
#     else:
#         # TODO haven't checks this is stacked on right axis
#         return jnp.stack(Kxxs, axis=-3)  # [..., P, N1, N2]


def separate_independent_cov_fn(
    params: dict,
    # kernels: List[Kernel],
    kernel: Kernel,
    X1: Input1,
    X2: Input2 = None,
    full_cov: Optional[bool] = False,
    full_output_cov: Optional[bool] = False,
) -> MultiOutputCovariance:
    # Kxxs = jax.tree_multimap(
    #     lambda kern, params_: kern(params_, X1, X2, full_cov=full_cov), kernels, params
    # )

    # def kern(params_):
    #     return kernel(params_, X1, X2, full_cov)

    Kxxs = jax.vmap(kernel, in_axes=(0, None, None, None))(params, X1, X2, full_cov)
    # Kxxs = jax.tree_multimap(
    #     lambda params_: kernel(params_, X1, X2, full_cov=full_cov), params
    # )
    # Kxxs = jnp.stack(Kxxs, axis=-1)  # [..., N1, N2, P] or [..., N1, P]
    # Kxx1 = kernels[0](params[0], X1, X2, full_cov=full_cov)
    # Kxx2 = kernels[1](params[1], X1, X2, full_cov=full_cov)
    # Kxxs = jnp.stack([Kxx1, Kxx2], axis=-1)  # [..., N1, N2, P] or [..., N1, P]
    if full_output_cov:
        diag = batched_diag(Kxxs)  # [..., N1, N2, P, P] or # [..., N1, P, P]
        if full_cov:
            return leading_transpose(diag, [..., -2, -3, -1])  # [..., N1, P, N2, P]
        return diag
    else:
        # TODO haven't checks this is stacked on right axis
        if full_cov:
            # TODO these if statemts were a hacky quick fix
            print("separate_ind_cov_fn")
            print(Kxxs.shape)
            # return jnp.transpose(Kxxs, [2, 0, 1])  # [..., P, N1, N2]
            return Kxxs  # [..., P, N1, N2]
            # if Kxxs.ndim > 2:
            # return leading_transpose(Kxxs, [..., -1, -3, -2])  # [..., P, N1, N2]
            # if Kxxs.ndim == 2:
            # return Kxxs.T  # [P, N1]
            # return leading_transpose(Kxxs, [..., -1, -3, -2])  # [..., P, N1, N2]
        else:
            return jnp.transpose(Kxxs, [1, 0])  # [..., N, P]
            # raise NotImplementedError()


# def multi_output_covariance_map(params: dict, kernels: List[Kernel], X1, X2):
#     Kxxs = jax.tree_multimap(
#         lambda kern1, params1: jnp.stack(
#             jax.tree_multimap(
#                 lambda kern2, params2: batched_covariance_map(p_, k, X1, X2),
#                 kernels,
#                 params,
#             ),
#             axis=-1,
#         ),
#         kernels,
#         params,
#         # lambda kern, params_: kern(params_, X1, X2, full_cov=full_cov), kernels, params
#     )
#     # Kxxs = jax.tree_multimap(
#     #     jax.tree_multimap(
#     #         lambda kernel, params_: jax.vmap(
#     #             lambda xi: jax.vmap(lambda xj: kernel(params_, xi, xj))(X2)
#     #         )(X1),
#     #         kernels,
#     #         params,
#     #     ),
#     #     kernels,
#     #     params,
#     #     # lambda kern, params_: kern(params_, X1, X2, full_cov=full_cov), kernels, params
#     # )
#     Kxxs = jnp.stack(Kxxs, axis=-1)  # [..., N1, N2, P] or [..., N1, P]
#     print("hiaidan")
#     print(Kxxs.shape)
#     return Kxxs


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
        X1: Input1,
        X2: Input2 = None,
        full_output_cov: Optional[bool] = True,
    ) -> MultiOutputCovariance:
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
        X: Input1,
        full_output_cov: Optional[bool] = True,
    ) -> MultiOutputCovariance:
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
        X1: Input1,
        X2: Input2 = None,
        full_cov: Optional[bool] = True,
        full_output_cov: Optional[bool] = False,
    ) -> MultiOutputCovariance:
        # TODO implement sliced
        if not full_cov and X2 is not None:
            raise ValueError(
                "Ambiguous inputs: passing in `X2` is not compatible with `full_cov=False`."
            )
        if not full_cov:
            return self.K_diag(
                params,
                self.kernel_fn,
                X1,
                full_output_cov=full_output_cov
                # params, self.kernels, X1, full_output_cov=full_output_cov
            )
        return self.K(params, self.kernel_fn, X1, X2, full_output_cov=full_output_cov)
        # return self.K(params, self.kernels, X1, X2, full_output_cov=full_output_cov)
        # return multi_output_covariance_map(params, self.kernels, X1, X2)


class SeparateIndependent(MultioutputKernel, Combination):
    """Separate independent kernels for each output dimension"""

    def __init__(self, kernels: List[Kernel], name: Optional[str] = None):
        super().__init__(kernels=kernels, name=name)
        self.kernel_fn = kernels[0]

    @staticmethod
    def K(
        params: dict,
        # kernels: Union[List[Kernel], Kernel],
        kernel: Kernel,
        X1: Input1,
        X2: Input2 = None,
        full_output_cov: Optional[bool] = False,
    ) -> MultiOutputCovariance:
        return separate_independent_cov_fn(
            params,
            kernel,
            X1,
            X2,
            full_cov=True,
            full_output_cov=full_output_cov
            # params, kernels, X1, X2, full_cov=True, full_output_cov=full_output_cov
        )
        # return separate_independent_cov_fn(params, kernels, X1, X2, full_output_cov)

    @staticmethod
    def K_diag(
        params: dict,
        # kernels: Union[List[Kernel], Kernel],
        kernel: Kernel,
        X: Input1,
        full_output_cov: Optional[bool] = False,
    ) -> MultiOutputCovariance:
        return separate_independent_cov_fn(
            params,
            kernel,
            X1=X,
            full_cov=False,
            full_output_cov=full_output_cov
            # params, kernels, X1=X, full_cov=False, full_output_cov=full_output_cov
        )
