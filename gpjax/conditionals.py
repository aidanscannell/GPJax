#!/usr/bin/env python3
from typing import Optional, Union

import jax
from jax import numpy as jnp
from jax import scipy as jsp
from multidispatch import multifunction
from tensor_annotations import jax as tjax

from gpjax.config import default_jitter
from gpjax.custom_types import (
    InputDim,
    MeanAndCovariance,
    NumData,
    NumInducing,
    OutputDim,
)

from gpjax.kernels import Kernel, MultioutputKernel, SeparateIndependent


@multifunction(None, None, None, Kernel)
def conditional(
    kernel_params: dict,
    Xnew: tjax.Array2[NumData, InputDim],
    X: tjax.Array2[NumInducing, InputDim],
    # inducing_variable: InducingVariable,
    kernel: Kernel,
    f: tjax.Array1[NumInducing],
    full_cov: Optional[bool] = False,
    full_output_cov: Optional[bool] = False,
    q_sqrt: Optional[
        Union[
            tjax.Array2[NumInducing, NumInducing],
            tjax.Array1[NumInducing],
        ]
    ] = None,
    white: Optional[bool] = False,
) -> MeanAndCovariance:
    """GP Conditional.

    Multidispatch handles changing implementation for multioutput etc
    """
    f_mean, f_cov = single_output_conditional(
        kernel_params,
        Xnew,
        inducing_variable,
        kernel,
        f=f,
        full_cov=full_cov,
        q_sqrt=q_sqrt,
        white=white,
    )
    return f_mean, f_cov


def single_output_conditional(
    kernel_params: dict,
    Xnew: tjax.Array2[NumData, InputDim],
    X: tjax.Array2[NumInducing, InputDim],
    # inducing_variable: InducingVariable,
    kernel: Kernel,
    f: tjax.Array1[NumInducing],
    full_cov: Optional[bool] = False,
    full_output_cov: Optional[bool] = False,
    q_sqrt: Optional[
        Union[
            tjax.Array2[NumInducing, NumInducing],
            tjax.Array1[NumInducing],
        ]
    ] = None,
    white: Optional[bool] = False,
) -> MeanAndCovariance:
    """Single-output GP conditional."""
    Kmm = (
        kernel(kernel_params, X, X)
        + jnp.eye(X.shape[-2], dtype=X.dtype) * default_jitter()
    )  # [..., M, M]
    Kmn = kernel(kernel_params, X, Xnew)  # [M, N]
    Knn = kernel(kernel_params, Xnew, full_cov=full_cov)  # [N, N]

    return base_conditional(
        Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white
    )  # [N]


# @conditional.dispatch(None, None, None, MultioutputKernel)
@conditional.dispatch(None, None, None, SeparateIndependent)
def independent_output_conditional(
    kernel_params: dict,
    Xnew: tjax.Array2[NumData, InputDim],
    X: tjax.Array2[NumInducing, InputDim],
    # inducing_variable: InducingVariable,
    kernel: Union[Kernel, MultioutputKernel],
    f: tjax.Array2[NumInducing, OutputDim],
    full_cov: Optional[bool] = False,
    full_output_cov: Optional[bool] = False,
    q_sqrt: Optional[
        Union[
            tjax.Array3[OutputDim, NumInducing, NumInducing],
            tjax.Array2[NumInducing, OutputDim],
        ]
    ] = None,
    white: Optional[bool] = False,
):
    """Multi-output GP conditional where outputs are assumed independent."""
    Kmm = (
        kernel(kernel_params, X, X)
        + jnp.eye(X.shape[-2], dtype=X.dtype) * default_jitter()
    )  # [P, M, M]
    Kmn = kernel(kernel_params, X, Xnew)  # [P, M, N]
    Knn = kernel(kernel_params, Xnew, full_cov=full_cov)  # [P, N, N] or [N, P]

    def base_conditional_wrapper(Kmn_, Kmm_, Knn_, f_, q_sqrt_):
        return base_conditional(
            Kmn_, Kmm_, Knn_, f_, full_cov=full_cov, q_sqrt=q_sqrt_, white=white
        )

    # setup axis containing output dim which are to be mapped over
    if full_cov:
        # [output_dim, num_data, num_data]
        out_axes = (-1, 0)
    else:
        # [num_data, output_dim]
        out_axes = (-1, -1)
    if q_sqrt is not None:
        if q_sqrt.ndim == 2:
            if full_cov:
                in_axes = (0, 0, 0, -1, -1)
            else:
                in_axes = (0, 0, -1, -1, -1)
        elif q_sqrt.ndim == 3:
            if full_cov:
                in_axes = (0, 0, 0, -1, 0)
            else:
                in_axes = (0, 0, -1, -1, 0)
        else:
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.ndim))

    F_mean, F_cov = jax.vmap(
        base_conditional_wrapper, in_axes=in_axes, out_axes=out_axes
    )(Kmn, Kmm, Knn, f, q_sqrt)
    return F_mean, F_cov


def fully_correlated_conditional(
    kernel_params: dict,
    Xnew: tjax.Array2[NumData, InputDim],
    X: tjax.Array2[NumInducing, InputDim],
    # inducing_variable: InducingVariable,
    kernel: Union[Kernel, MultioutputKernel],
    f: tjax.Array2[NumInducing, OutputDim],
    full_cov: Optional[bool] = False,
    full_output_cov: Optional[bool] = False,
    q_sqrt: Optional[
        Union[
            tjax.Array3[OutputDim, NumInducing, NumInducing],
            tjax.Array2[NumInducing, OutputDim],
        ]
    ] = None,
    white: Optional[bool] = False,
):
    """Multi-output GP conditional where the conditioning points are fully correlated."""
    raise NotImplementedError("Still needs to be implemented")


# def independent_output_conditional(
#     # def conditional(
#     kernel_params: dict,
#     Xnew: tjax.Array2[NumData, InputDim],
#     inducing_variable: InducingVariable,
#     kernel: Union[Kernel, MultioutputKernel],
#     f: tjax.Array2[NumInducing, OutputDim],
#     full_cov: Optional[bool] = False,
#     full_output_cov: Optional[bool] = False,
#     q_sqrt: Optional[
#         Union[
#             tjax.Array3[OutputDim, NumInducing, NumInducing],
#             tjax.Array2[NumInducing, OutputDim],
#         ]
#     ] = None,
#     white: Optional[bool] = False,
# ):
#     """Multi-output GP conditional where outputs are assumed independent."""
#     # TODO map over kernels
#     print("inside multi output gp conditional")
#     print(Xnew.shape)
#     print(inducing_variable.shape)
#     print(f.shape)
#     print(q_sqrt.shape)

#     def single_output_conditional_wrapper(f_, q_sqrt_=None):
#         print("vmap inside yo")
#         print(f_.shape)
#         print(q_sqrt_.shape)
#         f_mean, f_cov = single_output_conditional(
#             # kernel_params,
#             kernel_params[0],
#             Xnew,
#             inducing_variable,
#             kernel.kernels[0],
#             # kernel,
#             f_,
#             full_cov=full_cov,
#             q_sqrt=q_sqrt_,
#             white=white,
#         )
#         return f_mean, f_cov

#     # output should always be last dimension for mean
#     # but changes for covariance depending on full_cov
#     if full_cov:
#         # [output_dim, num_data, num_data]
#         out_axes = (-1, 0)
#     else:
#         # [num_data, output_dim]
#         out_axes = (-1, -1)
#     if q_sqrt is not None:
#         if q_sqrt.ndim == 2:
#             in_axes = (-1, -1)
#         elif q_sqrt.ndim == 3:
#             in_axes = (-1, 0)
#         else:
#             raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.ndim))
#         F_mean, F_cov = jax.vmap(
#             single_output_conditional_wrapper, in_axes=in_axes, out_axes=out_axes
#         )(f, q_sqrt)
#     else:
#         F_mean, F_cov = jax.vmap(
#             single_output_conditional_wrapper, in_axes=(-1), out_axes=out_axes
#         )(f)
#     return F_mean, F_cov


def base_conditional(
    Kmn: tjax.Array2[NumInducing, NumData],
    Kmm: tjax.Array2[NumInducing, NumInducing],
    Knn: Union[tjax.Array2[NumData, NumData], tjax.Array1[NumData]],
    f: tjax.Array1[NumInducing],
    full_cov: Optional[bool] = False,
    q_sqrt: Optional[
        Union[
            tjax.Array2[NumInducing, NumInducing],
            tjax.Array1[NumInducing],
        ]
    ] = None,
    white: Optional[bool] = False,
):
    r"""Base conditional for single outputs.

    Handling of output dimensions (independent/correlated) will be separate.

    Given a g1 and g2, and distribution p and q such that
      p(g2) = N(g2; 0, Kmm)
      p(g1) = N(g1; 0, Knn)
      p(g1 | g2) = N(g1; Knm (Kmm⁻¹) g2, Knn - Knm (Kmm⁻¹) Kmn)
    And
      q(g2) = N(g2; f, q_sqrt q_sqrtᵀ)
    This method computes the mean and (co)variance of
      q(g1) = ∫ q(g2) p(g1 | g2)
    :param Kmn: [M, N]
    :param Kmm: [M, M]
    :param Knn: [N, N]  or  [N]
    :param f: [M]
    :param full_cov: bool
    :param q_sqrt: [M, M] (lower triangular) or [M] (diagonal)
    :param white: bool
    :return: mean [N] and (co)variance [N]  or [N, N]
    """
    Lm = jsp.linalg.cholesky(Kmm, lower=True)
    return base_conditional_with_lm(
        Kmn=Kmn, Lm=Lm, Knn=Knn, f=f, full_cov=full_cov, q_sqrt=q_sqrt, white=white
    )


def base_conditional_with_lm(
    Kmn: tjax.Array2[NumInducing, NumData],
    Lm: tjax.Array2[NumInducing, NumInducing],
    Knn: Union[tjax.Array2[NumData, NumData], tjax.Array1[NumData]],
    f: tjax.Array1[NumInducing],
    full_cov: Optional[bool] = False,
    q_sqrt: Optional[
        Union[
            tjax.Array2[NumInducing, NumInducing],
            tjax.Array1[NumInducing],
        ]
    ] = None,
    white: Optional[bool] = False,
):
    """Same as base_conditional but expects the cholesky Lm instead of Kmm = Lm Lm.T

    Lm can be precomputed, improving performance.
    """
    A = jsp.linalg.solve_triangular(Lm, Kmn, lower=True)  # [M, N]

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - jnp.matmul(A.T, A)
    else:
        fvar = Knn - jnp.sum(jnp.square(A), -2)

    # another backsubstitution in the unwhitened case
    if not white:
        A = jsp.linalg.solve_triangular(Lm.T, A, lower=False)  # [M, N]

    # conditional mean
    fmean = A.T @ f  # [N]

    # covariance due to inducing variables
    if q_sqrt is not None:
        if q_sqrt.ndim == 1:
            LTA = jnp.expand_dims(q_sqrt, axis=-1) * A  # [M, N]
        elif q_sqrt.ndim == 2:
            LTA = q_sqrt.T @ A  # [M, N]
        else:
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.ndim))

        if full_cov:
            fvar = fvar + LTA.T @ LTA  # [N, N]
        else:
            fvar = fvar + jnp.sum(jnp.square(LTA), -2)  # [N]

    return fmean, fvar


# def independent_output_conditional(
# def conditional(
#     kernel_params: dict,
#     Xnew: jnp.DeviceArray,
#     inducing_variable: InducingVariable,
#     kernel: Union[Kernel, MultioutputKernel],
#     f: jnp.DeviceArray,
#     full_cov: bool = False,
#     full_output_cov: bool = False,
#     q_sqrt: jnp.DeviceArray = None,
#     white: bool = False,
#     jitter: jnp.float64 = 1e-4,
# ):
#     """Multi-output GP conditional where outputs are assumed independent."""
#     print("inside multi output gp conditional")
#     print(Xnew.shape)
#     print(inducing_variable.shape)
#     print(f.shape)
#     print(q_sqrt.shape)

#     def single_output_conditional_wrapper(f_, q_sqrt_=None):
#         print("vmap inside yo")
#         print(f_.shape)
#         print(q_sqrt_.shape)
#         f_mean, f_cov = single_output_conditional(
#             # kernel_params,
#             kernel_params[0],
#             Xnew,
#             inducing_variable,
#             kernel.kernels[0],
#             # kernel,
#             f_,
#             full_cov=full_cov,
#             q_sqrt=q_sqrt_,
#             white=white,
#         )
#         return f_mean, f_cov

#     if q_sqrt is not None:
#         if q_sqrt.ndim == 3:
#             F_mean, F_cov = jax.vmap(
#                 single_output_conditional_wrapper, in_axes=(-1, 0)
#             )(f, q_sqrt)
#         else:
#             raise NotImplementedError(
#                 "Need to implement conditionals for diagonal q_sqrt!"
#             )
#     else:
#         F_mean, F_cov = jax.vmap(single_output_conditional_wrapper, in_axes=(-1))(f)
#     return F_mean, F_cov


# def conditional_(
#     kernel_params: dict,
#     Xnew: tjax.Array2[NumData, InputDim],
#     inducing_variable: InducingVariable,
#     kernel: Kernel,
#     f: tjax.Array2[NumInducing, OutputDim],
#     full_cov: Optional[bool] = False,
#     full_output_cov: Optional[bool] = False,
#     q_sqrt: Optional[
#         Union[
#             tjax.Array3[OutputDim, NumInducing, NumInducing],
#             tjax.Array2[NumInducing, OutputDim],
#         ]
#     ] = None,
#     white: Optional[bool] = False,
# ):
#     """Single-output GP conditional."""
#     print("inside conditional")
#     print(Xnew.shape)
#     print(inducing_variable.shape)
#     print(f.shape)
#     print(q_sqrt.shape)
#     if isinstance(kernel, MultioutputKernel):
#         kernel = kernel.kernels[0]
#         kernel_params = kernel_params[0]

#         # TODO jitter should be noise (sampled)
#         Kmm = kernel(kernel_params, inducing_variable, inducing_variable)
#         print("Kmm")
#         print(Kmm.shape)
#         Kmn = kernel(kernel_params, inducing_variable, Xnew)
#         print(Kmn.shape)
#         Knn = kernel(kernel_params, Xnew, full_cov=False)
#         print(Knn.shape)
#         # if full_cov:
#         #     Knn = kernel(params["kernel"], Xnew)
#         #     # Knn += jitter * jnp.eye(Knn.shape[0], dtype=default_float())
#         # else:
#         #     Knn = kernel.K_diag(params["kernel"], Xnew)
#         #     # Knn = kernel.K_diag(Xnew) + jitter

#         # # TODO handle multioutput GPs
#         return base_conditional(
#             Kmn,
#             Kmm,
#             Knn,
#             f[:, 0],
#             full_cov=full_cov,
#             q_sqrt=q_sqrt[0, :, :],
#             white=white,
#         )
#     # if len(q_sqrt.shape) == 3 and q_sqrt.shape[0] == 1:
#     #     return base_conditional(
#     #         Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white
#     #     )
#     # else:
#     #     raise ValueError("Multioutput GPs not implemented yet")
