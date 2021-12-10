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
    # TODO implement dispatching for inducing variables
    kernel: Kernel,
    f: Union[tjax.Array2[NumInducing, OutputDim], tjax.Array1[NumInducing]],
    full_cov: Optional[bool] = False,
    full_output_cov: Optional[bool] = False,
    q_sqrt: Optional[
        Union[
            tjax.Array3[OutputDim, NumInducing, NumInducing],
            tjax.Array2[NumInducing, NumInducing],
            tjax.Array2[NumInducing, OutputDim],
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
        # inducing_variable,
        X,
        kernel,
        f=f,
        full_cov=full_cov,
        q_sqrt=q_sqrt,
        white=white,
    )
    # f_mean, f_cov = independent_output_conditional(
    #     kernel_params,
    #     Xnew,
    #     # inducing_variable,
    #     X,
    #     kernel,
    #     f=f,
    #     full_cov=full_cov,
    #     q_sqrt=q_sqrt,
    #     white=white,
    # )
    return f_mean, f_cov


# @multifunction(None, None, None, Kernel)
def _conditional(
    kernel_params: dict,
    xnew: tjax.Array1[InputDim],
    X: tjax.Array2[NumInducing, InputDim],
    # inducing_variable: InducingVariable,
    # TODO implement dispatching for inducing variables
    kernel: Kernel,
    f: Union[tjax.Array2[NumInducing, OutputDim], tjax.Array1[NumInducing]],
    full_cov: Optional[bool] = False,
    full_output_cov: Optional[bool] = False,
    q_sqrt: Optional[
        Union[
            tjax.Array3[OutputDim, NumInducing, NumInducing],
            tjax.Array2[NumInducing, NumInducing],
            tjax.Array2[NumInducing, OutputDim],
            tjax.Array1[NumInducing],
        ]
    ] = None,
    white: Optional[bool] = False,
) -> MeanAndCovariance:
    """GP Conditional for a single data point xnew [input_dim].

    Multidispatch handles changing implementation for multioutput etc
    """
    f_mean, f_cov = single_output_conditional(
        kernel_params,
        xnew,
        # inducing_variable,
        X,
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

    means, covs = base_conditional(
        Kmn, Kmm, Knn, f[:, 0], full_cov=full_cov, q_sqrt=q_sqrt[0, :, :], white=white
    )
    # if full_cov:
    #     means = jnp.expand_dims(means, 0)
    #     covs = jnp.expand_dims(covs, 0)
    # else:
    #     means = jnp.expand_dims(means, -1)
    #     covs = jnp.expand_dims(covs, -1)
    means = jnp.expand_dims(means, -1)
    covs = jnp.expand_dims(covs, -1)
    return means, covs
    # # setup axis containing output dim which are to be mapped over
    # if full_cov:  # [output_dim, num_data, num_data]
    #     out_axes = (-1, 0)
    # else:  # [num_data, output_dim]
    #     out_axes = (-1, -1)
    # if q_sqrt is not None:
    #     if q_sqrt.ndim == 2:
    #         in_axes = (-1, -1)
    #     elif q_sqrt.ndim == 3:
    #         in_axes = (-1, 0)
    #     else:
    #         raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.ndim))

    #     def base_conditional_wrapper(f_, q_sqrt_):
    #         return base_conditional(
    #             Kmn, Kmm, Knn, f_, full_cov=full_cov, q_sqrt=q_sqrt_, white=white
    #         )

    #     f_mean, f_cov = jax.vmap(
    #         base_conditional_wrapper, in_axes=in_axes, out_axes=out_axes
    #     )(f, q_sqrt)
    # else:

    #     def base_conditional_wrapper(f_):
    #         return base_conditional(
    #             Kmn, Kmm, Knn, f_, full_cov=full_cov, q_sqrt=q_sqrt, white=white
    #         )

    #     f_mean, f_cov = jax.vmap(
    #         base_conditional_wrapper, in_axes=-1, out_axes=out_axes
    #     )(f)
    # return f_mean, f_cov


# @conditional.dispatch(None, None, None, MultioutputKernel)
# @conditional.dispatch(None, None, None, SeparateIndependent)
def independent_output_conditional_new(
    # def conditional(
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
    # output_dim = q_sqrt.shape[0]
    # Kmn = jnp.tile(jnp.expand_dims(Kmn, 0), (output_dim, 1, 1))
    # Kmm = jnp.tile(jnp.expand_dims(Kmm, 0), (output_dim, 1, 1))
    # Knn = jnp.tile(jnp.expand_dims(Knn, 0), (output_dim, 1, 1))

    # setup axis containing output dim which are to be mapped over
    # [num_data, output_dim]
    out_axes = (-1, -1)
    in_axes = (0, 0, -1, -1, 0)

    def base_conditional_wrapper(Kmn_, Kmm_, Knn_, f_, q_sqrt_):
        return base_conditional(
            Kmn_, Kmm_, Knn_, f_, full_cov=full_cov, q_sqrt=q_sqrt_, white=white
        )

    F_mean, F_cov = [], []
    for i in range(2):
        f_mean, f_cov = base_conditional(
            Kmn[i, :, :],
            Kmm[i, :, :],
            Knn[i, :, :],
            f[:, i],
            q_sqrt=q_sqrt[i, :, :],
            white=white,
        )
        F_mean.append(f_mean)
        F_cov.append(f_cov)
    return jnp.stack(F_mean, 0), jnp.stack(F_cov, 0)

    # F_mean, F_cov = jax.vmap(
    #     base_conditional_wrapper, in_axes=in_axes, out_axes=out_axes
    # )(Kmn, Kmm, Knn, f, q_sqrt)
    # return F_mean, F_cov


# @conditional.dispatch(None, None, None, MultioutputKernel)
@conditional.dispatch(None, None, None, SeparateIndependent)
def independent_output_conditional(
    # def conditional(
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
    # output_dim = q_sqrt.shape[0]
    # Kmn = jnp.tile(jnp.expand_dims(Kmn, 0), (output_dim, 1, 1))
    # Kmm = jnp.tile(jnp.expand_dims(Kmm, 0), (output_dim, 1, 1))
    # Knn = jnp.tile(jnp.expand_dims(Knn, 0), (output_dim, 1, 1))

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
                # in_axes = (0, 0, 0, -1, 0)
        else:
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.ndim))

        def base_conditional_wrapper(Kmn_, Kmm_, Knn_, f_, q_sqrt_):
            return base_conditional(
                Kmn_, Kmm_, Knn_, f_, full_cov=full_cov, q_sqrt=q_sqrt_, white=white
            )

        F_mean, F_cov = jax.vmap(
            base_conditional_wrapper, in_axes=in_axes, out_axes=out_axes
        )(Kmn, Kmm, Knn, f, q_sqrt)
    else:

        def base_conditional_wrapper(Kmn_, Kmm_, Knn_, f_):
            return base_conditional(
                Kmn_, Kmm_, Knn_, f_, full_cov=full_cov, q_sqrt=q_sqrt, white=white
            )

        if full_cov:
            in_axes = (0, 0, 0, -1)
        else:
            in_axes = (0, 0, -1, -1)
        F_mean, F_cov = jax.vmap(
            base_conditional_wrapper, in_axes=in_axes, out_axes=out_axes
        )(Kmn, Kmm, Knn, f)
    return F_mean, F_cov


# @_conditional.dispatch(None, None, None, SeparateIndependent)
def _independent_output_conditional(
    kernel_params: dict,
    xnew: tjax.Array1[InputDim],
    X: tjax.Array2[NumInducing, InputDim],
    # inducing_variable: InducingVariable,
    kernel: Union[Kernel, MultioutputKernel],
    f: tjax.Array2[NumInducing, OutputDim],
    q_sqrt: Optional[
        Union[
            tjax.Array3[OutputDim, NumInducing, NumInducing],
            tjax.Array2[NumInducing, OutputDim],
        ]
    ] = None,
    white: Optional[bool] = False,
):
    """Independent multi-output GP conditional for single input xnew [input_dim]"""
    # Kmm = kernel(kernel_params, X, X)  # [P, M, M]
    # Kmm += jnp.eye(Kmm.shape, dtype=X.dtype) * default_jitter() + 1.0
    Kmm = (
        kernel(kernel_params, X, X)
        + jnp.eye(X.shape[-2], dtype=X.dtype) * default_jitter()
    )  # [P, M, M]
    Kmn = kernel(kernel_params, X, xnew)  # [P, M]
    Knn = kernel(kernel_params, xnew, full_cov=False)  # [P]
    print("inside _independent_output_conditional")
    print(xnew.shape)
    print(X.shape)
    print(Kmm.shape)
    print(Kmn.shape)
    print(Knn.shape)
    print(f.shape)
    print(q_sqrt.shape)

    # setup axis containing output dim which are to be mapped over
    out_axes = (-1, -1)
    if q_sqrt is not None:
        if q_sqrt.ndim == 2:
            # in_axes = (0, 0, 0, -1, -1)
            in_axes = (None, None, None, -1, -1)
        elif q_sqrt.ndim == 3:
            in_axes = (None, None, None, -1, 0)
        else:
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.ndim))

        def base_conditional_wrapper(Kmn_, Kmm_, Knn_, f_, q_sqrt_):
            return base_conditional(
                Kmn_, Kmm_, Knn_, f_, full_cov=False, q_sqrt=q_sqrt_, white=white
            )

        F_mean, F_cov = jax.vmap(
            base_conditional_wrapper, in_axes=in_axes, out_axes=out_axes
        )(Kmn, Kmm, Knn, f, q_sqrt)
    else:
        raise NotImplementedError("need to implement this")

        # def base_conditional_wrapper(Kmn_, Kmm_, Knn_, f_):
        #     return base_conditional(
        #         Kmn_, Kmm_, Knn_, f_, full_cov=full_cov, q_sqrt=q_sqrt, white=white
        #     )

        # if full_cov:
        #     in_axes = (0, 0, 0, -1)
        # else:
        #     in_axes = (0, 0, -1, -1)
        # F_mean, F_cov = jax.vmap(
        #     base_conditional_wrapper, in_axes=in_axes, out_axes=out_axes
        # )(Kmn, Kmm, Knn, f)
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
    # print("A")
    # print(A.shape)
    # print(Knn.shape)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - jnp.matmul(A.T, A)
    else:
        fvar = Knn - jnp.sum(jnp.square(A), 0)

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
            fvar = fvar + jnp.sum(jnp.square(LTA), 0)  # [N]

    return fmean, fvar
