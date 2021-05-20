#!/usr/bin/env python3
from typing import List, Optional, Union

import jax
import tensor_annotations.jax as tjax
from gpjax.conditionals import base_conditional, conditional
from gpjax.custom_types import (
    InducingVariable,
    InputData,
    InputDim,
    MeanAndCovariance,
    OutputData,
)
from gpjax.kernels import Kernel

from gpjax.mean_functions import MeanFunction
from jax import numpy as jnp
from jax import scipy as sp


# @jax.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def gp_predict_f(
    params: dict,
    Xnew: InputData,
    X: Union[InputData, InducingVariable],
    kernel: Kernel,
    mean_function: MeanFunction,
    f,
    full_cov: Optional[bool] = False,
    full_output_cov: Optional[bool] = False,
    q_sqrt: Optional = None,
    whiten: Optional[bool] = False,
) -> MeanAndCovariance:
    mean, cov = conditional(
        params["kernel"],
        Xnew,
        X,
        kernel,
        f=f,
        full_cov=full_cov,
        full_output_cov=full_output_cov,
        q_sqrt=q_sqrt,
        white=whiten,
    )
    return mean + mean_function(params["mean_function"], Xnew), cov


# @jax.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
# def gp_predict_f(
#     params: dict,
#     Xnew: InputData,
#     kernel: Kernel,
#     mean_function: MeanFunction,
#     full_cov: Optional[bool] = False,
#     full_output_cov: Optional[bool] = False,
#     whiten: Optional[bool] = False,
# ) -> MeanAndCovariance:
#     mean, cov = conditional(
#         params["kernel"],
#         Xnew,
#         params["inducing_variable"],
#         kernel,
#         f=params["q_mu"],
#         full_cov=full_cov,
#         full_output_cov=full_output_cov,
#         q_sqrt=params["q_sqrt"],
#         white=whiten,
#     )
#     return mean + mean_function(params["mean_function"], Xnew), cov
# def gp_predict(
#     Xnew: InputData,
#     X: InputData,
#     kernels,
#     mean_funcs: List[MeanFunc],
#     f: OutputData,
#     full_cov: bool = False,
#     q_sqrt=None,
#     jitter=1e-6,
#     white: bool = True,
# ) -> MeanAndVariance:
#     fmeans, fvars = [], []
#     # TODO this check for multioutput could be more robust
#     if isinstance(kernels, list):
#         for output_dim, (kernel, mean_func) in enumerate(zip(kernels, mean_funcs)):
#             fmu, fvar = gp_predict_single_output(
#                 Xnew,
#                 X,
#                 kernel,
#                 mean_func,
#                 f[:, output_dim : output_dim + 1],
#                 full_cov=full_cov,
#                 q_sqrt=q_sqrt[output_dim : output_dim + 1, :, :],
#                 jitter=jitter,
#                 white=white,
#             )
#             fmeans.append(fmu)
#             fvars.append(fvar)
#         fmeans = jnp.stack(fmeans)
#         fvars = jnp.stack(fvars)
#         return fmeans, fvars
#     else:
#         return gp_predict_single_output(
#             Xnew,
#             X,
#             kernels,
#             mean_funcs,
#             f,
#             full_cov=full_cov,
#             q_sqrt=q_sqrt,
#             jitter=jitter,
#             white=white,
#         )


# def gp_predict_single_output(
#     Xnew: InputData,
#     X: InputData,
#     kernel: Kernel,
#     mean_func: MeanFunc,
#     f: OutputData,
#     full_cov: bool = False,
#     q_sqrt=None,
#     jitter=1e-6,
#     white: bool = True,
# ) -> MeanAndVariance:
#     # TODO add noise???
#     Kmm = Kuu(X, kernel)

#     # Kmm += jitter * jnp.eye(Kmm.shape[0])
#     Kmn = kernel.K(X, Xnew)
#     if full_cov:
#         Knn = kernel.K(Xnew, Xnew)
#     else:
#         Knn = kernel.K_diag(Xnew)

#     if q_sqrt is not None:
#         # TODO map over output dimension
#         q_sqrt = jnp.squeeze(q_sqrt)
#         q_sqrt = q_sqrt.reshape([q_sqrt.shape[-1], q_sqrt.shape[-1]])

#     # TODO map over output dimension of Y??
#     # f += mean_func
#     fmean, fvar = base_conditional(
#         Kmn=Kmn,
#         Kmm=Kmm,
#         Knn=Knn,
#         f=f,
#         full_cov=full_cov,
#         q_sqrt=q_sqrt,
#         white=white,
#     )
#     # return fmean, fvar
#     return fmean + mean_func, fvar


def gp_jacobian_hard_coded(cov_fn, Xnew, X, Y, jitter=1e-4):
    Xnew = Xnew.reshape(1, -1)
    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    Kxx = cov_fn(X, X)
    Kxx = Kxx + jitter * jnp.eye(Kxx.shape[0])
    chol = sp.linalg.cholesky(Kxx, lower=True)
    # TODO check cholesky is implemented correctly
    kinvy = sp.linalg.solve_triangular(chol, Y, lower=True)

    # dk_dt0 = kernel.dK_dX(Xnew, X, 0)
    # dk_dt1 = kernel.dK_dX(Xnew, X, 1)
    # dk_dtT = jnp.stack([dk_dt0, dk_dt1], axis=1)
    # dk_dtT = jnp.squeeze(dk_dtT)
    dk_dtT = jacobian_cov_fn_wrt_x1(cov_fn, Xnew, X)

    v = sp.linalg.solve_triangular(chol, dk_dtT, lower=True)

    # TODO lengthscale shouldn't be hard codded
    lengthscale = jnp.array([0.4, 0.4])
    l2 = lengthscale ** 2
    # l2 = kernel.lengthscale**2
    l2 = jnp.diag(l2)
    d2k_dtt = -l2 * cov_fn(Xnew, Xnew)

    # calculate mean and variance of J
    # mu_j = jnp.dot(dk_dtT, kinvy)
    mu_j = v.T @ kinvy
    cov_j = d2k_dtt - jnp.matmul(v.T, v)  # d2Kd2t doesn't need to be calculated
    return mu_j, cov_j


def gp_jacobian(
    Xnew: InputData,
    X: InputData,
    kernels,
    mean_funcs: MeanFunction,
    f: OutputData,
    full_cov: bool = False,
    q_sqrt=None,
    jitter=1e-6,
    white: bool = True,
) -> MeanAndCovariance:
    mu_js, cov_js = [], []

    # TODO this check for multioutput could be more robust
    if isinstance(kernels, list):
        for output_dim, (kernel, mean_func) in enumerate(zip(kernels, mean_funcs)):
            if q_sqrt is not None:
                q_sqrt_ = q_sqrt[output_dim : output_dim + 1, :, :]
            else:
                q_sqrt_ = q_sqrt
            mu_j, cov_j = gp_jacobian_single_output(
                Xnew,
                X,
                kernel,
                mean_func,
                f[:, output_dim : output_dim + 1],
                full_cov=full_cov,
                q_sqrt=q_sqrt_,
                jitter=jitter,
                white=white,
            )
            mu_js.append(mu_j)
            cov_js.append(cov_j)
        mu_js = jnp.stack(mu_j)
        cov_js = jnp.stack(cov_j)
    else:
        if q_sqrt is not None:
            q_sqrt_ = jnp.squeeze(q_sqrt)
            q_sqrt_ = q_sqrt.reshape([q_sqrt_.shape[-1], q_sqrt_.shape[-1]])
        else:
            q_sqrt_ = q_sqrt
        mu_js, cov_js = gp_jacobian_single_output(
            Xnew,
            X,
            kernels,
            mean_funcs,
            f,
            full_cov=full_cov,
            q_sqrt=q_sqrt_,
            jitter=jitter,
            white=white,
        )
    return mu_js, cov_js


def gp_jacobian_single_output(
    Xnew: InputData,
    X: InputData,
    kernel,
    mean_func: MeanFunction,
    f: OutputData,
    full_cov: bool = False,
    q_sqrt=None,
    jitter=1e-6,
    white: bool = True,
) -> MeanAndCovariance:
    num_data = X.shape[0]
    input_dim = X.shape[1]
    assert Xnew.shape[1] == X.shape[1]
    Kxx = kernel.K(X, X)
    Kxx += jitter * jnp.eye(Kxx.shape[0])
    print("Kxx")
    print(Kxx.shape)
    # dKdx1 = jacobian_cov_fn_wrt_x1(kernel.K, Xnew, X)
    dKdx1 = jacobian_cov_fn_wrt_X1(kernel.K, Xnew, X)
    dKdx1 = dKdx1.reshape([num_data, input_dim])
    print("dKdx1")
    print(dKdx1.shape)
    # print(dKdx1)
    # d2K = hessian_cov_fn_wrt_x1x1(kernel.K, Xnew)
    d2K = hessian_cov_fn_wrt_X1X1(kernel.K, Xnew)
    d2K = d2K.reshape([input_dim, input_dim])
    # TODO cheating here - only works for RBF kernel
    # lengthscale = kernel.lengthscales.value
    # lengthscale = kernel.lengthscales

    # def cov_fn(x):
    #     x = x.reshape(1, -1)
    #     return kernel.K(x)

    # d2K = jax.hessian(cov_fn)(Xnew.reshape(-1))
    # input_dim = Xnew.shape[1]
    # d2K = d2K.reshape([input_dim, input_dim])
    # d2K = hessian_cov_fn_wrt_x1x1_hard_coded(kernel.K, lengthscale, Xnew)
    print("d2k")
    print(d2K.shape)
    print(d2K)

    if q_sqrt is not None:
        # TODO map over output dimension
        # q_sqrt = jnp.squeeze(q_sqrt)
        q_sqrt = q_sqrt.reshape([q_sqrt.shape[-1], q_sqrt.shape[-1]])

    mu_j, cov_j = base_conditional(
        Kmn=dKdx1,
        Kmm=Kxx,
        Knn=d2K,
        f=f,
        full_cov=full_cov,
        q_sqrt=q_sqrt,
        white=white,
    )
    # TODO add derivative of mean_func - for constant mean_func this is zero
    return mu_j, cov_j
