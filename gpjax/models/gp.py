#!/usr/bin/env python3
from typing import List, Tuple

from gpjax.conditionals import base_conditional
from gpjax.custom_types import InputData, MeanAndVariance, MeanFunc, OutputData
from jax import jacfwd, jacrev, jit
from jax import numpy as np
from jax import partial
from jax import scipy as sp
from jax import vmap


def gp_predict(
    Xnew: InputData,
    X: InputData,
    kernels,
    mean_funcs: List[MeanFunc],
    f: OutputData,
    full_cov: bool = False,
    q_sqrt=None,
    jitter=1e-6,
    white: bool = True,
) -> MeanAndVariance:
    # TODO mapping like this doesn't capture covariance
    fmeans, fvars = [], []
    for output_dim, (kernel, mean_func) in enumerate(zip(kernels, mean_funcs)):
        fmu, fvar = gp_predict_single_output(
            Xnew,
            X,
            kernel,
            mean_func,
            f[:, output_dim : output_dim + 1],
            full_cov=full_cov,
            q_sqrt=q_sqrt[output_dim : output_dim + 1, :, :],
            jitter=jitter,
            white=white,
        )
        fmeans.append(fmu)
        fvars.append(fvar)
    fmeans = np.stack(fmeans)
    fvars = np.stack(fvars)
    return fmeans, fvars


def gp_predict_single_output(
    Xnew: InputData,
    X: InputData,
    cov_fn,
    cov_fn_args,
    mean_func,
    mean_func_args,
    f: OutputData,
    full_cov: bool = False,
    q_sqrt=None,
    jitter=1e-6,
    white: bool = True,
) -> MeanAndVariance:
    # TODO add noise???
    Kmm = Kuu(X, kernel)
    Kmn = cov_fn(X, X, cov_fn_args, full_cov=True)

    # Kmm += jitter * np.eye(Kmm.shape[0])
    Kmn = cov_fn(X, Xnew, cov_fn_args, full_cov=True)
    Knn = cov_fn(Xnew, Xnew, cov_fn_args, full_cov=full_cov)

    print("q_sqrt")
    print(q_sqrt.shape)
    if q_sqrt is not None:
        # TODO map over output dimension
        q_sqrt = np.squeeze(q_sqrt)
        q_sqrt = q_sqrt.reshape([q_sqrt.shape[-1], q_sqrt.shape[-1]])

    # TODO map over output dimension of Y??
    # f += mean_func
    fmean, fvar = base_conditional(
        Kmn=Kmn,
        Kmm=Kmm,
        Knn=Knn,
        f=f,
        full_cov=full_cov,
        q_sqrt=q_sqrt,
        white=white,
    )
    # return fmean, fvar
    mean = mean_func(Xnew, mean_func_args)
    return fmean + mean, fvar


# def gp_predict(Xnew: InputData,
#                X: InputData,
#                kernel,
#                mean_func: MeanFunc,
#                f: OutputData,
#                *,
#                full_cov: bool = False,
#                q_sqrt=None,
#                jitter=1e-6,
#                white: bool = True) -> MeanAndVariance:
#     # TODO add noise???
#     Kmm = Kuu(X, kernel)

#     # Kmm += jitter * np.eye(Kmm.shape[0])
#     Kmn = kernel.K(X, Xnew)
#     if full_cov:
#         Knn = kernel.K(Xnew, Xnew)
#     else:
#         Knn = kernel.Kdiag(Xnew)

#     print('q_sqrt')
#     print(q_sqrt.shape)
#     # if q_sqrt is not None:
#     # TODO map over output dimension
#     # q_sqrt = np.squeeze(q_sqrt)
#     # q_sqrt = q_sqrt.reshape([q_sqrt.shape[-1], q_sqrt.shape[-1]])

#     # TODO map over output dimension of Y??
#     # f += mean_func
#     fmean, fvar = base_conditional(Kmn=Kmn,
#                                    Kmm=Kmm,
#                                    Knn=Knn,
#                                    f=f,
#                                    full_cov=full_cov,
#                                    q_sqrt=q_sqrt,
#                                    white=white)
#     # return fmean, fvar
#     return fmean + mean_func, fvar
