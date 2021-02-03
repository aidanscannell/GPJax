#!/usr/bin/env python3
from jax import numpy as jnp
from jax import scipy as sp

from gpjax.covariances import Kuf, Kuu
from gpjax.kernels import Kernel
from gpjax.utilities import leading_transpose

InducingVariable = None


def conditional(
    Xnew: jnp.ndarray,
    inducing_variable: InducingVariable,
    kernel: Kernel,
    f: jnp.ndarray,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt: jnp.ndarray = None,
    white: bool = False,
    jitter: jnp.float64 = 1e-4,
):
    # Kmm = Kuu(X, kernel)
    # # Kmm += jitter * jnp.eye(Kmm.shape[0])
    # Kmn = kernel.K(X, Xnew)
    # if full_cov:
    #     Knn = kernel.K(Xnew, Xnew)
    # else:
    #     Knn = kernel.K_diag(Xnew)

    # if q_sqrt is not None:
    #     # TODO map over output dimension
    #     q_sqrt = jnp.squeeze(q_sqrt)
    #     q_sqrt = q_sqrt.reshape([q_sqrt.shape[-1], q_sqrt.shape[-1]])

    """Single-output GP conditional."""
    print("inside conditional")
    print(Xnew.shape)
    Kmm = Kuu(inducing_variable, kernel, jitter=jitter)  # [M, M]
    Kmn = Kuf(inducing_variable, kernel, Xnew)  # [M, N]
    # Knn = kernel.K(Xnew, full_cov=full_cov)
    if full_cov:
        Knn = kernel.K(Xnew)
    else:
        Knn = kernel.K_diag(Xnew)

    # TODO handle multioutput GPs
    if len(q_sqrt.shape)== 3 and q_sqrt.shape[0] == 1:
        q_sqrt = q_sqrt[0, :, :]
    else:
        raise ValueError("Multioutput GPs not implemented yet")

    mean, cov = base_conditional(
        Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white
    )
    return mean, cov


def base_conditional(
    Kmn, Kmm, Knn, f, full_cov=False, q_sqrt=None, white=False
):
    # Lm = sp.linalg.cho_factor(Kmm)
    # print("Kmm")
    # print(Kmm.shape)
    Lm = sp.linalg.cholesky(Kmm, lower=True)
    return base_conditional_with_lm(
        Kmn=Kmn,
        Lm=Lm,
        # Lm=Kmm,
        Knn=Knn,
        f=f,
        full_cov=full_cov,
        q_sqrt=q_sqrt,
        white=white,
    )


def base_conditional_with_lm(
    Kmn, Lm, Knn, f, full_cov=False, q_sqrt=None, white=False
):
    # c, low = sp.linalg.cho_factor(Lm)
    # A = sp.linalg.cho_solve(Lm, Kmn)
    A = sp.linalg.solve_triangular(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    if full_cov:
        # fvar = Knn - A.T @ A
        # print("A")
        # print(A.shape)
        # print(Knn.shape)
        # print(Kmn.shape)
        # print(Lm.shape)
        # AT = jnp.transpose(A, [A.shape])
        AT = leading_transpose(A, perm=[..., -1, -2])
        # print(AT.shape)
        fvar = Knn - jnp.matmul(AT, A)
    else:
        fvar = Knn - jnp.sum(jnp.square(A), -2)
        print('fvar 1 ')
        print(fvar.shape)

    # print("f")
    # print(f.shape)
    # another backsubstitution in the unwhitened case
    # if not white:
    #     A = sp.linalg.solve_triangular(Lm.T, A, lower=False)

    # construct the conditional mean
    # Amean = sp.linalg.solve_triangular(Lm.T, f, lower=False)
    # print('Amean')
    # print(Amean.shape)
    # print(A.shape)
    # fmean = A.T @ Amean
    fmean = A.T @ f

    if q_sqrt is not None:
        q_sqrt_dims = len(q_sqrt.shape)
        if q_sqrt_dims == 2:
            # LTA = sp.linalg.solve_triangular(Lm, q_sqrt)
            # print("LTA")
            # print(q_sqrt.shape)
            # print(A.shape)
            # B = sp.linalg.cho_solve(Lm, q_sqrt)
            # print(LTA.shape)
            # LTALTA = A.T @ LTA
            # print(LTALTA.shape)
            # LTA = B.T @ A
            # LTA = q_sqrt.T @ A
            LTA = A.T @ q_sqrt
            # q_sqrtT = q_sqrt.T
            # print(q_sqrtT[:, :, None].shape)
            # LTA = A * q_sqrtT[:, :, None]
            # print('LTA')
            # print(LTA.shape)
            # LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # [R, M, N]
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt_dims))

        # fvar = fvar + LTA.T @ LTA
        # fvar = fvar + LTA @ LTA.T
        if full_cov:
            fvar = fvar + LTA @ LTA.T
            # fvar = fvar + LTA.T @ LTA
            # fvar = fvar + LTALTA @ LTALTA.T
            # print("fvar")
            # print(fvar.shape)
        else:
            # fvar = fvar + jnp.sum(jnp.square(LTA), -2)
            print('fvar 2 ')
            print(LTA.shape)
            fvar = fvar + jnp.sum(jnp.square(LTA), -1)
            print(fvar.shape)

    return fmean, fvar
