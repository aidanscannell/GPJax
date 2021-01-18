#!/usr/bin/env python3
from jax import numpy as np
from jax import scipy as sp


def base_conditional(Kmn, Kmm, Knn, f, full_cov=False, q_sqrt=None, white=False):
    # Lm = sp.linalg.cho_factor(Kmm)
    print("Kmm")
    print(Kmm.shape)
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


def base_conditional_with_lm(Kmn, Lm, Knn, f, full_cov=False, q_sqrt=None, white=False):
    # c, low = sp.linalg.cho_factor(Lm)
    # A = sp.linalg.cho_solve(Lm, Kmn)
    A = sp.linalg.solve_triangular(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - A.T @ A
    else:
        fvar = Knn - np.sum(np.square(A), -2)

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
            print("LTA")
            print(q_sqrt.shape)
            print(A.shape)
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
            print(LTA.shape)
            # LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # [R, M, N]
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt_dims))

        # fvar = fvar + LTA.T @ LTA
        # fvar = fvar + LTA @ LTA.T
        if full_cov:
            fvar = fvar + LTA @ LTA.T
            # fvar = fvar + LTA.T @ LTA
            # fvar = fvar + LTALTA @ LTALTA.T
            print("fvar")
            print(fvar.shape)
        else:
            # fvar = fvar + np.sum(np.square(LTA), -2)
            fvar = fvar + np.sum(np.square(LTA), -1)

    return fmean, fvar
