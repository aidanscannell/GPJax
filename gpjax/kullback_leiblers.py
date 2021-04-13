#!/usr/bin/env python3
from typing import Optional, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import tensor_annotations.jax as tjax

from gpjax.custom_types import InputDim, NumInducing, OutputDim
from gpjax.kernels import Kernel
from gpjax.config import default_jitter, default_float


def prior_kl(
    kernel_params: dict,
    inducing_variable: tjax.Array2[NumInducing, InputDim],
    kernel: Kernel,
    q_mu: tjax.Array2[NumInducing, OutputDim],
    q_sqrt: Union[
        tjax.Array3[OutputDim, NumInducing, NumInducing],
        tjax.Array2[NumInducing, OutputDim],
    ],
    whiten: Optional[bool] = False,
) -> jnp.float64:
    """KL divergence KL[q(x) || p(x)] between variational dist and prior.

    Dists are given by,
          q(x) = N(q_mu, q_sqrt q_sqrt^T)
    and
        if not whiten:
          p(x) = N(0, K(inducing_variable, inducing_variable))
        if whiten:
          p(x) = N(0, I)

    :param kernel_params: dict of params for kernel
    :param inducing_variable: array of inducing inputs [num_inducing, input_dim]
    :param kernel: Kernel associated with the prior p(x) = N(0, K(x,x))
    :param q_mu: [num_inducing, output_dim]
    :param q_sqrt: [output_dim, num_inducing, num_inducing] or [num_inducing, output_dim]
    :param whiten: whether or not to whiten
    :return: sum of KL[q(x) || p(x)] for each output_dim
    """
    if whiten:
        return gauss_kl(q_mu, q_sqrt)
    else:
        K = kernel(
            kernel_params, inducing_variable, inducing_variable, full_cov=True
        )  # [P, M, M]
        K += jnp.eye(K.shape[-2]) * default_jitter()
        return gauss_kl(q_mu, q_sqrt, Kp=K)


def gauss_kl(
    q_mu: tjax.Array2[NumInducing, OutputDim],
    q_sqrt: Union[
        tjax.Array3[OutputDim, NumInducing, NumInducing],
        tjax.Array2[NumInducing, OutputDim],
    ],
    Kp: Optional[tjax.Array3[OutputDim, NumInducing, NumInducing]] = None,
    Lp: Optional[tjax.Array3[OutputDim, NumInducing, NumInducing]] = None,
) -> jnp.float64:
    """KL divergence KL[q(x) || p(x)] between two multivariate normal dists.

    This function handles sets of independent multivariate normals, e.g.
    independent multivariate normals on each output dimension. It returns
    the sum of the divergences.

    Dists are given by,
          q(x) = N(q_mu, q_sqrt q_sqrt^T)
    and
          p(x) = N(0, Kp)  or  p(x) = N(0, Lp Lp^T)

    :param q_mu: [num_inducing, output_dim]
    :param q_sqrt: [output_dim, num_inducing, num_inducing] or [num_inducing, output_dim]
    :param Kp: [output_dim, num_inducing, num_inducing]
    :param Lp: [output_dim, num_inducing, num_inducing]
    :return: sum of KL[q(x) || p(x)] for each output_dim with shape []
    """
    if q_sqrt.ndim == 2:
        in_axes = (-1, -1, 0)
    elif q_sqrt.ndim == 3:
        in_axes = (-1, 0, 0)
    else:
        raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.ndim))

    if Kp is not None:
        if Kp.ndim == 2:
            Kp = jnp.expand_dims(Kp, 0)
        kls = jax.vmap(single_gauss_kl, in_axes=in_axes)(q_mu, q_sqrt, Kp)
    elif Lp is not None:
        raise NotImplementedError("Implement with a wrapper around single_gauss_kl")
        # if Lp.ndim == 2:
        #     Lp = jnp.expand_dims(Lp, 0)
        # kls = jax.vmap(single_gauss_kl, in_axes=in_axes)(q_mu, q_sqrt, Lp)
    else:
        kls = jax.vmap(single_gauss_kl, in_axes=in_axes[:-1])(q_mu, q_sqrt)
    kl_sum = jnp.sum(kls)
    return kl_sum


def single_gauss_kl(
    q_mu: tjax.Array1[NumInducing],
    q_sqrt: Union[tjax.Array2[NumInducing, NumInducing], tjax.Array1[NumInducing]],
    Kp: Optional[tjax.Array2[NumInducing, NumInducing]] = None,
    Lp: Optional[tjax.Array2[NumInducing, NumInducing]] = None,
) -> jnp.float64:
    """KL divergence KL[q(x) || p(x)] between two multivariate normal dists.

    Dists are given by,
          q(x) = N(q_mu, q_sqrt q_sqrt^T)
    and
          p(x) = N(0, Kp)  or  p(x) = N(0, Lp Lp^T)

    :param q_mu: [num_inducing]
    :param q_sqrt: [num_inducing, num_inducing] or [num_inducing]
    :param Kp: [num_inducing, num_inducing]
    :param Lp: [num_inducing, num_inducing]
    :return: KL[q(x) || p(x)] with shape []
    """
    q_diag = q_sqrt.ndim == 1
    whiten = (Kp is None) and (Lp is None)

    if whiten:
        alpha = q_mu  # [M]
    else:
        if Lp is None:
            if Kp is not None:
                Lp = jsp.linalg.cholesky(Kp, lower=True)
            else:
                raise ("what to use for p(x)??, implement N(0,I)")

        alpha = jsp.linalg.solve_triangular(Lp, q_mu, lower=True)  # [M]

    if q_diag:
        Lq = Lq_diag = q_sqrt
    else:
        Lq = q_sqrt
        Lq_diag = jnp.diag(Lq)  # [M]

    # Trace term: tr(Σp⁻¹ Σq)
    if whiten:
        trace = jnp.sum(jnp.square(q_sqrt))
    else:
        if q_diag:
            # Lq = Lq_diag = q_sqrt
            Lp_inv = jsp.linalg.solve_triangular(
                Lp, jnp.eye(q_mu.shape[0], dtype=default_float()), lower=True
            )  # [M, M]
            Kp_inv = jsp.linalg.solve_triangular(Lp.T, Lp_inv, lower=False)  # [M, M]
            Kp_inv = jnp.diag(Kp_inv)  # [M]
            trace = jnp.sum(Kp_inv * jnp.square(q_sqrt))
        else:
            LpiLq = jsp.linalg.solve_triangular(Lp, Lq, lower=True)  # [M, M]
            trace = jnp.sum(jnp.square(LpiLq))

    # Mahalanobis term: μqᵀ Σp⁻¹ μq
    mahalanobis = jnp.sum(jnp.square(alpha))

    # Constant term: M
    constant = q_mu.shape[0]

    # Log-determinant of the covariance of q(x):
    logdet_qcov = jnp.sum(jnp.log(jnp.square(Lq_diag)))

    two_kl = mahalanobis - constant - logdet_qcov + trace

    # Log-determinant of the covariance of p(x):
    if not whiten:
        log_det_pcov = jnp.sum(jnp.log(jnp.square(jnp.diag(Lp))))
        two_kl += log_det_pcov

    return two_kl / 2.0
