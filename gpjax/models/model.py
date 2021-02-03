#!/usr/bin/env python3
import abc
from typing import Optional

import jax
import jax.numpy as jnp
import objax
from gpjax.conditionals import base_conditional
from gpjax.covariances import (hessian_cov_fn_wrt_X1X1,
                               hessian_cov_fn_wrt_x1x1_hard_coded,
                               jacobian_cov_fn_wrt_X1)
from gpjax.custom_types import InputData, MeanAndVariance
from gpjax.kernels import Kernel
from gpjax.mean_functions import MeanFunction, Zero

jax.config.update("jax_enable_x64", True)

# create types
Likelihood = None
InducingVariable = None


class GPBase(objax.Module, abc.ABC):
    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: int = None,
        jitter=1e-6,
    ):
        assert (
            num_latent_gps is not None
        ), "GP requires specification of num_latent_gps"
        self.num_latent_gps = num_latent_gps
        self.kernel = kernel
        self.likelihood = likelihood
        if mean_function is None:
            mean_function = Zero()
        self.mean_function = mean_function
        self.jitter = jitter

    @abc.abstractmethod
    def predict_f(
        self,
        Xnew: InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
        """Compute mean and (co)variance of latent function at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :param full_cov:
            If True, draw correlated samples over Xnew. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.
        :param full_output_cov:
            TODO Not implemented
        :returns: tuple of Tensors (mean, variance),
            means.shape == [num_test, output_dim],
            If full_cov=True and full_output_cov=False,
                var.shape == [output_dim, num_test, num_test]
            If full_cov=False,
                var.shape == [num_test, output_dim]
        """
        raise NotImplementedError

    def predict_jacobian_f_wrt_Xnew(
        self,
        Xnew: InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
        """
        This only works for single output GPs
        TODO change this if GP or SVGP
        TODO implement full_output_cov functionality
        """
        f = self.q_mu
        X = self.inducing_variable

        Kxx = self.kernel.K(X, X)
        Kxx += self.jitter * jnp.eye(Kxx.shape[0])
        # print("Kxx")
        # print(Kxx.shape)
        if len(Xnew.shape) == 2:
            num_test = Xnew.shape[0]
            Kxx = jnp.broadcast_to(Kxx, [num_test, *Kxx.shape])
        # print(Kxx.shape)
        # print(type(Kxx))
        dKdx1 = jacobian_cov_fn_wrt_X1(self.kernel.K, Xnew, X)
        # print("dKdx1")
        # print(dKdx1.shape)
        # print(dKdx1)
        # print(type(dKdx1))
        # d2K = hessian_cov_fn_wrt_x1x1(kernel.K, Xnew)
        # TODO cheating here - only works for RBF kernel
        # lengthscale = kernel.lengthscales.value
        # lengthscale = self.kernel.lengthscales
        # d2K = hessian_cov_fn_wrt_x1x1_hard_coded(
        #     self.kernel.K, self.kernel.lengthscales, Xnew
        # )

        # def hessian_kernel_wrt_x1x1(Xnew: InputData):
        #     def cov_fn(x):
        #         x = x.reshape(1, -1)
        #         return kernel.K(x)
        #     d2K = jax.hessian(cov_fn)(Xnew.reshape(-1))
        #     input_dim = Xnew.shape[1]
        #     d2K = d2K.reshape([input_dim, input_dim])

        # d2K = hessian_cov_fn_wrt_x1x1(self.kernel.K, Xnew)
        d2K = hessian_cov_fn_wrt_X1X1(self.kernel.K, Xnew)
        # d2K = hessian_cov_fn_wrt_x1x1_hard_coded(self.kernel.K, Xnew)
        # print("d2k inside gp class")
        # print(d2K.shape)
        # print(d2K)
        # print(type(d2K))

        # if self.q_sqrt is not None:
        #     # todo map over output dimension
        #     # q_sqrt = jnp.squeeze(q_sqrt)
        # q_sqrt = self.q_sqrt.reshape(
        #     [self.q_sqrt.shape[-1], self.q_sqrt.shape[-1]]
        # )

        # TODO this is VERY hacky, only works for single output GPs
        # handle output dimension better
        self.num_inducing_points = self.q_sqrt.shape[-1]
        q_sqrt = self.q_sqrt.reshape(
            [self.num_inducing_points, self.num_inducing_points]
        )

        def single_base_conditional(dKdx1, Kxx, d2K):
            return base_conditional(
                Kmn=dKdx1,
                Kmm=Kxx,
                Knn=d2K,
                f=f,
                full_cov=full_cov,
                q_sqrt=q_sqrt,
                white=self.whiten,
            )

        if len(Xnew.shape) == 2:
            jac_mean, jac_cov = jax.vmap(single_base_conditional)(
                dKdx1, Kxx, d2K
            )
        else:
            jac_mean, jac_cov = single_base_conditional(dKdx1, Kxx, d2K)
        # jac_mean, jac_cov = base_conditional(
        #     Kmn=dKdx1,
        #     Kmm=Kxx,
        #     Knn=d2K,
        #     f=f,
        #     full_cov=full_cov,
        #     q_sqrt=q_sqrt,
        #     white=self.whiten,
        # )
        return jac_mean, jac_cov

    def predict_y(
        self,
        Xnew: InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
        """Compute the mean and (co)variance of function at Xnew."""
        if full_cov or full_output_cov:
            raise NotImplementedError(
                "The predict_y method currently supports only the argument values full_cov=False and full_output_cov=False"
            )

        f_mean, f_var = self.predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return self.likelihood.predict_mean_and_var(f_mean, f_var)
