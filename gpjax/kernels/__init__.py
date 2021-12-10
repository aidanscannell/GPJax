#!/usr/bin/env python3
from gpjax.kernels.base import Kernel, Combination, covariance_decorator
from gpjax.kernels.stationaries import (
    squared_exponential_cov_fn,
    SquaredExponential,
    Stationary,
    SigmoidRectangle,
)

RBF = SquaredExponential

from gpjax.kernels.multioutput import SeparateIndependent, MultioutputKernel
