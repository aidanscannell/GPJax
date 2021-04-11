#!/usr/bin/env python3
from gpjax.kernels.base import Kernel, Combination, kernel_decorator
from gpjax.kernels.stationaries import (
    squared_exponential_cov_fn,
    SquaredExponential,
    Stationary,
)

RBF = SquaredExponential

from gpjax.kernels.multioutput import SeparateIndependent, MultioutputKernel
