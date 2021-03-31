#!/usr/bin/env python3
# from gpjax.kernels.base import Kernel
from gpjax.kernels.kernels import (
    Kernel,
    # Cosine,
    SquaredExponential,
    Stationary,
)

RBF = SquaredExponential
