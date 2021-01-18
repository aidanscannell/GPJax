#!/usr/bin/env python3
from gpjax.kernels.base import Kernel
from gpjax.kernels.stationaries import ( 
    # base classes:
    Stationary,
    IsotropicStationary,
    AnisotropicStationary,
    # actual kernel classes:
    SquaredExponential,
)

RBF = SquaredExponential
