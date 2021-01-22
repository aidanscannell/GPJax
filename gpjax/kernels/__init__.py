#!/usr/bin/env python3
from gpjax.kernels.base import Kernel
from gpjax.kernels.stationaries import (  # base classes:; actual kernel classes:
    AnisotropicStationary, Cosine, IsotropicStationary, SquaredExponential,
    Stationary)

RBF = SquaredExponential
