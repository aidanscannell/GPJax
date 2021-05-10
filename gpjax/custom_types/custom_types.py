#!/usr/bin/env python3
from typing import Optional, Tuple, Union

import tensor_annotations.jax as tjax
from jax import numpy as jnp
from tensor_annotations.axes import Batch

from .shapes import N1, N2, InputDim, NumData, OutputDim, NumInducing

# MeanAndVariance = Tuple[jnp.ndarray, jnp.ndarray]
# InputData = jnp.ndarray
# OutputData = jnp.ndarray
InputData = tjax.Array2[NumData, InputDim]
OutputData = tjax.Array2[NumData, OutputDim]
# MeanFunc = jnp.float64

# Variance = Union[jnp.float64, jnp.ndarray]
# Lengthscales = Union[jnp.float64, jnp.ndarray]


# Data types
SingleInput = tjax.Array1[InputDim]
# Inputs = tjax.Array2[NumData, InputDim]
# BatchedInputs = tjax.Array3[Batch, NumData, InputDim]
# Output = tjax.Array1[OutputDim]
# Outputs = tjax.Array2[NumData, OutputDim]
# BatchedOutputs = tjax.Array3[Batch, NumData, OutputDim]

# AnyInput = Union[Input, Inputs, BatchedInputs]
# AnyOutput = Union[Output, Outputs, BatchedOutputs]


# Kernel parameter types
Lengthscales = Union[tjax.Array1[InputDim], jnp.float64]
KernVariance = tjax.Array1

Input1 = Union[
    tjax.Array1[InputDim],
    tjax.Array2[N1, InputDim],
    tjax.Array3[Batch, N1, InputDim],
]
Input2 = Optional[
    Union[
        tjax.Array1[InputDim],
        tjax.Array2[N2, InputDim],
        tjax.Array3[Batch, N2, InputDim],
    ]
]

Covariance = Union[
    tjax.Array1,
    tjax.Array2[N1, N2],
    tjax.Array2[N1, N1],  # if X2=None
    tjax.Array3[Batch, N1, N2],
    tjax.Array3[Batch, N1, N1],  # if X2=None
]

MultiOutputCovariance = Union[
    tjax.Array1[OutputDim],
    tjax.Array3[OutputDim, N1, N2],
    tjax.Array3[OutputDim, N1, N1],  # if X2=None
    tjax.Array4[OutputDim, N1, OutputDim, N2],
    tjax.Array4[OutputDim, N1, OutputDim, N1],  # if X2=None
    # tjax.Array5[Batch, OutputDim, N1, OutputDim, N2],
    # tjax.Array5[Batch, OutputDim, N1, OutputDim, N1],  # if X2=None
]

# Data types
Mean = tjax.Array2[NumData, OutputDim]
Variance = tjax.Array2[NumData, OutputDim]
# Covariance = tjax.Array3[OutputDim, NumData, NumData]
# MeanAndVariance = Union[Tuple[Mean, Variance], Tuple[Mean, Covariance]]
MeanAndCovariance = Union[Tuple[Mean, Variance], Tuple[Mean, Covariance]]

InducingVariable = tjax.Array2[NumInducing, InputDim]
