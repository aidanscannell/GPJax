#!/usr/bin/env python3
from jax import numpy as np
from typing import Tuple


MeanAndVariance = Tuple[np.array, np.array]
InputData = np.array
OutputData = np.array
MeanFunc = np.float64
# TODO replace DiffRBF with abstract kernel class
