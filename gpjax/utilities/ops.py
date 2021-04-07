#!/usr/bin/env python3
from typing import List, Union

import jax
import numpy as np
from jax import numpy as jnp

EllipsisType = type(...)


@jax.partial(jnp.vectorize, signature="(m)->(m,m)")
def batched_diag(K: jnp.DeviceArray) -> jnp.DeviceArray:
    return jnp.diag(K)


def leading_transpose(
    tensor: jnp.ndarray,
    perm: List[Union[int, EllipsisType]],
    leading_dim: int = 0,
) -> jnp.ndarray:
    """
    Transposes tensors with leading dimensions. Leading dimensions in
    permutation list represented via ellipsis `...`.
    When leading dimensions are found, `transpose` method
    considers them as a single grouped element indexed by 0 in `perm` list. So, passing
    `perm=[-2, ..., -1]`, you assume that your input tensor has [..., A, B] shape,
    and you want to move leading dims between A and B dimensions.
    Dimension indices in permutation list can be negative or positive. Valid positive
    indices start from 1 up to the tensor rank, viewing leading dimensions `...` as zero
    index.
    Example:
        a = tf.random.normal((1, 2, 3, 4, 5, 6))
            # [..., A, B, C],
            # where A is 1st element,
            # B is 2nd element and
            # C is 3rd element in
            # permutation list,
            # leading dimensions are [1, 2, 3]
            # which are 0th element in permutation
            # list
        b = leading_transpose(a, [3, -3, ..., -2])  # [C, A, ..., B]
        sess.run(b).shape
        output> (6, 4, 1, 2, 3, 5)
    :param tensor: TensorFlow tensor.
    :param perm: List of permutation indices.
    :returns: TensorFlow tensor.
    :raises: ValueError when `...` cannot be found.
    """
    idx = perm.index(...)
    perm[idx] = leading_dim
    perm = np.array(perm)
    rank = tensor.ndim
    perm_tf = perm % rank

    leading_dims = np.arange(rank - len(perm) + 1)
    perm = np.concatenate([perm_tf[:idx], leading_dims, perm_tf[idx + 1 :]], 0)
    return jnp.transpose(tensor, perm)
