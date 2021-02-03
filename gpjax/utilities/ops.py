#!/usr/bin/env python3
from typing import List, Union

import numpy as np
from jax import numpy as jnp

EllipsisType = type(...)


# def broadcasting_elementwise(
#     op, a: jnp.ndarray, b: jnp.ndarray
# ) -> jnp.ndarray:
#     """
#     Apply binary operation `op` to every pair in tensors `a` and `b`.
#     :param op: binary operator on tensors, e.g. jnp.add, jnp.substract
#     :param a: jnp.ndarray, shape [n_1, ..., n_a]
#     :param b: jnp.ndarray, shape [m_1, ..., m_b]
#     :return: jnp.ndarray, shape [n_1, ..., n_a, m_1, ..., m_b]
#     """
#     flatres = op(jnp.reshape(a, [-1, 1]), jnp.reshape(b, [1, -1]))
#     print("shape")
#     print(jnp.shape(a))
#     print(a.shape)
#     print(flatres.shape)
#     # shape = jnp.concatenate([jnp.array(jnp.shape(a)), jnp.array(jnp.shape(b))])
#     shape = [jnp.array(jnp.shape(a)), jnp.array(jnp.shape(b))]
#     print(shape)
#     return jnp.reshape(flatres, shape)


def square_distance(X: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
    """
    Returns ||X - X2ᵀ||²
    Due to the implementation and floating-point imprecision, the
    result may actually be very slightly negative for entries very
    close to each other.
    This function can deal with leading dimensions in X and X2.
    In the sample case, where X and X2 are both 2 dimensional,
    for example, X is [N, D] and X2 is [M, D], then a tensor of shape
    [N, M] is returned. If X is [N1, S1, D] and X2 is [N2, S2, D]
    then the output will be [N1, S1, N2, S2].
    """
    if X2 is None:
        Xs = jnp.sum(jnp.square(X), axis=-1, keepdims=True)
        XT = leading_transpose(X, perm=[..., -1, -2])
        dist = -2 * jnp.matmul(X, XT)
        XsT = leading_transpose(Xs, perm=[..., -1, -2])
        conj = jnp.conjugate(XsT)
        dist += Xs + conj
        return dist
    Xs = jnp.sum(jnp.square(X), axis=-1, keepdims=True)
    X2s = jnp.sum(jnp.square(X2), axis=-1, keepdims=True)
    X2sT = leading_transpose(X2s, perm=[..., -1, -2])
    X2T = leading_transpose(X2, perm=[..., -1, -2])
    dist = -2 * jnp.matmul(X, X2T)
    dist2 = Xs + X2sT  # broadcast
    dist += dist2
    return dist


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


def difference_matrix(X, X2=None):
    """
    Returns (X - X2ᵀ)
    This function can deal with leading dimensions in X and X2.
    For example, If X has shape [M, D] and X2 has shape [N, D],
    the output will have shape [M, N, D]. If X has shape [I, J, M, D]
    and X2 has shape [I, J, N, D], the output will have shape
    [I, J, M, N, D].
    """
    if X2 is None:
        X2 = X
    diff = jnp.expand_dims(X, -2) - jnp.expand_dims(X2, -3)
    return diff
