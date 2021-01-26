import jax
import pytest
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)

from gpjax.kernels import Cosine, SquaredExponential


class Data:
    x1 = jnp.linspace(0, 10, 20).reshape(10, 2)
    x2 = jnp.linspace(0, 10, 60).reshape(30, 2)


class Kern:
    lengthscales = jnp.array([1, 0.1], dtype=jnp.float64)
    variance = 2.0


class BatchedData:
    x1 = jnp.linspace(0, 10, 80 * 5).reshape(5, 4, 10, 2)
    x2 = jnp.linspace(0, 10, 240).reshape(5, 4, 6, 2)


def test_SquaredExponential():
    kernel = SquaredExponential(
        lengthscales=Kern.lengthscales, variance=Kern.variance
    )

    cov = kernel.K(Data.x1)
    assert cov.shape[-1] == Data.x1.shape[0]
    assert cov.shape[-2] == Data.x1.shape[0]
    cov = kernel.K(Data.x1, Data.x2)
    assert cov.shape[-2] == Data.x1.shape[0]
    assert cov.shape[-1] == Data.x2.shape[0]

    cov = kernel.K(BatchedData.x1)
    assert cov.shape[-1] == BatchedData.x1.shape[-2]
    assert cov.shape[-2] == BatchedData.x1.shape[-2]

    # TODO assetion to check leading dims are correct?
    cov = kernel.K(BatchedData.x1, BatchedData.x2)
    assert cov.shape[-1] == BatchedData.x2.shape[-2]
    assert cov.shape[-2] == BatchedData.x1.shape[-2]


def test_Cosine():
    kernel = Cosine(lengthscales=Kern.lengthscales, variance=Kern.variance)

    cov = kernel.K(Data.x1)
    assert cov.shape[-1] == Data.x1.shape[0]
    assert cov.shape[-2] == Data.x1.shape[0]
    cov = kernel.K(Data.x1, Data.x2)
    assert cov.shape[-2] == Data.x1.shape[0]
    assert cov.shape[-1] == Data.x2.shape[0]

    cov = kernel.K(BatchedData.x1)
    assert cov.shape[-1] == BatchedData.x1.shape[-2]
    assert cov.shape[-2] == BatchedData.x1.shape[-2]

    # TODO assetion to check leading dims are correct?
    cov = kernel.K(BatchedData.x1, BatchedData.x2)
    assert cov.shape[-1] == BatchedData.x2.shape[-2]
    assert cov.shape[-2] == BatchedData.x1.shape[-2]


if __name__ == "__main__":
    # test_Cosine()
    test_SquaredExponential()
