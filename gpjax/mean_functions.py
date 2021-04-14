# import jax
import abc
import jax.numpy as jnp
import jax

from gpjax.custom_types import InputData, OutputData
from gpjax.base import Module


class MeanFunction(Module, abc.ABC):
    """The base mean function class.

    Each mean function must implement the __call__ method.
    """

    def __init__(self, default_int=None):
        if default_int is None:
            self.default_int = jnp.int64
        else:
            self.default_int = default_int

    @jax.partial(jax.jit, static_argnums=0)
    def __call__(self, params: dict, X: InputData) -> OutputData:
        """Return the mean functon evaluated at the input tensor X

        This takes a tensor X and returns a tensor m(X).
        Note that MeanFunction classes can have trainable variables.
        Preserves leading dimensions in X.

        :param X: input tensor [..., num_test, input_dim]
        :returns: output tensor m(X) [..., num_test, output_dim]
        """
        raise NotImplementedError(
            "Implement the __call__ method for this mean function"
        )

    # def __add__(self, other):
    #     return Additive(self, other)

    # def __mul__(self, other):
    #     return Product(self, other)


# class Additive(MeanFunction):
#     def __init__(self, first_part, second_part):
#         MeanFunction.__init__(self)
#         self.add_1 = first_part
#         self.add_2 = second_part

#     def __call__(self, X):
#         return jnp.add(self.add_1(X), self.add_2(X))


# class Product(MeanFunction):
#     def __init__(self, first_part, second_part):
#         MeanFunction.__init__(self)

#         self.prod_1 = first_part
#         self.prod_2 = second_part

#     def __call__(self, X):
#         return jnp.multiply(self.prod_1(X), self.prod_2(X))


class Constant(MeanFunction):
    def __init__(self, c=None, output_dim=1):
        super().__init__()
        if c is None:
            c = jnp.zeros(1)
        self.c = c
        self.output_dim = output_dim

    def get_params(
        self,
    ) -> dict:
        return {"constant": self.c}

    @jax.partial(jax.jit, static_argnums=0)
    def __call__(self, params: dict, X: InputData) -> OutputData:
        """Returns constant with shape (X.shape[:-1], output_dim)

        Preserves leading dimensions in X.

        :param X: input tensor [..., num_test, input_dim]
        :returns: output tensor m(X) [..., num_test, output_dim]
        """
        c = params["constant"]
        output_shape = (*jnp.shape(X)[:-1], self.output_dim)
        return jnp.ones(output_shape, dtype=X.dtype) * params["constant"]


class Zero(Constant):
    def __init__(self, output_dim=1):
        Constant.__init__(self)
        self.output_dim = output_dim
        del self.c

    def get_params(
        self,
    ) -> dict:
        return None

    @jax.partial(jax.jit, static_argnums=(0))
    def __call__(self, params: dict, X: InputData) -> OutputData:
        """Returns zeros with shape the input tensor (X.shape[:-1], output_dim)

        Preserves leading dimensions in X.

        :param X: input tensor [..., num_test, input_dim]
        :returns: output tensor m(X) [..., num_test, output_dim]
        """
        output_shape = (*jnp.shape(X)[:-1], self.output_dim)
        return jnp.zeros(output_shape, dtype=X.dtype)


# class MeanFunction(objax.Module):
#     """The base mean function class.

#     Each mean function must implement the __call__ method.
#     """

#     def __init__(self, default_int=None):
#         if default_int is None:
#             self.default_int = jnp.int64
#         else:
#             self.default_int = default_int

#     def __call__(self, X: InputData) -> OutputData:
#         """Return the mean functon evaluated at the input tensor X

#         This takes a tensor X and returns a tensor m(X).
#         Note that MeanFunction classes can have trainable variables.

#         :param X: input tensor [num_test, input_dim]
#         :returns: output tensor m(X) [num_test, output_dim]
#         """
#         raise NotImplementedError(
#             "Implement the __call__ method for this mean function"
#         )

#     def __add__(self, other):
#         return Additive(self, other)

#     def __mul__(self, other):
#         return Product(self, other)


# class Additive(MeanFunction):
#     def __init__(self, first_part, second_part):
#         MeanFunction.__init__(self)
#         self.add_1 = first_part
#         self.add_2 = second_part

#     def __call__(self, X):
#         return jnp.add(self.add_1(X), self.add_2(X))


# class Product(MeanFunction):
#     def __init__(self, first_part, second_part):
#         MeanFunction.__init__(self)

#         self.prod_1 = first_part
#         self.prod_2 = second_part

#     def __call__(self, X):
#         return jnp.multiply(self.prod_1(X), self.prod_2(X))


# class Constant(MeanFunction):
#     def __init__(self, c=None):
#         super().__init__()
#         if c is None:
#             c = jnp.zeros(1)
#         # c = jnp.zeros(1) if c is None else c
#         self.const_train_var = objax.TrainVar(c)
#         self.c = objax.TrainRef(self.const_train_var)

#     def __call__(self, X):
#         tile_shape = jnp.concatenate(
#             [jnp.array(jnp.shape(X)[:-1]), jnp.array([1])],
#             axis=0,
#         )
#         reshape_shape = jnp.concatenate(
#             [
#                 jnp.ones(shape=(X.ndim - 1), dtype=self.default_int),
#                 jnp.array([-1]),
#             ],
#             axis=0,
#         )
#         return jnp.tile(jnp.reshape(self.c.value, reshape_shape), tile_shape)


# class Zero(Constant):
#     def __init__(self, output_dim=1):
#         Constant.__init__(self)
#         self.output_dim = output_dim
#         del self.c

#     def __call__(self, X):
#         output_shape = jnp.concatenate(
#             [jnp.array(jnp.shape(X)[:-1]), jnp.array([self.output_dim])],
#             axis=0,
#         )
#         return jnp.zeros(output_shape, dtype=X.dtype)
