#!/usr/bin/env python3
import abc
from typing import Optional, Union

import haiku as hk
from jax import numpy as jnp

PRNGKey = jnp.ndarray
ActiveDims = Union[slice, list]


class Kernel(hk.Module, metaclass=abc.ABCMeta):
    """
    The basic kernel class. Handles active dims.
    """

    def __init__(
        self, active_dims: Optional[ActiveDims] = None, name: Optional[str] = None
    ):
        """
        :param active_dims: active dimensions, either a slice or list of
            indices into the columns of X.
        :param name: optional kernel name.
        """
        super().__init__(name=name)
        self._active_dims = self._normalize_active_dims(active_dims)

    @staticmethod
    def _normalize_active_dims(value):
        if value is None:
            value = slice(None, None, None)
        if not isinstance(value, slice):
            value = jnp.array(value, dtype=int)
        return value

    @property
    def active_dims(self):
        return self._active_dims

    @active_dims.setter
    def active_dims(self, value):
        self._active_dims = self._normalize_active_dims(value)

    def _validate_ard_active_dims(self, ard_parameter):
        """
        Validate that ARD parameter matches the number of active_dims (provided active_dims
        has been specified as an array).
        """
        if self.active_dims is None or isinstance(self.active_dims, slice):
            # Can only validate parameter if active_dims is an array
            return

        if ard_parameter.shape.rank > 0 and ard_parameter.shape[0] != len(
            self.active_dims
        ):
            raise ValueError(
                f"Size of `active_dims` {self.active_dims} does not match "
                f"size of ard parameter ({ard_parameter.shape[0]})"
            )
