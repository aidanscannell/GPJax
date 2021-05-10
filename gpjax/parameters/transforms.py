#!/usr/bin/env python3
from typing import Callable

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


def sort_dict(base_dict: dict) -> dict:
    return dict(sorted(base_dict.items()))


def build_constrain_params(transforms: dict) -> Callable:
    transforms = sort_dict(transforms)

    def constrain_params(params: dict) -> dict:
        params = sort_dict(params)

        def transform_param(param, transform):
            if isinstance(transform, tfp.bijectors.Bijector):
                return jnp.array(transform.forward(param))
            else:
                return param

        return jax.tree_util.tree_multimap(transform_param, params, transforms)

    return constrain_params


def build_unconstrain_params(transforms: dict) -> Callable:
    transforms = sort_dict(transforms)

    def unconstrain_params(params: dict) -> dict:
        params = sort_dict(params)

        def transform_param(param, transform):
            if isinstance(transform, tfp.bijectors.Bijector):
                return jnp.array(transform.inverse(param))
            else:
                return param

        return jax.tree_util.tree_multimap(transform_param, params, transforms)

    return unconstrain_params
