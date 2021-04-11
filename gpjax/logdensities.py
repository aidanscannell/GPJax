#!/usr/bin/env python3
import jax.numpy as jnp


def gaussian(x, mu, var):
    return -0.5 * (jnp.log(2 * jnp.pi) + jnp.log(var) + jnp.square(mu - x) / var)
