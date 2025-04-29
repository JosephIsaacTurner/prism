import jax.numpy as jnp
from jax import jit
import numpy as np


@jit
def zscore(arr):
    mean = jnp.mean(arr, axis=0)
    std = jnp.std(arr, axis=0)
    return jnp.nan_to_num((arr - mean) / std)

