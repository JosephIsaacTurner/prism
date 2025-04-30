import jax.numpy as jnp
from jax import jit
import numpy as np
from scipy.stats import spearmanr


@jit
def zscore(arr):
    mean = jnp.mean(arr, axis=0)
    std = jnp.std(arr, axis=0)
    return jnp.nan_to_num((arr - mean) / std)


def spearman_r(x, y): 
    return spearmanr(x, y).statistic