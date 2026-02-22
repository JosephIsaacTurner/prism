import jax.numpy as jnp
from jax import jit
import numpy as np
from scipy.stats import spearmanr


@jit
def zscore(arr):
    """
    Compute the z-score of an array.

    Args:
        arr (jnp.ndarray): The input array.

    Returns:
        jnp.ndarray: The z-scored array.
    """
    mean = jnp.mean(arr, axis=0)
    std = jnp.std(arr, axis=0)
    return jnp.nan_to_num((arr - mean) / std)


def spearman_r(x, y): 
    """
    Compute the Spearman rank correlation coefficient.

    Args:
        x (np.ndarray): First input array.
        y (np.ndarray): Second input array.

    Returns:
        float: The Spearman correlation coefficient.
    """
    return spearmanr(x, y).statistic