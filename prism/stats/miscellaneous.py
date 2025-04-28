import jax.numpy as jnp
from jax import jit
import numpy as np


@jit
def zscore(arr):
    mean = jnp.mean(arr, axis=0)
    std = jnp.std(arr, axis=0)
    return jnp.nan_to_num((arr - mean) / std)


def partition_model(design, contrast):
    """
    Partition design matrix into regressors of interest and nuisances
    using the Beckmann method defined by contrast.

    Parameters
    ----------
    design : array, shape (n_samples, n_regressors) or (n_samples, n_regressors, n_nodes)
        Design matrix. If 2D, treated as single “node” (t=1).
    contrast : array, shape (n_regressors,) or (n_regressors, n_contrasts)
        Contrast vector or matrix.

    Returns
    -------
    regressors_of_interest : array, shape (n_samples, n_contrasts[, n_nodes])
        Regressors of interest.
    nuisance_regressors : array, shape (n_samples, n_regressors - n_contrasts[, n_nodes])
        Nuisance regressors.
    effective_contrast_overall : array, shape (n_contrasts, n_regressors)
        Effective contrast for [regressors_of_interest, nuisance_regressors].
    effective_contrast_interest : array, shape (n_contrasts, n_contrasts)
        Effective contrast for regressors_of_interest only.
    """
    design = np.asarray(design)
    if design.ndim == 2:
        design = design[:, :, None]
    n, p, t = design.shape

    contrast = np.asarray(contrast)
    if contrast.ndim == 1:
        contrast = contrast[:, None]
    elif contrast.ndim == 2 and contrast.shape[0] != p and contrast.shape[1] == p:
        contrast = contrast.T
    if contrast.ndim != 2 or contrast.shape[0] != p:
        raise ValueError(f"Contrast must be shape ({p}, q), got {contrast.shape}")
    q = contrast.shape[1]

    def null_space(A, tol=None):
        U, s, Vh = np.linalg.svd(A, full_matrices=True)
        if tol is None:
            tol = max(A.shape) * np.finfo(s.dtype).eps * s[0]
        rank = (s > tol).sum()
        return Vh[rank:].T

    Cu = null_space(contrast.T)  # (p, p-q)
    r_nuis = Cu.shape[1]

    regressors_of_interest = np.zeros((n, q, t), dtype=design.dtype)
    nuisance_regressors = np.zeros((n, r_nuis, t), dtype=design.dtype)

    for i in range(t):
        Di = design[:, :, i]
        D = np.linalg.pinv(Di.T @ Di)
        CDC = contrast.T @ D @ contrast
        CDCi = np.linalg.pinv(CDC)
        Pc = contrast @ CDCi @ contrast.T @ D
        Cv = Cu - Pc @ Cu
        F3 = np.linalg.pinv(Cv.T @ D @ Cv)

        regressors_of_interest[:, :, i] = Di @ D @ contrast @ CDCi
        nuisance_regressors[:, :, i] = Di @ D @ Cv @ F3

    # build and transpose effective contrasts
    effective_contrast_overall = np.vstack(
        [np.eye(q), np.zeros((r_nuis, q))]
    ).T  # (q, p)
    effective_contrast_interest = effective_contrast_overall[:, :q]  # (q, q)

    if t == 1:
        regressors_of_interest = regressors_of_interest[:, :, 0]
        nuisance_regressors = nuisance_regressors[:, :, 0]

    return (
        regressors_of_interest,
        nuisance_regressors,
        effective_contrast_overall,
        effective_contrast_interest,
    )
