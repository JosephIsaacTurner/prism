import warnings
from functools import partial

import jax.numpy as jnp
from jax import jit, vmap
from jax.numpy.linalg import matrix_rank, pinv
from jax.ops import segment_sum
from jax.scipy.special import betainc
from jax.scipy.stats import norm
import numpy as np

"""
Generalized linear model (GLM) statistics.
"""


@jit
def t(Y, X, C):
    """
    Compute GLM t‐statistics for a single contrast.

    Parameters
    ----------
    Y : array, (n, p)
        Response data.
    X : array, (n, k)
        Design matrix.
    C : array, (k,)
        Contrast vector.

    Returns
    -------
    t_vals : array, (p,)
        t = (Cᵀβ) / SE, where
          β = (XᵀX)⁻¹ Xᵀ Y,
          df = n − k,
          MSE = ∑(resid²)/df,
          var_C = Cᵀ(XᵀX)⁻¹ C,
          SE = √(var_C · MSE).

    Notes
    -----
    0/0→0, k/0→±∞.
    """
    n, p = Y.shape
    C = jnp.ravel(C)
    k = X.shape[1]
    # fit GLM
    XtX_inv = pinv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    # residuals & df
    resid = Y - X @ beta
    df = n - k
    mse = jnp.maximum(jnp.sum(resid**2, axis=0) / df, jnp.finfo(resid.dtype).tiny)
    # t‐statistic
    var_C = jnp.maximum(C @ XtX_inv @ C, 0.0)
    t_vals = (C @ beta) / jnp.sqrt(var_C * mse)
    return jnp.nan_to_num(t_vals, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf), 1, df


@partial(jit, static_argnums=(4,))
def aspin_welch_v(Y, X, C, groups, J_max):
    """
    Compute Aspin–Welch v‐statistics for a single contrast.

    Parameters
    ----------
    Y : array, (n, p)
        Response data.
    X : array, (n, k)
        Design matrix.
    C : array, (k,)
        Contrast vector.
    groups : int array, (n,)
        Group membership.
    J_max : int
        Max number of groups.

    Returns
    -------
    v_vals : array, (p,)
        v = (Cᵀ β) / √den, where
          β        = (Xᵀ X)⁻¹ Xᵀ Y,
          Hᵢᵢ      = diagonal of X (Xᵀ X)⁻¹ Xᵀ,
          d_b      = ∑₍i∈group_b₎ (1 − Hᵢᵢ),
          RSS_b    = ∑₍i∈group_b₎ resid_i²,
          W_b      = d_b / RSS_b,
          M_b      = ∑₍i∈group_b₎ x_i x_iᵀ,
          cte      = ∑₍b=1…J₎ M_b · W_b,
          den      = Cᵀ [pinv(cte)] C,
          resid    = Y − X β.

    Notes
    -----
    0/0 → 0, k/0 → ±∞.
    """
    n, p = Y.shape
    C = jnp.ravel(C)
    # fit GLM
    XtX_inv = pinv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    resid = Y - X @ beta
    # group leverages
    _, g = jnp.unique(groups, return_inverse=True, size=J_max)
    H_diag = jnp.diag(X @ XtX_inv @ X.T)
    d = jnp.maximum(segment_sum(1 - H_diag, g, num_segments=J_max), 1e-12)
    # weights
    rss = jnp.maximum(segment_sum(resid**2, g, num_segments=J_max), 1e-12)
    W = d[:, None] / rss
    # Sattherthwaite df
    df = (W.sum(axis=0) ** 2) / ((W**2 / d[:, None]).sum(axis=0))
    # denom
    Mb = segment_sum(jnp.einsum("ij,ik->ijk", X, X), g, num_segments=J_max)
    cte = jnp.sum(Mb.reshape(J_max, -1, 1) * W[:, None, :], axis=0)
    den = vmap(lambda A: C @ (pinv(A) @ C))(
        cte.reshape(X.shape[1], X.shape[1], -1).transpose(2, 0, 1)
    )
    den = jnp.maximum(den, 1e-12)
    v_vals = (C @ beta) / jnp.sqrt(den)
    return jnp.nan_to_num(v_vals, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf), 1, df


@jit
def F(Y, X, C):
    """
    Compute GLM F‐statistics for multiple contrasts.

    Parameters
    ----------
    Y : array, (n, p)
        Response data.
    X : array, (n, k)
        Design matrix.
    C : array, (m, k)
        Contrast matrix.

    Returns
    -------
    F_vals : array, (p,)
        F = [(Cβ)' (C (XᵀX)⁻¹ Cᵀ)⁻¹ (Cβ) / m] / MSE, where
          β    = (XᵀX)⁻¹ Xᵀ Y,
          df₂  = n − rank(X),
          MSE  = ∑(resid²)/df₂,
          m    = rank(C).

    Notes
    -----
    0/0 → 0, k/0 → ±∞.
    """
    n, p = Y.shape
    # fit GLM
    XtX_inv = pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ Y)

    # residuals & MSE
    resid = Y - X @ beta
    df1 = matrix_rank(C)
    df2 = n - matrix_rank(X)
    mse = jnp.maximum(jnp.sum(resid**2, axis=0) / df2, jnp.finfo(resid.dtype).tiny)

    # contrast DF
    m = matrix_rank(C)

    # numerator SS
    CB = C @ beta
    CXX = C @ XtX_inv @ C.T
    ss = jnp.maximum(jnp.sum(CB * (pinv(CXX) @ CB), axis=0), 0.0)

    # F‐statistic
    F_vals = (ss / m) / mse
    return jnp.nan_to_num(F_vals, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf), df1, df2


@partial(jit, static_argnums=(4,))
def G(Y, X, C, groups, J_max):
    """
    Compute Aspin–Welch G‐statistics for multiple contrasts.

    Parameters
    ----------
    Y : array, (n, p)
        Response data.
    X : array, (n, k)
        Design matrix.
    C : array, (m, k)
        Contrast matrix.
    groups : int array, (n,)
        Group membership.
    J_max : int
        Max number of groups (static for jitting).

    Returns
    -------
    G_vals : array, (p,)
        G = [(Cβ)' V⁻¹ (Cβ) / m] / [1 + 2·(m−1)·b], where
          β      = (XᵀX)⁻¹ Xᵀ Y,
          V      = C · pinv(cte) · Cᵀ,
          cte    = ∑₍b=1…J_max₎ Mᵦ · W_int[ᵦ],
          Mᵦ     = ∑₍i∈groupᵦ₎ xᵢ xᵢᵀ,
          W_intᵦ = dᵦ / rssᵦ,
          dᵦ     = ∑₍i∈groupᵦ₎ (1 − hᵢᵢ),
          rssᵦ   = ∑₍i∈groupᵦ₎ residᵢ²,
          W_finᵦ = W_intᵦ · countᵦ,
          sW     = ∑₍ᵦ₎ W_finᵦ,
          b      = [∑₍ᵦ₎ (1 − W_finᵦ/sW)² / dᵦ] / [m·(m+2)],
          m      = rank(C).

    Notes
    -----
    0/0 → 0, k/0 → ±∞.
    """
    n, p = Y.shape
    # fit GLM
    XtX_inv = pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ Y)
    resid = Y - X @ beta

    # group indexing (static J_max)
    _, g = jnp.unique(groups, return_inverse=True, size=J_max)
    counts = jnp.bincount(g, length=J_max)

    # leverages
    H_diag = jnp.diag(X @ XtX_inv @ X.T)
    d = jnp.maximum(segment_sum(1 - H_diag, g, num_segments=J_max), 1e-12)

    # residual sums & weights
    rss = jnp.maximum(segment_sum(resid**2, g, num_segments=J_max), 1e-12)
    W_int = d[:, None] / rss
    W_fin = W_int * counts[:, None]

    # build cte
    Mb = segment_sum(jnp.einsum("ij,ik->ijk", X, X), g, num_segments=J_max)
    cte = jnp.sum(Mb.reshape(J_max, -1, 1) * W_int[:, None, :], axis=0)
    cte_mats = cte.reshape(X.shape[1], X.shape[1], -1).transpose(2, 0, 1)

    # numerator SS
    def quad(b_col, A):
        v = C @ b_col
        return jnp.maximum(v.T @ pinv(C @ pinv(A) @ C.T) @ v, 0.0)

    num_ss = vmap(quad)(beta.T, cte_mats)

    # Welch correction
    m = matrix_rank(C)
    sW = jnp.sum(W_fin, axis=0)
    b = jnp.sum((1 - W_fin / sW) ** 2 * (1 / d)[:, None], axis=0)
    b = b / (m * (m + 2))

    G_vals = (num_ss / m) / jnp.maximum(1 + 2 * (m - 1) * b, 1e-12)

    # Ddegrees of freedom
    df2 = (W_int.sum(axis=0) ** 2) / ((W_int**2 / d[:, None]).sum(axis=0))

    return jnp.nan_to_num(G_vals, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf), m, df2


@jit
def pearson_r(Y, X, C, *args, **kwargs):
    """
    Compute GLM-based Pearson r for a single contrast.

    Parameters
    ----------
    Y : array, (n, p)
        Response data.
    X : array, (n, k)
        Design matrix.
    C : array, (k,)
        Contrast vector.

    Returns
    -------
    r_vals : array, (p,)
        r = CB / √(CB² + df·var_C·MSE), where
          β = (XᵀX)⁻¹ Xᵀ Y,
          CB = C β,
          var_C = Cᵀ(XᵀX)⁻¹ C,
          df = n − k,
          MSE = ∑(Y − Xβ)²/df.

    Notes
    -----
    r in [−1, 1];  0/0→0, ±/0→±∞.
    """
    n, p = Y.shape
    C = jnp.ravel(C)
    k = X.shape[1]
    # fit GLM
    XtX_inv = pinv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    # residuals & mse
    resid = Y - X @ beta
    df = n - k
    mse = jnp.maximum(jnp.sum(resid**2, axis=0) / df, jnp.finfo(resid.dtype).tiny)
    # contrast effect and variance
    CB = C @ beta
    var_C = jnp.maximum(C @ XtX_inv @ C, 0.0)
    # compute Pearson r
    denom = jnp.sqrt(CB**2 + df * var_C * mse)
    r_vals = CB / denom
    return jnp.nan_to_num(r_vals, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf), 1, df


@jit
def r_squared(Y, X, C, *args, **kwargs):
    """
    Compute GLM-based coefficient of determination (R²) for one or multiple contrasts.

    Parameters
    ----------
    Y : array, (n, p)
        Response data.
    X : array, (n, k)
        Design matrix.
    C : array, (k,) or (m, k)
        Contrast vector or matrix.

    Returns
    -------
    r2_vals : array, (p,)
        R² = SS_mod / (SS_mod + df·MSE), where
          β = (XᵀX)⁻¹ Xᵀ Y,
          CB = C β,
          V = C (XᵀX)⁻¹ Cᵀ,
          SS_mod = CBᵀ V⁻¹ CB,
          df = n − k,
          MSE = ∑(Y − Xβ)²/df.

    Notes
    -----
    R² in [0, 1];  0/0→0.
    """
    n, p = Y.shape
    k = X.shape[1]
    # fit GLM
    XtX_inv = pinv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    # residuals & mse
    resid = Y - X @ beta
    df = n - k
    mse = jnp.maximum(jnp.sum(resid**2, axis=0) / df, jnp.finfo(resid.dtype).tiny)
    # contrast effect and covariance
    CB = C @ beta
    V = C @ XtX_inv @ C.T
    V_inv = pinv(V)
    # model sum of squares per voxel
    ss_mod = jnp.einsum("mp,mn,np->p", CB, V_inv, CB)
    # compute R²
    r2_vals = ss_mod / (ss_mod + df * mse)
    return jnp.nan_to_num(r2_vals, nan=0.0, posinf=1.0, neginf=0.0), matrix_rank(C), df


"""
Zstat equivalent functions for each of the above.
"""


@jit
def t_z(Y, X, C):
    """
    t→z via incomplete‐beta CDF and norm.ppf
    """
    t_vals, df1, df2 = t(Y, X, C)
    # compute I_{v/(v+t²)}(v/2,1/2)
    z2 = df2 / (df2 + t_vals**2)
    ib = betainc(df2 / 2, 0.5, z2)
    cdf_t = jnp.where(t_vals > 0, 1 - 0.5 * ib, 0.5 * ib)
    cdf_t = jnp.clip(
        cdf_t, jnp.finfo(t_vals.dtype).eps, 1 - jnp.finfo(t_vals.dtype).eps
    )
    z_vals = norm.ppf(cdf_t)
    return jnp.nan_to_num(z_vals, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf), 1, df2


@partial(jit, static_argnums=(4,))
def aspin_welch_v_z(Y, X, C, groups, J_max):
    """
    v→z via incomplete‐beta CDF and norm.ppf
    """
    v_vals, df1, df2 = aspin_welch_v(Y, X, C, groups, J_max)
    z2 = df2 / (df2 + v_vals**2)
    ib = betainc(df2 / 2, 0.5, z2)
    cdf_v = jnp.where(v_vals > 0, 1 - 0.5 * ib, 0.5 * ib)
    cdf_v = jnp.clip(
        cdf_v, jnp.finfo(v_vals.dtype).eps, 1 - jnp.finfo(v_vals.dtype).eps
    )
    z_vals = norm.ppf(cdf_v)
    return jnp.nan_to_num(z_vals, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf), 1, df2


@jit
def F_z(Y, X, C):
    """
    Convert F‐stats to z‐scores via the F‐CDF → normal‐ppf mapping.
    """
    F_vals, df1, df2 = F(Y, X, C)

    # CDF argument for F(df1,df2)
    x = (df1 * F_vals) / (df2 + df1 * F_vals)
    cdf_F = betainc(df1 / 2, df2 / 2, x)

    # guard boundaries
    eps = jnp.finfo(F_vals.dtype).eps
    cdf_F = jnp.clip(cdf_F, eps, 1 - eps)

    # inverse‐normal
    z = norm.ppf(cdf_F)
    return jnp.nan_to_num(z, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf), df1, df2


@partial(jit, static_argnums=(4,))
def G_z(Y, X, C, groups, J_max):
    """
    G→z via F‐CDF (incomplete‐beta) and norm.ppf.
    """
    G_vals, df1, df2 = G(Y, X, C, groups, J_max)

    x = (df1 * G_vals) / (df2 + df1 * G_vals)
    cdf_G = betainc(df1 / 2, df2 / 2, x)
    eps = jnp.finfo(G_vals.dtype).eps
    cdf_G = jnp.clip(cdf_G, eps, 1 - eps)
    z_vals = norm.ppf(cdf_G)

    return jnp.nan_to_num(z_vals, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf), df1, df2


@jit
def fisher_z(Y, X, C, *args, **kwargs):
    """
    Compute Fisher's z-statistic for a single contrast.

    Parameters
    ----------
    Y : array, (n, p)
        Response data.
    X : array, (n, k)
        Design matrix.
    C : array, (k,)
        Contrast vector.

    Returns
    -------
    z_vals : array, (p,)
        fisher_z = inverse hyperbolic tangent of pearson_r.
        See pearson_r docstring for details.

    Notes
    -----
    0/0→0, ±/0→±∞.
    """
    r, df1, df2 = pearson_r(Y, X, C)
    return jnp.arctanh(r), 1, df2


@jit
def r_squared_z(Y, X, C, *args, **kwargs):
    """
    Convert R² to z-statistic.

    Parameters
    ----------
    Y : array, (n, p)
        Response data.
    X : array, (n, k)
        Design matrix.
    C : array, (k,)
        Contrast vector.

    Returns
    -------
    z_vals : array, (p,)
        Use normally distributed percent point function to convert R² to z-statistic.
        See r_squared docstring for details.

    """
    r2, df1, df2 = r_squared(Y, X, C)
    return norm.ppf(r2), df1, df2


@jit
def residualize_data(data, design_matrix):
    """
    Residualize data against a set of regressors.

    Parameters
    ----------
    data : array, shape (n_samples, n_features)
        Observed data (response).
    design_matrix : array, shape (n_samples, n_regressors)
        Nuisance or covariate regressors.

    Returns
    -------
    residuals : array, shape (n_samples, n_features)
        Unexplained variation (data minus predictions).
    fitted_values : array, shape (n_samples, n_features)
        Model predictions (ŷ) from the regressors.
    """
    coeffs = jnp.linalg.lstsq(design_matrix, data, rcond=None)[0]
    fitted_values = design_matrix @ coeffs
    residuals = data - fitted_values
    return residuals, fitted_values


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
