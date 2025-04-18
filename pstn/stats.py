from jax.numpy.linalg import pinv, matrix_rank
from jax import jit, vmap
import jax.numpy as jnp
from jax.ops import segment_sum
from functools import partial
import warnings


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
    C = jnp.atleast_1d(jnp.squeeze(C))
    k = X.shape[1]
    # fit GLM
    XtX_inv = pinv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    # residuals & df
    resid = Y - X @ beta
    df = n - k
    if df <= 0:
        warnings.warn("Non-positive df; returning NaNs.")
        return jnp.full(p, jnp.nan)
    # mse
    mse = jnp.maximum(jnp.sum(resid**2, axis=0) / df, jnp.finfo(resid.dtype).tiny)
    # t‐statistic
    var_C = jnp.maximum(C @ XtX_inv @ C, 0.0)
    t_vals = (C @ beta) / jnp.sqrt(var_C * mse)
    return jnp.nan_to_num(t_vals, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf)


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
        v = (Cᵀβ) / √den.
    """
    n, p = Y.shape
    C = jnp.atleast_1d(jnp.squeeze(C))
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
    # denom
    Mb = segment_sum(jnp.einsum("ij,ik->ijk", X, X), g, num_segments=J_max)
    cte = jnp.sum(Mb.reshape(J_max, -1, 1) * W[:, None, :], axis=0)
    den = vmap(lambda A: C @ (pinv(A) @ C))(
        cte.reshape(X.shape[1], X.shape[1], -1).transpose(2, 0, 1)
    )
    den = jnp.maximum(den, 1e-12)
    v_vals = (C @ beta) / jnp.sqrt(den)
    return jnp.nan_to_num(v_vals, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf)


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
        F = [(Cβ)' (C (XᵀX)⁻¹ Cᵀ)⁻¹ (Cβ) / m] / MSE.

    Notes
    -----
    df2 = n − rank(X).
    """
    n, p = Y.shape
    # fit GLM
    XtX_inv = pinv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    # residuals & mse
    resid = Y - X @ beta
    df2 = n - matrix_rank(X)
    mse = jnp.maximum(jnp.sum(resid**2, axis=0) / df2, jnp.finfo(resid.dtype).tiny)
    # contrast SS
    CB = C @ beta
    CXX = C @ XtX_inv @ C.T
    ss = jnp.maximum(jnp.sum(CB * (pinv(CXX) @ CB), axis=0), 0.0)
    # F‐statistic
    m = C.shape[0]
    F_vals = (ss / m) / mse
    return jnp.nan_to_num(F_vals, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf)


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
        Max number of groups.

    Returns
    -------
    G_vals : array, (p,)
        G = [(Cβ)' V⁻¹ (Cβ)/m] / [1 + 2*(m−1)*b].
    """
    n, p = Y.shape
    m, k = C.shape
    # fit GLM
    XtX_inv = pinv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    resid = Y - X @ beta
    # group leverages
    _, g = jnp.unique(groups, return_inverse=True, size=J_max, fill_value=-1)
    H_diag = jnp.diag(X @ XtX_inv @ X.T)
    d = jnp.maximum(segment_sum(1 - H_diag, g, num_segments=J_max), 1e-12)
    counts = jnp.bincount(g, length=J_max)
    # weights
    rss = jnp.maximum(segment_sum(resid**2, g, num_segments=J_max), 1e-12)
    Wint = d[:, None] / rss
    Wfinal = Wint * counts[:, None]
    # accumulate weighted X'X
    Mb = segment_sum(jnp.einsum("ij,ik->ijk", X, X), g, num_segments=J_max)
    cte = jnp.sum(Mb.reshape(J_max, -1, 1) * Wint[:, None, :], axis=0)
    cte_mats = cte.reshape(k, k, -1).transpose(2, 0, 1)

    # numerator SS
    def quad(b_col, A):
        cvec = C @ b_col
        return jnp.maximum(cvec.T @ pinv(C @ pinv(A) @ C.T) @ cvec, 0.0)

    num_ss = vmap(quad)(beta.T, cte_mats)
    # b‐sum
    sW = jnp.sum(Wfinal, axis=0)
    b = jnp.sum(
        (1 - Wfinal / sW) ** 2 * jnp.where(d > 1e-12, 1 / d, 0)[:, None], axis=0
    )
    b = b / (m * (m + 2))
    # G‐statistic
    G_vals = (num_ss / m) / jnp.maximum(1 + 2 * (m - 1) * b, 1e-12)
    return jnp.nan_to_num(G_vals, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf)


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
    C = jnp.atleast_1d(jnp.squeeze(C))
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
    return jnp.nan_to_num(r_vals, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf)


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
    C = jnp.atleast_1d(jnp.squeeze(C))
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
    ss_mod = jnp.einsum("mi,ij,mj->i", CB, V_inv, CB)
    # compute R²
    r2_vals = ss_mod / (ss_mod + df * mse)
    return jnp.nan_to_num(r2_vals, nan=0.0, posinf=1.0, neginf=0.0)
