import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as jss
from functools import partial
from sklearn.utils import Bunch


@jax.jit
def _compute_auc_from_probs(y_true, y_pred, eps=1e-9):
    """Compute AUC via rank-based (Mann–Whitney U) method."""
    ranks = jss.rankdata(y_pred)
    n_pos = jnp.sum(y_true)
    n_neg = y_pred.shape[0] - n_pos
    sum_pos = jnp.sum(ranks * y_true)
    return jnp.where(
        (n_pos > 0) & (n_neg > 0),
        (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg),
        0.5,
    )


@partial(jax.jit, static_argnames=("max_iter",))
def _jit_core_logistic_regression(Y, X, C, max_iter, tol, ridge, eps, calc_auc):
    """IRLS logistic regression. Returns z-stats, p-values, and AUCs."""
    # Initialize
    n_samples, n_features = X.shape
    beta = jnp.zeros(n_features, dtype=X.dtype)

    def cond_fn(state):
        _, it, diff, bad, _ = state
        return (it < max_iter) & (diff > tol) & ~bad

    def body_fn(state):
        b_old, it, _, _, _ = state
        mu = jsp.special.expit(X @ b_old)
        w = jnp.maximum(mu * (1 - mu), eps)
        I = (X.T * w) @ X + jnp.eye(n_features) * ridge
        grad = X.T @ (Y - mu)
        delta = jnp.linalg.solve(I, grad)
        b_new = b_old + delta
        bad = jnp.any(jnp.isnan(b_new)) | jnp.any(jnp.isinf(b_new))
        diff = jnp.sum(jnp.abs(delta))
        return (
            jnp.where(bad, b_old, b_new),
            it + 1,
            jnp.where(bad, 0.0, diff),
            bad,
            mu,
        )

    init_mu = jsp.special.expit(X @ beta)
    state = (beta, 0, tol + 1.0, False, init_mu)
    beta_final, _, _, bad_end, _ = jax.lax.while_loop(cond_fn, body_fn, state)

    mu_final = jsp.special.expit(X @ beta_final)
    # Covariance and stats
    w_final = jnp.maximum(mu_final * (1 - mu_final), eps)
    XtWX = (X.T * w_final) @ X + jnp.eye(n_features) * ridge
    cov_beta = jax.lax.cond(
        (jnp.all(jnp.linalg.eigvalsh(XtWX) > eps)) & ~bad_end,
        lambda M: jnp.linalg.inv(M),
        lambda M: jnp.full_like(M, jnp.nan),
        XtWX,
    )

    def stats_per_contrast(c):
        eff = c @ beta_final
        var = c @ cov_beta @ c
        se = jnp.where(var > eps, jnp.sqrt(var), jnp.nan)
        z = jnp.where(se > eps, eff / se, jnp.nan)
        p = jnp.where(jnp.isnan(z), jnp.nan, 2 * jss.norm.sf(jnp.abs(z)))
        valid = ~bad_end
        return jnp.where(valid, z, jnp.nan), jnp.where(valid, p, jnp.nan)

    z_stats, p_vals = jax.vmap(stats_per_contrast, in_axes=1, out_axes=0)(C)

    # AUC per contrast
    def auc_per_contrast(c):
        beta_mask = beta_final * (c != 0)
        mu_c = jsp.special.expit(X @ beta_mask)
        return _compute_auc_from_probs(Y, mu_c, eps)

    aucs = jnp.where(
        (~bad_end) & calc_auc,
        jax.vmap(auc_per_contrast, in_axes=1)(C),
        jnp.full(C.shape[1], jnp.nan),
    )

    return z_stats, p_vals, aucs


def mass_univariate_logistic_regression(
    Y, X, C, max_iter=100, tol=1e-7, ridge=1e-8, eps=1e-9, calculate_auc=True
):
    """
    Run IRLS logistic regression per outcome unit.

    Returns only z-statistics, p-values, and optional AUCs.
    """
    Y_j = jnp.asarray(Y, dtype=jnp.result_type(float))
    X_j = jnp.asarray(X, dtype=jnp.result_type(float))
    C_j = jnp.asarray(C, dtype=jnp.result_type(float))

    if Y_j.ndim != 2 or X_j.ndim != 2:
        raise ValueError("Y and X must be 2D arrays")
    if Y_j.shape[0] != X_j.shape[0]:
        raise ValueError("Sample size mismatch between Y and X")

    n_feat = X_j.shape[1]
    # Prepare contrast matrix
    if C_j.ndim == 1:
        C_mat = C_j[:, None]
    elif C_j.ndim == 2:
        C_mat = C_j.T if C_j.shape[1] == n_feat else C_j
    else:
        raise ValueError("C must be 1D or 2D")

    z_map, p_map, auc_map = jax.vmap(
        _jit_core_logistic_regression,
        in_axes=(1, None, None, None, None, None, None, None),
        out_axes=(0, 0, 0),
    )(Y_j, X_j, C_mat, max_iter, tol, ridge, eps, calculate_auc)

    n_contrasts = C_mat.shape[1]
    out = {}
    if n_contrasts == 1:
        out["zstat"] = z_map[:, 0]
        out["pval"] = p_map[:, 0]
        if calculate_auc:
            out["auc"] = auc_map[:, 0]
    else:
        for i in range(n_contrasts):
            suf = f"_c{i+1}"
            out[f"zstat{suf}"] = z_map[:, i]
            out[f"pval{suf}"] = p_map[:, i]
            if calculate_auc:
                out[f"auc{suf}"] = auc_map[:, i]
    return Bunch(**out)