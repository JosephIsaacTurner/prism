from jax.numpy.linalg import pinv, matrix_rank
from jax import jit, vmap
import jax.numpy as jnp
from jax.ops import segment_sum
from functools import partial
import warnings


@jit
def t(data, design_matrix, contrast_vector):
    """
    Computes t-statistics for a GLM based on provided data, design, and contrast.

    Assumes data is (n_samples, n_features), design_matrix is (n_samples, n_regressors),
    and contrast_vector is (n_regressors,). Calculates based on standard OLS
    using pseudo-inverse.
    NOTE: Residual degrees of freedom are calculated as n_samples - n_regressors,
    which assumes the design matrix has full column rank.

    Args:
        data (array): Data array (n_samples, n_features).
        design_matrix (array): Design matrix (n_samples, n_regressors).
        contrast_vector (array): Contrast vector (n_regressors,).

    Returns:
        array: Resulting t-values (n_features,).
    """
    n_samples, n_features = data.shape
    n_regressors = design_matrix.shape[1]

    # Ensure contrast_vector is 1D (n_regressors,)
    contrast_vector = jnp.squeeze(contrast_vector)
    # Handle case where contrast_vector might become scalar after squeeze if n_regressors=1
    if contrast_vector.ndim == 0 and n_regressors == 1:
        contrast_vector = contrast_vector[jnp.newaxis]
    # Basic check - more robust error handling could be added if needed
    if contrast_vector.ndim != 1 or contrast_vector.shape[0] != n_regressors:
        # Or raise ValueError("Contrast vector shape mismatch")
        warnings.warn(f"Contrast vector shape {contrast_vector.shape} might be incompatible with design matrix columns {n_regressors}.")
        # Attempt to proceed, or return NaNs immediately
        # return jnp.full(n_features, jnp.nan)


    # --- GLM Parameter Estimation ---
    XTX = design_matrix.T @ design_matrix
    XTX_inv_pseudo = pinv(XTX)  # Shape (n_regressors, n_regressors)
    # betas = (X'X)^-1 @ X' @ Y -> (r, n_features)
    betas = XTX_inv_pseudo @ design_matrix.T @ data

    # --- Residuals ---
    # predictions = X @ betas -> (n_samples, n_features)
    model_predictions = design_matrix @ betas
    residuals = data - model_predictions # Shape (n_samples, n_features)

    # --- Degrees of Freedom (using n_regressors, assumes full rank) ---
    df_residual = n_samples - n_regressors
    if df_residual <= 0:
        # If degrees of freedom are non-positive, variance is undefined.
        # Return NaNs as t-statistic cannot be meaningfully computed.
        warnings.warn(f"Residual degrees of freedom ({df_residual}) is non-positive. Returning NaNs.", stacklevel=2)
        return jnp.full(n_features, jnp.nan)

    # --- Mean Squared Error ---
    sum_sq_residuals = jnp.sum(residuals**2, axis=0) # Shape (n_features,)
    mse = sum_sq_residuals / df_residual
    # Clamp MSE to be non-negative (and ideally slightly above zero for stability)
    mse = jnp.maximum(mse, jnp.finfo(mse.dtype).tiny) # Shape (n_features,)

    # --- Contrast Effect and Variance ---
    contrast_values = contrast_vector @ betas # Shape (n_features,)

    # var_contrast = c @ (X'X)^-1 @ c' -> should be scalar
    var_contrast = contrast_vector @ XTX_inv_pseudo @ contrast_vector
    # Clamp variance to be non-negative
    var_contrast = jnp.maximum(var_contrast, 0.0) # Scalar

    # --- Standard Error of Contrast ---
    # se = sqrt(var_contrast * mse)
    se_contrast = jnp.sqrt(var_contrast * mse) # Shape (n_features,)
    # Note: se_contrast can be zero if var_contrast is 0 or mse is ~0.

    # --- T-Statistic Calculation with Division-by-Zero Handling ---
    # Standard division - may produce NaN (0/0) or Inf (k/0)
    t_values_raw = contrast_values / se_contrast

    # Handle NaN/Inf:
    # Where SE is zero (or tiny):
    #   If contrast_value is also zero -> t = 0 (interpret 0/0 as 0)
    #   If contrast_value is non-zero -> t = +/- Inf (interpret k/0 as Inf)
    # jnp.nan_to_num maps NaN -> 0.0, posinf -> large_float, neginf -> -large_float
    # We want to preserve Inf, so nan_to_num is suitable.
    t_values = jnp.nan_to_num(t_values_raw, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf)

    # The original function had .flatten(), but if data is (n_samples, n_features),
    # the result t_values is already (n_features,), so flatten is redundant.
    return t_values


@partial(jit, static_argnums=(4,))
def aspin_welch_v(data, design_matrix, contrast_vector, group_ids, max_n_groups):
    """
    Computes the Aspin-Welch v statistic and degrees of freedom (df2), including using
    the diagonal of (I - H) (where H is the hat matrix) for group degrees-of-freedom.

    Parameters:
      data            : array of shape (n_samples, nT), response variable(s).
      design_matrix   : array of shape (n_samples, r), design matrix.
                        Assumed constant across tests (nT).
      contrast_vector : 1D array of length r, the contrast vector (rank-1).
      group_ids       : 1D integer array of length n_samples; specifies variance group membership.
      max_n_groups    : Static integer, the maximum possible number of unique groups.

    Returns:
      G   : array of shape (nT,), the Aspin-Welch v statistic.
      df2 : array of shape (nT,), the adjusted degrees of freedom (second df, with df1 = 1).
    """
    n_samples, nT = data.shape
    n_regressors = design_matrix.shape[1]

    # Ensure contrast_vector is 1D (n_regressors,)
    contrast_vector = jnp.squeeze(contrast_vector)

    # --- GLM Fit & Residuals ---
    M = design_matrix
    MtM = M.T @ M
    MtM_inv = pinv(MtM)  # (r, r)
    betas = MtM_inv @ M.T @ data   # (r, nT)
    predictions = M @ betas        # (n_samples, nT)
    residuals = data - predictions # (n_samples, nT)

    # --- dRmb Calculation via the Hat Matrix ---
    # Use size argument for jnp.unique
    # Note: unique_groups is not explicitly needed later, only group_idx
    _, group_idx = jnp.unique(group_ids, return_inverse=True, size=max_n_groups)

    # Use max_n_groups for segment operations
    group_counts = jnp.bincount(group_idx, length=max_n_groups)  # (max_n_groups,)

    # Compute hat matrix H and residual-forming matrix R = I - H.
    H = M @ MtM_inv @ M.T  # (n_samples, n_samples)
    diag_R = 1.0 - jnp.diag(H)  # (n_samples,)
    diag_R = jnp.maximum(diag_R, 1e-12)  # Clamp to avoid numerical issues
    # dRmb per group = sum(diag_R) for observations in that group.
    dRmb = segment_sum(diag_R, group_idx, num_segments=max_n_groups)  # (max_n_groups,)
    dRmb = jnp.maximum(dRmb, 1e-12)

    # --- Group-level Residual Sum-of-Squares and Intermediate Weight ---
    group_res_sum = segment_sum(residuals**2, group_idx, num_segments=max_n_groups)  # (max_n_groups, nT)
    group_res_sum = jnp.maximum(group_res_sum, 1e-12)
    # Intermediate weight: W_int = dRmb / (sum of squared residuals for the group).
    # This division should be safe due to the jnp.maximum calls above.
    W_int = dRmb[:, None] / group_res_sum  # (max_n_groups, nT)

    # --- Accumulate Design Matrix Contributions (cte) ---
    # Compute outer product for each observation.
    outer_products = jnp.einsum('ij,ik->ijk', M, M)  # (n_samples, r, r)
    # Sum these outer products within each group.
    group_Mb = segment_sum(outer_products, group_idx, num_segments=max_n_groups)  # (max_n_groups, r, r)
    # Flatten each group's Mb to a vector.
    group_Mb_flat = group_Mb.reshape(max_n_groups, -1)  # (max_n_groups, r*r)
    # Accumulate cte using the intermediate weight (W_int).
    cte = jnp.sum(group_Mb_flat[:, :, None] * W_int[:, None, :], axis=0)  # (r*r, nT)

    # --- Final Weight for df2 Calculation ---
    # MATLAB updates W by multiplying by the group count.
    W_final = W_int * group_counts[:, None]  # (max_n_groups, nT)

    # --- Denominator Calculation via vmap ---
    cte_reshaped = jnp.moveaxis(cte.reshape((n_regressors, n_regressors, -1)), -1, 0)  # (nT, r, r)
    def compute_den_matrix(A):
        return contrast_vector @ (pinv(A) @ contrast_vector)
    den = vmap(compute_den_matrix)(cte_reshaped)  # (nT,)
    den = jnp.maximum(den, 1e-12)  # Ensure non-negative

    # --- Compute the Aspin-Welch v Statistic (G) ---
    contrast_est = contrast_vector @ betas  # (nT,)
    G = contrast_est / jnp.sqrt(den)

    # --- Adjusted Degrees of Freedom (df2) ---
    # Sum over the max_n_groups dimension; padded groups contribute 0.
    sW1 = jnp.sum(W_final, axis=0)  # (nT,)
    sW1 = jnp.maximum(sW1, 1e-12)
    # bsum = sum_b [(1 - (W_final[b,t] / sW1[t]))^2 / dRmb[b]]
    # Need to handle potential division by zero if dRmb is zero for some groups
    # The jnp.maximum(dRmb, 1e-12) should prevent division by strict zero.
    # However, if a group is unused (padded), dRmb might be zero (or near zero).
    # W_final for that group will also be zero, leading to (1 - 0/sW1)^2 / dRmb = 1 / dRmb.
    # We should ensure we only sum over valid groups or that the division is safe.
    # A safe way is to divide only where dRmb is non-negligible.
    safe_inv_dRmb = jnp.where(dRmb > 1e-12, 1.0 / dRmb, 0.0)
    bsum = jnp.sum(((1 - (W_final / sW1))**2) * safe_inv_dRmb[:, None], axis=0) # (nT,)

    bsum = jnp.maximum(bsum, 1e-12) # Avoid division by zero in the final df2 calculation
    # Incorporate the 1/3 factor as in MATLAB.
    df2 = (1/3) / bsum

    return G


@jit
def F(data, design_matrix, contrast_matrix):
    """
    Computes F-statistics for a GLM based on provided data, design, and contrast matrices.

    This function assumes a standard OLS GLM framework and calculates F-statistics
    consistent with the methodology used in the reference t-test function
    (beta estimation via pseudo-inverse, standard MSE calculation, and
    OLS residual degrees of freedom). Assumes data is (n_samples, n_features/voxels).

    Args:
        data (array): The data array (n_samples, n_features).
        design_matrix (array): The design matrix (n_samples, n_regressors).
        contrast_matrix (array): The contrast matrix (n_contrasts, n_regressors).

    Returns:
        array: The resulting F-values (n_features,).
    """
    n_samples = data.shape[0]
    C = contrast_matrix # Shape (n_contrasts, n_regressors)

    # --- GLM Parameter Estimation (Harmonized with t-function) ---
    XTX = design_matrix.T @ design_matrix
    XTX_inv_pseudo = pinv(XTX)  # Shape (n_regressors, n_regressors)
    # betas = (X'X)^-1 @ X' @ Y
    # -> (r, r) @ (r, n_samples) @ (n_samples, n_features) -> (r, n_features)
    betas = XTX_inv_pseudo @ design_matrix.T @ data # Shape (n_regressors, n_features)

    # --- Residuals and Variance Estimation (Harmonized with t-function) ---
    # predictions = X @ betas
    # -> (n_samples, r) @ (r, n_features) -> (n_samples, n_features)
    model_predictions = design_matrix @ betas
    residuals = data - model_predictions # Shape (n_samples, n_features)

    # Residual degrees of freedom (using matrix_rank for robustness, consistent with F-test logic)
    df_residual = n_samples - matrix_rank(design_matrix)

    # Mean Squared Error (sum over samples dimension: axis=0)
    # mse shape (n_features,)
    sum_sq_residuals = jnp.sum(residuals**2, axis=0)
    mse = sum_sq_residuals / df_residual
    # Prevent division by zero or negative MSE in edge cases
    mse = jnp.maximum(mse, jnp.finfo(mse.dtype).tiny)

    # --- F-statistic Calculation ---
    # Numerator degrees of freedom
    df_contrast = matrix_rank(C) # Shape ()

    # Contrast effect: C @ beta
    # -> (n_contrasts, r) @ (r, n_features) -> (n_contrasts, n_features)
    CB = C @ betas

    # Variance term: C @ (X'X)^-1 @ C'
    # -> (n_contrasts, r) @ (r, r) @ (r, n_contrasts) -> (n_contrasts, n_contrasts)
    CXX = C @ XTX_inv_pseudo @ C.T
    # Use pinv for the inverse in case CXX is rank deficient
    CXX_inv = pinv(CXX) # Shape (n_contrasts, n_contrasts)

    # Calculate the quadratic form numerator of the F-statistic for each feature
    # Numerator Sum of Squares = (CB)' @ CXX_inv @ CB element-wise for each feature
    # term1 = CXX_inv @ CB -> (n_contrasts, n_contrasts) @ (n_contrasts, n_features) -> (n_contrasts, n_features)
    term1 = CXX_inv @ CB
    # Sum Sq Contrast = sum(CB * term1, axis=0) -> sum over contrasts -> (n_features,)
    numerator_ss = jnp.sum(CB * term1, axis=0)
    # Clamp numerator SS to be non-negative
    numerator_ss = jnp.maximum(numerator_ss, 0.0)

    # F = (NumeratorSS / df_contrast) / MSE
    F_values = (numerator_ss / df_contrast) / mse

    # Handle potential NaN/Inf resulting from division (e.g., if mse was zero)
    F_values = jnp.nan_to_num(F_values, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf) # Or nan=jnp.nan if preferred

    return F_values


@partial(jit, static_argnums=(4,))  # max_n_groups is the 5th argument (index 4)
def G(data, design_matrix, contrast_matrix, group_ids, max_n_groups):
    """
    Computes a G-statistic (generalization of F-test for heteroscedasticity
    using Aspin-Welch approach).

    Args:
        data (array): Data array (n_samples, n_features).
        design_matrix (array): Design matrix (n_samples, n_regressors).
        contrast_matrix (array): Contrast matrix (n_contrasts, n_regressors).
        group_ids (array): 1D integer array (n_samples,) specifying variance group membership.
        max_n_groups (int): Static integer, the maximum possible number of unique groups.

    Returns:
        G_values (array): Resulting G-statistic values (n_features,).
        df_den (array): Adjusted denominator degrees of freedom (n_features,).
    """
    n_samples, n_features = data.shape
    n_regressors = design_matrix.shape[1]
    C = contrast_matrix  # Shape (n_contrasts, n_regressors)

    # --- GLM Parameter Estimation (Standard) ---
    M = design_matrix
    MtM = M.T @ M
    MtM_inv = pinv(MtM)
    betas = MtM_inv @ M.T @ data  # (n_regressors, n_features)

    # --- Residuals (Standard) ---
    predictions = M @ betas
    residuals = data - predictions

    # --- Aspin-Welch Variance Components ---
    # Get group indices and counts
    unique_groups, group_idx = jnp.unique(group_ids, return_inverse=True, size=max_n_groups, fill_value=-1)
    # Ensure we only use valid group indices returned by unique
    num_actual_groups = jnp.sum(unique_groups != -1)
    # Use num_actual_groups if segment_sum needs precise segment count, or ensure max_n_groups is sufficient
    group_counts = jnp.bincount(group_idx, length=max_n_groups)  # (max_n_groups,)

    # Calculate effective group degrees-of-freedom using the hat matrix
    H = M @ MtM_inv @ M.T
    diag_R = 1.0 - jnp.diag(H)
    diag_R = jnp.maximum(diag_R, 1e-12) # Avoid division by zero or negative values
    # Sum diag_R within each valid group
    dRmb = segment_sum(diag_R, group_idx, num_segments=max_n_groups)
    # Ensure dRmb is non-zero for groups that exist, might remain 0 for unused segments
    dRmb = jnp.maximum(dRmb, 1e-12) # Avoid division by zero later

    # Compute group-level residual sum of squares
    group_res_sum = segment_sum(residuals**2, group_idx, num_segments=max_n_groups)
    # Add safeguard against division by zero or small numbers
    group_res_sum = jnp.maximum(group_res_sum, 1e-12) # Avoid division by zero

    # --- Adjusted Group Weights (Split logic based on MATLAB) ---
    # Initial weights used for cte calculation (MATLAB: W before update)
    W_initial = dRmb[:, None] / group_res_sum  # Shape: (max_n_groups, n_features)

    # Final weights used for df_den calculation (MATLAB: W after multiplication by count)
    W_final = W_initial * group_counts[:, None] # Shape: (max_n_groups, n_features)

    # --- Construct cte (Weighted X'X based on group variances) ---
    # Compute outer products for every sample: (n_samples, r, r)
    outer_products = jnp.einsum('ij,ik->ijk', M, M)
    # Sum outer products within each group: (max_n_groups, r, r)
    group_Mb = segment_sum(outer_products, group_idx, num_segments=max_n_groups)
    # Flatten group_Mb to shape (max_n_groups, r*r)
    group_Mb_flat = group_Mb.reshape(max_n_groups, -1)

    # Accumulate cte over groups using the *initial* weights W_initial
    # Shape W_initial for broadcasting: (max_n_groups, 1, n_features)
    # Shape group_Mb_flat:             (max_n_groups, r*r) -> (max_n_groups, r*r, 1)
    cte = jnp.sum(group_Mb_flat[:, :, None] * W_initial[:, None, :], axis=0)  # (r*r, n_features)
    cte_reshaped = jnp.moveaxis(cte.reshape((n_regressors, n_regressors, -1)), -1, 0)  # (n_features, r, r)

    # --- Numerator Degrees of Freedom ---
    df_num = matrix_rank(C)

    # --- G-Statistic Calculation (Quadratic form with scaling) ---
    # Inner function to compute quadratic form for one feature's beta column and cte matrix
    def compute_quadratic_form(beta_col_r, cte_matrix_rr):
        cb_vec = C @ beta_col_r  # Contrast effect vector: (n_contrasts,)
        # Compute variance structure and invert twice (mimicking MATLAB mrdiv behavior)
        # Note: Ensure cte_matrix_rr is invertible or handle appropriately
        try:
            pinv_cte = pinv(cte_matrix_rr)
            variance_matrix_inv = C @ pinv_cte @ C.T
            # Add regularization if variance_matrix_inv might be singular
            variance_matrix_inv = variance_matrix_inv + jnp.eye(variance_matrix_inv.shape[0]) * 1e-10
            variance_matrix_inv_inv = pinv(variance_matrix_inv)
            num_ss = cb_vec.T @ variance_matrix_inv_inv @ cb_vec
        except jnp.linalg.LinAlgError:
             # Handle cases where inversion fails, e.g., return 0 or NaN
             num_ss = 0.0 # Or jnp.nan if preferred
        return num_ss

    # Vectorize the quadratic form computation over all features
    numerator_ss = vmap(compute_quadratic_form)(betas.T, cte_reshaped) # Pass betas as (n_features, r)
    numerator_ss = jnp.maximum(numerator_ss, 0.0) # Ensure non-negative result

    # --- Adjusted Denominator Degrees of Freedom ---
    # Compute sW1 as the total *final* weight per feature
    sW1 = jnp.sum(W_final, axis=0) # Use W_final here
    sW1 = jnp.maximum(sW1, 1e-12) # Avoid division by zero

    # Calculate Welch-Satterthwaite intermediate term 'bsum' using W_final
    # Use W_final / sW1 for the relative weights
    term_in_sum = (1.0 - (W_final / sW1))**2  # element-wise difference squared using W_final
    # dRmb already has safeguard, use safe inverse
    safe_inv_dRmb = jnp.where(dRmb > 1e-12, 1.0 / dRmb, 0.0)  # (max_n_groups,)
    # Note: reshape safe_inv_dRmb for broadcasting over features
    # Sum across groups (axis=0)
    bsum = jnp.sum(term_in_sum * safe_inv_dRmb[:, None], axis=0) # Uses W_final via term_in_sum
    bsum = jnp.maximum(bsum, 1e-12) # Safeguard bsum

    # MATLAB scales bsum by dividing by plm.rC0 (contrast df) and (contrast df + 2)
    # Here, we assume plm.rC0 is equivalent to df_num.
    bsum_corrected = bsum / (df_num * (df_num + 2))

    # Calculate final denominator df
    df_den = (1.0 / 3.0) / bsum_corrected
    df_den = jnp.nan_to_num(df_den, nan=jnp.nan, posinf=jnp.inf, neginf=-jnp.inf) # Handle potential division by zero in correction

    # --- Final G-statistic with extra scaling ---
    # MATLAB divides the quadratic form by plm.rC0 then applies:
    #       G = G / (1 + 2*(plm.rC0 - 1)*bsum_corrected)
    # Here, we mimic that using df_num as plm.rC0.
    # Ensure df_num is float for calculations
    df_num_float = df_num.astype(jnp.float32)
    G_numerator_scaled = numerator_ss / df_num_float # Scale numerator SS by df
    # Ensure denominator is non-zero and calculation is safe
    g_denom_scaling = 1.0 + 2.0 * (df_num_float - 1.0) * bsum_corrected
    g_denom_scaling = jnp.maximum(g_denom_scaling, 1e-12) # Avoid division by zero

    G_values = G_numerator_scaled / g_denom_scaling
    G_values = jnp.nan_to_num(G_values, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf) # Handle potential NaNs

    return G_values