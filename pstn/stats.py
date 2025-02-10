from jax.numpy.linalg import pinv
from jax import jit
import jax.numpy as jnp

@jit
def welchs_t_glm(data, design_matrix, contrast_matrix):
    # Solve the GLM
    XTX = design_matrix.T @ design_matrix
    XTX_inv_pseudo = pinv(XTX)  # Pseudo-inverse
    betas = XTX_inv_pseudo @ design_matrix.T @ data  # Beta coefficients

    # Calculate residuals
    model_predictions = design_matrix @ betas
    residuals = data - model_predictions

    # Contrast values
    contrast_values = contrast_matrix @ betas

    # Variance of the contrast estimate
    var_contrast = contrast_matrix @ XTX_inv_pseudo @ contrast_matrix.T

    # Degrees of freedom using Welch–Satterthwaite equation
    mse = jnp.sum(residuals**2, axis=0) / (data.shape[0] - design_matrix.shape[1])
    se_contrast = jnp.sqrt(var_contrast * mse)
    
    # Calculate t-values
    t_values = contrast_values / se_contrast

    return t_values.flatten()


@jit
def fdr_bh_correction(p_values):
    """
    Perform FDR correction using the Benjamini-Hochberg procedure with JAX.
    
    Parameters:
    p_values (array-like): Array of p-values to be corrected.
    
    Returns:
    jax.numpy.array: FDR-corrected p-values.
    """
    # Convert input to JAX array
    p_values = jnp.array(p_values)
    
    # Number of tests
    m = p_values.size
    
    # Sort the p-values and keep track of the original indices
    sorted_indices = jnp.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    # Calculate the FDR adjusted p-values
    rank = jnp.arange(1, m + 1)
    adjusted_p_values = sorted_p_values * m / rank
    
    # Ensure the adjusted p-values are less than or equal to 1
    adjusted_p_values = jnp.minimum(adjusted_p_values, 1)
    
    # Ensure the adjusted p-values are non-decreasing
    min_ufunc = jnp.frompyfunc(jnp.minimum, nin=2, nout=1)
    adjusted_p_values_monotonic = min_ufunc.accumulate(adjusted_p_values[::-1])[::-1]
    
    # Place the adjusted p-values back in the original order
    original_order_indices = jnp.argsort(sorted_indices)
    adjusted_p_values = adjusted_p_values_monotonic[original_order_indices]
    
    return adjusted_p_values
