import numpy as np
from scipy.stats import genpareto, goodness_of_fit
from tqdm import tqdm
from statsmodels.stats.multitest import fdrcorrection
from .loading import load_nifti_if_not_already_nifti, is_nifti_like, Dataset, load_data, prepare_glm_data
from .stats import t, aspin_welch_v, F, G, zscore
from .tfce import apply_tfce
from nilearn.maskers import NiftiMasker
import nibabel as nib
from jax import jit, random
import warnings
from typing import List, Optional, Union, Callable, Generator, Any, Tuple
from sklearn.utils import Bunch



def permutation_analysis(data, design, contrast, stat_function='auto', n_permutations=1000, random_state=42, two_tailed=True, exchangeability_matrix=None, vg_auto=False, vg_vector=None, within=True, whole=False, flip_signs=False, accel_tail=False, demean=False, f_stat_function='auto', f_contrast_indices=None, f_only=False, correct_across_contrasts=False, on_permute_callback=None, permute=True, save_fn=None, save_1minusp=False, save_neglog10p=False, zstat=False):
    """
    Performs permutation testing on the provided data using a specified statistical function.

    The function calculates the observed (true) test statistics, then generates a null distribution
    by permuting the design matrix (optionally respecting an exchangeability structure). It computes:
      - Empirical (uncorrected) p-values,
      - FDR-corrected p-values via the Benjamini-Hochberg procedure,
      - FWE-corrected p-values using a max-statistic approach (with an option for accelerated tail estimation via a GPD fit).

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_elements_per_sample)
        The data matrix, where each row represents a sample and each column an element/feature.
    design : np.ndarray, shape (n_samples, n_features)
        The design matrix for the GLM, where each row corresponds to a sample and each column to a regressor.
    contrast : np.ndarray, shape (n_features,) or (n_contrasts, n_features)
        The contrast vector or matrix specifying the hypothesis to be tested.
    stat_function : function or 'auto'
        A function that calculates the test statistic (e.g., t-statistic). It must accept the arguments
        (data, design, contrast, [vg_vector, n_groups] if variance groups used)
        and return an array of test statistics. 'auto' selects based on variance groups.
    n_permutations : int
        The number of permutations to perform.
    random_state : int
        Seed for the random number generator to ensure reproducibility.
    two_tailed : bool, default True
        If True, p-values are computed in a two-tailed manner using absolute values of statistics.
        Applies only to symmetric statistics like t-statistics, ignored for F-statistics.
    exchangeability_matrix : np.ndarray, optional
        Defines the exchangeability blocks for permutation testing. Expected shapes:
          - If provided as a vector: (n_samples,) or (n_samples, 1).
          - If provided as a matrix: (n_samples, n_permutation_groups).
    vg_auto : bool, default False
        If True, automatically generates a variance group (VG) vector based on the exchangeability matrix.
        Requires `exchangeability_matrix` to be provided.
    vg_vector : np.ndarray, optional
        A 1D array defining the variance groups for each observation. Overrides `vg_auto`.
    within : bool, default True
        For a 1D exchangeability matrix, indicates whether to permute within blocks.
    whole : bool, default False
        For a 1D exchangeability matrix, indicates whether to permute whole blocks.
    flip_signs : bool, default False
        If True, randomly flips the signs of the data residuals for each permutation (assumes independent and symmetric errors (ISE)).
    accel_tail : bool, default True
        If True, applies the accelerated tail method (GPD approximation) to compute FWE p-values
        for cases with low empirical exceedance counts.
    demean : bool, default True
        If True, the data is demeaned before applying the statistical function.
    f_stat_function : function or 'auto'
        A function that calculates the F-like statistic. Must accept arguments similar to `stat_function`.
        'auto' selects based on variance groups.
    f_contrast_indices : np.ndarray or list, optional
        Indices (0-based) or boolean mask specifying which rows of the `contrast` matrix
        to include in the F-test. If None, no F-test is performed unless `f_only` is True
        and `contrast` itself is suitable for an F-test.
    f_only : bool, default False
        If True, only performs the F-test specified by `f_contrast_indices` (or using the full
        `contrast` matrix if `f_contrast_indices` is None but the contrast is suitable).
        Skips individual contrast tests.
    on_permute_callback: function (optional)
       If provided, calls this function on the permuted stats for each permutation.
       Signature: callback(permuted_stats, permutation_index, is_two_tailed)
    permute : bool, default True
        If True, performs permutation testing. If False, only computes the observed statistics.
    correct_across_contrasts : bool, default False
        If True, applies FWE corrections across all contrasts.
    save_fn : callable, optional
        Expects a callable that is called on "results" each time a key is added to the results dict/Bunch.
    save_1minusp : bool, default True
        If True, stores the 1 - p-values instead of the raw p-values.
    save_neglog10p : bool, default False
        If True, stores the -log10(p-values) instead of the raw p-values.

    Returns
    -------
    results : sklearn.utils.Bunch
        A Bunch object containing the calculated p-values.
        - For each individual contrast `i` (if `f_only` is False):
            - `c{i+1}_unc_p`: Uncorrected p-values.
            - `c{i+1}_fdr_p`: FDR-corrected p-values.
            - `c{i+1}_fwe_p`: FWE-corrected p-values.
            - `c{i+1}_true_stat`: The calculated true statistic values.
        - If an F-test is performed:
            - `f_unc_p`: Uncorrected p-values for the F-test.
            - `f_fdr_p`: FDR-corrected p-values for the F-test.
            - `f_fwe_p`: FWE-corrected p-values for the F-test.
            - `f_true_stat`: The calculated true F-statistic values.

    Notes
    -----
    - The first dimension of both `data` and `design` is assumed to correspond to the samples.
    - FWE correction uses the max-statistic approach (distribution of the maximum statistic across elements).
    - F-tests are inherently one-tailed (upper tail). `two_tailed` parameter is ignored for F-tests.
    - If using variance groups (`vg_auto` or `vg_vector`), ensure the corresponding stat functions (`aspin_welch_v`, `G`) are used or selected via 'auto'.
    """
    # Step Zero: Check inputs and setup
    if n_permutations <= 0:
        raise ValueError("Number of permutations must be positive")
    if data.shape[0] != design.shape[0]:
        raise ValueError(f"Data ({data.shape[0]}) and design ({design.shape[0]}) must have the same number of samples")
    if contrast.ndim == 1:
        if contrast.shape[0] != design.shape[1]:
            raise ValueError("1D contrast dimensions must match number of regressors in design matrix")
    elif contrast.ndim == 2:
        if contrast.shape[1] != design.shape[1]:
            raise ValueError("2D contrast dimensions must match number of regressors in design matrix")
    else: # contrast.ndim > 2
        raise ValueError(f"Contrast must be 1D or 2D. Got {contrast.ndim}D. (Shape: {contrast.shape})")
    if exchangeability_matrix is not None and exchangeability_matrix.shape[0] != data.shape[0]:
        raise ValueError("Exchangeability matrix length must match number of samples")
    if vg_auto and exchangeability_matrix is None:
        raise ValueError("exchangeability_matrix must be provided if vg_auto is True")
    if f_contrast_indices is not None:
        f_contrast_indices = np.ravel(f_contrast_indices).astype(bool).astype(int)
    if f_only and f_contrast_indices is None and contrast.ndim == 1:
         warnings.warn("f_only is True, but f_contrast_indices is None and only one base contrast is provided. Performing F-test on this single contrast.")
         # Treat the single contrast as the one to test with F
         f_contrast_indices = np.array([0])
    elif f_only and f_contrast_indices is None and contrast.ndim == 2 and contrast.shape[0] == 1:
         warnings.warn("f_only is True, but f_contrast_indices is None and only one base contrast is provided. Performing F-test on this single contrast.")
         f_contrast_indices = np.array([0])
    elif f_only and f_contrast_indices is None and contrast.ndim == 2 and contrast.shape[0] > 1:
         warnings.warn("f_only is True, but f_contrast_indices is None. Performing F-test on *all* provided contrasts.")
         # Use all contrasts for the F-test
         f_contrast_indices = np.arange(contrast.shape[0])
    elif f_only and f_contrast_indices is not None:
         pass # User specified indices for F-test
    elif not f_only and f_contrast_indices is None and contrast.ndim == 2 and contrast.shape[0] > 1:
         warnings.warn("Multiple contrasts provided, but f_contrast_indices is None. No F-test will be performed.")
         # No F-test by default if multiple contrasts and no indices

    if demean:
        data, design, contrast, f_contrast_indices = prepare_glm_data(data, design, contrast, f_contrast_indices)

    # Ensure contrast is 2D
    original_contrast = np.atleast_2d(contrast)
    n_contrasts = original_contrast.shape[0]
    n_elements = data.shape[1] # Number of voxels/features etc.


    # Prepare f_contrast if F-test is needed
    perform_f_test = False
    f_contrast = None
    if f_contrast_indices is not None or f_only:
        perform_f_test = True
        if f_contrast_indices is not None:
             f_contrast_indices = np.ravel(np.asarray(f_contrast_indices)).astype(bool)
             if f_contrast_indices.ndim > 1:
                 raise ValueError("f_contrast_indices must be 1D array or list of indices/booleans.")
             if f_contrast_indices.shape[0] > n_contrasts:
                f_contrast_indices = f_contrast_indices[:n_contrasts]
             if f_contrast_indices.dtype == bool:
                 if len(f_contrast_indices) != n_contrasts:
                      raise ValueError(f"Boolean f_contrast_indices length ({len(f_contrast_indices)}) must match number of contrasts ({n_contrasts})")
                 f_contrast = original_contrast[f_contrast_indices, :]
             else: # Integer indices
                 if np.max(f_contrast_indices) >= n_contrasts or np.min(f_contrast_indices) < 0:
                     raise ValueError(f"f_contrast_indices values out of bounds for {n_contrasts} contrasts.")
                 f_contrast = original_contrast[f_contrast_indices, :]
        elif f_only: # Indices were None, f_only is True -> use all original contrasts
            f_contrast = original_contrast

        if f_contrast is None or f_contrast.shape[0] == 0:
            raise ValueError("Cannot perform F-test: f_contrast_indices resulted in an empty set of contrasts.")
        if f_contrast.shape[0] == 1 and not f_only:
             warnings.warn("F-test requested for a single contrast vector. This is equivalent to a squared t-test (or similar for other stats).")

    # Determine variance groups if needed
    use_variance_groups = (exchangeability_matrix is not None and vg_auto) or (vg_vector is not None)
    calculated_vg_vector = None
    n_groups = None
    if use_variance_groups:
        if vg_vector is not None:
            calculated_vg_vector = np.asarray(vg_vector)
            if calculated_vg_vector.shape[0] != data.shape[0]:
                 raise ValueError("Provided vg_vector length must match number of samples")
        else: # vg_auto is True and exchangeability_matrix is not None
            calculated_vg_vector = get_vg_vector(exchangeability_matrix, within=within, whole=whole)
        n_groups = len(np.unique(calculated_vg_vector))
        if n_groups <= 1:
            warnings.warn("Variance groups were requested or auto-detected, but only one group was found. Standard statistics will be used.")
            use_variance_groups = False # Revert to standard stats
            calculated_vg_vector = None
            n_groups = None

    # Determine which stat functions to use
    actual_stat_function = None
    if stat_function == 'auto':
        actual_stat_function = aspin_welch_v if use_variance_groups else t
    elif callable(stat_function):
        actual_stat_function = stat_function
    else:
        raise ValueError("stat_function must be 'auto' or a callable function")

    actual_f_stat_function = None
    if perform_f_test:
        if f_stat_function == 'auto':
            actual_f_stat_function = G if use_variance_groups else F
        elif callable(f_stat_function):
            actual_f_stat_function = f_stat_function
        else:
            raise ValueError("f_stat_function must be 'auto' or a callable function")

    if save_fn is not None:
        def save_fn_wrapper(results):
            try:
                save_fn(results)
            except Exception as e:
                print(f"Error saving results: {e}")
    else:
        save_fn_wrapper = None

    # Initialize results
    results = Bunch()

    # Do ground truth t tests
    if not f_only:
        for i in range(n_contrasts):
            contrast_label = f"c{i+1}"
            current_contrast = np.ravel(original_contrast[i:i+1])
            if use_variance_groups:
                true_stats = np.ravel(actual_stat_function(data, design, current_contrast, calculated_vg_vector, n_groups))
            else:
                true_stats = np.ravel(actual_stat_function(data, design, current_contrast))

            if true_stats.ndim == 0: # Handle case where stat_function returns scalar (e.g., 1 element)
                true_stats = true_stats.reshape(1,)
            if true_stats.shape[0] != n_elements:
                raise RuntimeError(f"Stat function returned unexpected shape {true_stats.shape} for contrast {i+1}, expected ({n_elements},)")

            results[f"stat_{contrast_label}"] = true_stats
            if save_fn is not None:
                save_fn_wrapper(results)

    # Do ground truth F tests
    if perform_f_test:
        if use_variance_groups:
            true_stats_f = np.ravel(actual_f_stat_function(data, design, f_contrast, calculated_vg_vector, n_groups))
        else:
            true_stats_f = np.ravel(actual_f_stat_function(data, design, f_contrast))

        if true_stats_f.ndim == 0: # Handle case where stat_function returns scalar (e.g., 1 element)
            true_stats_f = true_stats_f.reshape(1,)
        if true_stats_f.shape[0] != n_elements:
            raise RuntimeError(f"F Stat function returned unexpected shape {true_stats_f.shape}, expected ({n_elements},)")
            
        results["stat_f"] = true_stats_f
        if save_fn is not None:
            save_fn_wrapper(results)

    # --- Permutation Testing for T Test ---
    if not f_only and permute:
        # Iterate over each contrast
        for i in range(n_contrasts):
            current_contrast = np.ravel(original_contrast[i:i+1])
            contrast_label = f"c{i+1}"
            print(f"--- Processing Contrast {i+1}/{n_contrasts} ---")

            true_stats = results[f"stat_{contrast_label}"]
            exceedances = np.zeros_like(true_stats, dtype=float)
            max_stat_dist = np.zeros(n_permutations) # Store max statistic from each permutation

            permutation_generator = yield_permuted_stats(
                data, design, current_contrast, # Pass the current contrast
                stat_function=actual_stat_function,
                n_permutations=n_permutations,
                random_state=random_state + i, # Offset seed for different contrasts
                exchangeability_matrix=exchangeability_matrix,
                vg_auto=vg_auto, # Pass vg_auto for generator's internal logic if needed
                vg_vector=calculated_vg_vector, # Pass calculated vector
                within=within,
                whole=whole,
                flip_signs=flip_signs,
                zstat=zstat
            )

            for j in tqdm(range(n_permutations), desc=f"Permuting {contrast_label}", leave=False):
                permuted_stats = np.ravel(next(permutation_generator))
                    
                if on_permute_callback is not None:
                    # Pass the permuted stats for processing somewhere else, if requested
                    on_permute_callback(permuted_stats, j, i, two_tailed)

                if two_tailed:
                    abs_perm_stats = np.abs(permuted_stats)
                    exceedances += abs_perm_stats >= np.abs(true_stats)
                    max_stat_dist[j] = np.max(abs_perm_stats) if len(abs_perm_stats) > 0 else -np.inf
                else:
                    exceedances += permuted_stats >= true_stats
                    max_stat_dist[j] = np.max(permuted_stats) if len(permuted_stats) > 0 else -np.inf

            # Step Three: Calculate uncorrected p-values
            unc_p = (exceedances + 1.0) / (n_permutations + 1.0)

            # Step Four: Correct using FDR (Benjamini-Hochberg)
            _, fdr_p = fdrcorrection(unc_p, alpha=0.05, method='indep', is_sorted=False) # Store only corrected p-values

            # Step Five: Correct using FWE (max-stat i.e. Westfall-Young)
            if accel_tail:
                # Use a generalized Pareto distribution to estimate p-values for the tail.
                # Pass the distribution of maximum statistics collected during permutations.
                fwe_p = compute_p_values_accel_tail(true_stats, max_stat_dist, two_tailed=two_tailed)
            else:
                # Calculate directly from the empirical distribution of max statistics
                if two_tailed:
                    fwe_p = (np.sum(max_stat_dist[None, :] >= np.abs(true_stats[:, None]), axis=1) + 1.0) / (n_permutations + 1.0)
                else:
                    fwe_p = (np.sum(max_stat_dist[None, :] >= true_stats[:, None], axis=1) + 1.0) / (n_permutations + 1.0)

            unc_p, fdr_p, fwe_p = process_p_values((unc_p, fdr_p, fwe_p), save_1minusp=save_1minusp, save_neglog10p=save_neglog10p)
            results[f"max_stat_dist_{contrast_label}"] = max_stat_dist
            results[f"stat_uncp_{contrast_label}"] = unc_p
            results[f"stat_fdrp_{contrast_label}"] = fdr_p
            results[f"stat_fwep_{contrast_label}"] = fwe_p
            
            if save_fn is not None:
                save_fn_wrapper(results)

        if correct_across_contrasts:
            # Retrieve max-stat and unc p vectors for all contrasts
            global_uncp_matrix = np.array(reverse_process_p_values((tuple([results[f"stat_uncp_{contrast_label}"] for contrast_label in [f"c{i+1}" for i in range(n_contrasts)]])), save_1minusp=save_1minusp, save_neglog10p=save_neglog10p))
            global_uncp_vector_flat = np.ravel(global_uncp_matrix)
            _, global_fdrp_vector = fdrcorrection(global_uncp_vector_flat, alpha=0.05, method='indep', is_sorted=False)
            global_fdrp_matrix = global_fdrp_vector.reshape(global_uncp_matrix.shape)
            
            if two_tailed:
                global_max_stat_dist = np.max(np.abs(global_max_stat_dist), axis=0)
            else:
                global_max_stat_dist = np.max(global_max_stat_dist, axis=0)

            results["global_max_stat_dist"] = global_max_stat_dist

            for i in range(n_contrasts):
                contrast_label = f"c{i+1}"
                observed_values = results[f"stat_{contrast_label}"]

                # Apply the global max-stat distribution for FWE correction
                if accel_tail:
                    cfwe_p = compute_p_values_accel_tail(observed_values, global_max_stat_dist, two_tailed=two_tailed)
                else:
                    if two_tailed:
                        cfwe_p = (np.sum(global_max_stat_dist[None, :] >= np.abs(observed_values[:, None]), axis=1) + 1.0) / (n_permutations + 1.0)
                    else:
                        cfwe_p = (np.sum(global_max_stat_dist[None, :] >= observed_values[:, None], axis=1) + 1.0) / (n_permutations + 1.0)

                # Store corrected p-values
                fdr_p = global_fdrp_matrix[i, :]
                cfwe_p = process_p_values(cfwe_p, save_1minusp=save_1minusp, save_neglog10p=save_neglog10p)
                results[f"stat_fdrp_{contrast_label}"] = fdr_p
                results[f"stat_cfwep_{contrast_label}"] = cfwe_p
            
            if save_fn is not None:
                save_fn_wrapper(results)

    # --- F-Test Permutations ---
    if perform_f_test and permute:
        print(f"--- Processing F-Test ---")
        # F-tests are always one-tailed
        f_two_tailed = False

        exceedances_f = np.zeros_like(true_stats_f, dtype=float)
        max_stat_dist_f = np.zeros(n_permutations) # Store max F statistic from each permutation

        permutation_generator_f = yield_permuted_stats(
            data, design, f_contrast, # Pass the specific F-contrast
            stat_function=actual_f_stat_function, # Use the F stat function
            n_permutations=n_permutations,
            random_state=random_state - 1, # Use a different seed offset for F-test
            exchangeability_matrix=exchangeability_matrix,
            vg_auto=vg_auto,
            vg_vector=calculated_vg_vector,
            within=within,
            whole=whole,
            flip_signs=flip_signs, # Note: sign flipping might be conceptually odd for F-tests depending on permutation scheme
            zstat=zstat
        )

        for j in tqdm(range(n_permutations), desc="Permuting F-Test", leave=False):
            permuted_stats_f = next(permutation_generator_f)
            permuted_stats_f = np.squeeze(permuted_stats_f) # Ensure 1D
            if permuted_stats_f.ndim == 0:
                permuted_stats_f = permuted_stats_f.reshape(1,)

            if on_permute_callback is not None:
                on_permute_callback(permuted_stats_f, j, -1, f_two_tailed)
                
            # F-statistic is positive, compare directly
            exceedances_f += permuted_stats_f >= true_stats_f
            max_stat_dist_f[j] = np.max(permuted_stats_f) if len(permuted_stats_f) > 0 else -np.inf

        # Step Three: Calculate uncorrected p-values for F-test
        unc_p_f = (exceedances_f + 1.0) / (n_permutations + 1.0)

        # Step Four: Correct using FDR (Benjamini-Hochberg) for F-test
        _, fdr_p_f = fdrcorrection(unc_p_f, alpha=0.05, method='indep', is_sorted=False)

        # Step Five: Correct using FWE (max-stat) for F-test
        if accel_tail:
            # Pass the distribution of maximum F statistics. two_tailed is False for F-test.
            fwe_p_f = compute_p_values_accel_tail(true_stats_f, max_stat_dist_f, two_tailed=f_two_tailed)
        else:
            # Calculate directly from the empirical distribution of max F statistics
            fwe_p_f = (np.sum(max_stat_dist_f[None, :] >= true_stats_f[:, None], axis=1) + 1.0) / (n_permutations + 1.0)

        # Store F-test results
        unc_p_f, fdr_p_f, fwe_p_f = process_p_values((unc_p_f, fdr_p_f, fwe_p_f), save_1minusp=save_1minusp, save_neglog10p=save_neglog10p)
        results["stat_uncp_f"] = unc_p_f
        results["stat_fdrp_f"] = fdr_p_f
        results["stat_fwep_f"] = fwe_p_f
        
        if save_fn is not None:
            save_fn_wrapper(results)

    if not results:
         raise RuntimeError("No results were generated. Check settings and inputs.")

    return results


def permutation_analysis_volumetric_dense(imgs, mask_img,
                                          design, contrast, 
                                          stat_function='auto', n_permutations=1000, random_state=42,
                                          two_tailed=True, exchangeability_matrix=None, vg_auto=False, vg_vector=None,
                                          within=True, whole=False, flip_signs=False,
                                          accel_tail=False,
                                          demean=False,
                                          f_stat_function='auto', f_contrast_indices=None, f_only=False,
                                          correct_across_contrasts=False,
                                          on_permute_callback=None,
                                          tfce=False,
                                          save_1minusp=True,
                                          save_neglog10p=False,
                                          save_fn=None,
                                          zstat=False):
    """
    Parameters
    ----------
    imgs : list of str
        List of file paths to the volumetric images (NIfTI format). Can also be a single file path for a 4d image.
    mask_img : str
        File path to the mask image (NIfTI format).
    design : np.ndarray, shape (n_samples, n_features)
        The design matrix for the GLM, where each row corresponds to a sample and each column to a regressor.
    contrast : np.ndarray, shape (n_features,) or (n_contrasts, n_features)
        The contrast vector or matrix specifying the hypothesis to be tested.
    stat_function : function
        A function that calculates the test statistic. It must accept the arguments (data, design, contrast)
        and return an array of test statistics. The shape of its output defines the shape of the result.
    n_permutations : int
        The number of permutations to perform.
    random_state : int
        Seed for the random number generator to ensure reproducibility.
    two_tailed : bool, default True
        If True, p-values are computed in a two-tailed manner using absolute values of statistics.
    exchangeability_matrix : np.ndarray, optional
        Defines the exchangeability blocks for permutation testing. Expected shapes:
          - If provided as a vector: (n_samples,) or (n_samples, 1).
          - If provided as a matrix: (n_samples, n_permutation_groups).
    vg_auto : bool, default False
        If True, automatically generates a variance group (VG) vector based on the exchangeability matrix.
        If False, the user must provide a VG vector via `vg_vector` if they want to define variance groups.
    vg_vector : np.ndarray, optional
        A 1D array defining the variance groups for each observation. If exchangeability blocks provided and vg_auto is True, vg_vector is calculated automatically.
    within : bool, default True
        For a 1D exchangeability matrix, indicates whether to permute within blocks.
    whole : bool, default False
        For a 1D exchangeability matrix, indicates whether to permute whole blocks.
    flip_signs : bool, default False
        If True, randomly flips the signs of the data for each permutation (assume independent and symmetric errors (ISE)).
    accel_tail : bool, default True
        If True, applies the accelerated tail method (GPD approximation) to compute p-values for cases with 
        low empirical exceedance counts.
    f_stat_function : function
        A function that calculates the F-like statistic. Must accept arguments similar to `stat_function`. Can be 'auto'.
    f_contrast_indices : np.ndarray or list, optional
        Indices (0-based) or boolean mask specifying which rows of the `contrast` matrix
        to include in the F-test. If None, no F-test is performed.
    f_only : bool, default False
        If True, only performs the F-test specified by `f_contrast_indices`.
        Skips individual contrast tests.
    correct_across_contrasts : bool, default False
        If True, applies FWE corrections across all contrasts.
    on_permute_callback: function (optional)
        If provided, calls this function on the permuted stats for each permutation.
        Signature: callback(permuted_stats, permutation_index, contrast_index, is_two_tailed)
    tfce : bool, default True
        If True, applies Threshold-Free Cluster Enhancement (TFCE) to the test statistics.
    save_1minusp : bool, default True
        If True, stores the 1 - p-values instead of the raw p-values.
    save_neglog10p : bool, default False
        If True, stores the -log10(p-values) instead of the raw p-values.
    save_fn : callable, optional
        Expects a callable that is called on "results" each time a key is added to the results dict/Bunch.
    """
    # Step One: Load volumetric images into a 2d matrix (n_samples x n_voxels)
    if mask_img is None:
        print("Warning: No mask image provided. Using the whole image. Unexpected results may occur.")
        masker = NiftiMasker()
    else:
        masker = NiftiMasker(mask_img=mask_img)
    data = masker.fit_transform(imgs)
    mask_img = masker.mask_img_
    n_contrasts = np.atleast_2d(contrast).shape[0]

    if tfce:
        tfce_manager = TfceStatsManager(n_permutations=n_permutations, 
                                        mask_img=load_nifti_if_not_already_nifti(mask_img), 
                                        two_tailed=two_tailed, 
                                        n_contrasts=n_contrasts, 
                                        save_1minusp=save_1minusp, 
                                        save_neglog10p=save_neglog10p,
                                        demean=demean,
                                        correct_across_contrasts=correct_across_contrasts,
                                        accel_tail=accel_tail)   

    def save_fn_wrapper(results):
        """Handle updating the tfce manager with the ground truth stats."""
        if tfce:
            stat_c_keys = {f"stat_c{idx}" for idx in range(1, n_contrasts + 1)}
            stat_keys   = stat_c_keys | {"stat_f"}

            for key, value in results.items():
                # only care about stat_c1…stat_cN  or stat_f
                if key not in stat_keys:
                    continue

                # figure out contrast_idx and label
                if key == "stat_f":
                    contrast_idx, contrast_label = -1, "f"
                else:
                    # splits "stat_c3" → ["stat", "c3"], take "3"
                    contrast_idx = int(key.split("_c")[-1]) - 1
                    contrast_label = f"c{contrast_idx + 1}"

                tfce_key = f"tfce_stat_{contrast_label}"
                # mark *before* processing so we never recurse here
                if tfce_key in tfce_manager.true_stats_tfce:
                    continue

                try:
                    tfce_manager.process_true_stats(value, contrast_idx)
                except Exception as e:
                    print(f"Error processing TFCE for {key!r}: {type(e).__name__}: {e.args}")

        # Handle the actual saving of results
        if save_fn is not None:
            save_fn(results)                                     

    def on_permute_callback_wrapper(permuted_stats, permutation_idx, contrast_idx, two_tailed, *args, **kwargs):
        """Handle updating the tfce manager with the permuted stats."""
        if tfce:
            tfce_manager.update(permuted_stats, permutation_idx, contrast_idx)
        if on_permute_callback is not None:
            on_permute_callback(permuted_stats, permutation_idx, contrast_idx, two_tailed, *args, **kwargs)

    results = permutation_analysis(
        data=data, design=design, contrast=contrast, stat_function=stat_function, f_contrast_indices=f_contrast_indices, f_only=f_only, f_stat_function=f_stat_function,
        n_permutations=n_permutations, random_state=random_state, two_tailed=two_tailed, exchangeability_matrix=exchangeability_matrix, vg_auto=vg_auto, vg_vector=vg_vector,
        within=within, whole=whole, flip_signs=flip_signs, accel_tail=accel_tail, demean=False,
        on_permute_callback=on_permute_callback_wrapper, save_1minusp=save_1minusp, save_neglog10p=save_neglog10p,
        save_fn=save_fn_wrapper, zstat=zstat,
    )

    if tfce:
        tfce_results = tfce_manager.finalize()
        results.update(tfce_results)

    save_fn_wrapper(results)

    return results


def spatial_correlation_permutation_analysis(
    datasets: Union[Dataset, List[Dataset]],
    reference_maps: Optional[Union[str, nib.Nifti1Image, np.ndarray, List[Union[str, nib.Nifti1Image, np.ndarray]]]] = None,
    two_tailed: bool = True,
    compare_func: Optional[Callable] = None,
    ) -> Optional[Bunch]:
    """
    Computes spatial correlations between dataset statistic maps and reference maps,
    using permutation testing to assess significance.

    Parameters
    ----------
    datasets : Dataset or list of Dataset objects
        The datasets to compare. Each Dataset object should contain necessary
        parameters for statistic calculation and permutations.
    reference_maps : NIfTI path/object, np.ndarray, or list thereof, optional
        Reference maps to compare against. Must be compatible (in feature space)
        with the dataset statistic maps after potential masking.
    two_tailed : bool, default True
        If True, computes two-tailed p-values. If False, computes one-tailed
        (right-tailed) p-values.

    Returns
    -------
    results : Bunch or None
        An sklearn.Bunch object containing the results, or None if the analysis cannot proceed
        (e.g., due to insufficient inputs). The object contains:
        - 'corr_matrix_ds_ds': (N_datasets x N_datasets) array of true correlations, or None.
        - 'corr_matrix_ds_ref': (N_datasets x N_references) array of true correlations, or None.
        - 'p_matrix_ds_ds': (N_datasets x N_datasets) array of p-values (diag=NaN), or None.
        - 'p_matrix_ds_ref': (N_datasets x N_references) array of p-values, or None.
        - 'corr_matrix_perm_ds_ds': (N_perm x N_datasets x N_datasets) array, or None.
        - 'corr_matrix_perm_ds_ref': (N_perm x N_datasets x N_references) array, or None.
    """
    analyzer = _SpatialCorrelationAnalysis(datasets, reference_maps, two_tailed, compare_func)
    results = analyzer.run_analysis()
    return results


@jit
def flip_data(data, key):
    n_samples = data.shape[0]
    flip_signs = random.randint(key, (n_samples,), 0, 2) * 2 - 1  # Maps 0,1 to -1, +1.
    return data * flip_signs[:, None]


def yield_sign_flipped_data(data, n_permutations, random_state):
    """
    Generator function that yields sign-flipped versions of the input data one by one.

    Parameters:
      data : array-like, shape (n_samples, n_elements_per_sample)
          Input data matrix.
      n_permutations : int
          Number of sign-flipped permutations to generate.
      random_state : int
          Random seed for reproducibility.

    Yields:
      A sign-flipped version of data for each permutation.
    """
    key = random.PRNGKey(random_state)
    for _ in range(n_permutations):
        key, subkey = random.split(key)
        yield flip_data(data, subkey)


def yield_permuted_stats(data, design, contrast, stat_function, n_permutations, random_state, exchangeability_matrix=None, vg_auto=False, vg_vector=None, within=True, whole=False, flip_signs=False, zstat=False):
    """Generator function for permutation testing.
    data: Shape (n_samples, n_elements_per_sample)
    design: Shape (n_samples, n_features)
    contrast: Shape (n_features,) or (n_contrasts, n_features)
    stat_function: Function that calculates the test statistic. Must take data, design, contrast as arguments.
    n_permutations: Number of permutations to perform.
    random_state: Random seed for reproducibility.
    exchangeability_matrix (Optional): Exchangeability matrix for permutation testing. Shape (n_samples,) or (n_samples, n_permutation_groups).
    vg_auto (Optional): If True, automatically generates a variance group (VG) vector based on the exchangeability matrix.
    vg_vector (Optional): A 1D array defining the variance groups for each observation. If exchangeability blocks provided and vg_auto is True, vg_vector is calculated automatically.
    within (Optional): For a 1D exchangeability matrix, indicates whether to permute within blocks.
    whole (Optional): For a 1D exchangeability matrix, indicates whether to permute whole blocks.
    flip_signs (Optional): If True, randomly flips the signs of the data for each permutation.
    zstat (Optional): If True, calculates z-statistics instead of t-statistics. Uses Z normalization.
    """
    calculate = stat_function
    permuted_design_generator = yield_permuted_design(design=design, n_permutations=n_permutations, contrast=contrast, exchangeability_matrix=exchangeability_matrix, within=within, whole=whole, random_state=random_state)
    if flip_signs:
        sign_flipped_data_generator = yield_sign_flipped_data(data, n_permutations, random_state)
    for i in range(n_permutations):
        if flip_signs:
            data = next(sign_flipped_data_generator)
        if (exchangeability_matrix is not None and vg_auto) or vg_vector:
            if vg_vector is None:
                vg_vector = get_vg_vector(exchangeability_matrix, within=within, whole=whole)
            permuted_value = calculate(data, next(permuted_design_generator), contrast, vg_vector, len(np.unique(vg_vector)))
        else:
            permuted_value = calculate(data, next(permuted_design_generator), contrast)
        if zstat:
            permuted_value = zscore(permuted_value)
        yield permuted_value


def permute_indices_recursive(current_original_indices, level, eb_matrix, rng, parent_instructed_fix_order=False):
    """
    Recursively permute indices based on the exchangeability matrix.
    Args:
        current_original_indices (np.ndarray): Current indices to permute.
        level (int): Current level in the exchangeability matrix.
        eb_matrix (np.ndarray): Exchangeability matrix.
        rng (np.random.Generator): Random number generator.
        parent_instructed_fix_order (bool): Whether the parent instructed to fix order.
    Returns:
        np.ndarray: Permuted indices.
    """
    if len(current_original_indices) == 0: return np.array([], dtype=int)
    if level >= eb_matrix.shape[1]: return rng.permutation(current_original_indices)

    is_last_defined_level = (level == eb_matrix.shape[1] - 1)
    current_eb_level_values = eb_matrix[current_original_indices, level]
    unique_blocks, block_inverse_indices = np.unique(current_eb_level_values, return_inverse=True)
    n_unique_blocks_at_level = len(unique_blocks)

    # --- Single Block Logic ---
    if n_unique_blocks_at_level == 1:
        block_val = unique_blocks[0]
        if block_val == 0:
             raise ValueError(f"Block index 0 found at level {level} for indices subset {current_original_indices[:5]}..., which is not supported.")

        if is_last_defined_level: # Explicit termination for last level
             if block_val > 0: return rng.permutation(current_original_indices)
             else: return np.copy(current_original_indices) # Neg@Last = Identity
        else: # Not last defined level - recurse
             instruct_fix_next = (block_val < 0) # Instruction for NEXT level
             if block_val > 0:
                  # Positive: Shuffle order of sub-blocks found at next level.
                  next_eb_level_values = eb_matrix[current_original_indices, level + 1]
                  unique_sub_blocks, sub_block_inverse, sub_block_counts = np.unique(next_eb_level_values, return_inverse=True, return_counts=True)
                  n_sub_blocks = len(unique_sub_blocks)
                  if n_sub_blocks <= 1:
                      # Pass parent_fix based on current block sign (False here)
                      return permute_indices_recursive(current_original_indices, level + 1, eb_matrix, rng, parent_instructed_fix_order=instruct_fix_next)
                  if len(np.unique(sub_block_counts)) > 1:
                      # Corrected Msg
                      raise ValueError(
                            f"Level {level} (positive index {block_val}) requires sub-blocks "
                            f"defined by level {level + 1} to be uniform size for whole-block shuffling. "
                            f"Indices subset starting with: {current_original_indices[:5]}.... "
                            f"Sub-block IDs: {unique_sub_blocks}. "
                            f"Sub-block sizes: {sub_block_counts}."
                         )
                  sub_block_indices_list = [ current_original_indices[sub_block_inverse == i] for i in range(n_sub_blocks) ]
                  shuffled_sub_block_order = rng.permutation(n_sub_blocks)
                  # Pass down fix instruction based on the SUB-BLOCK'S sign
                  permuted_sub_blocks = [ permute_indices_recursive(sub_block_indices_list[i], level + 1, eb_matrix, rng, parent_instructed_fix_order=(unique_sub_blocks[i]<0)) for i in range(n_sub_blocks) ]
                  return np.concatenate([permuted_sub_blocks[idx] for idx in shuffled_sub_block_order])
             else: # block_val < 0
                  # Negative: Recurse, instructing next level to fix order.
                  return permute_indices_recursive(current_original_indices, level + 1, eb_matrix, rng, parent_instructed_fix_order=True)

    # --- Multi Block Logic ---
    else: # n > 1
        signs = np.sign(unique_blocks)
        if np.any(unique_blocks == 0):
             raise ValueError(f"Block index 0 found at level {level} among {unique_blocks} for indices {current_original_indices[:5]}..., which is not supported.")
        if len(np.unique(signs)) > 1:
             raise ValueError(
                f"Level {level}: Mixed positive/negative block indices found "
                f"({unique_blocks}) within the same parent block structure "
                f"for indices starting with {current_original_indices[:5]}..., which is ambiguous and not supported by PALM."
            )

        # *** Prioritize Parent Instruction & Last Level Check ***
        if parent_instructed_fix_order:
             # Parent said fix order -> MUST concatenate this level in order. Recurse within.
             permuted_indices_list = []
             for i, block_val_i in enumerate(unique_blocks):
                  mask = (block_inverse_indices == i)
                  indices_in_this_block_i = current_original_indices[mask]
                  # Recurse, passing instruction based on this block's sign
                  instruct_fix_i = (block_val_i < 0)
                  permuted_subset = permute_indices_recursive(indices_in_this_block_i, level + 1, eb_matrix, rng, parent_instructed_fix_order=instruct_fix_i)
                  permuted_indices_list.append(permuted_subset)
             return np.concatenate(permuted_indices_list) # Concat in order

        elif is_last_defined_level:
             # Parent allowed shuffle AND this is the last level. Terminate based on signs.
             if signs[0] > 0:
                  # Freely permute all involved indices together.
                  return rng.permutation(current_original_indices)
             else: # signs[0] < 0
                  # Identity for each block, concatenate in order.
                  permuted_indices_list = [np.copy(current_original_indices[block_inverse_indices == i])
                                           for i in range(n_unique_blocks_at_level)]
                  return np.concatenate(permuted_indices_list)
        else:
             # Intermediate level AND parent allowed shuffle.
             # Recurse within each block. Concatenate results based on *this* level's signs (Original V1 logic).
             permuted_indices_list = []
             for i, block_val_i in enumerate(unique_blocks):
                  mask = (block_inverse_indices == i)
                  indices_in_this_block_i = current_original_indices[mask]
                  instruct_fix_i = (block_val_i < 0)
                  permuted_subset = permute_indices_recursive(indices_in_this_block_i, level + 1, eb_matrix, rng, parent_instructed_fix_order=instruct_fix_i)
                  permuted_indices_list.append(permuted_subset)

             # Use original concatenation logic because parent_fix is False
             if signs[0] > 0: # Shuffle order
                  shuffled_block_order = rng.permutation(n_unique_blocks_at_level)
                  return np.concatenate([permuted_indices_list[idx] for idx in shuffled_block_order])
             else: # Preserve order
                  return np.concatenate(permuted_indices_list)


def yield_permuted_design(design, n_permutations, contrast=None, exchangeability_matrix=None, within=None, whole=None, random_state=None):
    """Generator for permuting the design matrix per PALM documentation.

    Handles free exchange, within-block, whole-block, combined within/whole,
    and multi-level exchangeability via positive/negative indices.
    Docs: https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/PALM(2f)ExchangeabilityBlocks.html

    Args:
        design (np.ndarray): Design matrix. Shape (n_samples, n_features).
        n_permutations (int): Number of permutations to generate.
        random_state (int or np.random.Generator or None): Seed or Generator
            for the random number generator.
        exchangeability_matrix (np.ndarray or None): Matrix or vector defining
            exchangeability blocks. Shape (n_samples,) or (n_samples, n_levels).
            If None, free exchange is assumed. Defaults to None.
        within (bool | None): For single-column blocks, allow shuffling within blocks.
                       If None (default): Behavior depends on 'whole'. If 'whole' is also None or False,
                       defaults to True. If 'whole' is True, defaults to False.
                       Ignored if exchangeability_matrix has >1 column or if None.
        whole (bool | None): For single-column blocks, shuffle blocks as wholes.
                      If None (default): Defaults to False.
                      Ignored if exchangeability_matrix has >1 column or if None.

    Yields:
        np.ndarray: A permuted version of the design matrix.

    Raises:
        ValueError: If inputs are inconsistent (e.g., non-uniform block sizes
                    required for whole-block shuffling, ambiguous multi-col structure,
                    zero indices in eb_matrix).
        TypeError: If design or exchangeability_matrix is not a numpy array or
                   if eb_matrix contains non-numeric data.
    """
    # --- Input Validation ---
    if not isinstance(design, np.ndarray):
        raise TypeError("design must be a numpy array.")
    if design.ndim != 2:
        raise ValueError(f"design must be 2D (samples x features), got shape {design.shape}")

    n_samples = design.shape[0]
    if n_samples == 0:
         # Handle empty design matrix - yield nothing or raise error?
         # Let's yield nothing as n_permutations would be irrelevant.
         return

    # Initialize RNG
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    original_indices = np.arange(n_samples)

    # --- Preprocess exchangeability_matrix (eb_matrix) ---
    is_eb_provided = exchangeability_matrix is not None
    eb_matrix = None
    n_levels = 0
    use_flags = False # Default

    if is_eb_provided:
        if not isinstance(exchangeability_matrix, np.ndarray):
            raise TypeError("exchangeability_matrix must be a numpy array.")
        if exchangeability_matrix.size == 0 and n_samples > 0 :
             raise ValueError("exchangeability_matrix is empty but design matrix is not.")
        if exchangeability_matrix.size > 0:
            if not np.issubdtype(exchangeability_matrix.dtype, np.number):
                 raise TypeError("exchangeability_matrix must contain numeric indices.")

            # Check for non-integer values that aren't trivially convertible (e.g., 1.5 vs 1.0)
            if not np.all(np.mod(exchangeability_matrix, 1) == 0):
                 # Check more robustly if conversion is possible without loss
                 try:
                     int_eb = exchangeability_matrix.astype(int)
                     if not np.all(np.isclose(exchangeability_matrix, int_eb)):
                         raise ValueError("Non-integer values found in exchangeability_matrix.")
                     eb_matrix = int_eb
                 except (ValueError, TypeError):
                      raise ValueError("Non-integer values found in exchangeability_matrix.")
            else:
                 eb_matrix = exchangeability_matrix.astype(int)

            if eb_matrix.shape[0] != n_samples:
                raise ValueError(f"exchangeability_matrix rows ({eb_matrix.shape[0]}) "
                                 f"must match design matrix rows ({n_samples}).")
            if eb_matrix.ndim == 1:
                eb_matrix = eb_matrix.reshape(-1, 1)
            elif eb_matrix.ndim > 2:
                raise ValueError("exchangeability_matrix cannot have more than 2 dimensions.")
            elif eb_matrix.ndim == 0: # Should be caught by shape[0] check if n_samples > 0
                 raise ValueError("exchangeability_matrix cannot be 0-dimensional.")

            n_levels = eb_matrix.shape[1]
            use_flags = (n_levels == 1) # Flags only relevant if effectively single-level

            # Final check for 0 indices which are unsupported by PALM logic
            if np.any(eb_matrix == 0):
                 raise ValueError("Exchangeability matrix contains index 0, which is not supported (use positive/negative integers).")

    # --- Determine Effective within/whole for single-level ---
    eff_within = within
    eff_whole = whole
    if use_flags:
        # Apply default logic only if flags are relevant (single level)
        if eff_whole is None:
            eff_whole = False
        if eff_within is None:
            eff_within = not eff_whole # Default within=True unless whole=True

    # --- Define the permutation function for one iteration ---
    def get_permuted_indices():
        if not is_eb_provided or eb_matrix is None:
            # Case 0: Free exchange (no eb_matrix provided or it was empty)
            return rng.permutation(original_indices)

        # --- Determine strategy based on levels and flags ---
        if use_flags:
            # --- Case 1: Single Level - Use Flags ---
            block_ids = eb_matrix[:, 0]
            unique_blocks, inverse = np.unique(block_ids, return_inverse=True)
            n_unique_blocks = len(unique_blocks)

            # Trivial case: only one block behaves like free exchange
            if n_unique_blocks <= 1:
                 return rng.permutation(original_indices)

            if eff_within and eff_whole:
                # Simultaneous within & whole -> Treat as free exchange
                # (Based on VG interpretation suggesting equivalence to simplest case)
                return rng.permutation(original_indices)
            elif eff_whole:
                # Whole-block shuffling
                unique_blocks, inverse, counts = np.unique(block_ids, return_inverse=True, return_counts=True)
                if len(np.unique(counts)) > 1:
                     raise ValueError(
                        "Whole-block shuffling requires all blocks to be the same size. "
                        f"Found sizes: {counts} for blocks {unique_blocks}"
                     )
                n_blocks = len(unique_blocks)
                # Group original indices by block ID
                blocks_indices = [original_indices[inverse == i] for i in range(n_blocks)]
                # Shuffle the order of the blocks
                shuffled_block_order = rng.permutation(n_blocks)
                # Concatenate blocks in the new shuffled order
                return np.concatenate([blocks_indices[i] for i in shuffled_block_order])
            elif eff_within:
                # Within-block shuffling
                permuted_indices = np.copy(original_indices) # Start with identity
                for i in range(n_unique_blocks):
                    mask = (inverse == i)
                    indices_this_block = original_indices[mask]
                    # Permute the indices *within* this block
                    shuffled_subset = rng.permutation(indices_this_block)
                    # Assign the permuted indices back to the original positions of the block
                    permuted_indices[mask] = shuffled_subset
                return permuted_indices
            else: # within=False, whole=False explicitly set
                # This state isn't clearly defined by PALM for permutations.
                # Defaulting to free exchange as the least restrictive assumption.
                return rng.permutation(original_indices)
        else:
            # --- Case 2: Multi-Level (Ignore flags) ---
            # Call the recursive helper starting at level 0
            return permute_indices_recursive(original_indices, 0, eb_matrix, rng)

    # --- Generator Loop ---
    for i in range(n_permutations):
        permuted_row_indices = get_permuted_indices()
        # Check if the permutation is valid before yielding
        if len(permuted_row_indices) != n_samples:
             raise RuntimeError(f"Permutation {i+1} generated incorrect number of indices: "
                                f"{len(permuted_row_indices)}, expected {n_samples}")
        if contrast is not None:
            # If a contrast is provided, permute only the subset of columns in the design matrix that are being tested.
            # Note: This method is the Draper-Stoneman method, which is not what Anderson Winkler recommends.
            # To me, it makes more sense and is easier to implement than the Freedman-Lane method recommended by Anderson Winkler.
            contrast = np.atleast_2d(contrast)
            contrast_indices = np.ravel(contrast[0,:]).astype(bool)
            design_subset = design[:, contrast_indices]
            design_subset = design_subset[permuted_row_indices, :]
            design[:, contrast_indices] = design_subset
        else:
            design = design[permuted_row_indices, :]
        yield design


def gpdpvals(exceedances, scale, shape):
    """
    Compute GPD tail survival p-values for exceedances, robust to invalid-power errors.

    Parameters
    ----------
    exceedances : array-like
        Values above the threshold (x - threshold), assumed >= 0.
    scale : float
        GPD scale parameter (must be > 0).
    shape : float
        GPD shape parameter.

    Returns
    -------
    p_values : ndarray
        Survival-function p-values for each exceedance, clipped to [0, 1].
    """
    exceedances = np.asarray(exceedances, dtype=float)
    if scale <= 0:
        raise ValueError("`scale` must be positive.")

    with np.errstate(divide='ignore', invalid='ignore'):
        if abs(shape) < np.finfo(float).eps:
            p = np.exp(- exceedances / scale)
        else:
            base = 1 - shape * exceedances / scale
            p = np.where(base > 0, base ** (1.0 / shape), 0.0)

    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(p, 0.0, 1.0)


def compute_p_values_accel_tail(
    obs,
    null_dist,
    two_tailed=True,
    p_threshold=0.1,
    min_tail_size=20,
    start_quantile=0.75,
    method="MOM",
    n_mc_samples=5000
):
    """
    Convert observed statistics to p-values via permutations + optional GPD tail.

    Parameters
    ----------
    obs : array-like, shape (n,)
        Observed test statistics.
    null_dist : array-like, shape (m,)
        Null distribution stats from permutations.
    two_tailed : bool, default=True
        If True, use absolute values of obs and null_dist.
    p_threshold : float, default=0.1
        Empirical p-value cutoff for applying GPD tail refinement.
    min_tail_size : int, default=20
        Minimum number of exceedances required for GPD fitting.
    start_quantile : float in (0,1), default=0.75
        Initial quantile of null_dist at which to define the tail.
    method : {'MOM', 'MLE'}, default='MOM'
        'MOM'  – Hosking–Wallis method-of-moments fit
        'MLE'  – Maximum-likelihood fit via SciPy
    n_mc_samples : int, default=5000
        Monte-Carlo samples for the Anderson–Darling goodness-of-fit.

    Returns
    -------
    p_values : ndarray, shape (n,)
        Estimated p-values for each observed statistic.
    """
    print("Using accelerated tail method for p-value computation.", flush=True)
    obs = np.asarray(obs)
    null_dist = np.asarray(null_dist)
    if two_tailed:
        obs, null_dist = np.abs(obs), np.abs(null_dist)

    m = null_dist.size
    if m == 0:
        return np.ones_like(obs, dtype=float)

    # 1) Empirical p-values
    empirical_p = (np.sum(null_dist[None] >= obs[:, None], axis=1) + 1) / (m + 1)
    if not np.any(empirical_p <= p_threshold):
        return empirical_p

    # 2) GPD tail refinement
    for q in np.arange(start_quantile, 0.992, 0.01):
        threshold = np.percentile(null_dist, q * 100)
        tail = null_dist[null_dist >= threshold]
        if tail.size < min_tail_size:
            break

        exceedances = tail - threshold
        try:
            if method.upper() == "MOM":
                mean_exc = exceedances.mean()
                var_exc = exceedances.var(ddof=0)
                if var_exc <= 0:
                    continue
                scale = mean_exc * (mean_exc**2 / var_exc + 1) / 2
                shape = (mean_exc**2 / var_exc - 1) / 2
                loc = 0
            elif method.upper() == "MLE":
                shape, loc, scale = genpareto.fit(exceedances, floc=0, method="MLE")
            else:
                raise ValueError(f"Unknown method '{method}'")

            if scale <= 0:
                continue

            # 3) Goodness-of-fit (AD test with known params)
            gof = goodness_of_fit(
                genpareto,
                exceedances,
                known_params={"c": shape, "loc": loc, "scale": scale},
                statistic="ad",
                n_mc_samples=n_mc_samples
            )
            if gof.pvalue <= 0.05:
                continue
            print("Found good GPD fit.", flush=True)

            # 4) Refine tail p-values
            tail_prob = np.mean(null_dist >= threshold)
            mask = (empirical_p <= p_threshold) & (obs >= threshold)
            if np.any(mask):
                obs_exc = obs[mask] - threshold
                if method.upper() == "MOM":
                    tail_p = gpdpvals(obs_exc, scale, shape)
                else:
                    tail_p = genpareto.sf(obs_exc, c=shape, loc=loc, scale=scale)
                emp_tail_p = tail_prob * tail_p
                empirical_p[mask] = np.minimum(
                    empirical_p[mask],
                    np.maximum(emp_tail_p, np.finfo(float).tiny)
                )
            print("Succesfully refined tail p-values.", flush=True)

            break

        except Exception:
            continue

    return empirical_p

class TfceStatsManager:
    def __init__(self, n_permutations, mask_img, two_tailed=True, n_contrasts=1, save_1minusp=False, save_neglog10p=False, correct_across_contrasts=False, accel_tail=False, zstat=False):

        self.n_permutations = n_permutations
        self.mask_img = mask_img
        self.masker = NiftiMasker(mask_img).fit()
        self.two_tailed = two_tailed
        self.n_contrasts = n_contrasts
        self.save_1minusp = save_1minusp
        self.save_neglog10p = save_neglog10p
        self.correct_across_contrasts = correct_across_contrasts    
        self.accel_tail = accel_tail

        self.exceedances_tfce = Bunch()
        self.max_stat_dist_tfce = Bunch()
        self.true_stats_tfce = Bunch()
        self.results = Bunch()

    def process_true_stats(self, true_stats, contrast_idx):
        if contrast_idx is None or contrast_idx == -1:
            contrast_label = "f"
        else:
            contrast_label = f"c{contrast_idx + 1}"

        # Compute the TFCE-transformed statistics
        true_stats_tfce = apply_tfce(self.masker.inverse_transform(true_stats), two_tailed=self.two_tailed).get_fdata()
        true_stats_tfce = true_stats_tfce[self.mask_img.get_fdata() != 0]
        self.results[f"tfce_stat_{contrast_label}"] = true_stats_tfce
        self.true_stats_tfce[f"tfce_stat_{contrast_label}"] = true_stats_tfce

    def update(self, permuted_stats, permutation_idx, contrast_idx=None):
        # Determine the contrast label
        if contrast_idx is None or contrast_idx == -1:
            contrast_label = "f"
        else:
            contrast_label = f"c{contrast_idx + 1}"
        
        # Transform the permuted stats with TFCE
        permuted_stats_tfce = apply_tfce(self.masker.inverse_transform(permuted_stats), two_tailed=self.two_tailed).get_fdata()
        permuted_stats_tfce = permuted_stats_tfce[self.mask_img.get_fdata() != 0]
        
        # On the first iteration, initialize state arrays/scalars
        if permutation_idx == 0:
            self.exceedances_tfce[f"tfce_stat_{contrast_label}"] = np.zeros_like(permuted_stats_tfce)
            if self.two_tailed:
                self.exceedances_tfce[f"tfce_stat_{contrast_label}"] += (np.abs(permuted_stats_tfce) >= np.abs(self.true_stats_tfce[f"tfce_stat_{contrast_label}"]))
                self.max_stat_dist_tfce[f"tfce_stat_{contrast_label}"] = np.max(np.abs(permuted_stats_tfce))
            else:
                self.exceedances_tfce[f"tfce_stat_{contrast_label}"] += (permuted_stats_tfce >= self.true_stats_tfce[f"tfce_stat_{contrast_label}"])
                self.max_stat_dist_tfce[f"tfce_stat_{contrast_label}"] = np.max(permuted_stats_tfce)
        
        # If not the first iteration, update the exceedances and max stat distribution
        else:
            if self.two_tailed:
                # Update exceedances: count where abs(permuted) >= abs(true)
                self.exceedances_tfce[f"tfce_stat_{contrast_label}"] += (np.abs(permuted_stats_tfce) >= np.abs(self.true_stats_tfce[f"tfce_stat_{contrast_label}"]))
                # Concatenate the new max value using pd.concat equivalent (if arrays) or np.hstack
                self.max_stat_dist_tfce[f"tfce_stat_{contrast_label}"] = np.hstack([
                    np.max(np.abs(permuted_stats_tfce)),
                    self.max_stat_dist_tfce[f"tfce_stat_{contrast_label}"]
                ])
            else:
                # Update exceedances: count where permuted >= true
                self.exceedances_tfce[f"tfce_stat_{contrast_label}"] += (permuted_stats_tfce >= self.true_stats_tfce[f"tfce_stat_{contrast_label}"])
                # Concatenate the new max value using pd.concat equivalent (if arrays) or np.hstack
                self.max_stat_dist_tfce[f"tfce_stat_{contrast_label}"] = np.hstack([
                    np.max(permuted_stats_tfce),
                    self.max_stat_dist_tfce[f"tfce_stat_{contrast_label}"]
                ])

    def finalize(self):
        # Process each t contrast, if applicable
        valid_contrast_labels = [f"c{i + 1}" for i in range(self.n_contrasts)]
        valid_contrast_labels = [label for label in valid_contrast_labels if f"tfce_stat_{label}" in self.true_stats_tfce]
        
        for contrast_label in valid_contrast_labels:
            exceedances = self.exceedances_tfce[f"tfce_stat_{contrast_label}"]
            tfce_stat_uncp = (exceedances + 1) / (self.n_permutations + 1)
            _, tfce_stat_fdrp = fdrcorrection(tfce_stat_uncp)

            if self.accel_tail:
                tfce_stat_fwep = compute_p_values_accel_tail(self.true_stats_tfce[f"tfce_stat_{contrast_label}"],
                                                        self.max_stat_dist_tfce[f"tfce_stat_{contrast_label}"],
                                                        two_tailed=self.two_tailed)
            else:
                if self.two_tailed:
                    tfce_stat_fwep = (np.sum(self.max_stat_dist_tfce[f"tfce_stat_{contrast_label}"][None, :] >= np.abs(self.true_stats_tfce[f"tfce_stat_{contrast_label}"][:, None]), axis=1) + 1) / (self.n_permutations + 1)
                else:
                    tfce_stat_fwep = (np.sum(self.max_stat_dist_tfce[f"tfce_stat_{contrast_label}"][None, :] >= self.true_stats_tfce[f"tfce_stat_{contrast_label}"][:, None], axis=1) + 1) / (self.n_permutations + 1)

            # Store the results in the Bunch object
            self.results[f"tfce_stat_uncp_{contrast_label}"] = tfce_stat_uncp
            self.results[f"tfce_stat_fdrp_{contrast_label}"] = tfce_stat_fdrp
            self.results[f"tfce_stat_fwep_{contrast_label}"] = tfce_stat_fwep

        if self.correct_across_contrasts:  

            # Create tfce FDR p values corrected across all contrasts
            global_uncp_matrix = np.stack([self.results[f"tfce_stat_uncp_{label}"] for label in valid_contrast_labels], axis=0)
            global_uncp_vector_flat = np.ravel(global_uncp_matrix)
            _, global_fdrp_vector_flat = fdrcorrection(global_uncp_vector_flat)
            global_fdrp_matrix = np.reshape(global_fdrp_vector_flat, global_uncp_matrix.shape)
            

            # Create tfce FWE p values corrected across all contrasts
            global_max_stat_dist_tfce = np.stack([self.max_stat_dist_tfce[f"tfce_stat_{label}"] for label in valid_contrast_labels], axis=0)
    
            if self.two_tailed:
                global_max_stat_dist = np.max(np.abs(global_max_stat_dist), axis=0)
            else:
                global_max_stat_dist = np.max(global_max_stat_dist, axis=0)

            for i, label in enumerate(valid_contrast_labels):
                cfdrp_vector = global_fdrp_matrix[i, :]
                observed_tfce_values = self.true_stats_tfce[f"tfce_stat_{label}"]
                if self.accel_tail:
                    cfwe_p = compute_p_values_accel_tail(observed_tfce_values, global_max_stat_dist_tfce[i, :], two_tailed=self.two_tailed)
                else:
                    if self.two_tailed:
                        cfwe_p = (np.sum(global_max_stat_dist_tfce[i, :] >= np.abs(observed_tfce_values), axis=1) + 1) / (self.n_permutations + 1)
                    else:
                        cfwe_p = (np.sum(global_max_stat_dist_tfce[i, :] >= observed_tfce_values, axis=1) + 1) / (self.n_permutations + 1)
                
                self.results[f"tfce_stat_cfdrp_{label}"] = cfdrp_vector
                self.results[f"tfce_stat_cfwep_{label}"] = cfwe_p

        # Process the f contrast results, if they exist:
        if f"tfce_stat_f" in self.true_stats_tfce:
            # Process the f contrast results
            tfce_stat_uncp = self.exceedances_tfce["tfce_stat_f"]
            tfce_stat_uncp = (tfce_stat_uncp + 1) / (self.n_permutations + 1)
            _, tfce_stat_fdrp = fdrcorrection(tfce_stat_uncp)

            if self.accel_tail:
                tfce_stat_fwep = compute_p_values_accel_tail(self.true_stats_tfce["tfce_stat_f"],
                                                        self.max_stat_dist_tfce["tfce_stat_f"],
                                                        two_tailed=self.two_tailed)
            else:
                if self.two_tailed:
                    tfce_stat_fwep = (np.sum(self.max_stat_dist_tfce["tfce_stat_f"] >= np.abs(self.true_stats_tfce["tfce_stat_f"]), axis=1) + 1) / (self.n_permutations + 1)
                else:
                    tfce_stat_fwep = (np.sum(self.max_stat_dist_tfce["tfce_stat_f"] >= self.true_stats_tfce["tfce_stat_f"], axis=1) + 1) / (self.n_permutations + 1)

            # Store the results in the Bunch object
            tfce_stat_uncp, tfce_stat_fdrp, tfce_stat_fwep = process_p_values((tfce_stat_uncp, tfce_stat_fdrp, tfce_stat_fwep), self.save_1minusp, self.save_neglog10p)
            self.results["tfce_stat_uncp_f"] = tfce_stat_uncp
            self.results["tfce_stat_fdrp_f"] = tfce_stat_fdrp
            self.results["tfce_stat_fwep_f"] = tfce_stat_fwep

    
        # Finally, let's process all the p values from the valid contrasts:
        for label in valid_contrast_labels:
            tfce_stat_uncp, tfce_stat_fdrp, tfce_stat_fwep = process_p_values((self.results[f"tfce_stat_uncp_{label}"], self.results[f"tfce_stat_fdrp_{label}"], self.results[f"tfce_stat_fwep_{label}"]), self.save_1minusp, self.save_neglog10p)
            self.results[f"tfce_stat_uncp_{label}"] = tfce_stat_uncp
            self.results[f"tfce_stat_fdrp_{label}"] = tfce_stat_fdrp
            self.results[f"tfce_stat_fwep_{label}"] = tfce_stat_fwep

        return self.results

    
def get_vg_vector(exchangeability_matrix, within=True, whole=False):
    """
    Calculates the variance group (VG) vector based on exchangeability rules.

    Args:
        exchangeability_matrix (np.ndarray): 
            A 1D or 2D numpy array defining exchangeability blocks.
            - For 1D: Integer indices defining blocks. 'within' and 'whole' flags matter.
            - For 2D: Defines nested exchangeability. Flags are ignored.
              - Positive index in col k: Sub-indices in col k+1 shuffle as a whole.
              - Negative index in col k: Sub-indices in col k+1 shuffle within block.
        within (bool, optional): 
            If True and exchangeability_matrix is 1D and whole=False, 
            indicates within-block exchangeability. Defaults to True.
        whole (bool, optional): 
            If True and exchangeability_matrix is 1D, indicates whole-block
            exchangeability. Overrides 'within' if both are True for VG calc.
            Defaults to False.

    Returns:
        np.ndarray: A 1D numpy array of unique integer identifiers (starting from 1)
                    defining the variance groups (vg_vector) for each observation.

    Raises:
        ValueError: If inputs are inconsistent (e.g., non-uniform block sizes
                    required for whole-block shuffling, ambiguous multi-col structure).
        TypeError: If exchangeability_matrix is not a numpy array.
    """

    if not isinstance(exchangeability_matrix, np.ndarray):
        raise TypeError("exchangeability_matrix must be a numpy array.")

    # Check if dtype is already integer
    if not np.issubdtype(exchangeability_matrix.dtype, np.integer):
        # If not integer, check if it contains only integer-like values (e.g., floats like 1.0)
        try:
            # Use np.mod and check closeness to 0 for float precision issues
            is_integer_like = np.all(np.isclose(np.mod(exchangeability_matrix, 1), 0))
        except TypeError:
            # This catches errors if np.mod fails (e.g., non-numeric types)
            raise ValueError("exchangeability_matrix must contain numeric integer-like indices.")

        if is_integer_like:
            # If all are integer-like, convert safely
            exchangeability_matrix = exchangeability_matrix.astype(int)
        else:
            # If any are truly non-integer floats (like 1.5), raise specific error
            raise ValueError("Non-integer values found in exchangeability_matrix.")

    # Store original dimension and force to 2D for consistent processing
    original_ndim = exchangeability_matrix.ndim
    if original_ndim == 0:
        raise ValueError("exchangeability_matrix cannot be 0-dimensional.")
    elif original_ndim == 1:
        # Reshape 1D array to a 2D array with one column
        eb_matrix = exchangeability_matrix.reshape(-1, 1)
    else:
        eb_matrix = exchangeability_matrix

    n_observations = eb_matrix.shape[0]
    n_levels = eb_matrix.shape[1]

    if n_observations == 0:
        return np.array([], dtype=int)
    if n_observations == 1:
        return np.ones(1, dtype=int)

    # --- Determine the effective VG rule ---
    
    use_flags = (original_ndim == 1)
    # According to the description, multi-column structure overrides flags
    if n_levels > 1:
         use_flags = False
         
    vg_vector = np.ones(n_observations, dtype=int) # Default to single group

    # --- Case 1: Use Flags (Original matrix was 1D) ---
    if use_flags:
        block_ids = eb_matrix[:, 0]
        # Handle potentially non-contiguous block IDs by mapping them
        unique_blocks, block_indices = np.unique(block_ids, return_inverse=True)
        
        # Calculate counts based on the mapped indices
        block_counts = np.bincount(block_indices)

        # If only one effective block, it's always a single VG
        if len(unique_blocks) <= 1:
             return np.ones(n_observations, dtype=int) # Correctly handles single block case

        if whole and within:
            # Simultaneous whole- and within-block => freely exchangeable => single VG
            return np.ones(n_observations, dtype=int)
        elif whole:
            # Whole-block shuffling (-whole flag)
            # Check for uniform block sizes using the calculated counts
            if len(np.unique(block_counts)) > 1:
                raise ValueError(
                    "Whole-block shuffling requires all blocks to be the same size. "
                    f"Found sizes: {block_counts}" # Show counts for unique blocks
                )
            block_size = block_counts[0]
            # VG = position within block (1 to block_size)
            # Generate VG based on position within original blocks
            temp_vg = np.zeros(n_observations, dtype=int)
            current_pos_in_block = {} # Key: block_id, Value: next position
            for i in range(n_observations):
                block_val = block_ids[i]
                pos = current_pos_in_block.get(block_val, 0)
                temp_vg[i] = pos + 1
                current_pos_in_block[block_val] = pos + 1
            vg_vector = temp_vg

        elif within:
            # Within-block shuffling (-within flag, default)
            # VG = block index (1-based) based on unique values encountered
            vg_vector = block_indices + 1
        else:
            # Neither within nor whole specified -> freely exchangeable -> single VG
             return np.ones(n_observations, dtype=int)


    # --- Case 2: Multi-Column Matrix (Flags ignored) ---
    elif n_levels > 1:
        col_0 = eb_matrix[:, 0]
        col_1 = eb_matrix[:, 1]
        
        # Determine unique groups based on first column
        unique_l0, indices_l0 = np.unique(col_0, return_inverse=True)

        # Check if the first level implies whole or within block shuffling
        # Assuming uniformity *within each block* defined by unique_l0
        
        # Check for mixed signs *across* blocks if multiple l0 blocks exist
        if len(unique_l0) > 1 and (np.any(unique_l0 > 0) and np.any(unique_l0 < 0)):
             raise ValueError(
                "Multi-column exchangeability matrix contains mixed positive/negative "
                "indices in the first column across different top-level blocks. "
                "Automatic VG determination for this specific structure is not supported."
             )
        
        # Determine effective rule (all positive or all negative in first relevant column)
        first_sign = col_0[0] # Check based on the first entry's sign
        all_positive = np.all(col_0 > 0)
        all_negative = np.all(col_0 < 0)

        if not (all_positive or all_negative):
             # If not uniformly positive or negative, check if it's just one block type
             if len(unique_l0) == 1:
                 all_positive = unique_l0[0] > 0
                 all_negative = unique_l0[0] < 0
             else: # Mixed signs within a block or across blocks was checked earlier
                 raise ValueError(
                     "Ambiguous multi-column structure: first column indices are not "
                     "consistently positive or negative."
                 )


        if all_positive:
             # Positive indices in col 0 -> Whole-block shuffling implied for col 1 groups
             # VG = position within the blocks defined by col 1
             
             # Need to determine block sizes based on col_1 *within* each col_0 group
             block_sizes = []
             temp_vg = np.zeros(n_observations, dtype=int)
             
             # Map positions within each unique block defined by col_1
             current_pos_in_sub_block = {} # key=col_1 value, val=next_pos
             for i in range(n_observations):
                 sub_block_val = col_1[i]
                 pos = current_pos_in_sub_block.get(sub_block_val, 0)
                 temp_vg[i] = pos + 1
                 current_pos_in_sub_block[sub_block_val] = pos + 1

             # Now check uniformity of sizes for blocks defined by col_1
             unique_sub_blocks, sub_block_indices, sub_block_counts = np.unique(
                 col_1, return_inverse=True, return_counts=True
             )
             
             # Special case: If overall structure results in only one group -> VG=1
             # e.g. [[1,1],[1,2],[1,3]] -> sub_block_counts = [1,1,1] -> block_size=1
             if len(np.unique(sub_block_counts)) > 1:
                  # Check if it's just trivial blocks of size 1
                  if not np.all(sub_block_counts == 1):
                      raise ValueError(
                         "Whole-block shuffling implied by positive indices requires "
                         "effective sub-blocks (from the second level) to be the same size. "
                         f"Found sizes based on column 1: {sub_block_counts}"
                      )
             
             block_size = sub_block_counts[0]
             # Special case check: If block size is 1, it's like free exchangeability
             if block_size == 1:
                 return np.ones(n_observations, dtype=int)
             
             vg_vector = temp_vg # Use the calculated positions

        elif all_negative:
            # Negative indices in col 0 -> Within-block shuffling implied for col 1 groups
            # VG = index of the block defined by col 1 (make unique IDs 1-based)
            unique_sub_blocks, sub_block_indices = np.unique(
                col_1, return_inverse=True
            )
            
            # If only one effective sub-block, implies single VG
            if len(unique_sub_blocks) <= 1:
                return np.ones(n_observations, dtype=int)
                
            vg_vector = sub_block_indices + 1
            
        # The case where neither all_positive nor all_negative should be caught by prior checks

    # --- Fallback for single level (if somehow missed) ---
    elif n_levels == 1:
         # Treat as 1D case with default flags (within=True, whole=False)
         # This case should theoretically be handled by use_flags=True path
         block_ids = eb_matrix[:, 0]
         unique_blocks, block_indices = np.unique(block_ids, return_inverse=True)
         if len(unique_blocks) <= 1:
             return np.ones(n_observations, dtype=int)
         else:
             vg_vector = block_indices + 1 # Default 'within' logic


    return vg_vector.astype(int)


class _SpatialCorrelationAnalysis:
    """
    Manages spatial correlation analysis between datasets and reference maps.

    Calculates correlations/similarities and performs permutation testing.
    Supports standard Pearson correlation or a custom comparison function.
    """

    def __init__(self, datasets_input: Union['Dataset', List['Dataset']],
                 reference_maps_input: Optional[Union[str, nib.Nifti1Image, np.ndarray, List[Union[str, nib.Nifti1Image, np.ndarray]]]],
                 two_tailed: bool,
                 comparison_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None):
        """
        Initializes the analysis manager.

        Args:
            datasets_input: One or more Dataset objects.
            reference_maps_input: Optional reference map(s) (path, Nifti1Image, or ndarray).
            two_tailed: If True, use two-tailed tests for p-values.
            comparison_func: Optional custom function(vec1, vec2) -> float. Defaults to Pearson correlation.
        """
        self.datasets_input = datasets_input
        self.reference_maps_input = reference_maps_input
        self.two_tailed = two_tailed
        self.comparison_func = comparison_func

        # Internal state
        self.datasets: List['Dataset'] = []
        self.final_reference_maps: List[np.ndarray] = [] # 1D arrays
        self.n_datasets: int = 0
        self.n_references: int = 0
        self.n_permutations: int = 0
        self.target_feature_shape: Optional[int] = None
        self.common_masker: Optional[NiftiMasker] = None

        # Results storage
        self.true_stats_list: List[np.ndarray] = [] # 1D stat maps
        self.true_corr_ds_ds: Optional[np.ndarray] = None
        self.true_corr_ds_ref: Optional[np.ndarray] = None
        self.permuted_corrs_ds_ds: Optional[np.ndarray] = None # (n_perm, n_ds, n_ds)
        self.permuted_corrs_ds_ref: Optional[np.ndarray] = None # (n_perm, n_ds, n_ref)

    def _setup_and_validate(self) -> bool:
        """Loads data, standardizes inputs, handles masking, validates shapes."""
        # 1. Standardize Datasets
        self.datasets = self.datasets_input if isinstance(self.datasets_input, list) else [self.datasets_input]
        if not self.datasets: raise ValueError("Dataset list cannot be empty.")
        for i, ds in enumerate(self.datasets):
             if not isinstance(ds, Dataset): raise TypeError(f"Item {i} is not a Dataset object.")
        self.n_datasets = len(self.datasets)

        # 2. Standardize Reference Maps
        ref_maps_list_raw: List[Any] = []
        if self.reference_maps_input is not None:
            ref_maps_list_raw = self.reference_maps_input if isinstance(self.reference_maps_input, list) else [self.reference_maps_input]
        self.n_references = len(ref_maps_list_raw)

        # 3. Check Trivial Case
        if self.n_datasets == 0 or (self.n_datasets < 2 and self.n_references == 0):
            warnings.warn("Insufficient inputs for correlation analysis.")
            return False

        # 4. Load Dataset Data
        for i, ds in enumerate(self.datasets):
            try:
                # Assumes ds.load_data() is implemented in the Dataset class
                ds.load_data()
                if ds.data is None: raise RuntimeError("Dataset.data is None after loading.")
            except Exception as e:
                 raise RuntimeError(f"Failed loading data for dataset {i+1}: {e}") from e

        # 5. Prepare Masker & Process References
        self._prepare_common_masker()
        self._process_reference_maps(ref_maps_list_raw)

        # 6. Validate Feature Shapes
        if not self._validate_shapes(): return False

        # 7. Determine Number of Permutations
        if any(ds.n_permutations <= 0 for ds in self.datasets):
             warnings.warn("n_permutations <= 0 found. Permutation testing skipped.")
             self.n_permutations = 0
        else:
             self.n_permutations = min(ds.n_permutations for ds in self.datasets)
             print(f"Running analysis with {self.n_permutations} permutations.") # Keep one informative print

        return True

    def _prepare_common_masker(self):
        """Determines and stores a common NiftiMasker if multiple NIfTI datasets exist."""
        nifti_datasets = [ds for ds in self.datasets if ds.is_nifti]
        self.common_masker = None
        if len(nifti_datasets) > 1:
            first_masker = next((ds.masker for ds in nifti_datasets if ds.masker is not None), None)
            if first_masker:
                self.common_masker = first_masker
            else:
                warnings.warn("Multiple NIfTI datasets found, but no common masker identified. Ensure consistency.")
        elif len(nifti_datasets) == 1:
            self.common_masker = nifti_datasets[0].masker # Use single NIfTI dataset's masker for refs

    def _process_reference_maps(self, ref_maps_list_raw: List):
        """Loads, masks (if NIfTI & common_masker exists), and flattens reference maps."""
        self.final_reference_maps = []
        if not ref_maps_list_raw: return

        for i, ref_map_input in enumerate(ref_maps_list_raw):
            ref_map_data: Optional[np.ndarray] = None
            try:
                loaded_ref = load_data(ref_map_input) # Assumes load_data handles paths/objects

                if is_nifti_like(loaded_ref): # Assumes is_nifti_like checks paths/objects
                    ref_img = load_nifti_if_not_already_nifti(loaded_ref) # Assumes this loads/returns Nifti1Image
                    if self.common_masker:
                        if not hasattr(self.common_masker, 'mask_img_') or self.common_masker.mask_img_ is None:
                             warnings.warn(f"Common masker for ref map {i+1} seems unfit.")
                        masked_ref = self.common_masker.transform(ref_img)
                        ref_map_data = masked_ref.ravel()
                    else:
                         warnings.warn(f"NIfTI ref map {i+1} processed raw (no common masker).")
                         ref_map_data = ref_img.get_fdata().ravel()
                elif isinstance(loaded_ref, np.ndarray):
                    ref_map_data = loaded_ref.ravel()
                    if self.common_masker:
                        warnings.warn(f"NumPy ref map {i+1} used; ensure it matches masked space.")
                else:
                    raise TypeError(f"Unsupported type for ref map {i+1}: {type(loaded_ref)}")

                if ref_map_data is not None:
                    self.final_reference_maps.append(ref_map_data)
            except Exception as e:
                raise ValueError(f"Failed processing ref map {i+1}: {e}") from e

    def _validate_shapes(self) -> bool:
        """Checks consistency of feature dimensions across stats maps and reference maps."""
        self.target_feature_shape = None
        # Determine target shape from first dataset's stat map
        if self.n_datasets > 0:
             first_ds = self.datasets[0]
             try:
                if not all([first_ds.data is not None, first_ds.design is not None,
                            first_ds.contrast is not None, first_ds.stat_function is not None]):
                     raise ValueError("Dataset 1 missing components needed for shape validation.")
                # Calculate temporary map just for shape
                stat_args = [first_ds.data, first_ds.design, first_ds.contrast]
                temp_stat_map = first_ds.stat_function(*stat_args)
                self.target_feature_shape = temp_stat_map.ravel().shape[0]
             except Exception as e:
                 warnings.warn(f"Could not get shape from dataset 1 stat map: {e}. Trying refs.")

        # Fallback to first reference map if needed
        if self.target_feature_shape is None and self.final_reference_maps:
            self.target_feature_shape = self.final_reference_maps[0].shape[0]

        # Final check if shape could be determined
        if self.target_feature_shape is None:
             if self.n_datasets > 0 or self.n_references > 0: # Only error if inputs existed
                 raise RuntimeError("Could not determine target feature shape from any source.")
             else: return True # No inputs, technically no mismatch

        # Validate reference maps against target shape
        for i, ref_map in enumerate(self.final_reference_maps):
             if ref_map.shape[0] != self.target_feature_shape:
                 raise ValueError(f"Shape mismatch: Ref map {i+1} ({ref_map.shape[0]}) != target ({self.target_feature_shape}).")

        return True

    def calculate_true_statistics(self):
        """Calculates the true statistic map for each dataset."""
        if self.target_feature_shape is None: raise RuntimeError("Target shape unknown.")
        self.true_stats_list = []

        for i, dataset in enumerate(self.datasets):
            if not all([dataset.data is not None, dataset.design is not None,
                        dataset.contrast is not None, dataset.stat_function is not None]):
                 raise ValueError(f"Dataset {i+1} missing components for stat calculation.")

            stat_args = [dataset.data, dataset.design, dataset.contrast]
            # Handle variance groups if specified
            effective_vg_vector = dataset.vg_vector
            if effective_vg_vector is None and dataset.exchangeability_matrix is not None and dataset.vg_auto:
                 # Assumes get_vg_vector is available
                 effective_vg_vector = get_vg_vector(dataset.exchangeability_matrix,
                                                     within=dataset.within, whole=dataset.whole)
            if effective_vg_vector is not None:
                n_groups = len(np.unique(effective_vg_vector))
                stat_args.extend([effective_vg_vector, n_groups]) # Assumes stat_func signature adapts

            # Calculate stats
            true_stats_raw = dataset.stat_function(*stat_args)
            true_stats_flat = true_stats_raw.ravel()

            # Validate shape
            if true_stats_flat.shape[0] != self.target_feature_shape:
                raise ValueError(f"Shape mismatch: Dataset {i+1} stat map ({true_stats_flat.shape[0]}) != target ({self.target_feature_shape}).")

            dataset.true_stats = true_stats_flat
            self.true_stats_list.append(true_stats_flat)

    def _compute_correlation_matrix(self, data1: np.ndarray, data2: Optional[np.ndarray] = None) -> np.ndarray:
        """Computes correlation/similarity matrix using np.corrcoef or custom function."""
        if data1.ndim == 1: data1 = data1[:, np.newaxis]
        if data2 is not None and data2.ndim == 1: data2 = data2[:, np.newaxis]
        n_items1 = data1.shape[1]
        n_items2 = data2.shape[1] if data2 is not None else 0

        if self.comparison_func:
            # Pairwise calculation using custom function
            if data2 is None: # Self-similarity
                if n_items1 == 0: return np.array([]).reshape(0,0)
                res = np.zeros((n_items1, n_items1))
                for i in range(n_items1):
                    for j in range(i, n_items1):
                        sim = self.comparison_func(data1[:, i], data1[:, j])
                        res[i, j] = sim
                        if i != j: res[j, i] = sim # Assume symmetry
                return res
            else: # Cross-similarity
                if n_items1 == 0 or n_items2 == 0: return np.array([]).reshape(n_items1, n_items2)
                res = np.zeros((n_items1, n_items2))
                for i in range(n_items1):
                    for j in range(n_items2):
                        res[i, j] = self.comparison_func(data1[:, i], data2[:, j])
                return res
        else:
            # Default Pearson correlation using np.corrcoef
            if data2 is None: # Self-correlation
                if n_items1 <= 1: return np.array([[1.0]]) if n_items1 == 1 else np.array([]).reshape(0,0)
                with warnings.catch_warnings(): # Suppress warnings for now, handle NaNs below
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    corr = np.corrcoef(data1, rowvar=False)
                corr = np.nan_to_num(corr, nan=0.0) # Replace NaN with 0
                np.fill_diagonal(corr, 1.0) # Ensure diagonal is 1
                return corr
            else: # Cross-correlation
                if n_items1 == 0 or n_items2 == 0: return np.array([]).reshape(n_items1, n_items2)
                combined = np.hstack((data1, data2))
                with warnings.catch_warnings(): # Suppress warnings
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    full_corr = np.corrcoef(combined, rowvar=False)
                full_corr = np.nan_to_num(full_corr, nan=0.0)
                # Check shape before slicing
                if full_corr.shape == (n_items1 + n_items2, n_items1 + n_items2):
                    return full_corr[:n_items1, n_items1:]
                else: # Handle unexpected scalar or shape mismatch
                    warnings.warn(f"Unexpected corrcoef shape {full_corr.shape}. Returning NaNs.")
                    return np.full((n_items1, n_items2), np.nan)

    def calculate_true_correlations(self):
        """Calculates the true correlation/similarity matrices."""
        if not self.true_stats_list:
             warnings.warn("No true stats calculated, cannot compute correlations.")
             return
        try:
            stacked_ds_stats = np.stack(self.true_stats_list, axis=-1)
        except ValueError as e:
             raise RuntimeError(f"Failed stacking true stats: {e}") from e

        # DS-DS
        if self.n_datasets > 1:
            self.true_corr_ds_ds = self._compute_correlation_matrix(stacked_ds_stats)
        elif self.n_datasets == 1:
            self.true_corr_ds_ds = np.array([[1.0]])

        # DS-Ref
        if self.n_datasets > 0 and self.final_reference_maps:
            try:
                 stacked_ref_maps = np.stack(self.final_reference_maps, axis=-1)
                 self.true_corr_ds_ref = self._compute_correlation_matrix(stacked_ds_stats, stacked_ref_maps)
            except ValueError as e:
                 raise RuntimeError(f"Failed stacking reference maps: {e}") from e
        elif self.n_datasets > 0 and self.n_references > 0: # Refs provided but failed processing
             self.true_corr_ds_ref = np.array([]).reshape(self.n_datasets, 0)

    def run_permutations(self):
        """Runs permutations and collects permuted correlations/similarities."""
        if self.n_permutations <= 0: return # Already warned in setup
        if self.target_feature_shape is None: raise RuntimeError("Target shape unknown.")

        # Initialize storage
        self.permuted_corrs_ds_ds = None
        if self.n_datasets > 1:
            self.permuted_corrs_ds_ds = np.zeros((self.n_permutations, self.n_datasets, self.n_datasets))
        self.permuted_corrs_ds_ref = None
        if self.n_datasets > 0 and self.n_references > 0:
            self.permuted_corrs_ds_ref = np.zeros((self.n_permutations, self.n_datasets, self.n_references))

        # Setup generators
        for i, dataset in enumerate(self.datasets):
            if not all([dataset.data is not None, dataset.design is not None,
                        dataset.contrast is not None, dataset.stat_function is not None]):
                 raise ValueError(f"Dataset {i+1} missing components for permutation.")
            # Assumes yield_permuted_stats exists and handles these args
            dataset.permuted_stat_generator = yield_permuted_stats(
                data=dataset.data, design=dataset.design, contrast=dataset.contrast,
                stat_function=dataset.stat_function,
                n_permutations=self.n_permutations,
                random_state=dataset.random_state,
                exchangeability_matrix=dataset.exchangeability_matrix,
                vg_auto=dataset.vg_auto, vg_vector=dataset.vg_vector,
                within=dataset.within, whole=dataset.whole, flip_signs=dataset.flip_signs
            )
            if not isinstance(dataset.permuted_stat_generator, Generator):
                 raise RuntimeError(f"Failed creating permutation generator for dataset {i+1}.")

        # Pre-stack references if needed
        stacked_ref_maps = None
        if self.permuted_corrs_ds_ref is not None and self.final_reference_maps:
             try:
                 stacked_ref_maps = np.stack(self.final_reference_maps, axis=-1)
             except ValueError as e:
                  raise RuntimeError(f"Failed stacking reference maps for permutations: {e}") from e

        # Permutation loop with progress bar
        permuted_stats_current = np.zeros((self.target_feature_shape, self.n_datasets))
        for perm_idx in tqdm(range(self.n_permutations), desc="Permutations", unit="perm", leave=False):
            # Get stats for current permutation
            for i, dataset in enumerate(self.datasets):
                try:
                    perm_stat = next(dataset.permuted_stat_generator)
                    permuted_stats_current[:, i] = perm_stat.ravel()
                except StopIteration:
                    raise RuntimeError(f"Perm generator ended early for dataset {i+1} at perm {perm_idx+1}.")
                except Exception as e:
                     raise RuntimeError(f"Error getting perm stat for dataset {i+1} at perm {perm_idx+1}: {e}") from e

            # Compute and store permuted correlations/similarities
            if self.permuted_corrs_ds_ds is not None:
                self.permuted_corrs_ds_ds[perm_idx] = self._compute_correlation_matrix(permuted_stats_current)
            if self.permuted_corrs_ds_ref is not None and stacked_ref_maps is not None:
                self.permuted_corrs_ds_ref[perm_idx] = self._compute_correlation_matrix(permuted_stats_current, stacked_ref_maps)

    def _calculate_p_values_internal(self, true_values: Optional[np.ndarray],
                                     permuted_values: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Calculates p-values based on true and permuted values."""
        if permuted_values is None or true_values is None: return None # Cannot calculate
        n_perm_actual = permuted_values.shape[0]
        if n_perm_actual == 0: return np.full_like(true_values, np.nan) # No perms run

        try:
            if self.two_tailed:
                exceedances = np.sum(np.abs(permuted_values) >= np.abs(true_values)[np.newaxis, :, :], axis=0)
            else: # One-tailed (right)
                exceedances = np.sum(permuted_values >= true_values[np.newaxis, :, :], axis=0)
            p_values = (exceedances + 1.0) / (n_perm_actual + 1.0)
            return p_values
        except Exception as e:
            warnings.warn(f"Error during p-value calculation: {e}. Returning NaNs.")
            return np.full_like(true_values, np.nan)

    def run_analysis(self) -> Bunch[str, Optional[np.ndarray]]:
        """
        Orchestrates the full analysis pipeline.

        Returns:
            sklearn Bunch obj containing results ('corr_matrix_ds_ds', 'corr_matrix_ds_ref',
            'p_matrix_ds_ds', 'p_matrix_ds_ref', 'corr_matrix_perm_ds_ds',
            'corr_matrix_perm_ds_ref').
        """
        results: Bunch[str, Optional[np.ndarray]] = Bunch()
        results['corr_matrix_ds_ds'] = None
        results['corr_matrix_ds_ref'] = None
        results['p_matrix_ds_ds'] = None
        results['p_matrix_ds_ref'] = None
        results['corr_matrix_perm_ds_ds'] = None
        results['corr_matrix_perm_ds_ref'] = None

        # 1. Setup & Validate
        if not self._setup_and_validate(): return results

        # 2. True Statistics
        self.calculate_true_statistics()

        # 3. True Correlations/Similarities
        self.calculate_true_correlations()
        results['corr_matrix_ds_ds'] = self.true_corr_ds_ds
        results['corr_matrix_ds_ref'] = self.true_corr_ds_ref

        # 4. Permutations
        self.run_permutations() # Runs only if self.n_permutations > 0
        results['corr_matrix_perm_ds_ds'] = self.permuted_corrs_ds_ds
        results['corr_matrix_perm_ds_ref'] = self.permuted_corrs_ds_ref

        # 5. P-values
        results['p_matrix_ds_ds'] = self._calculate_p_values_internal(
            results['corr_matrix_ds_ds'], results['corr_matrix_perm_ds_ds'])
        results['p_matrix_ds_ref'] = self._calculate_p_values_internal(
            results['corr_matrix_ds_ref'], results['corr_matrix_perm_ds_ref'])

        # Set diagonal of ds-ds p-values to NaN
        if results['p_matrix_ds_ds'] is not None and self.n_datasets > 1:
                p_matrix_ds_ds = results['p_matrix_ds_ds'].copy() # Ensure writeable
                np.fill_diagonal(p_matrix_ds_ds, np.nan)
                results['p_matrix_ds_ds'] = p_matrix_ds_ds

        print("Analysis finished.")
        return results
    

def process_p_values(p_values, save_1minusp=False, save_neglog10p=False):
    if save_1minusp and save_neglog10p:
        raise ValueError("Only one of save_1minusp/save_neglog10p may be True")

    def _neglog(p):
        with np.errstate(divide='ignore'):
            out = -np.log10(p)
        if np.any(np.isnan(out)):
            warnings.warn("NaN produced by -log10(p)", RuntimeWarning)
        return out

    # choose transformation
    fn = (
        (lambda p: 1 - p) if save_1minusp
        else _neglog if save_neglog10p
        else (lambda p: p)
    )

    # normalize to a tuple for uniform processing
    is_single = not isinstance(p_values, (list, tuple))
    inputs = (p_values,) if is_single else p_values

    # apply and wrap back
    results = tuple(fn(np.asarray(p)) for p in inputs)
    return results[0] if is_single else results


def reverse_process_p_values(p_values, save_1minusp=False, save_neglog10p=False):
    """Reverses the transformation applied in process_p_values."""
    if save_1minusp and save_neglog10p:
        raise ValueError("Only one of save_1minusp/save_neglog10p may be True")

    def _inv_neglog(x):
        with np.errstate(over='ignore'):
            return 10 ** (-x)

    # pick inverse transform
    fn = (
        (lambda x: 1 - x) if save_1minusp
        else _inv_neglog if save_neglog10p
        else (lambda x: x)
    )

    # unify to tuple
    is_single = not isinstance(p_values, (list, tuple))
    inputs = (p_values,) if is_single else p_values

    # apply and restore shape/structure
    results = tuple(fn(np.asarray(p)) for p in inputs)
    return results[0] if is_single else results
