import warnings
import numpy as np
from jax import jit, random, numpy as jnp
from scipy.stats import genpareto, goodness_of_fit
from sklearn.utils import Bunch
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
from nilearn.maskers import NiftiMasker
from .preprocessing import load_nifti_if_not_already_nifti, ResultSaver
from .stats import (
    t, aspin_welch_v, F, G, pearson_r, r_squared,
    t_z, aspin_welch_v_z, F_z, G_z, fisher_z, r_squared_z,
    residualize_data, partition_model, demean_glm_data
)
from .tfce import apply_tfce
from .permutation_logic import get_vg_vector, yield_permuted_indices


def permutation_analysis(
    data,
    design,
    contrast,
    output_prefix=None,
    f_contrast_indices=None,
    two_tailed=True,
    exchangeability_matrix=None,
    vg_auto=False,
    variance_groups=None,
    within=True,
    whole=False,
    flip_signs=False,
    stat_function="auto",
    f_stat_function="auto",
    f_only=False,
    n_permutations=1000,
    accel_tail=False,
    save_1minusp=False,
    save_neglog10p=False,
    correct_across_contrasts=False,
    random_state=42,
    demean=False,
    zstat=False,
    save_fn=None,
    permute_fn=None,
    save_permutations=False,
    mask_img=None,
):
    """
    Perform permutation-based inference on a general linear model.

    Implements:
      1. **Beckmann partitioning** of the GLM into regressors of interest vs. nuisance.
      2. **Freedman–Lane permutation** strategy:
         a. Fit to estimate residuals wrt nuisance regressors.
         b. Permute or sign-flip those residuals.
         c. Add permuted residuals back onto the fitted nuisance part.
         d. Recompute test statistics to build a null distribution.
      3. Compute observed (true) statistics.
      4. Derive empirical (uncorrected) p-values.
      5. Apply Benjamini–Hochberg FDR correction.
      6. Apply Westfall–Young max-statistic FWE correction (optionally with GPD tail acceleration).

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_elements)
        Observations matrix: rows are samples, columns are features/voxels.
    design : np.ndarray, shape (n_samples, n_features)
        Full design matrix for the GLM.
    contrast : np.ndarray, shape (n_features,) or (n_contrasts, n_features)
        Contrast vector or matrix defining hypotheses.
    output_prefix : str or None
        Prefix for any output files or saved arrays.
    f_contrast_indices : array-like or None
        Indices (or boolean mask) selecting rows of `contrast` for an F-test.
    two_tailed : bool, default True
        If True, compute two-tailed p-values (uses absolute statistics).
    exchangeability_matrix : np.ndarray or None
        Block structure for permutation; shape (n_samples,) or (n_samples, n_groups).
    vg_auto : bool, default False
        If True, auto-derive variance-group labels from `exchangeability_matrix`.
    variance_groups : np.ndarray or None
        Predefined variance-group labels for each sample.
    within : bool, default True
        When `exchangeability_matrix` is 1D, permute within blocks.
    whole : bool, default False
        When `exchangeability_matrix` is 1D, permute whole blocks.
    flip_signs : bool, default False
        If True, also sign-flip residuals (assumes symmetric errors).
    stat_function : callable or 'auto', default 'auto'
        Function to compute per-contrast test statistics (e.g., t).  
        Signature: `stat, df1, df2 = stat_function(data, design, contrast[, variance_groups, n_groups])`
    f_stat_function : callable or 'auto', default 'auto'
        Function to compute F-statistics. Same signature as `stat_function`.
    f_only : bool, default False
        If True, skip individual contrasts and run only the specified F-test.
    n_permutations : int, default 1000
        Number of permutations to generate null distributions.
    accel_tail : bool, default False
        If True, use GPD tail acceleration for FWE p-values when exceedances are low.
    save_1minusp : bool, default False
        If True, store 1–p instead of raw p-values.
    save_neglog10p : bool, default False
        If True, store –log₁₀(p) instead of raw p-values.
    correct_across_contrasts : bool, default False
        If True, apply FWE correction jointly over all contrasts.
    random_state : int, default 42
        Seed for reproducibility.
    demean : bool, default False
        If True, demean the data before computing statistics.
    zstat : bool, default False
        If True, convert t-statistics to z-scores before p-value calculation.
    save_fn : callable or None
        If provided, called as `save_fn(results, key)` whenever a result is added.
    permute_fn : callable or None
        If provided, called each permutation as  
        `permute_fn(permuted_stats, perm_index, is_two_tailed)`.
    save_permutations : bool, default False
        If True, retain and return all permuted statistic arrays.
    mask_img : Niimg-like or None
        Optional mask to apply to data before analysis.

    Returns
    -------
    results : sklearn.utils.Bunch
        A Bunch containing:
          - Observed stats: `stat_c{i}`, `stat_f` (if F-test run).
          - Uncorrected p-values: `stat_uncp_c{i}`, `stat_uncp_f`.
          - FDR p-values: `stat_fdrp_c{i}`, `stat_fdrp_f`.
          - FWE p-values: `stat_fwep_c{i}`, `stat_fwep_f`.
          - Max-stat distributions: `max_stat_dist_c{i}`, `max_stat_dist_f`.
          - If `correct_across_contrasts`: joint `stat_cfdrp_c{i}`, `stat_cfwep_c{i}`, `global_max_stat_dist`.
          - (Optional) Saved permutations if `save_permutations=True`.

    Notes
    -----
    - F-tests are one-tailed by construction; `two_tailed` is ignored for F-statistics.
    - When using variance groups (`vg_auto` or `variance_groups`), ensure your stat functions support them.
    """
    # Step Zero: Check inputs and setup
    if n_permutations <= 0:
        raise ValueError("Number of permutations must be positive")
    if data.shape[0] != design.shape[0]:
        raise ValueError(
            f"Data ({data.shape[0]}) and design ({design.shape[0]}) must have the same number of samples"
        )
    if contrast.ndim == 1:
        if contrast.shape[0] != design.shape[1]:
            raise ValueError(
                "1D contrast dimensions must match number of regressors in design matrix"
            )
    elif contrast.ndim == 2:
        if contrast.shape[1] != design.shape[1]:
            raise ValueError(
                "2D contrast dimensions must match number of regressors in design matrix"
            )
    else:  # contrast.ndim > 2
        raise ValueError(
            f"Contrast must be 1D or 2D. Got {contrast.ndim}D. (Shape: {contrast.shape})"
        )
    if (
        exchangeability_matrix is not None
        and exchangeability_matrix.shape[0] != data.shape[0]
    ):
        raise ValueError("Exchangeability matrix length must match number of samples")
    if vg_auto and exchangeability_matrix is None:
        raise ValueError("exchangeability_matrix must be provided if vg_auto is True")
    if f_contrast_indices is not None:
        f_contrast_indices = np.ravel(f_contrast_indices).astype(bool).astype(int)
    if f_only and f_contrast_indices is None and contrast.ndim == 1:
        warnings.warn(
            "f_only is True, but f_contrast_indices is None and only one base contrast is provided. Performing F-test on this single contrast."
        )
        # Treat the single contrast as the one to test with F
        f_contrast_indices = np.array([0])
    elif (
        f_only
        and f_contrast_indices is None
        and contrast.ndim == 2
        and contrast.shape[0] == 1
    ):
        warnings.warn(
            "f_only is True, but f_contrast_indices is None and only one base contrast is provided. Performing F-test on this single contrast."
        )
        f_contrast_indices = np.array([0])
    elif (
        f_only
        and f_contrast_indices is None
        and contrast.ndim == 2
        and contrast.shape[0] > 1
    ):
        warnings.warn(
            "f_only is True, but f_contrast_indices is None. Performing F-test on *all* provided contrasts."
        )
        # Use all contrasts for the F-test
        f_contrast_indices = np.arange(contrast.shape[0])
    elif f_only and f_contrast_indices is not None:
        pass  # User specified indices for F-test
    elif (
        not f_only
        and f_contrast_indices is None
        and contrast.ndim == 2
        and contrast.shape[0] > 1
    ):
        warnings.warn(
            "Multiple contrasts provided, but f_contrast_indices is None. No F-test will be performed."
        )
        # No F-test by default if multiple contrasts and no indices

    if demean:
        data, design, contrast, f_contrast_indices = demean_glm_data(
            data, design, contrast, f_contrast_indices
        )

    # Ensure contrast is 2D
    original_contrast = np.atleast_2d(contrast)
    n_t_contrasts = original_contrast.shape[0]
    n_elements = data.shape[1]  # Number of voxels/features etc.

    # Prepare f_contrast if F-test is needed
    perform_f_test = False
    f_contrast = None
    if f_contrast_indices is not None or f_only:
        perform_f_test = True
        if f_contrast_indices is not None:
            f_contrast_indices = np.ravel(np.asarray(f_contrast_indices)).astype(bool)
            if f_contrast_indices.ndim > 1:
                raise ValueError(
                    "f_contrast_indices must be 1D array or list of indices/booleans."
                )
            if f_contrast_indices.shape[0] > n_t_contrasts:
                f_contrast_indices = f_contrast_indices[:n_t_contrasts]
            if f_contrast_indices.dtype == bool:
                if len(f_contrast_indices) != n_t_contrasts:
                    raise ValueError(
                        f"Boolean f_contrast_indices length ({len(f_contrast_indices)}) must match number of contrasts ({n_t_contrasts})"
                    )
                f_contrast = original_contrast[f_contrast_indices, :]
            else:  # Integer indices
                if (
                    np.max(f_contrast_indices) >= n_t_contrasts
                    or np.min(f_contrast_indices) < 0
                ):
                    raise ValueError(
                        f"f_contrast_indices values out of bounds for {n_t_contrasts} contrasts."
                    )
                f_contrast = original_contrast[f_contrast_indices, :]
        elif f_only:  # Indices were None, f_only is True -> use all original contrasts
            f_contrast = original_contrast

        if f_contrast is None or f_contrast.shape[0] == 0:
            raise ValueError(
                "Cannot perform F-test: f_contrast_indices resulted in an empty set of contrasts."
            )
        if f_contrast.shape[0] == 1 and not f_only:
            warnings.warn(
                "F-test requested for a single contrast vector. This is equivalent to a squared t-test (or similar for other stats)."
            )

    # Assemble contrasts for output
    contrasts = Bunch()
    if not f_only:
        for i in range(n_t_contrasts):
            contrasts[f"c{i+1}"] = np.atleast_2d(original_contrast[i, :])
    if perform_f_test:
        contrasts["f"] = np.atleast_2d(f_contrast)

    contrast_labels = list(contrasts.keys())

    # Determine variance groups if needed
    use_variance_groups = (exchangeability_matrix is not None and vg_auto) or (
        variance_groups is not None
    )
    calculated_variance_groups = None
    n_groups = None
    if use_variance_groups:
        if variance_groups is not None:
            calculated_variance_groups = np.asarray(variance_groups)
            if calculated_variance_groups.shape[0] != data.shape[0]:
                raise ValueError(
                    "Provided variance_groups length must match number of samples"
                )
        else:  # vg_auto is True and exchangeability_matrix is not None
            calculated_variance_groups = get_vg_vector(
                exchangeability_matrix, within=within, whole=whole
            )
        n_groups = len(np.unique(calculated_variance_groups))
        if n_groups <= 1:
            warnings.warn(
                "Variance groups were requested or auto-detected, but only one group was found. Standard statistics will be used."
            )
            use_variance_groups = False  # Revert to standard stats
            calculated_variance_groups = None
            n_groups = None

    # Determine which stat functions to use
    actual_stat_function, actual_f_stat_function = select_stat_functions(
        stat_function=stat_function,
        f_stat_function=f_stat_function,
        use_variance_groups=use_variance_groups,
        zstat=zstat,
        perform_f_test=perform_f_test,
    )

    # Initialize saver object
    results_saver = ResultSaver(
        output_prefix=output_prefix,
        variance_groups=calculated_variance_groups if use_variance_groups else None,
        stat_function=stat_function,
        f_stat_function=f_stat_function,
        zstat=zstat,
        save_permutations=save_permutations,
        mask_img=mask_img,
        n_t_contrasts=n_t_contrasts,
    )

    # Setup wrapper functions for saving results/permutations if provided
    def save_fn_wrapper(results):
        try:
            if output_prefix is not None:
                results_saver.save_results(results)
            if save_fn is not None:
                save_fn(results)
        except Exception as e:
            print(f"Error saving results: {e}")

    def permute_fn_wrapper(permuted_stats, perm_idx, contrast_idx, two_tailed):
        """Function to handle permutation results."""
        # Save permutations if output output_prefix and save_permutations are provided
        if output_prefix is not None and save_permutations:
            results_saver.save_permutation(permuted_stats, perm_idx, contrast_idx)

        # Call the user-defined callback if provided
        if permute_fn is not None:
            permute_fn(permuted_stats, perm_idx, contrast_idx, two_tailed)

    # Initialize results
    results = Bunch()

    # Do ground truth calculations
    for label, contrast in contrasts.items():
        stat_function_ = (
            actual_stat_function if label != "f" else actual_f_stat_function
        )
        (
            regressors_of_interest_,
            nuisance_regressors_,
            effective_contrast_overall_,
            effective_contrast_interest_,
        ) = partition_model(design, contrast)
        effective_design_ = np.hstack([regressors_of_interest_, nuisance_regressors_])
        if use_variance_groups:
            observed_stats, df1, df2 = stat_function_(
                data,
                effective_design_,
                effective_contrast_overall_,
                calculated_variance_groups,
                n_groups,
            )
        else:
            observed_stats, df, df2 = stat_function_(
                data, effective_design_, effective_contrast_overall_
            )

        observed_stats = np.ravel(observed_stats)
        if (
            observed_stats.ndim == 0
        ):  # Handle case where stat_function returns scalar (e.g., 1 element)
            observed_stats = observed_stats.reshape(
                1,
            )
        if observed_stats.shape[0] != n_elements:
            raise RuntimeError(
                f"Stat function returned unexpected shape {observed_stats.shape} for contrast {label}, expected ({n_elements},)"
            )
        results[f"stat_{label}"] = observed_stats
        save_fn_wrapper(results)

    # Do permutation testing
    for label, contrast in contrasts.items():
        current_idx = contrast_labels.index(label)
        print(
            f"--- Processing Contrast {label} ({current_idx + 1}/{len(contrast_labels)}) ---"
        )

        stat_function_ = (
            actual_stat_function if label != "f" else actual_f_stat_function
        )

        two_tailed = False if label == "f" else two_tailed

        observed_stats = results[f"stat_{label}"]
        exceedances = np.zeros_like(observed_stats, dtype=float)
        max_stat_dist = np.zeros(n_permutations)


        permutation_generator = yield_permuted_stats(
            data,
            design,
            contrast,  # Pass the current contrast
            stat_function=stat_function_,
            n_permutations=n_permutations,
            random_state=random_state + current_idx,
            exchangeability_matrix=exchangeability_matrix,
            vg_auto=vg_auto,  # Pass vg_auto for generator's internal logic if needed
            variance_groups=calculated_variance_groups,  # Pass calculated vector
            within=within,
            whole=whole,
            flip_signs=flip_signs,
        )


        for i in tqdm(range(n_permutations), desc=f"Permuting {label}", leave=False):

            permuted_stats = np.ravel(next(permutation_generator))

            permute_fn_wrapper(permuted_stats, i, current_idx, two_tailed)

            if two_tailed:
                abs_perm_stats = np.abs(permuted_stats)
                exceedances += abs_perm_stats >= np.abs(observed_stats)
                max_stat_dist[i] = (
                    np.max(abs_perm_stats) if len(abs_perm_stats) > 0 else -np.inf
                )
            else:
                exceedances += permuted_stats >= observed_stats
                max_stat_dist[i] = (
                    np.max(permuted_stats) if len(permuted_stats) > 0 else -np.inf
                )

        # Step Three: Calculate uncorrected p-values
        unc_p = (exceedances + 1.0) / (n_permutations + 1.0)
        # Step Four: Correct using FDR (Benjamini-Hochberg)
        _, fdr_p = fdrcorrection(np.where(unc_p == 1.0 / (n_permutations + 1.0), 0, unc_p), alpha=0.05, method="indep", is_sorted=False)
        # Step Five: Correct using FWE (max-stat i.e. Westfall-Young)
        if accel_tail:
            # Use a generalized Pareto distribution to estimate p-values for the tail.
            # Pass the distribution of maximum statistics collected during permutations.
            fwe_p = compute_p_values_accel_tail(
                observed_stats, max_stat_dist, two_tailed=two_tailed
            )
        else:
            # Calculate directly from the empirical distribution of max statistics
            if two_tailed:
                fwe_p = (
                    np.sum(
                        max_stat_dist[None, :] >= np.abs(observed_stats[:, None]), axis=1
                    )
                    + 1.0
                ) / (n_permutations + 1.0)
            else:
                fwe_p = (
                    np.sum(max_stat_dist[None, :] >= observed_stats[:, None], axis=1) + 1.0
                ) / (n_permutations + 1.0)
        unc_p, fdr_p, fwe_p = process_p_values(
            (unc_p, fdr_p, fwe_p),
            save_1minusp=save_1minusp,
            save_neglog10p=save_neglog10p,
        )
        results[f"max_stat_dist_{label}"] = max_stat_dist
        results[f"stat_uncp_{label}"] = unc_p
        results[f"stat_fdrp_{label}"] = fdr_p
        results[f"stat_fwep_{label}"] = fwe_p
        save_fn_wrapper(results)

    if correct_across_contrasts:
        # Recover global uncp matrix from all contrasts
        global_uncp_matrix = np.vstack(
            [results[f"stat_uncp_{lbl}"] for lbl in contrast_labels]
        )
        flat_uncp = global_uncp_matrix.ravel()
        flat_uncp = np.array(
            reverse_process_p_values(flat_uncp, save_1minusp, save_neglog10p)
        )

        # Recover all max stat distributions from all contrasts
        global_max_stat_dist = np.vstack(
            [results[f"max_stat_dist_{label}"] for label in contrast_labels]
        )
        if two_tailed:
            global_max_stat_dist = np.max(np.abs(global_max_stat_dist), axis=0)
        else:
            global_max_stat_dist = np.max(global_max_stat_dist, axis=0)
        results["global_max_stat_dist"] = global_max_stat_dist

        # Correct using FDR (Benjamini-Hochberg)
        _, global_fdrp_vector = fdrcorrection(
            np.where(flat_uncp == 1.0 / (n_permutations + 1.0), 0, flat_uncp), alpha=0.05, method="indep", is_sorted=False
        )

        global_fdrp_matrix = global_fdrp_vector.reshape(global_uncp_matrix.shape)

        # Iterate over each contrast to compute cfdrp and cfwep
        for i, contrast_label in enumerate(contrast_labels):
            observed_values = results[f"stat_{contrast_label}"]
            if accel_tail:
                cfwe_p = compute_p_values_accel_tail(
                    observed_values, global_max_stat_dist, two_tailed=two_tailed
                )
            else:
                if two_tailed:
                    cfwe_p = (
                        np.sum(
                            global_max_stat_dist[None, :]
                            >= np.abs(observed_values[:, None]),
                            axis=1,
                        )
                        + 1.0
                    ) / (n_permutations + 1.0)
                else:
                    cfwe_p = (
                        np.sum(
                            global_max_stat_dist[None, :] >= observed_values[:, None],
                            axis=1,
                        )
                        + 1.0
                    ) / (n_permutations + 1.0)

            # Store corrected p-values
            cfdr_p = global_fdrp_matrix[i, :]
            cfdr_p, cfwe_p = process_p_values(
                (cfdr_p, cfwe_p),
                save_1minusp=save_1minusp,
                save_neglog10p=save_neglog10p,
            )
            results[f"stat_cfdrp_{contrast_label}"] = cfdr_p
            results[f"stat_cfwep_{contrast_label}"] = cfwe_p
            save_fn_wrapper(results)

    results = results_saver.finalize_results(results)

    if not results:
        raise RuntimeError("No results were generated. Check settings and inputs.")

    return results


def permutation_analysis_nifti(
    imgs,
    design,
    contrast,
    output_prefix=None,
    f_contrast_indices=None,
    two_tailed=True,
    exchangeability_matrix=None,
    vg_auto=False,
    variance_groups=None,
    within=True,
    whole=False,
    flip_signs=False,
    stat_function="auto",
    f_stat_function="auto",
    f_only=False,
    n_permutations=1000,
    accel_tail=False,
    save_1minusp=True,
    save_neglog10p=False,
    correct_across_contrasts=False,
    random_state=42,
    demean=False,
    zstat=False,
    save_fn=None,
    permute_fn=None,
    save_permutations=False,
    mask_img=None,
    tfce=False,
):
    """
    Perform dense volumetric permutation analysis on a set of 3D images or a 4D image.

    Parameters
    ----------
    imgs : str or list of str or Niimg-like
        File path(s) or Niimg-like object(s) for the input volumetric images.
        Can be a list of 3D NIfTI image paths or a single 4D NIfTI path.
    design : np.ndarray, shape (n_samples, n_features)
        Design matrix for the GLM; rows correspond to samples, columns to regressors.
    contrast : np.ndarray, shape (n_features,) or (n_contrasts, n_features)
        Contrast vector or matrix defining the hypothesis tests.
    output_prefix : str or None
        Prefix to prepend to any output files or saved maps.
    f_contrast_indices : array-like or None
        Indices or boolean mask selecting rows of `contrast` for F-tests.
    two_tailed : bool, default True
        If True, compute two-tailed p-values (using absolute statistic values).
    exchangeability_matrix : np.ndarray or None
        Defines permutation blocks. 1D shape `(n_samples,)` or 2D shape `(n_samples, n_groups)`.
    vg_auto : bool, default False
        If True, derive variance-group labels automatically from `exchangeability_matrix`.
    variance_groups : np.ndarray or None
        Explicit variance-group labels per sample; overrides `vg_auto`.
    within : bool, default True
        When using a 1D exchangeability vector, permute within each block.
    whole : bool, default False
        When using a 1D exchangeability vector, permute whole blocks.
    flip_signs : bool, default False
        If True, also randomly flip residual signs (assumes symmetric errors).
    stat_function : callable or 'auto', default 'auto'
        Function to compute per-contrast test statistics (e.g., t-values).  
        Signature: `stat, df1, df2 = stat_function(data, design, contrast[, variance_groups, n_groups])`
    f_stat_function : callable or 'auto', default 'auto'
        Function to compute F-statistics. Same signature as `stat_function`.
    f_only : bool, default False
        If True, skip individual contrast tests and run only the F-test(s).
    n_permutations : int, default 1000
        Number of permutations for null distribution.
    accel_tail : bool, default False
        If True, apply GPD tail acceleration for FWE p-values when exceedances are low.
    save_1minusp : bool, default True
        If True, store 1–p rather than raw p-values.
    save_neglog10p : bool, default False
        If True, store –log₁₀(p) rather than raw p-values.
    correct_across_contrasts : bool, default False
        If True, apply FWE correction jointly across all contrasts.
    random_state : int, default 42
        Seed for reproducibility.
    demean : bool, default False
        If True, demean the data before computing statistics.
    zstat : bool, default False
        If True, convert t-statistics to z-scores before p-value computation.
    save_fn : callable or None
        If provided, called as `save_fn(results, key)` whenever a result is added.
    permute_fn : callable or None
        If provided, called each permutation as  
        `permute_fn(permuted_stats, perm_index, contrast_index, is_two_tailed)`.
    save_permutations : bool, default False
        If True, retain and return all permuted statistic arrays.
    mask_img : str or Niimg-like or None
        File path or Niimg-like mask to apply to `imgs` before analysis.
    tfce : bool, default False
        If True, apply TFCE enhancement to test statistics instead of voxelwise stat.

    Returns
    -------
    results : sklearn.utils.Bunch
        A Bunch containing observed statistics, uncorrected/FDR/FWE p-values,
        max-stat distributions, and (optionally) saved permutations,
        following the same structure as `permutation_analysis()`.
    """
    # Step One: Load volumetric images into a 2d matrix (n_samples x n_voxels)
    if mask_img is None:
        print(
            "Warning: No mask image provided. Using the whole image. Unexpected results may occur."
        )
        masker = NiftiMasker()
    else:
        masker = NiftiMasker(mask_img=mask_img).fit()
    if not isinstance(imgs, np.ndarray):
        data = masker.fit_transform(imgs)
    else:
        data = imgs
    mask_img = masker.mask_img_
    n_t_contrasts = np.atleast_2d(contrast).shape[0]

    if tfce:
        tfce_manager = TfceStatsManager(
            f_contrast_indices=f_contrast_indices,
            two_tailed=two_tailed,
            n_t_contrasts=n_t_contrasts,
            f_only=f_only,
            n_permutations=n_permutations,
            accel_tail=accel_tail,
            save_1minusp=save_1minusp,
            save_neglog10p=save_neglog10p,
            correct_across_contrasts=correct_across_contrasts,
            mask_img=load_nifti_if_not_already_nifti(mask_img),
        )
        tfce_result_saver = ResultSaver(
            output_prefix=output_prefix,
            variance_groups=(vg_auto or variance_groups is not None),
            stat_function=stat_function,
            f_stat_function=f_stat_function,
            zstat=zstat,
            save_permutations=save_permutations,
            mask_img=mask_img,
            n_t_contrasts=n_t_contrasts,
        )

    def save_fn_wrapper(results):
        """Handle updating the tfce manager with the ground truth stats."""
        if tfce:
            stat_c_keys = {f"stat_c{idx}" for idx in range(1, n_t_contrasts + 1)}
            stat_keys = stat_c_keys | {"stat_f"}

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
                if tfce_key in tfce_manager.observed_stats_tfce:
                    continue

                try:
                    tfce_manager.process_observed_stats(value, contrast_idx)
                except Exception as e:
                    print(
                        f"Error processing TFCE for {key!r}: {type(e).__name__}: {e.args}"
                    )

        # Handle the actual saving of results
        if save_fn is not None:
            save_fn(results)

    def permute_fn_wrapper(
        permuted_stats, permutation_idx, contrast_idx, two_tailed, *args, **kwargs
    ):
        """Handle updating the tfce manager with the permuted stats."""
        if tfce:
            tfce_manager.update(permuted_stats, permutation_idx, contrast_idx)
        if permute_fn is not None:
            permute_fn(
                permuted_stats,
                permutation_idx,
                contrast_idx,
                two_tailed,
                *args,
                **kwargs,
            )

    results = permutation_analysis(
        data=data,
        design=design,
        contrast=contrast,
        output_prefix=output_prefix,
        f_contrast_indices=f_contrast_indices,
        two_tailed=two_tailed,
        exchangeability_matrix=exchangeability_matrix,
        vg_auto=vg_auto,
        variance_groups=variance_groups,
        within=within,
        whole=whole,
        flip_signs=flip_signs,
        stat_function=stat_function,
        f_stat_function=f_stat_function,
        f_only=f_only,
        n_permutations=n_permutations,
        accel_tail=accel_tail,
        save_1minusp=save_1minusp,
        save_neglog10p=save_neglog10p,
        correct_across_contrasts=correct_across_contrasts,
        random_state=random_state,
        demean=demean,
        zstat=zstat,
        save_fn=save_fn_wrapper,
        permute_fn=permute_fn_wrapper,
        save_permutations=save_permutations,
        mask_img=mask_img,
    )

    if tfce:
        tfce_results = tfce_manager.finalize()
        tfce_result_saver.save_results(tfce_results)
        tfce_results = tfce_result_saver.finalize_results(tfce_results)
        save_fn_wrapper(tfce_results)
        results.update(tfce_results)

    return results

class TfceStatsManager:
    def __init__(
        self,
        f_contrast_indices=None,
        two_tailed=True,
        n_t_contrasts=1,
        f_only=False,
        n_permutations=1000,
        accel_tail=False,
        save_1minusp=False,
        save_neglog10p=False,
        correct_across_contrasts=False,
        mask_img=None,
    ):
        self.f_contrast_indices = f_contrast_indices
        self.two_tailed = two_tailed
        self.n_t_contrasts = n_t_contrasts
        self.f_only = f_only
        self.n_permutations = n_permutations
        self.accel_tail = accel_tail
        self.save_1minusp = save_1minusp
        self.save_neglog10p = save_neglog10p
        self.correct_across_contrasts = correct_across_contrasts

        self.mask_img = mask_img
        self.masker = NiftiMasker(mask_img).fit() if mask_img is not None else None

        self.exceedances_tfce = Bunch()
        self.max_stat_dist_tfce = Bunch()
        self.observed_stats_tfce = Bunch()
        self.results = Bunch()

    def process_observed_stats(self, observed_stats, contrast_idx):
        if contrast_idx is None or contrast_idx == -1:
            contrast_label = "f"
        else:
            contrast_label = f"c{contrast_idx + 1}"

        # Compute the TFCE-transformed statistics
        observed_stats_tfce = apply_tfce(
            self.masker.inverse_transform(observed_stats), two_tailed=self.two_tailed
        ).get_fdata()
        observed_stats_tfce = observed_stats_tfce[self.mask_img.get_fdata() != 0]
        self.results[f"tfce_stat_{contrast_label}"] = observed_stats_tfce
        self.observed_stats_tfce[f"tfce_stat_{contrast_label}"] = observed_stats_tfce

    def update(self, permuted_stats, permutation_idx, contrast_idx=None):
        # Determine the contrast label
        if contrast_idx is None or contrast_idx == -1:
            contrast_label = "f"
        else:
            contrast_label = f"c{contrast_idx + 1}"

        # Transform the permuted stats with TFCE
        permuted_stats_tfce = apply_tfce(
            self.masker.inverse_transform(permuted_stats), two_tailed=self.two_tailed
        ).get_fdata()
        permuted_stats_tfce = permuted_stats_tfce[self.mask_img.get_fdata() != 0]

        # On the first iteration, initialize state arrays/scalars
        if permutation_idx == 0:
            self.exceedances_tfce[f"tfce_stat_{contrast_label}"] = np.zeros_like(
                permuted_stats_tfce
            )
            if self.two_tailed:
                self.exceedances_tfce[f"tfce_stat_{contrast_label}"] += np.abs(
                    permuted_stats_tfce
                ) >= np.abs(self.observed_stats_tfce[f"tfce_stat_{contrast_label}"])
                self.max_stat_dist_tfce[f"tfce_stat_{contrast_label}"] = np.max(
                    np.abs(permuted_stats_tfce)
                )
            else:
                self.exceedances_tfce[f"tfce_stat_{contrast_label}"] += (
                    permuted_stats_tfce
                    >= self.observed_stats_tfce[f"tfce_stat_{contrast_label}"]
                )
                self.max_stat_dist_tfce[f"tfce_stat_{contrast_label}"] = np.max(
                    permuted_stats_tfce
                )

        # If not the first iteration, update the exceedances and max stat distribution
        else:
            if self.two_tailed:
                # Update exceedances: count where abs(permuted) >= abs(true)
                self.exceedances_tfce[f"tfce_stat_{contrast_label}"] += np.abs(
                    permuted_stats_tfce
                ) >= np.abs(self.observed_stats_tfce[f"tfce_stat_{contrast_label}"])
                # Concatenate the new max value using pd.concat equivalent (if arrays) or np.hstack
                self.max_stat_dist_tfce[f"tfce_stat_{contrast_label}"] = np.hstack(
                    [
                        np.max(np.abs(permuted_stats_tfce)),
                        self.max_stat_dist_tfce[f"tfce_stat_{contrast_label}"],
                    ]
                )
            else:
                # Update exceedances: count where permuted >= true
                self.exceedances_tfce[f"tfce_stat_{contrast_label}"] += (
                    permuted_stats_tfce
                    >= self.observed_stats_tfce[f"tfce_stat_{contrast_label}"]
                )
                # Concatenate the new max value using pd.concat equivalent (if arrays) or np.hstack
                self.max_stat_dist_tfce[f"tfce_stat_{contrast_label}"] = np.hstack(
                    [
                        np.max(permuted_stats_tfce),
                        self.max_stat_dist_tfce[f"tfce_stat_{contrast_label}"],
                    ]
                )

    def finalize(self):
        # Determine all valid contrast labels (c1…cN and f if present)
        contrast_labels = [f"c{i+1}" for i in range(self.n_t_contrasts)]
        if "tfce_stat_f" in self.observed_stats_tfce:
            contrast_labels.append("f")
        valid_labels = [
            lbl
            for lbl in contrast_labels
            if f"tfce_stat_{lbl}" in self.exceedances_tfce
        ]

        # Per‐contrast p-value processing
        for lbl in valid_labels:
            key = f"tfce_stat_{lbl}"
            stat = self.observed_stats_tfce[key]
            max_dist = self.max_stat_dist_tfce[key]
            exc = self.exceedances_tfce[key]

            # 1. Uncorrected
            uncp = (exc + 1) / (self.n_permutations + 1)
            # 2. FDR
            _, fdrp = fdrcorrection(np.where(uncp == 1.0 / (self.n_permutations +1), 0, uncp), alpha=0.05, method="indep", is_sorted=False)
            # 3. FWE
            if self.accel_tail:
                fwep = compute_p_values_accel_tail(
                    stat, max_dist, two_tailed=self.two_tailed
                )
            else:
                cmp_arr = np.abs(stat) if self.two_tailed else stat
                # broadcast to (n_samples, n_permutations)
                cmp_arr = cmp_arr[:, None]
                fwep = (np.sum(max_dist[None, :] >= cmp_arr, axis=1) + 1) / (
                    self.n_permutations + 1
                )

            # Store max‐stat distribution
            self.results[f"tfce_max_stat_dist_{lbl}"] = max_dist

            # Apply any 1-minus-p or –log10 transforms
            uncp, fdrp, fwep = process_p_values(
                (uncp, fdrp, fwep), self.save_1minusp, self.save_neglog10p
            )
            self.results[f"tfce_stat_uncp_{lbl}"] = uncp
            self.results[f"tfce_stat_fdrp_{lbl}"] = fdrp
            self.results[f"tfce_stat_fwep_{lbl}"] = fwep
            self.results[f"tfce_max_stat_dist_{lbl}"] = max_dist

        # Global correction across contrasts (if requested)
        if self.correct_across_contrasts:
            # Recover raw uncp matrix
            uncp_matrix = np.vstack(
                [self.results[f"tfce_stat_uncp_{lbl}"] for lbl in valid_labels]
            )
            flat_uncp = uncp_matrix.ravel()
            flat_uncp = np.array(
                reverse_process_p_values(
                    flat_uncp, self.save_1minusp, self.save_neglog10p
                )
            )
            _, flat_fdr = fdrcorrection(
                np.where(flat_uncp==1.0/(self.n_permutations + 1), 0, flat_uncp), alpha=0.05, method="indep", is_sorted=False
            )
            fdr_matrix = flat_fdr.reshape(uncp_matrix.shape)

            # Build global max‐stat distribution
            dist_stack = [
                self.max_stat_dist_tfce[f"tfce_stat_{lbl}"] for lbl in valid_labels
            ]
            stack = np.stack(dist_stack, axis=0)
            if self.two_tailed:
                global_max = np.max(np.abs(stack), axis=0)
            else:
                global_max = np.max(stack, axis=0)

            # Compute and store per‐contrast cfdrp & cfwep
            for idx, lbl in enumerate(valid_labels):
                key = f"tfce_stat_{lbl}"
                stat = self.observed_stats_tfce[key]

                # cfdrp from FDR matrix
                cfdrp = fdr_matrix[idx]

                # cfwep
                if self.accel_tail:
                    cfwep = compute_p_values_accel_tail(
                        stat, global_max, two_tailed=self.two_tailed
                    )
                else:
                    cmp_arr = np.abs(stat) if self.two_tailed else stat
                    cmp_arr = cmp_arr[:, None]
                    if self.two_tailed:
                        cfwep = (
                            np.sum(global_max[None, :] >= np.abs(cmp_arr), axis=1) + 1
                        ) / (self.n_permutations + 1)
                    else:
                        cfwep = (np.sum(global_max[None, :] >= cmp_arr, axis=1) + 1) / (
                            self.n_permutations + 1
                        )

                cfwep, cfdrp = process_p_values(
                    (cfwep, cfdrp), self.save_1minusp, self.save_neglog10p
                )
                self.results[f"tfce_stat_cfwep_{lbl}"] = cfwep
                self.results[f"tfce_stat_cfdrp_{lbl}"] = cfdrp

            # Keep the global distribution for later inspection
            self.results["tfce_global_max_stat_dist"] = global_max

        return self.results
    

def yield_permuted_stats(
    data,
    design,
    contrast,
    stat_function,
    n_permutations=1000,
    random_state=42,
    exchangeability_matrix=None,
    vg_auto=False,
    variance_groups=None,
    within=True,
    whole=False,
    flip_signs=False,
):
    """
    Generator for Freedman–Lane permutation test statistics
    using Beckmann partitioning of the GLM.

    This implements:
      1. **Beckmann’s method** to split the design into
         - regressors_of_interest (X)
         - nuisance_regressors (Z)
      2. **Freedman–Lane’s permutation strategy**:
         a. Fit the full model to get residuals wrt nuisance regressors
         b. Permute those residuals (and/or flip signs)
         c. Add permuted residuals back onto the fitted nuisance part
         d. Recompute the test statistic on this permuted dataset
         e. Repeat to build the null distribution

    Args:
        data (array, shape (n_samples, n_elements)):
            Observations to test.
        design (array, shape (n_samples, n_features)):
            Full design matrix.
        contrast (array, shape (n_features,) or (n_contrasts, n_features)):
            Contrast vector(s).
        stat_function (callable):
            Function to compute the test statistic. Signature:
                stat, df1, df2 = stat_function(data, design, contrast[, variance_groups, n_groups])
        n_permutations (int):
            Number of permutations.
        random_state (int):
            Seed for reproducibility.
        exchangeability_matrix (array, optional):
            Defines exchangeable blocks.
        vg_auto (bool):
            If True, derive variance_groups automatically from exchangeability_matrix.
        variance_groups (array, optional):
            Precomputed variance-group labels.
        within (bool):
            Permute within blocks when using 1D exchangeability.
        whole (bool):
            Permute whole blocks when using 1D exchangeability.
        flip_signs (bool):
            Also generate sign-flipped residuals.

    Yields:
        permuted_stat (array):
            The permuted statistic map for each iteration.
    """
    # ─── Prepare design and residuals ──────────────────────────────────────────────
    X_interest, Z_nuisance, C_eff, _ = partition_model(design, contrast)
    full_design = np.hstack([X_interest, Z_nuisance])

    base_residuals, fitted_values = residualize_data(data, Z_nuisance)

    # ─── Initialize permutation and sign-flip generators ─────────────────────────
    perm_index_gen = yield_permuted_indices(
        design=full_design,
        n_permutations=n_permutations,
        contrast=contrast,
        exchangeability_matrix=exchangeability_matrix,
        within=within,
        whole=whole,
        random_state=random_state,
    )

    if exchangeability_matrix is not None and vg_auto and variance_groups is None:
        variance_groups = get_vg_vector(exchangeability_matrix, within=within, whole=whole)

    if flip_signs:
        sign_flip_gen = yield_sign_flipped_data(
            base_residuals,
            n_permutations,
            random_state + 1,
            group_ids=variance_groups,
            whole=whole,
        )

    # ─── Permutation loop ───────────────────────────────────────────────────────
    for _ in range(n_permutations):
        # 1. Next permutation of residual indices
        perm_indices = next(perm_index_gen)

        # 2. Optionally flip residual signs
        current_resid = next(sign_flip_gen) if flip_signs else base_residuals

        # 3. Build permuted dataset under H0
        permuted_data = apply_freedman_lane_permutation(
            current_resid, fitted_values, perm_indices
        )

        # 4. Compute statistic, with vg info if provided
        if variance_groups is not None:
            n_groups = len(np.unique(variance_groups))
            stat, df1, df2 = stat_function(
                permuted_data, full_design, C_eff, variance_groups, n_groups
            )
        else:
            stat, df1, df2 = stat_function(permuted_data, full_design, C_eff)

        yield stat


@jit
def flip_data_rows(data, key):
    n = data.shape[0]
    signs = random.randint(key, (n,), 0, 2) * 2 - 1
    return data * signs[:, None]


@jit
def flip_data_groups(data, key, group_ids):
    # group_ids: shape (n_samples,), int IDs for each row’s variance group
    unique_ids, inv = jnp.unique(group_ids, return_inverse=True)
    n_groups = unique_ids.shape[0]
    group_signs = random.randint(key, (n_groups,), 0, 2) * 2 - 1
    signs = group_signs[inv]  # map each row → its group’s sign
    return data * signs[:, None]


def yield_sign_flipped_data(
    data, n_permutations, random_state, group_ids=None, whole=False
):
    """
    Generator yielding sign-flipped versions of `data`.

    If `group_ids` is provided and `whole=True`, then each unique group in
    `group_ids` gets one random ±1 flip and all rows with that ID are flipped together.
    Otherwise flips each row independently as before.
    """
    key = random.PRNGKey(random_state)
    if group_ids is not None:
        group_ids = jnp.asarray(group_ids)

    for _ in range(n_permutations):
        key, subkey = random.split(key)
        if whole and group_ids is not None:
            yield flip_data_groups(data, subkey, group_ids)
        else:
            yield flip_data_rows(data, subkey)


@jit
def apply_freedman_lane_permutation(residuals, fitted_values, permutation_order):
    """Apply Freedman–Lane: permute residuals, then add back the fit."""
    permuted_resid = residuals[permutation_order, :]
    return permuted_resid + fitted_values


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

    with np.errstate(divide="ignore", invalid="ignore"):
        if abs(shape) < np.finfo(float).eps:
            p = np.exp(-exceedances / scale)
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
    n_mc_samples=5000,
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
                n_mc_samples=n_mc_samples,
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
                    empirical_p[mask], np.maximum(emp_tail_p, np.finfo(float).tiny)
                )
            print("Succesfully refined tail p-values.", flush=True)

            break

        except Exception:
            continue

    return empirical_p


def process_p_values(p_values, save_1minusp=False, save_neglog10p=False):
    if save_1minusp and save_neglog10p:
        raise ValueError("Only one of save_1minusp/save_neglog10p may be True")

    def _neglog(p):
        with np.errstate(divide="ignore"):
            out = -np.log10(p)
        if np.any(np.isnan(out)):
            warnings.warn("NaN produced by -log10(p)", RuntimeWarning)
        return out

    # choose transformation
    fn = (
        (lambda p: 1 - p)
        if save_1minusp
        else _neglog if save_neglog10p else (lambda p: p)
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
        with np.errstate(over="ignore"):
            return 10 ** (-x)

    # pick inverse transform
    fn = (
        (lambda x: 1 - x)
        if save_1minusp
        else _inv_neglog if save_neglog10p else (lambda x: x)
    )

    # unify to tuple
    is_single = not isinstance(p_values, (list, tuple))
    inputs = (p_values,) if is_single else p_values

    # apply and restore shape/structure
    results = tuple(fn(np.asarray(p)) for p in inputs)
    return results[0] if is_single else results


def select_stat_functions(
    stat_function="auto",
    f_stat_function="auto",
    use_variance_groups=False,
    zstat=False,
    perform_f_test=True,
):
    """
    Return (actual_t_stat_fn, actual_f_stat_fn) based on:
      - stat_function ∈ {"auto", "pearson"} or a callable
      - f_stat_function ∈ {"auto", "pearson"} or a callable
      - use_variance_groups, zstat, perform_f_test
    """
    # ─── pick the “t‐like” function ─────────────────────────────────
    if stat_function == "auto":
        if not zstat:
            actual_t = aspin_welch_v if use_variance_groups else t
        else:
            actual_t = aspin_welch_v_z if use_variance_groups else t_z
    elif stat_function == "pearson":
        actual_t = fisher_z if zstat else pearson_r
    elif callable(stat_function):
        actual_t = stat_function
    else:
        raise ValueError("stat_function must be 'auto', 'pearson', or a callable")

    # ─── pick the “F‐like” function ─────────────────────────────────
    if not perform_f_test:
        actual_f = None
    elif f_stat_function == "auto":
        if not zstat:
            actual_f = G if use_variance_groups else F
        else:
            actual_f = G_z if use_variance_groups else F_z
    elif f_stat_function == "pearson":
        actual_f = r_squared_z if zstat else r_squared
    elif callable(f_stat_function):
        actual_f = f_stat_function
    else:
        raise ValueError("f_stat_function must be 'auto', 'pearson', or a callable")

    return actual_t, actual_f
