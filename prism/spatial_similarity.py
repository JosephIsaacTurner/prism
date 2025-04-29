from .data_wrangling import is_nifti_like, load_data, load_nifti_if_not_already_nifti
from .datasets import Dataset
from .permutation_inference import yield_permuted_stats, compute_p_values_accel_tail
from .permutation_logic import get_vg_vector
import nibabel as nib
import numpy as np
from typing import List, Optional, Union, Callable, Any, Generator
from sklearn.utils import Bunch
from nilearn.maskers import NiftiMasker
from tqdm import tqdm
import warnings


def spatial_similarity_permutation_analysis(
    datasets: Union[Dataset, List[Dataset]],
    reference_maps: Optional[
        Union[
            str,
            nib.Nifti1Image,
            np.ndarray,
            List[Union[str, nib.Nifti1Image, np.ndarray]],
        ]
    ] = None,
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
    analyzer = _SpatialCorrelationAnalysis(
        datasets, reference_maps, two_tailed, compare_func
    )
    results = analyzer.run_analysis()
    return results


class _SpatialCorrelationAnalysis:
    """
    Manages spatial correlation analysis between datasets and reference maps.

    Calculates correlations/similarities and performs permutation testing.
    Supports standard Pearson correlation or a custom comparison function.
    """

    def __init__(
        self,
        datasets_input: Union["Dataset", List["Dataset"]],
        reference_maps_input: Optional[
            Union[
                str,
                nib.Nifti1Image,
                np.ndarray,
                List[Union[str, nib.Nifti1Image, np.ndarray]],
            ]
        ],
        two_tailed: bool,
        comparison_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ):
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
        self.datasets: List["Dataset"] = []
        self.final_reference_maps: List[np.ndarray] = []  # 1D arrays
        self.n_datasets: int = 0
        self.n_references: int = 0
        self.n_permutations: int = 0
        self.target_feature_shape: Optional[int] = None
        self.common_masker: Optional[NiftiMasker] = None

        # Results storage
        self.true_stats_list: List[np.ndarray] = []  # 1D stat maps
        self.true_corr_ds_ds: Optional[np.ndarray] = None
        self.true_corr_ds_ref: Optional[np.ndarray] = None
        self.permuted_corrs_ds_ds: Optional[np.ndarray] = None  # (n_perm, n_ds, n_ds)
        self.permuted_corrs_ds_ref: Optional[np.ndarray] = None  # (n_perm, n_ds, n_ref)

    def _setup_and_validate(self) -> bool:
        """Loads data, standardizes inputs, handles masking, validates shapes."""
        # 1. Standardize Datasets
        self.datasets = (
            self.datasets_input
            if isinstance(self.datasets_input, list)
            else [self.datasets_input]
        )
        if not self.datasets:
            raise ValueError("Dataset list cannot be empty.")
        for i, ds in enumerate(self.datasets):
            if not isinstance(ds, Dataset):
                raise TypeError(f"Item {i} is not a Dataset object.")
        self.n_datasets = len(self.datasets)

        # 2. Standardize Reference Maps
        ref_maps_list_raw: List[Any] = []
        if self.reference_maps_input is not None:
            ref_maps_list_raw = (
                self.reference_maps_input
                if isinstance(self.reference_maps_input, list)
                else [self.reference_maps_input]
            )
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
                ds.select_stat_functions()
                if ds.data is None:
                    raise RuntimeError("Dataset.data is None after loading.")
            except Exception as e:
                raise RuntimeError(f"Failed loading data for dataset {i+1}: {e}") from e

        # 5. Prepare Masker & Process References
        self._prepare_common_masker()
        self._process_reference_maps(ref_maps_list_raw)

        # 6. Validate Feature Shapes
        if not self._validate_shapes():
            return False

        # 7. Determine Number of Permutations
        if any(ds.n_permutations <= 0 for ds in self.datasets):
            warnings.warn("n_permutations <= 0 found. Permutation testing skipped.")
            self.n_permutations = 0
        else:
            self.n_permutations = min(ds.n_permutations for ds in self.datasets)
            print(
                f"Running analysis with {self.n_permutations} permutations."
            )  # Keep one informative print

        return True

    def _prepare_common_masker(self):
        """Determines and stores a common NiftiMasker if multiple NIfTI datasets exist."""
        nifti_datasets = [ds for ds in self.datasets if ds.is_nifti]
        self.common_masker = None
        if len(nifti_datasets) > 1:
            first_masker = next(
                (ds.masker for ds in nifti_datasets if ds.masker is not None), None
            )
            if first_masker:
                self.common_masker = first_masker
            else:
                warnings.warn(
                    "Multiple NIfTI datasets found, but no common masker identified. Ensure consistency."
                )
        elif len(nifti_datasets) == 1:
            self.common_masker = nifti_datasets[
                0
            ].masker  # Use single NIfTI dataset's masker for refs

    def _process_reference_maps(self, ref_maps_list_raw: List):
        """Loads, masks (if NIfTI & common_masker exists), and flattens reference maps."""
        self.final_reference_maps = []
        if not ref_maps_list_raw:
            return

        for i, ref_map_input in enumerate(ref_maps_list_raw):
            ref_map_data: Optional[np.ndarray] = None
            try:
                loaded_ref = load_data(
                    ref_map_input
                )  # Assumes load_data handles paths/objects

                if is_nifti_like(
                    loaded_ref
                ):  # Assumes is_nifti_like checks paths/objects
                    ref_img = load_nifti_if_not_already_nifti(
                        loaded_ref
                    )  # Assumes this loads/returns Nifti1Image
                    if self.common_masker:
                        if (
                            not hasattr(self.common_masker, "mask_img_")
                            or self.common_masker.mask_img_ is None
                        ):
                            warnings.warn(
                                f"Common masker for ref map {i+1} seems unfit."
                            )
                        masked_ref = self.common_masker.transform(ref_img)
                        ref_map_data = masked_ref.ravel()
                    else:
                        warnings.warn(
                            f"NIfTI ref map {i+1} processed raw (no common masker)."
                        )
                        ref_map_data = ref_img.get_fdata().ravel()
                elif isinstance(loaded_ref, np.ndarray):
                    ref_map_data = loaded_ref.ravel()
                    if self.common_masker:
                        warnings.warn(
                            f"NumPy ref map {i+1} used; ensure it matches masked space."
                        )
                else:
                    raise TypeError(
                        f"Unsupported type for ref map {i+1}: {type(loaded_ref)}"
                    )

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
                if not all(
                    [
                        first_ds.data is not None,
                        first_ds.design is not None,
                        first_ds.contrast is not None,
                        first_ds.stat_function is not None,
                    ]
                ):
                    raise ValueError(
                        "Dataset 1 missing components needed for shape validation."
                    )
                # Calculate temporary map just for shape
                stat_args = [first_ds.data, first_ds.design, first_ds.contrast]
                temp_stat_map = first_ds.stat_function(*stat_args)[0]
                self.target_feature_shape = temp_stat_map.ravel().shape[0]
            except Exception as e:
                warnings.warn(
                    f"Could not get shape from dataset 1 stat map: {e}. Trying refs."
                )

        # Fallback to first reference map if needed
        if self.target_feature_shape is None and self.final_reference_maps:
            self.target_feature_shape = self.final_reference_maps[0].shape[0]

        # Final check if shape could be determined
        if self.target_feature_shape is None:
            if (
                self.n_datasets > 0 or self.n_references > 0
            ):  # Only error if inputs existed
                raise RuntimeError(
                    "Could not determine target feature shape from any source."
                )
            else:
                return True  # No inputs, technically no mismatch

        # Validate reference maps against target shape
        for i, ref_map in enumerate(self.final_reference_maps):
            if ref_map.shape[0] != self.target_feature_shape:
                raise ValueError(
                    f"Shape mismatch: Ref map {i+1} ({ref_map.shape[0]}) != target ({self.target_feature_shape})."
                )

        return True

    def calculate_true_statistics(self):
        """Calculates the true statistic map for each dataset."""
        if self.target_feature_shape is None:
            raise RuntimeError("Target shape unknown.")
        self.true_stats_list = []

        for i, dataset in enumerate(self.datasets):
            if not all(
                [
                    dataset.data is not None,
                    dataset.design is not None,
                    dataset.contrast is not None,
                    dataset.stat_function is not None,
                ]
            ):
                raise ValueError(
                    f"Dataset {i+1} missing components for stat calculation."
                )

            if dataset.f_only:
                stat_function = dataset.f_stat_function
                contrast = dataset.contrast[dataset.f_contrast_indices.astype(bool), :] if dataset.f_contrast_indices is not None else dataset.contrast
            else:
                stat_function = dataset.stat_function
                contrast = dataset.contrast[0, :] # Use only the first contrast

            stat_args = [dataset.data, dataset.design, contrast]
            # Handle variance groups if specified
            effective_variance_groups = dataset.variance_groups
            if (
                effective_variance_groups is None
                and dataset.exchangeability_matrix is not None
                and dataset.vg_auto
            ):
                # Assumes get_variance_groups is available
                effective_variance_groups = get_vg_vector(
                    dataset.exchangeability_matrix,
                    within=dataset.within,
                    whole=dataset.whole,
                )
            if effective_variance_groups is not None:
                n_groups = len(np.unique(effective_variance_groups))
                stat_args.extend(
                    [effective_variance_groups, n_groups]
                )  # Assumes stat_func signature adapts

            # Calculate stats
            true_stats_raw = stat_function(*stat_args)[0]
            true_stats_flat = true_stats_raw.ravel()

            # Validate shape
            if true_stats_flat.shape[0] != self.target_feature_shape:
                raise ValueError(
                    f"Shape mismatch: Dataset {i+1} stat map ({true_stats_flat.shape[0]}) != target ({self.target_feature_shape})."
                )

            dataset.true_stats = true_stats_flat
            self.true_stats_list.append(true_stats_flat)

    def _compute_correlation_matrix(
        self, data1: np.ndarray, data2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Computes correlation/similarity matrix using np.corrcoef or custom function."""
        if data1.ndim == 1:
            data1 = data1[:, np.newaxis]
        if data2 is not None and data2.ndim == 1:
            data2 = data2[:, np.newaxis]
        n_items1 = data1.shape[1]
        n_items2 = data2.shape[1] if data2 is not None else 0

        if self.comparison_func:
            # Pairwise calculation using custom function
            if data2 is None:  # Self-similarity
                if n_items1 == 0:
                    return np.array([]).reshape(0, 0)
                res = np.zeros((n_items1, n_items1))
                for i in range(n_items1):
                    for j in range(i, n_items1):
                        sim = self.comparison_func(data1[:, i], data1[:, j])
                        res[i, j] = sim
                        if i != j:
                            res[j, i] = sim  # Assume symmetry
                return res
            else:  # Cross-similarity
                if n_items1 == 0 or n_items2 == 0:
                    return np.array([]).reshape(n_items1, n_items2)
                res = np.zeros((n_items1, n_items2))
                for i in range(n_items1):
                    for j in range(n_items2):
                        res[i, j] = self.comparison_func(data1[:, i], data2[:, j])
                return res
        else:
            # Default Pearson correlation using np.corrcoef
            if data2 is None:  # Self-correlation
                if n_items1 <= 1:
                    return (
                        np.array([[1.0]])
                        if n_items1 == 1
                        else np.array([]).reshape(0, 0)
                    )
                with warnings.catch_warnings():  # Suppress warnings for now, handle NaNs below
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    corr = np.corrcoef(data1, rowvar=False)
                corr = np.nan_to_num(corr, nan=0.0)  # Replace NaN with 0
                np.fill_diagonal(corr, 1.0)  # Ensure diagonal is 1
                return corr
            else:  # Cross-correlation
                if n_items1 == 0 or n_items2 == 0:
                    return np.array([]).reshape(n_items1, n_items2)
                combined = np.hstack((data1, data2))
                with warnings.catch_warnings():  # Suppress warnings
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    full_corr = np.corrcoef(combined, rowvar=False)
                full_corr = np.nan_to_num(full_corr, nan=0.0)
                # Check shape before slicing
                if full_corr.shape == (n_items1 + n_items2, n_items1 + n_items2):
                    return full_corr[:n_items1, n_items1:]
                else:  # Handle unexpected scalar or shape mismatch
                    warnings.warn(
                        f"Unexpected corrcoef shape {full_corr.shape}. Returning NaNs."
                    )
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
                self.true_corr_ds_ref = self._compute_correlation_matrix(
                    stacked_ds_stats, stacked_ref_maps
                )
            except ValueError as e:
                raise RuntimeError(f"Failed stacking reference maps: {e}") from e
        elif (
            self.n_datasets > 0 and self.n_references > 0
        ):  # Refs provided but failed processing
            self.true_corr_ds_ref = np.array([]).reshape(self.n_datasets, 0)

    def run_permutations(self):
        """Runs permutations and collects permuted correlations/similarities."""
        if self.n_permutations <= 0:
            return  # Already warned in setup
        if self.target_feature_shape is None:
            raise RuntimeError("Target shape unknown.")

        # Initialize storage
        self.permuted_corrs_ds_ds = None
        if self.n_datasets > 1:
            self.permuted_corrs_ds_ds = np.zeros(
                (self.n_permutations, self.n_datasets, self.n_datasets)
            )
        self.permuted_corrs_ds_ref = None
        if self.n_datasets > 0 and self.n_references > 0:
            self.permuted_corrs_ds_ref = np.zeros(
                (self.n_permutations, self.n_datasets, self.n_references)
            )

        # Setup generators
        for i, dataset in enumerate(self.datasets):
            if not all(
                [
                    dataset.data is not None,
                    dataset.design is not None,
                    dataset.contrast is not None,
                    dataset.stat_function is not None,
                ]
            ):
                raise ValueError(f"Dataset {i+1} missing components for permutation.")
            # Assumes yield_permuted_stats exists and handles these args

            if dataset.f_only:
                stat_function = dataset.f_stat_function
                contrast = dataset.contrast[dataset.f_contrast_indices.astype(bool), :] if dataset.f_contrast_indices is not None else dataset.contrast
            else:
                stat_function = dataset.stat_function
                contrast = dataset.contrast[0, :] # Use only the first contrast


            dataset.permuted_stat_generator = yield_permuted_stats(
                data=dataset.data,
                design=dataset.design,
                contrast=contrast,
                stat_function=stat_function,
                n_permutations=self.n_permutations,
                random_state=dataset.random_state,
                exchangeability_matrix=dataset.exchangeability_matrix,
                vg_auto=dataset.vg_auto,
                variance_groups=dataset.variance_groups,
                within=dataset.within,
                whole=dataset.whole,
                flip_signs=dataset.flip_signs,
            )
            if not isinstance(dataset.permuted_stat_generator, Generator):
                raise RuntimeError(
                    f"Failed creating permutation generator for dataset {i+1}."
                )

        # Pre-stack references if needed
        stacked_ref_maps = None
        if self.permuted_corrs_ds_ref is not None and self.final_reference_maps:
            try:
                stacked_ref_maps = np.stack(self.final_reference_maps, axis=-1)
            except ValueError as e:
                raise RuntimeError(
                    f"Failed stacking reference maps for permutations: {e}"
                ) from e

        # Permutation loop with progress bar
        permuted_stats_current = np.zeros((self.target_feature_shape, self.n_datasets))
        for perm_idx in tqdm(
            range(self.n_permutations), desc="Permutations", unit="perm", leave=False
        ):
            # Get stats for current permutation
            for i, dataset in enumerate(self.datasets):
                try:
                    perm_stat = next(dataset.permuted_stat_generator)
                    permuted_stats_current[:, i] = perm_stat.ravel()
                except StopIteration:
                    raise RuntimeError(
                        f"Perm generator ended early for dataset {i+1} at perm {perm_idx+1}."
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Error getting perm stat for dataset {i+1} at perm {perm_idx+1}: {e}"
                    ) from e

            # Compute and store permuted correlations/similarities
            if self.permuted_corrs_ds_ds is not None:
                self.permuted_corrs_ds_ds[perm_idx] = self._compute_correlation_matrix(
                    permuted_stats_current
                )
            if self.permuted_corrs_ds_ref is not None and stacked_ref_maps is not None:
                self.permuted_corrs_ds_ref[perm_idx] = self._compute_correlation_matrix(
                    permuted_stats_current, stacked_ref_maps
                )

    def _calculate_p_values_internal(
        self, true_values: Optional[np.ndarray], permuted_values: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Calculates p-values based on true and permuted values."""
        if permuted_values is None or true_values is None:
            return None  # Cannot calculate
        n_perm_actual = permuted_values.shape[0]
        if n_perm_actual == 0:
            return np.full_like(true_values, np.nan)  # No perms run

        try:
            if self.two_tailed:
                exceedances = np.sum(
                    np.abs(permuted_values) >= np.abs(true_values)[np.newaxis, :, :],
                    axis=0,
                )
            else:  # One-tailed (right)
                exceedances = np.sum(
                    permuted_values >= true_values[np.newaxis, :, :], axis=0
                )
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
        results["corr_matrix_ds_ds"] = None
        results["corr_matrix_ds_ref"] = None
        results["p_matrix_ds_ds"] = None
        results["p_matrix_ds_ref"] = None
        results["corr_matrix_perm_ds_ds"] = None
        results["corr_matrix_perm_ds_ref"] = None

        # 1. Setup & Validate
        if not self._setup_and_validate():
            return results

        # 2. True Statistics
        self.calculate_true_statistics()

        # 3. True Correlations/Similarities
        self.calculate_true_correlations()
        results["corr_matrix_ds_ds"] = self.true_corr_ds_ds
        results["corr_matrix_ds_ref"] = self.true_corr_ds_ref

        # 4. Permutations
        self.run_permutations()  # Runs only if self.n_permutations > 0
        results["corr_matrix_perm_ds_ds"] = self.permuted_corrs_ds_ds
        results["corr_matrix_perm_ds_ref"] = self.permuted_corrs_ds_ref

        # 5. P-values
        results["p_matrix_ds_ds"] = self._calculate_p_values_internal(
            results["corr_matrix_ds_ds"], results["corr_matrix_perm_ds_ds"]
        )
        results["p_matrix_ds_ref"] = self._calculate_p_values_internal(
            results["corr_matrix_ds_ref"], results["corr_matrix_perm_ds_ref"]
        )

        # Set diagonal of ds-ds p-values to NaN
        if results["p_matrix_ds_ds"] is not None and self.n_datasets > 1:
            p_matrix_ds_ds = results["p_matrix_ds_ds"].copy()  # Ensure writeable
            np.fill_diagonal(p_matrix_ds_ds, np.nan)
            results["p_matrix_ds_ds"] = p_matrix_ds_ds

        print("Analysis finished.")
        return results
