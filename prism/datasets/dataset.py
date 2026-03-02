import warnings
import os
import numpy as np
import nibabel as nib
from typing import Callable, Optional, Union
from nilearn.maskers import NiftiMasker
from prism.preprocessing import load_data, load_nifti_if_not_already_nifti, is_nifti_like
from prism.permutation_inference import permutation_analysis, permutation_analysis_nifti
from prism.permutation_logic import yield_permuted_indices
from prism.stats import t, t_z, aspin_welch_v, aspin_welch_v_z, F, F_z, G, G_z, r_squared, r_squared_z, pearson_r, fisher_z, demean_glm_data
import json
import sys

class Dataset:
    """
    Represents a single dataset for analysis, handling data loading and masking.

    Attributes:
        data (np.ndarray): Masked data matrix (samples x features).
        design (np.ndarray): Design matrix (samples x regressors).
        contrast (np.ndarray): Contrast vector or matrix.
        mask_img (nib.Nifti1Image): Mask image for NIfTI data.
        masker (NiftiMasker): Nilearn masker object.
        is_nifti (bool): Whether the input data is NIfTI-like.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        data: Union[str, np.ndarray, nib.Nifti1Image]=None,
        design: Union[str, np.ndarray]=None,
        contrast: Union[str, np.ndarray]=None,
        output_prefix: Optional[str] = None,
        f_contrast_indices: Optional[Union[str, np.ndarray]] = None,
        two_tailed: bool = True,
        exchangeability_matrix: Optional[Union[str, np.ndarray]] = None,
        vg_auto: bool = False,
        variance_groups: Optional[Union[str, np.ndarray]] = None,
        within: bool = True,
        whole: bool = False,
        flip_signs: bool = False,
        stat_function: Union[str, Callable] = "auto",
        f_stat_function: Union[str, Callable] = "auto",
        f_only: bool = False,
        n_permutations: int = 1000,
        accel_tail: bool = False,
        save_1minusp: bool = True,
        save_neglog10p: bool = False,
        correct_across_contrasts: bool = False,
        random_state: Optional[Union[int, np.random.RandomState]] = 42,
        demean: bool = False,
        zstat: bool = False,
        save_fn: Optional[Callable] = None,
        permute_fn: Optional[Callable] = None,
        save_permutations: bool = False,
        return_permutations: bool = False,
        mask_img: Optional[Union[str, nib.Nifti1Image]] = None,
        tfce: bool = False,
        quiet: bool = False,
    ):
        """
        Initializes the Dataset object with the provided parameters.

        Args:
            config_path: Path to a JSON configuration file. If provided, other arguments are ignored.
            data: Input data. Can be a path to a NIfTI, CSV, or NPY file, or a NumPy array/Nifti1Image.
            design: Design matrix. Can be a path to a CSV/NPY file or a NumPy array.
            contrast: Contrast vector or matrix. Can be a path to a CSV/NPY file or a NumPy array.
            output_prefix: Prefix for output files.
            f_contrast_indices: Boolean array or path to one, indicating which contrasts to include in an F-test.
            two_tailed: Whether to perform a two-tailed test. Defaults to True.
            exchangeability_matrix: Structure defining exchangeable blocks for permutations.
            vg_auto: Whether to automatically derive variance groups from the exchangeability matrix.
            variance_groups: Predefined variance groups for each sample.
            within: Whether to permute within blocks defined by the exchangeability matrix.
            whole: Whether to permute blocks as a whole.
            flip_signs: Whether to also perform sign-flipping permutations.
            stat_function: Function or name ("auto", "pearson") for T-like statistics.
            f_stat_function: Function or name ("auto", "pearson") for F-like statistics.
            f_only: If True, only perform F-tests.
            n_permutations: Number of permutations to perform.
            accel_tail: If True, use GPD tail-fitting for p-values.
            save_1minusp: Whether to save 1-p maps.
            save_neglog10p: Whether to save -log10(p) maps.
            correct_across_contrasts: Whether to apply FWE correction across all contrasts.
            random_state: Seed or RandomState for reproducibility.
            demean: Whether to demean data and design before analysis.
            zstat: Whether to convert statistics to z-scores.
            save_fn: Callback function called when a result is saved.
            permute_fn: Callback function called for each permutation.
            save_permutations: Whether to save all permuted statistic maps.
            return_permutations: Whether to return all permuted statistic maps in the results.
            mask_img: Mask image for NIfTI data.
            tfce: Whether to apply Threshold-Free Cluster Enhancement.
            quiet: If True, suppress progress bars and warnings.

        Raises:
            ValueError: If neither config_path nor data/design/contrast are provided.
        """
        if config_path is None and (data is None or design is None or contrast is None):
            raise ValueError(
                "Either config_path or data, design, and contrast must be provided."
            )
        if config_path is not None:
            data, design, contrast, output_prefix, f_contrast_indices, two_tailed, exchangeability_matrix, vg_auto, variance_groups, within, whole, flip_signs, stat_function, f_stat_function, f_only, n_permutations, accel_tail, save_1minusp, save_neglog10p, correct_across_contrasts, random_state, demean, zstat, save_fn, permute_fn, save_permutations, return_permutations, mask_img, tfce, quiet = self.parse_config(config_path)
        
        # ─── Store raw inputs ───────────────────────────────────────────
        self._data_input = data
        self._design_input = design
        self._contrast_input = contrast
        self._output_prefix_input = output_prefix
        self._f_contrast_indices_input = f_contrast_indices
        self._two_tailed_input = two_tailed
        self._exchangeability_matrix_input = exchangeability_matrix
        self._vg_auto_input = vg_auto
        self._variance_groups_input = variance_groups
        self._within_input = within
        self._whole_input = whole
        self._flip_signs_input = flip_signs
        self._stat_function_input = stat_function
        self._f_stat_function_input = f_stat_function
        self._f_only_input = f_only
        self._n_permutations_input = n_permutations
        self._accel_tail_input = accel_tail
        self._save_1minusp_input = save_1minusp
        self._save_neglog10p_input = save_neglog10p
        self._correct_across_contrasts_input = correct_across_contrasts
        self._random_state_input = random_state
        self._demean_input = demean
        self._zstat_input = zstat
        self._save_fn_input = save_fn
        self._permute_fn_input = permute_fn
        self._save_permutations_input = save_permutations
        self._return_permutations_input = return_permutations
        self._mask_img_input = mask_img
        self._tfce_input = tfce
        self._quiet_input = quiet

        # ─── Core analysis parameters ──────────────────────────────────
        self.stat_function = stat_function
        self.f_stat_function = f_stat_function
        self.two_tailed = two_tailed
        self.f_only = f_only
        self.n_permutations = n_permutations
        self.accel_tail = accel_tail
        self.demean = demean
        self.zstat = zstat
        self.quiet = quiet

        # ensure reproducible RNG
        if random_state is None or isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = 42

        # ─── Permutation control flags ─────────────────────────────────
        self.exchangeability_matrix = None
        self.vg_auto = vg_auto
        self.variance_groups = None
        self.within = within
        self.whole = whole
        self.flip_signs = flip_signs
        self.tfce = tfce

        # ─── Loaded data/state (initialized empty) ────────────────────
        self.data = None
        self.design = None
        self.contrast = None
        self.mask_img = None
        self.masker = None
        self.is_nifti = False

        # ─── Results storage placeholders ──────────────────────────────
        self.true_stats = None
        self.permuted_stat_generator = None

        # ─── Saving flags & callbacks ──────────────────────────────────
        self.save_1minusp = save_1minusp
        self.save_neglog10p = save_neglog10p
        self.correct_across_contrasts = correct_across_contrasts
        self.save_permutations = save_permutations
        self.return_permutations = return_permutations
        self.save_fn = save_fn
        self.permute_fn = permute_fn
        self.output_prefix = output_prefix
        self.f_contrast_indices = f_contrast_indices

    def load_data(self):
        """
        Loads data, design, contrast, and optional arrays from inputs.

        Raises:
            TypeError: If loaded data is not a NumPy array or NIfTI image.
            ValueError: If there's a shape mismatch between data, design, or contrast.
        """
        # --- Load Main Data (and handle masking if NIfTI) ---
        loaded_data = load_data(self._data_input)  # Assumes load_data utility exists
        self.is_nifti = is_nifti_like(
            loaded_data
        )  # Assumes is_nifti_like utility exists

        if self.is_nifti:
            if self._mask_img_input and self.mask_img is None:
                self.mask_img = load_nifti_if_not_already_nifti(self._mask_img_input)
                self.masker = NiftiMasker(mask_img=self.mask_img)
            elif self.mask_img is not None:
                self.masker = NiftiMasker(mask_img=self.mask_img)
            else:
                if not self.quiet:
                    warnings.warn(
                        "NIfTI data provided without mask; using NiftiMasker for auto-masking."
                    )
                self.masker = NiftiMasker()  # Default strategy
            # Fit masker (if needed) and transform data
            self.data = self.masker.fit_transform(loaded_data)
            if self.mask_img is None:
                self.mask_img = self.masker.mask_img_  # Store auto-generated mask
            # Ensure data is 2D (samples x features)
            if self.data.ndim == 1:
                self.data = self.data[:, np.newaxis]
            elif self.data.ndim != 2:
                raise ValueError(
                    f"Masked NIfTI data shape {self.data.shape} unexpected."
                )
        else:
            # Handle non-NIfTI data (must be ndarray)
            if not isinstance(loaded_data, np.ndarray):
                raise TypeError(
                    f"Loaded data is not NumPy array or NIfTI (type: {type(loaded_data)})."
                )
            self.data = loaded_data
            self.masker = None
            self.mask_img = None
            if self._mask_img_input:
                if not self.quiet:
                    warnings.warn("Mask provided but data is not NIfTI; Interesting choice. Behavior might be unexpected.")
                self.mask_img = load_nifti_if_not_already_nifti(self._mask_img_input)
            # Ensure data is 2D (samples x features)
            if self.data.ndim == 1:
                self.data = self.data[np.newaxis, :]  # Assume 1 sample
            elif self.data.ndim != 2:
                raise ValueError(
                    f"Non-NIfTI data must be 1D or 2D; got {self.data.shape}."
                )

        # --- Load Design and Contrast ---
        self.design = load_data(self._design_input)
        self.contrast = load_data(self._contrast_input)
        if not isinstance(self.design, np.ndarray):
            raise TypeError("Design must load as NumPy array.")
        if not isinstance(self.contrast, np.ndarray):
            raise TypeError("Contrast must load as NumPy array.")
        
        # --- Load f_contrast_indices, if provided, and cast as bool array ---
        if self.f_contrast_indices is not None:
            if isinstance(self.f_contrast_indices, str):
                self.f_contrast_indices = load_data(self.f_contrast_indices)
            self.f_contrast_indices = np.ravel(np.array(self.f_contrast_indices)).astype(bool)

        # --- Load Optional Permutation Arrays ---
        if self._exchangeability_matrix_input is not None:
            self.exchangeability_matrix = load_data(self._exchangeability_matrix_input)
            if not isinstance(self.exchangeability_matrix, np.ndarray):
                raise TypeError("Exchangeability matrix must load as NumPy array.")
        if self._variance_groups_input is not None:
            self.variance_groups = load_data(self._variance_groups_input)
            if not isinstance(self.variance_groups, np.ndarray):
                raise TypeError("Variance group vector must load as NumPy array.")

        # If demean is True, demean the data
        if self.demean:
            self.data, self.design, self.contrast, f_contrast_indices = demean_glm_data(
                self.data, self.design, self.contrast
            )

        # --- Final Shape Validation ---
        n_samples = self.data.shape[0]
        n_regressors = self.design.shape[1]
        if self.design.shape[0] != n_samples:
            raise ValueError(
                f"Shape mismatch: Data samples ({n_samples}) != Design samples ({self.design.shape[0]})."
            )
        if self.contrast.ndim == 1 and self.contrast.shape[0] != n_regressors:
            raise ValueError(
                f"Shape mismatch: Contrast vector ({self.contrast.shape[0]}) != Design regressors ({n_regressors})."
            )
        elif self.contrast.ndim == 2 and self.contrast.shape[1] != n_regressors:
            raise ValueError(
                f"Shape mismatch: Contrast matrix cols ({self.contrast.shape[1]}) != Design regressors ({n_regressors})."
            )
        if (
            self.exchangeability_matrix is not None
            and self.exchangeability_matrix.shape[0] != n_samples
        ):
            raise ValueError(
                f"Shape mismatch: Exch. matrix samples ({self.exchangeability_matrix.shape[0]}) != Data samples ({n_samples})."
            )
        if self.variance_groups is not None and self.variance_groups.shape[0] != n_samples:
            raise ValueError(
                f"Shape mismatch: VG vector samples ({self.variance_groups.shape[0]}) != Data samples ({n_samples})."
            )
            
    @property
    def params(self):
        """
        Returns a dictionary of all analysis parameters.
        """
        return {
            "data": self.data,
            "design": self.design,
            "contrast": self.contrast,
            "output_prefix": self.output_prefix,
            "f_contrast_indices": self.f_contrast_indices,
            "two_tailed": self.two_tailed,
            "exchangeability_matrix": self.exchangeability_matrix,
            "vg_auto": self.vg_auto,
            "variance_groups": self.variance_groups,
            "within": self.within,
            "whole": self.whole,
            "flip_signs": self.flip_signs,
            "stat_function": self.stat_function,
            "f_stat_function": self.f_stat_function,
            "f_only": self.f_only,
            "n_permutations": self.n_permutations,
            "accel_tail": self.accel_tail,
            "save_1minusp": self.save_1minusp,
            "save_neglog10p": self.save_neglog10p,
            "correct_across_contrasts": self.correct_across_contrasts,
            "random_state": self.random_state,
            "demean": self.demean,
            "zstat": self.zstat,
            "save_fn": self.save_fn,
            "permute_fn": self.permute_fn,
            "save_permutations": self.save_permutations,
            "return_permutations": self.return_permutations,
            "mask_img": self.mask_img,
            "tfce": self.tfce,
            "quiet": self.quiet,
        }
    

    def permutation_analysis(self):
        """
        Perform permutation test on the loaded data.

        Returns:
            sklearn.utils.Bunch: Results of the permutation test, containing observed statistics,
                uncorrected p-values, and corrected p-values (FDR, FWE).
        """
        self.load_data()  # Ensure data is loaded
        params = self.params.copy()
        if self.is_nifti:
            params['imgs'] = params.pop('data') # Rename data to imgs for NIfTI
            results = permutation_analysis_nifti(
                **params
            )
        else:
            params.pop('tfce') # This key is not used in the non-volumetric case
            results = permutation_analysis(
                **params
            )
        return results
    
    def generate_permutation_indices(self):
        """
        Generates and returns the permuted indices used for each contrast.

        Returns:
            dict: A dictionary mapping contrast labels (e.g., 'c1', 'f') to
                  a numpy array of shape (n_permutations, n_samples).
        """
        self.load_data()

        # Assemble contrast labels exactly as in permutation_inference.py
        original_contrast = np.atleast_2d(self.contrast)
        n_t_contrasts = original_contrast.shape[0]

        contrast_labels = []
        if not self.f_only:
            for i in range(n_t_contrasts):
                contrast_labels.append(f"c{i+1}")

        perform_f_test = self.f_only or self.f_contrast_indices is not None
        if perform_f_test:
            contrast_labels.append("f")

        indices_results = {}
        # Base random state
        base_rs = self.random_state if self.random_state is not None else 42

        for idx, label in enumerate(contrast_labels):
            # Mirror the random state shift used in permutation_inference.py
            if isinstance(base_rs, (int, np.integer)):
                current_rs = base_rs + idx
            else:
                current_rs = base_rs

            gen = yield_permuted_indices(
                design=self.design,
                n_permutations=self.n_permutations,
                exchangeability_matrix=self.exchangeability_matrix,
                within=self.within,
                whole=self.whole,
                random_state=current_rs
            )

            # Collect all permutations for this contrast
            indices_results[label] = np.array(list(gen), dtype=int)

        # Optional saving logic
        if self.output_prefix:
            if os.path.dirname(self.output_prefix):
                os.makedirs(os.path.dirname(self.output_prefix), exist_ok=True)
            for label, indices in indices_results.items():
                save_path = f"{self.output_prefix}_permuted_indices_{label}.csv"
                np.savetxt(save_path, indices, delimiter=",", fmt='%d')
                if not self.quiet:
                    print(f"Saved permuted indices for {label} to {save_path}")

        return indices_results
    
    def save_config(self):
        """
        Saves the current configuration to a JSON file and returns the path.

        Returns:
            str: Path to the saved JSON configuration file.
        """
        input_params = {
            "data": self._data_input,
            "design": self._design_input,
            "contrast": self._contrast_input,
            "output_prefix": self.output_prefix,
            "f_contrast_indices": self._f_contrast_indices_input,
            "two_tailed": self.two_tailed,
            "exchangeability_matrix": self._exchangeability_matrix_input,
            "vg_auto": self.vg_auto,
            "variance_groups": self._variance_groups_input,
            "within": self.within,
            "whole": self.whole,
            "flip_signs": self.flip_signs,
            "stat_function": self._stat_function_input,
            "f_stat_function": self._f_stat_function_input,
            "f_only": self.f_only,
            "n_permutations": self.n_permutations,
            "accel_tail": self.accel_tail,
            "save_1minusp": self.save_1minusp,
            "save_neglog10p": self.save_neglog10p,
            "correct_across_contrasts": self.correct_across_contrasts,
            "random_state": self.random_state,
            "demean": self.demean,
            "zstat": self.zstat,
            "save_fn": self._save_fn_input,
            "permute_fn": self._permute_fn_input,
            "save_permutations": self._save_permutations_input,
            "return_permutations": self._return_permutations_input,
            "mask_img": self.mask_img,
            "tfce": self._tfce_input,
            "quiet": self._quiet_input,
            "cmd": " ".join(sys.argv),
        }

        if self.output_prefix is None:
            output_prefix = f"{os.getcwd()}/prism"
        else:
            output_prefix = self.output_prefix

        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

        # We need to iterate over input_params, and if they are ndarrays, we need to save them as 
        # headerless CSV files with the prefix output_prefix
        for key, value in input_params.items():
            if isinstance(value, np.ndarray):
                # Save as CSV without header
                csv_path = f"{output_prefix}_{key}.csv"
                np.savetxt(csv_path, value, delimiter=",", header="", comments="")
                if not self.quiet:
                    print(f"Saved {key} to {csv_path}")
                input_params[key] = csv_path  # Update to path for JSON saving

        # Save a json at the prefix config.json
        config_path = f"{output_prefix}_config.json"
        with open(config_path, "w") as f:
            json.dump(input_params, f, indent=4)
        if not self.quiet:
            print(f"Configuration saved to {config_path}")
        return config_path

    def parse_config(self, config_path: str):
        """
        Parses a configuration file and updates the Dataset instance.

        Args:
            config_path: Path to the JSON configuration file.

        Returns:
            tuple: All parameters loaded from the configuration file.
        """
        with open(config_path, "r") as f:
            params = json.load(f)
        params.pop("cmd", None)  # Remove command line args if present
        return tuple(params.values())
    
    def select_stat_functions(self):
        """
        Return (actual_t_stat_fn, actual_f_stat_fn) based on:
        - stat_function ∈ {"auto", "pearson"} or a callable
        - f_stat_function ∈ {"auto", "pearson"} or a callable
        - use_variance_groups, zstat, perform_f_test

        Returns:
            tuple: (actual_t_stat_fn, actual_f_stat_fn)
        """
        use_variance_groups = self.vg_auto or self.variance_groups is not None
        perform_f_test = self.f_only or self.f_contrast_indices is not None

        # ─── pick the “t‐like” function ─────────────────────────────────
        if self.stat_function == "auto":
            if not self.zstat:
                actual_t = aspin_welch_v if use_variance_groups else t
            else:
                actual_t = aspin_welch_v_z if use_variance_groups else t_z
        elif self.stat_function == "pearson":
            actual_t = fisher_z if self.zstat else pearson_r
        elif callable(self.stat_function):
            actual_t = self.stat_function
        else:
            raise ValueError("stat_function must be 'auto', 'pearson', or a callable")

        # ─── pick the “F‐like” function ─────────────────────────────────
        if not perform_f_test:
            actual_f = None
        elif self.f_stat_function == "auto":
            if not self.zstat:
                actual_f = G if use_variance_groups else F
            else:
                actual_f = G_z if use_variance_groups else F_z
        elif self.f_stat_function == "pearson":
            actual_f = r_squared_z if self.zstat else r_squared
        elif callable(self.f_stat_function):
            actual_f = self.f_stat_function
        else:
            raise ValueError("f_stat_function must be 'auto', 'pearson', or a callable")
        
        self.stat_function = actual_t
        self.f_stat_function = actual_f

        return actual_t, actual_f
