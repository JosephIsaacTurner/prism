import warnings
import os
import numpy as np
import nibabel as nib
from typing import Callable, Optional, Union
from nilearn.maskers import NiftiMasker
from prism.data_wrangling import load_data, load_nifti_if_not_already_nifti, is_nifti_like
from prism.preprocessing import demean_glm_data
from prism.permutation_inference import permutation_analysis_volumetric_dense, permutation_analysis
from prism.stats import t, t_z, aspin_welch_v, aspin_welch_v_z, F, F_z, G, G_z, r_squared, r_squared_z, pearson_r, fisher_z
import json

class Dataset:
    """
    Represents a single dataset for analysis, handling data loading and masking.
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
        mask_img: Optional[Union[str, nib.Nifti1Image]] = None,
        tfce: bool = False,
    ):
        """
        Initializes the Dataset object with the provided parameters.
        """
        if config_path is None and (data is None or design is None or contrast is None):
            raise ValueError(
                "Either config_path or data, design, and contrast must be provided."
            )
        if config_path is not None:
            data, design, contrast, output_prefix, f_contrast_indices, two_tailed, exchangeability_matrix, vg_auto, variance_groups, within, whole, flip_signs, stat_function, f_stat_function, f_only, n_permutations, accel_tail, save_1minusp, save_neglog10p, correct_across_contrasts, random_state, demean, zstat, save_fn, permute_fn, save_permutations, mask_img, tfce = self.parse_config(config_path)
        
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
        self._mask_img_input = mask_img
        self._tfce_input = tfce

        # ─── Core analysis parameters ──────────────────────────────────
        self.stat_function = stat_function
        self.f_stat_function = f_stat_function
        self.two_tailed = two_tailed
        self.f_only = f_only
        self.n_permutations = n_permutations
        self.accel_tail = accel_tail
        self.demean = demean
        self.zstat = zstat

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
        self.save_fn = save_fn
        self.permute_fn = permute_fn
        self.output_prefix = output_prefix
        self.f_contrast_indices = f_contrast_indices

    def load_data(self):
        """Loads data, design, contrast, and optional arrays from inputs."""
        # --- Load Main Data (and handle masking if NIfTI) ---
        loaded_data = load_data(self._data_input)  # Assumes load_data utility exists
        self.is_nifti = is_nifti_like(
            loaded_data
        )  # Assumes is_nifti_like utility exists

        if self.is_nifti:
            nifti_data = load_nifti_if_not_already_nifti(
                loaded_data
            )  # Assumes utility exists
            if self._mask_img_input:
                self.mask_img = load_nifti_if_not_already_nifti(self._mask_img_input)
                self.masker = NiftiMasker(mask_img=self.mask_img)
            else:
                warnings.warn(
                    "NIfTI data provided without mask; using NiftiMasker for auto-masking."
                )
                self.masker = NiftiMasker()  # Default strategy
            # Fit masker (if needed) and transform data
            self.data = self.masker.fit_transform(nifti_data)
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
                warnings.warn("Mask ignored for non-NIfTI data.")
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
            "mask_img": self.mask_img,
            "tfce": self.tfce,
        }
    

    def permutation_analysis(self):
        """
        Perform permutation test on the loaded data.
        Returns:
        - results: sklearn.Bunch, results of the permutation test
        """
        self.load_data()  # Ensure data is loaded
        params = self.params.copy()
        if self.is_nifti:
            params['imgs'] = params.pop('data') # Rename data to imgs for NIfTI
            results = permutation_analysis_volumetric_dense(
                **params
            )
        else:
            params.pop('tfce') # This key is not used in the non-volumetric case
            results = permutation_analysis(
                **params
            )
        return results
    
    def save_config(self):
        input_params = {
            "data": self._data_input,
            "design": self._design_input,
            "contrast": self._contrast_input,
            "output_prefix": self._output_prefix_input,
            "f_contrast_indices": self._f_contrast_indices_input,
            "two_tailed": self._two_tailed_input,
            "exchangeability_matrix": self._exchangeability_matrix_input,
            "vg_auto": self._vg_auto_input,
            "variance_groups": self._variance_groups_input,
            "within": self._within_input,
            "whole": self._whole_input,
            "flip_signs": self._flip_signs_input,
            "stat_function": self._stat_function_input,
            "f_stat_function": self._f_stat_function_input,
            "f_only": self._f_only_input,
            "n_permutations": self._n_permutations_input,
            "accel_tail": self._accel_tail_input,
            "save_1minusp": self._save_1minusp_input,
            "save_neglog10p": self._save_neglog10p_input,
            "correct_across_contrasts": self._correct_across_contrasts_input,
            "random_state": self._random_state_input,
            "demean": self._demean_input,
            "zstat": self._zstat_input,
            "save_fn": self._save_fn_input,
            "permute_fn": self._permute_fn_input,
            "save_permutations": self._save_permutations_input,
            "mask_img": self._mask_img_input,
            "tfce": self._tfce_input,
        }

        if self.output_prefix is None:
            output_prefix = f"{os.path.getcwd()}/prism"
        else:
            output_prefix = self.output_prefix

        # Save a json at the prefix config.json
        config_path = f"{output_prefix}_config.json"
        with open(config_path, "w") as f:
            json.dump(input_params, f, indent=4)
        print(f"Configuration saved to {config_path}")

    def parse_config(self, config_path: str):
        """
        Parses a configuration file and updates the Dataset instance.
        """
        with open(config_path, "r") as f:
            params = json.load(f)
        return tuple(params.values())
    
    def select_stat_functions(self):
        """
        Return (actual_t_stat_fn, actual_f_stat_fn) based on:
        - stat_function ∈ {"auto", "pearson"} or a callable
        - f_stat_function ∈ {"auto", "pearson"} or a callable
        - use_variance_groups, zstat, perform_f_test
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
