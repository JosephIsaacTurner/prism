from nilearn.mass_univariate._utils import calculate_tfce
from scipy.ndimage import generate_binary_structure
from nilearn.maskers import NiftiMasker
import numpy as np
import os
import nibabel as nib
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from copy import deepcopy
from nibabel.affines import apply_affine
import pandas as pd
from typing import Optional, Union, Callable, Generator
import warnings

pst_source_dir = os.path.dirname(os.path.realpath(__file__))

def atleast_4d(arr):
    """
    Function to ensure that the input array is 4D.
    """
    if arr.ndim == 3:
        return np.expand_dims(arr, axis=-1)  # Adds an extra dimension at the end
    elif arr.ndim == 4:
        return arr
    else:
        raise ValueError("Input array must be either 3D or 4D.")

def apply_tfce(img):
    """
    Apply TFCE to volumetric neuroimaging data.
    Parameters:
    - img: Nifti1Img of the data to apply TFCE to
    
    Returns: 
    - tfce_data_vector: 1d numpy array of data_vector after applying TFCE
    """
    tfce_data = np.nan_to_num(calculate_tfce(atleast_4d(img.get_fdata()), generate_binary_structure(3, 1))[:,:,:,0])
    return nib.Nifti1Image(tfce_data, img.affine)

def fetch_mni152_2mm_mask_img():
    """
    Fetches the MNI152 2mm mask.
    Returns:
    - masker: NiftiMasker object
    """
    return nib.load(os.path.join(pst_source_dir, 'data', 'MNI152_T1_2mm_brain_mask.nii.gz'))

def fetch_mni152_2mm_masker():
    """
    Fetches the MNI152 2mm masker.
    Returns:
    - masker: NiftiMasker object
    """
    return NiftiMasker(mask_img=os.path.join(pst_source_dir, 'data', 'MNI152_T1_2mm_brain_mask.nii.gz')).fit()

def generate_null_data_vector(mask_img, random_state=None):
    """
    Generates 1d np vector of random data with some spatial autocorrelation imposed, in the shape of the provided mask image.
    Parameters
    ----------
    mask_img : str or nibabel.Nifti1Image
        Path to mask image or mask image object.
    Returns
    -------
    data : np.ndarray
        Random data vector.
    """
    if random_state:
            np.random.seed(random_state)
    mask_img = deepcopy(mask_img) if isinstance(mask_img, nib.Nifti1Image) else nib.load(mask_img)
    data = mask_img.get_fdata()
    mask = data > 0; data[mask] = np.random.randn(np.sum(mask))  # fill brain with noise
    coords = np.array(np.nonzero(mask)).T; center = coords.mean(axis=0); dists = np.sqrt(((coords-center)**2).sum(axis=1))
    norm = (dists - dists.min()) / (dists.max() - dists.min())  # normalize distances [0,1]
    m1, m2 = np.zeros(data.shape), np.zeros(data.shape); m1[mask], m2[mask] = 1 - norm, norm  # soft masks for center/periphery
    data = m1 * gaussian_filter(data, sigma=3) + m2 * gaussian_filter(data, sigma=7)  # blend two smoothing levels
    data[data != 0] = zscore(data[data != 0])  # z-score the data
    return np.squeeze(data[data != 0])

def generate_null_brain_map(mask_img, random_state=None):
    """
    Returns totally random data in the shape of the provided mask image with some spatial autocorrelation imposed.
    Parameters
    ----------
    mask_img : str or nibabel.Nifti1Image
        Path to mask image or mask image object.
    Returns
    -------
    brain_map : nibabel.Nifti1Image
        Random brain map.
    """
    mask_img = deepcopy(mask_img) if isinstance(mask_img, nib.Nifti1Image) else nib.load(mask_img)
    data_vector = generate_null_data_vector(mask_img, random_state)
    masker = NiftiMasker(mask_img).fit()
    return masker.inverse_transform(data_vector)

def get_data_vector_and_coord_matrix(img, mask_img):
    """
    For a provided nifti image, return a flattened array (vector) of values contained within the brain, and a matrix of corresponding coordinates.
    Parameters
    ----------
    img : str or nibabel.Nifti1Image
        Path to image of interest or image object
    mask_img : str or nibabel.Nifti1Image
        Path to mask image or mask image object.
    Returns
    ----------
    img_data_vector : np.ndarray
        1d vector of shape (n_voxels,) for the voxels contained in the mask img
    coords: np.ndarray
        2d vector of shape (n_voxels, 3) for the corresponding 3d coordinates in world space (typically MNI152)
    """
    if mask_img == None:
        img = nib.load(img) if type(img) == str else img
        if type(img) != nib.Nifti1Image:
            print(f"Provided img was not Nifti1Image or path,str; instead got {type(img)}")
        mask_img = img
    else:
        mask_img = nib.load(mask_img) if type(mask_img) == str else mask_img
        masker = NiftiMasker(mask_img=mask_img).fit()
        img = masker.inverse_transform(np.squeeze(masker.fit_transform(img)))
    coords = np.indices(img.shape).reshape(3, -1).T
    coords = apply_affine(img.affine, coords) # Convert voxel coordinates to world coordinates (usually MNI coordinate)
    coords = coords[mask_img.get_fdata().flatten() != 0] # Remove data outside brain
    img_data_vector = img.get_fdata().flatten()[mask_img.get_fdata().flatten() != 0]
    return img_data_vector, coords

def load_data(input):
    """
    Load the input data from a file.
    Parameters:
    - input: str, path to the input file
    
    Returns:
    - data: numpy array, loaded data
    """
    if not isinstance(input, str):
        # print("Warning: Input is not a string. Probably already loaded data.")
        return input
    if input.endswith('.csv'):
        data = pd.read_csv(input, header=None).values
        if data.shape[1] == 1:
            data = data[:, 0] # Convert to 1D array if only one column
    elif input.endswith('.npy'):
        data = np.load(input)
    elif input.endswith('.txt'):
        data = pd.read_csv(input, sep='\t').values
    elif input.endswith('.nii') or input.endswith('.nii.gz'):
        data = nib.load(input)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv, .npy, .txt, .nii, or .nii.gz file.")
    return data

def load_nifti_if_not_already_nifti(img):
    """
    Load the image if it is not already a Nifti1Image.
    Parameters:
    - img: str or nibabel.Nifti1Image, path to the image or the image object
    
    Returns:
    - img: nibabel.Nifti1Image, loaded image
    """
    if isinstance(img, str):
        img = nib.load(img)
    elif not isinstance(img, nib.Nifti1Image):
        raise ValueError("Input must be a Nifti1Image or a file path. Got: {}".format(type(img)))
    return img

def is_nifti_like(data):
    """
    Check if the input data is Nifti-like.
    Parameters:
    - data: object to check
    
    Returns:
    - bool: True if data is Nifti-like, False otherwise
    """
    return isinstance(data, nib.Nifti1Image) or (isinstance(data, str) and (data.endswith('.nii') or data.endswith('.nii.gz')))

class Dataset:
    """
    Represents a single dataset for analysis, handling data loading and masking.
    """
    def __init__(
        self,
        data: Union[str, np.ndarray, nib.Nifti1Image],
        design: Union[str, np.ndarray],
        contrast: Union[str, np.ndarray],
        stat_function: Callable,
        n_permutations: int,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        mask_img: Optional[Union[str, nib.Nifti1Image]] = None,
        demean=True,
        # Permutation options
        exchangeability_matrix: Optional[Union[str, np.ndarray]] = None,
        vg_auto: bool = False,
        vg_vector: Optional[Union[str, np.ndarray]] = None,
        within: bool = True,
        whole: bool = False,
        flip_signs: bool = False,
        # Stat function options (handled by stat_function itself)
        tfce: bool = False,
        # Output saving flags (not directly used in analysis logic here)
        save_1minusp: bool = True,
        save_neglog10p: bool = False,
        save_permutations: bool = False
    ):
        """
        Initializes a Dataset object.

        Args:
            data: Path to data file or loaded data (ndarray/Nifti1Image).
            design: Path to design matrix file or loaded array.
            contrast: Path to contrast file or loaded array.
            stat_function: Callable that computes statistics (e.g., t-stats).
            n_permutations: Number of permutations for testing.
            random_state: Seed or RandomState object for reproducibility.
            mask_img: Optional mask (path or Nifti1Image) for NIfTI data.
            demean: Flag to demean data (default: True).
            exchangeability_matrix: Optional array defining permutation blocks.
            vg_auto: If True, auto-generate variance groups from exch. matrix.
            vg_vector: Optional array defining variance groups.
            within: Permute within blocks (if exch. matrix is 1D).
            whole: Permute whole blocks (if exch. matrix is 1D).
            flip_signs: Use sign-flipping permutations.
            tfce: Flag indicating if stat_function uses TFCE (handled internally by function).
            save_*: Flags for saving specific output types (informational).
        """
        # Store inputs (paths or raw data)
        self._data_input = data
        self._design_input = design
        self._contrast_input = contrast
        self._mask_img_input = mask_img
        self._exchangeability_matrix_input = exchangeability_matrix
        self._vg_vector_input = vg_vector

        # Core analysis parameters
        self.stat_function = stat_function
        self.n_permutations = n_permutations
        self.demean = demean
        # Ensure random_state is a numpy RandomState object
        if random_state is None or isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)

        # Permutation control flags
        self.vg_auto = vg_auto
        self.within = within
        self.whole = whole
        self.flip_signs = flip_signs
        self.tfce = tfce # Store TFCE flag, but logic is in stat_function

        # Loaded data/state (initialized to None)
        self.data: Optional[np.ndarray] = None # (n_samples, n_features)
        self.design: Optional[np.ndarray] = None # (n_samples, n_regressors)
        self.contrast: Optional[np.ndarray] = None # (n_regressors,) or (n_contrasts, n_regressors)
        self.mask_img: Optional[nib.Nifti1Image] = None
        self.masker: Optional[NiftiMasker] = None
        self.is_nifti: bool = False
        self.exchangeability_matrix: Optional[np.ndarray] = None
        self.vg_vector: Optional[np.ndarray] = None # (n_samples,)

        # Results storage within dataset (populated by analysis steps)
        self.true_stats: Optional[np.ndarray] = None # (n_features,)
        self.permuted_stat_generator: Optional[Generator] = None

        # Store saving flags
        self.save_1minusp = save_1minusp
        self.save_neglog10p = save_neglog10p
        self.save_permutations = save_permutations

    def load_data(self):
        """Loads data, design, contrast, and optional arrays from inputs."""
        # --- Load Main Data (and handle masking if NIfTI) ---
        loaded_data = load_data(self._data_input) # Assumes load_data utility exists
        self.is_nifti = is_nifti_like(loaded_data) # Assumes is_nifti_like utility exists

        if self.is_nifti:
            nifti_data = load_nifti_if_not_already_nifti(loaded_data) # Assumes utility exists
            if self._mask_img_input:
                self.mask_img = load_nifti_if_not_already_nifti(self._mask_img_input)
                self.masker = NiftiMasker(mask_img=self.mask_img)
            else:
                warnings.warn("NIfTI data provided without mask; using NiftiMasker for auto-masking.")
                self.masker = NiftiMasker() # Default strategy
            # Fit masker (if needed) and transform data
            self.data = self.masker.fit_transform(nifti_data)
            if self.mask_img is None: self.mask_img = self.masker.mask_img_ # Store auto-generated mask
            # Ensure data is 2D (samples x features)
            if self.data.ndim == 1: self.data = self.data[:, np.newaxis]
            elif self.data.ndim != 2: raise ValueError(f"Masked NIfTI data shape {self.data.shape} unexpected.")
        else:
            # Handle non-NIfTI data (must be ndarray)
            if not isinstance(loaded_data, np.ndarray):
                 raise TypeError(f"Loaded data is not NumPy array or NIfTI (type: {type(loaded_data)}).")
            self.data = loaded_data
            self.masker = None
            self.mask_img = None
            if self._mask_img_input: warnings.warn("Mask ignored for non-NIfTI data.")
            # Ensure data is 2D (samples x features)
            if self.data.ndim == 1: self.data = self.data[np.newaxis, :] # Assume 1 sample
            elif self.data.ndim != 2: raise ValueError(f"Non-NIfTI data must be 1D or 2D; got {self.data.shape}.")

        # --- Load Design and Contrast ---
        self.design = load_data(self._design_input)
        self.contrast = load_data(self._contrast_input)
        if not isinstance(self.design, np.ndarray): raise TypeError("Design must load as NumPy array.")
        if not isinstance(self.contrast, np.ndarray): raise TypeError("Contrast must load as NumPy array.")

        # --- Load Optional Permutation Arrays ---
        if self._exchangeability_matrix_input is not None:
            self.exchangeability_matrix = load_data(self._exchangeability_matrix_input)
            if not isinstance(self.exchangeability_matrix, np.ndarray): raise TypeError("Exchangeability matrix must load as NumPy array.")
        if self._vg_vector_input is not None:
            self.vg_vector = load_data(self._vg_vector_input)
            if not isinstance(self.vg_vector, np.ndarray): raise TypeError("Variance group vector must load as NumPy array.")

        # If demean is True, demean the data
        if self.demean:
            self.data, self.design, self.contrast, f_contrast_indices = prepare_glm_data(self.data, self.design, self.contrast)

        # --- Final Shape Validation ---
        n_samples = self.data.shape[0]
        n_regressors = self.design.shape[1]
        if self.design.shape[0] != n_samples:
             raise ValueError(f"Shape mismatch: Data samples ({n_samples}) != Design samples ({self.design.shape[0]}).")
        if self.contrast.ndim == 1 and self.contrast.shape[0] != n_regressors:
             raise ValueError(f"Shape mismatch: Contrast vector ({self.contrast.shape[0]}) != Design regressors ({n_regressors}).")
        elif self.contrast.ndim == 2 and self.contrast.shape[1] != n_regressors:
             raise ValueError(f"Shape mismatch: Contrast matrix cols ({self.contrast.shape[1]}) != Design regressors ({n_regressors}).")
        if self.exchangeability_matrix is not None and self.exchangeability_matrix.shape[0] != n_samples:
             raise ValueError(f"Shape mismatch: Exch. matrix samples ({self.exchangeability_matrix.shape[0]}) != Data samples ({n_samples}).")
        if self.vg_vector is not None and self.vg_vector.shape[0] != n_samples:
             raise ValueError(f"Shape mismatch: VG vector samples ({self.vg_vector.shape[0]}) != Data samples ({n_samples}).")
        

def prepare_glm_data(Y, X, C, f_contrast_indices=None):
    """
    Prepare data for GLM analysis by:
      1. Demeaning Y.
      2. Removing constant columns from X and centering continuous covariates.
      3. Adjusting contrast matrix C and optional F-contrast mask.

    Parameters
    ----------
    Y : array, shape (n_samples, n_features)
    X : array, shape (n_samples, n_regressors)
    C : array, shape (n_contrasts, n_regressors) or (n_regressors,)
    f_contrast_indices : array-like of {0,1}, optional

    Returns
    -------
    Y_d : array
    X_d : array
    C_d : array
    f_d : 1D bool array or None
    """
    import numpy as np, warnings

    # ——— 1. Validate dimensions ———
    C_arr = np.atleast_2d(C)
    n_obs, k = X.shape
    q, k_c = C_arr.shape
    if k != k_c:
        raise ValueError(f"X has {k} cols but C has {k_c}")
    if Y.shape[0] != n_obs:
        raise ValueError(f"X has {n_obs} rows but Y has {Y.shape[0]}")

    # ——— 2. Demean Y ———
    Y_d = Y - Y.mean(axis=0)

    # ——— 3. Trim constant columns from X ———
    const_mask = X.ptp(axis=0) == 0
    if const_mask.any():
        removed = np.nonzero(const_mask)[0].tolist()
        warnings.warn(f"Removing constant X cols at indices {removed}", UserWarning)
    X_trim = X[:, ~const_mask]
    if X_trim.ndim == 1:
        X_trim = X_trim[:, None]

    # ——— 4. Center continuous covariates ———
    uniq_counts = [np.unique(X_trim[:, i]).size for i in range(X_trim.shape[1])]
    cont_mask = np.array(uniq_counts) > 2
    X_d = X_trim.astype(float)
    if cont_mask.any():
        means = X_trim[:, cont_mask].mean(axis=0)
        X_d[:, cont_mask] -= means

    # ——— 5. Adjust contrast C ———
    C_trim = C_arr[:, ~const_mask]
    nonzero_rows = ~np.all(C_trim == 0, axis=1)
    if not nonzero_rows.all():
        dropped = np.nonzero(~nonzero_rows)[0].tolist()
        warnings.warn(f"Dropping zero-contrast rows {dropped}", UserWarning)
    C_kept = C_trim[nonzero_rows]
    C_d = C_kept.ravel() if C.ndim == 1 else C_kept

    # ——— 6. Update F-contrast indices ———
    f_d = None
    if f_contrast_indices is not None and C.ndim > 1:
        f = np.asarray(f_contrast_indices).flatten()
        if f.size != q:
            warnings.warn("Truncating f_contrast_indices to match C rows", UserWarning)
            f = f[:q]
        f_kept = f[nonzero_rows].astype(bool)
        if f_kept.sum() >= 2:
            f_d = f_kept
        else:
            warnings.warn("Not enough F-contrast components; setting f_d=None", UserWarning)

    return Y_d, X_d, C_d, f_d
