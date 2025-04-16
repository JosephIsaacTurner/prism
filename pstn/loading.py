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
        data = pd.read_csv(input).values
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
        raise ValueError("Input must be a Nifti1Image or a file path.")
    return img

def is_nifti_like(data):
    """
    Check if the input data is Nifti-like.
    Parameters:
    - data: object to check
    
    Returns:
    - bool: True if data is Nifti-like, False otherwise
    """
    return isinstance(data, (nib.Nifti1Image)) or (isinstance(data, str) and (data.endswith('.nii') or data.endswith('.nii.gz')))

class Dataset:
    def __init__(
        self,
        data: Union[str, np.ndarray, nib.Nifti1Image],
        design: Union[str, np.ndarray],
        contrast: Union[str, np.ndarray],
        stat_function: Callable,
        n_permutations: int,
        random_state: Optional[int] = None,
        mask_img: Optional[Union[str, nib.Nifti1Image]] = None,
        # Optional permutation parameters
        exchangeability_matrix: Optional[Union[str, np.ndarray]] = None,
        vg_auto: bool = False,
        vg_vector: Optional[Union[str, np.ndarray]] = None,
        within: bool = True,
        whole: bool = False,
        flip_signs: bool = False,
        # Stat function related
        tfce: bool = False, # Assumed to be used by stat_function if True
        # Output/Saving related (kept as requested, but not used here)
        save_1minusp: bool = True,
        save_neglog10p: bool = False,
        save_permutations: bool = False
    ):
        # Store paths/initial data
        self._data_input = data
        self._design_input = design
        self._contrast_input = contrast
        self._mask_img_input = mask_img
        self._exchangeability_matrix_input = exchangeability_matrix
        self._vg_vector_input = vg_vector

        # Core parameters
        self.stat_function = stat_function
        self.n_permutations = n_permutations
        self.random_state = random_state

        # Permutation control
        self.exchangeability_matrix: Optional[np.ndarray] = None
        self.vg_auto = vg_auto
        self.vg_vector: Optional[np.ndarray] = None
        self.within = within
        self.whole = whole
        self.flip_signs = flip_signs

        # Stat function related
        self.tfce = tfce # Needs to be handled by the stat_function

        # Loaded data/state
        self.data: Optional[np.ndarray] = None
        self.design: Optional[np.ndarray] = None
        self.contrast: Optional[np.ndarray] = None
        self.mask_img: Optional[nib.Nifti1Image] = None
        self.masker: Optional[NiftiMasker] = None
        self.is_nifti: bool = False

        # Results storage within dataset
        self.true_stats: Optional[np.ndarray] = None
        self.permuted_stat_generator: Optional[Generator] = None

        # Output saving flags (not used in analysis function itself)
        self.save_1minusp = save_1minusp
        self.save_neglog10p = save_neglog10p
        self.save_permutations = save_permutations


    def load_data(self):
        """Loads all data components from paths or uses provided arrays."""
        print(f"Loading data for dataset...") # Add identifier if possible

        loaded_data = load_data(self._data_input)
        self.is_nifti = isinstance(loaded_data, nib.spatialimages.SpatialImage)

        if self.is_nifti:
            nifti_data = loaded_data
            if self._mask_img_input:
                self.mask_img = load_nifti_if_not_already_nifti(self._mask_img_input)
                self.masker = NiftiMasker(mask_img=self.mask_img)
                print("  - Applying provided mask to NIfTI data.")
                self.data = self.masker.fit_transform(nifti_data)
            else:
                warnings.warn("NIfTI data provided without mask. Auto-masking.")
                self.masker = NiftiMasker()
                self.data = self.masker.fit_transform(nifti_data)
                self.mask_img = self.masker.mask_img_
                print(f"  - Auto mask applied. Mask shape: {self.mask_img.shape}, Data shape: {self.data.shape}")
            if self.data.ndim == 1: self.data = self.data[:, np.newaxis]
        else:
            if isinstance(loaded_data, np.ndarray):
                 self.data = loaded_data
                 print(f"  - Loaded non-NIfTI data. Shape: {self.data.shape}")
            else:
                 raise TypeError(f"Loaded data is not NumPy array or NIfTI. Type: {type(loaded_data)}")
            if self._mask_img_input:
                warnings.warn("Mask image provided, but data not NIfTI. Mask ignored.")
            if self.data.ndim == 1: self.data = self.data[:, np.newaxis]
            elif self.data.ndim != 2:
                 raise ValueError(f"Non-NIfTI data must be 1D or 2D. Got shape {self.data.shape}")

        self.design = load_data(self._design_input)
        self.contrast = load_data(self._contrast_input)
        if not isinstance(self.design, np.ndarray): raise TypeError("Design must load as NumPy array.")
        if not isinstance(self.contrast, np.ndarray): raise TypeError("Contrast must load as NumPy array.")

        if self._exchangeability_matrix_input is not None:
            self.exchangeability_matrix = load_data(self._exchangeability_matrix_input)
            if not isinstance(self.exchangeability_matrix, np.ndarray): raise TypeError("Exchangeability matrix must load as NumPy array.")
        if self._vg_vector_input is not None:
            self.vg_vector = load_data(self._vg_vector_input)
            if not isinstance(self.vg_vector, np.ndarray): raise TypeError("Variance group vector must load as NumPy array.")

        print("  - Data loading complete.")
