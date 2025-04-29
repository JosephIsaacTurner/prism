from nilearn.maskers import NiftiMasker
import numpy as np
import os
import nibabel as nib
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from copy import deepcopy

pst_source_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def fetch_mni152_2mm_mask_img():
    """
    Fetches the MNI152 2mm mask.
    Returns:
    - masker: NiftiMasker object
    """
    return nib.load(
        os.path.join(pst_source_dir, "data", "MNI152_T1_2mm_brain_mask.nii.gz")
    )


def fetch_mni152_2mm_masker():
    """
    Fetches the MNI152 2mm masker.
    Returns:
    - masker: NiftiMasker object
    """
    return NiftiMasker(
        mask_img=os.path.join(pst_source_dir, "data", "MNI152_T1_2mm_brain_mask.nii.gz")
    ).fit()


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
    mask_img = (
        deepcopy(mask_img)
        if isinstance(mask_img, nib.Nifti1Image)
        else nib.load(mask_img)
    )
    data = mask_img.get_fdata()
    mask = data > 0
    data[mask] = np.random.randn(np.sum(mask))  # fill brain with noise
    coords = np.array(np.nonzero(mask)).T
    center = coords.mean(axis=0)
    dists = np.sqrt(((coords - center) ** 2).sum(axis=1))
    norm = (dists - dists.min()) / (
        dists.max() - dists.min()
    )  # normalize distances [0,1]
    m1, m2 = np.zeros(data.shape), np.zeros(data.shape)
    m1[mask], m2[mask] = 1 - norm, norm  # soft masks for center/periphery
    data = m1 * gaussian_filter(data, sigma=3) + m2 * gaussian_filter(
        data, sigma=7
    )  # blend two smoothing levels
    data[data != 0] = zscore(data[data != 0])  # z-score the data
    return np.ravel(data[data != 0])


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
    mask_img = (
        deepcopy(mask_img)
        if isinstance(mask_img, nib.Nifti1Image)
        else nib.load(mask_img)
    )
    data_vector = generate_null_data_vector(mask_img, random_state)
    masker = NiftiMasker(mask_img).fit()
    return masker.inverse_transform(data_vector)


