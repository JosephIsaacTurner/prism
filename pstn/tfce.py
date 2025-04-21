from nilearn.mass_univariate._utils import calculate_tfce
import numpy as np
import nibabel as nib
from scipy.ndimage import generate_binary_structure
import warnings


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


def calculate_auto_dh(map_vector, two_tailed=True):
  """
  Calculates the automatic TFCE step height (dh).

  Args:
    map_vector: List or 1D NumPy array of map values.
    two_tailed: If True, use max absolute value. If False, use max value.

  Returns:
    float: Calculated step height (dh), or 0.0 on failure/invalid input.
  """
  vector_np = np.asarray(map_vector) # Convert input to numpy array

  if vector_np.size == 0:
    return 0.0 # Return 0.0 for empty input

  # Find the relevant maximum value, ignoring NaNs.
  try:
      # Use max absolute value for two-tailed, max original value for one-tailed
      max_value = np.nanmax(np.abs(vector_np)) if two_tailed else np.nanmax(vector_np)

      # Ensure max_value is a valid positive number for dh calculation
      # If max_value is not finite or <= 0, no valid steps can be taken.
      if not np.isfinite(max_value) or max_value <= 0:
          return 0.0 # Return 0.0 if no positive range exists
  except ValueError: # Handles cases where nanmax might raise error (e.g., all NaNs)
      return 0.0 # Return 0.0 on error

  # Calculate dh based on 100 steps
  dh = max_value / 100.0

  return dh


def apply_tfce(img, two_tailed=True):
    """
    Apply TFCE to volumetric neuroimaging data.
    Parameters:
    - img: Nifti1Img of the data to apply TFCE to
    
    Returns: 
    - tfce_data_vector: 1d numpy array of data_vector after applying TFCE
    """
    img_data = img.get_fdata()

    # # If we were to replicate FSL's PALM precisely, we'd do this if two_tailed:
    # if two_tailed:
    #     # Take the absolute value of the data
    #     img_data = np.abs(img_data)
    #     # But nilearn's TFCE function is capable of doing two-tailed TFCE, 
    #     # Which I think is a better approach. This is why we don't do the above.
    
    tfce_data = np.nan_to_num(calculate_tfce(atleast_4d(img_data), generate_binary_structure(rank=3, connectivity=1), E=0.5, H=2, two_sided_test=two_tailed)[:,:,:,0])
    step_height = calculate_auto_dh(img_data[img_data != 0], two_tailed=two_tailed)
    if step_height:
        tfce_data = tfce_data * np.abs(step_height)
    return nib.Nifti1Image(tfce_data, img.affine)