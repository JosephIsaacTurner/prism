from .datasets import generate_null_data_vector
from .preprocessing import get_data_vector_and_coord_matrix
import numpy as np
from tqdm import tqdm
from nilearn.maskers import NiftiMasker
from copy import deepcopy
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import zscore
from hilbertcurve.hilbertcurve import HilbertCurve


"""
Everything in this file is experimental and not intended for production use.
Mostly just features/ideas that I'm playing around with.
"""

def pearsonr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def correlate_maps_experimental(map1, map2, mask_img, n_permutations, random_state, two_tailed=True, comparison_function=pearsonr):
    """"""
    # One: Find true correlation value
    compare = comparison_function
    mask_img = deepcopy(mask_img) if isinstance(mask_img, nib.Nifti1Image) else nib.load(mask_img)
    masker = NiftiMasker(mask_img=mask_img).fit()
    map1_vector = np.squeeze(masker.fit_transform(map1).flatten())
    map2_vector = np.squeeze(masker.fit_transform(map2).flatten())
    true_comparison_value = compare(map1_vector, map2_vector)
    
    # Two: Create synthetic null data of 1000 samples each
    null_data_one = np.vstack([generate_null_data_vector(mask_img) for _ in tqdm(range(1000), desc="Generating null data one")])
    null_data_two = np.vstack([generate_null_data_vector(mask_img) for _ in tqdm(range(1000), desc="Generating null data two")])

    # Three: Simulate both maps using null data
    simulated_map_one, weights_one = generate_simulated_vector(map1_vector, null_data_one)
    simulated_map_two, weights_two = generate_simulated_vector(map2_vector, null_data_two)

    observed_comparison_values = compare(simulated_map_one, simulated_map_two)
    print(f"Observed comparison value: {observed_comparison_values}")

    # Four: Set up generators
    permuted_map_one_generator = yield_permuted_vectors(null_data_one, weights_one, n_permutations, random_state)
    permuted_map_two_generator = yield_permuted_vectors(null_data_two, weights_two, n_permutations, random_state)

    # Five: Calculate null distribution
    exceedances = 0
    for i in tqdm(range(n_permutations), desc="Calculating null distribution"):
        permuted_map_one = next(permuted_map_one_generator)
        permuted_map_two = next(permuted_map_two_generator)
        permuted_comparison_value = compare(permuted_map_one, permuted_map_two)
        if two_tailed:
            if np.abs(permuted_comparison_value) >= np.abs(true_comparison_value):
                exceedances += 1
        else:
            if permuted_comparison_value >= true_comparison_value:
                exceedances += 1

    # Six: Calculate p-value
    p_value = (exceedances +1) / (n_permutations + 1)
    return true_comparison_value, p_value
                

def generate_simulated_vector(target_map_vector, null_data, alpha=1e-2):
    """
    Reconstructs null_data_one as a weighted combination of null_data, with weights regularized
    towards uniformity.
    
    Parameters:
        null_data (np.ndarray): Array of shape (n_null_data, n_voxels)
        null_data_one (np.ndarray): Array of shape (n_voxels,)
        alpha (float): Regularization strength.
    
    Returns:
        simulated_map (np.ndarray): The weighted combination (simulated map).
        weights (np.ndarray): The weights used (summing to 1).
    """
    n = null_data.shape[0]
    uniform = np.ones(n) / n
    A = 2 * (null_data @ null_data.T + alpha * np.eye(n))
    b = 2 * (null_data @ target_map_vector + alpha * uniform)
    
    # Set up KKT system for constraint sum(w) = 1.
    KKT = np.block([[A, np.ones((n, 1))],
                    [np.ones((1, n)), np.zeros((1, 1))]])
    sol = np.linalg.solve(KKT, np.concatenate([b, [1]]))
    weights = sol[:n]
    simulated_map = weights @ null_data
    return simulated_map, weights


def yield_permuted_vectors(null_data, weights, n_permutations, random_state):
    """
    Yields permuted vectors using the provided null data and weights.

    Parameters:
        null_data (np.ndarray): Shape (n_samples, n_elements_per_sample).
        weights (np.ndarray): The weights to permute. Shape (n_elements_per_sample,).
        n_permutations (int): The number of permutations to yield.
        random_state (int): The random seed to use for reproducibility.

    Yields:
        permuted_map_vector (np.ndarray): The permuted map vector
    """
    np.random.seed(random_state)
    permuted_weights = np.random.permutation(weights)
    for _ in range(n_permutations):
        np.random.shuffle(permuted_weights)
        permuted_map_vector = permuted_weights @ null_data
        yield permuted_map_vector


def compute_variogram_sampled(map_vector, coord_matrix, n_bins, n_samples, random_state):
    """
    Computes an approximate experimental variogram via random sampling of point pairs.
    
    Parameters:
        map_vector (np.ndarray): 1D array of shape (n_elements,) with data values.
        coord_matrix (np.ndarray): 2D array of shape (n_elements, 3) with spatial coordinates.
        n_bins (int): Number of bins for grouping distances.
        n_samples (int): Number of random pairs to sample.
    
    Returns:
        bin_centers (np.ndarray): 1D array of bin centers (mean distance in each bin).
        gamma (np.ndarray): 1D array of semivariance estimates for each bin.
    """

    np.random.seed(random_state)

    n = len(map_vector)
    # Randomly sample indices for pairs. (Sampling with replacement is fine for estimation)
    idx1 = np.random.randint(0, n, n_samples)
    idx2 = np.random.randint(0, n, n_samples)
    
    # Compute Euclidean distances for the sampled pairs.
    dists = np.linalg.norm(coord_matrix[idx1] - coord_matrix[idx2], axis=1)
    
    # Compute semivariance for the sampled pairs: 0.5 * (difference^2)
    semivars = 0.5 * (map_vector[idx1] - map_vector[idx2])**2
    
    # Define bins over the range of distances.
    bin_edges = np.linspace(dists.min(), dists.max(), n_bins + 1)
    bin_indices = np.digitize(dists, bin_edges)
    
    bin_centers = []
    gamma = []
    
    # Compute the average semivariance within each distance bin.
    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        if np.any(mask):
            center = (bin_edges[i - 1] + bin_edges[i]) / 2.0
            bin_centers.append(center)
            gamma.append(np.mean(semivars[mask]))
    
    return np.array(bin_centers), np.array(gamma)


def compare_spatial_autocorrelation(img_one, img_two, mask_img, random_state, visualize):
    """
    For two maps, compares their spatial autocorrelation. 

    Parameters
    ----------
    img_one : str or nibabel.Nifti1Image
        Path to image of interest or image object
    img_two : str or nibabel.Nifti1Image
        Path to second image of interest or image object.
    mask_img: str of nibabel.Nifti1Image
        Path to mask img/ mask img object
    random_state: int
        Seed for the random number generator to ensure reproducibility.
    visualize: bool
        Whether to visualize the variograms

    Returns
    ----------
    mse: float
        Mean squared error variogram data
    """
    data_vector_one, coords_one = get_data_vector_and_coord_matrix(img_one, mask_img)
    data_vector_two, coords_two = get_data_vector_and_coord_matrix(img_two, mask_img)
    bin_centers_one, gamma_one = compute_variogram_sampled(zscore(data_vector_one), coords_one, n_bins=100, n_samples=10000000, random_state=random_state)
    bin_centers_two, gamma_two = compute_variogram_sampled(zscore(data_vector_two), coords_two, n_bins=100, n_samples=10000000, random_state=random_state)

    if visualize:
        # Plot using matplotlib
        plt.plot(bin_centers_one, gamma_one, 'o-')
        plt.plot(bin_centers_two, gamma_two, 'o-')
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.show()

    mse = float(np.sqrt(np.mean((gamma_one - gamma_two) ** 2)))
    return mse


def circular_spin(arr, random_shift=None):
    """
    Takes a 1D array, treats it as circular, and rotates it by a random amount.
    
    Parameters:
    arr: 1D numpy array
    random_shift: Optional int or None. If None, generates random shift.
    
    Returns:
    Rotated array
    """
    if random_shift is None:
        random_shift = np.random.randint(0, len(arr))
    
    # Using roll is the most straightforward way
    return np.roll(arr, random_shift)


def random_rotation_3d(points, random_state=None):
    """
    Apply a random 3D rotation to an array of points while preserving their structure.
    
    Parameters:
    points: numpy array of shape (n_samples, 3) containing x,y,z coordinates
    
    Returns:
    rotated_points: numpy array of shape (n_samples, 3) with randomly rotated coordinates
    """
    if random_state is not None:
        np.random.seed(random_state)
    # Generate a random rotation matrix using QR decomposition
    # This ensures a uniform random rotation
    random_matrix = np.random.normal(size=(3, 3))
    Q, R = np.linalg.qr(random_matrix)
    
    # Ensure we have a proper rotation matrix (determinant = 1)
    # If determinant is -1, we flip the last column
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
        
    # Apply rotation to all points
    rotated_points = points @ Q.T
    
    return rotated_points, Q


def reorder_brain_data_hilbert(values, coords, p=10, verbose=True):
    """
    Reorder a 1D vector 'values' (shape: (n_elements,)) and its corresponding 
    3D coordinates 'coords' (shape: (n_elements, 3)) using a 3D Hilbert curve mapping.
    Also returns an inverse permutation to restore the original order.
    
    This function first normalizes the 3D coordinates to a discrete grid based on
    the specified number of bits per dimension (p). It then computes the Hilbert index
    for each point, sorts the points by that index, and uses that ordering to rearrange
    the data. The parameter 'p' determines the resolution of the Hilbert curve: higher p
    means finer resolution.
    
    Parameters:
        values (np.ndarray): 1D array of data values with shape (n_elements,).
        coords (np.ndarray): 2D array of spatial coordinates with shape (n_elements, 3).
        p (int): Number of iterations (or bits per dimension) for the Hilbert curve.
                 The side length of the grid will be 2**p. (Default: 10)
    
    Returns:
        tuple: (reordered_values, reordered_coords, inverse_order)
            - reordered_values (np.ndarray): The reordered 1D data vector.
            - reordered_coords (np.ndarray): The corresponding reordered coordinate matrix.
            - inverse_order (np.ndarray): An array such that applying it to the reordered data 
                                          returns the data in the original order.
    """
    n = len(values)
    
    # Determine the bounding box of the coordinates.
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    
    # Compute the scaling factor to normalize coordinates to the grid [0, 2**p - 1].
    grid_size = 2**p
    scale = (grid_size - 1) / (max_coords - min_coords)
    
    # Normalize and convert coordinates to integer grid indices.
    coords_int = np.floor((coords - min_coords) * scale).astype(int)
    coords_int = np.clip(coords_int, 0, grid_size - 1)
    
    # Create a Hilbert curve instance for 3 dimensions.
    hilbert_curve = HilbertCurve(p, 3)
    
    # Compute the Hilbert index for each point.
    hilbert_indices = np.empty(n, dtype=np.int64)
    for i in tqdm(range(n), desc="Computing Hilbert indices") if verbose else range(n):
        point = list(coords_int[i])
        try:
            # Try the primary method name.
            hilbert_indices[i] = hilbert_curve.distance_from_coordinates(point)
        except AttributeError:
            # Fallback to an alternate method name if necessary.
            hilbert_indices[i] = hilbert_curve.distance_from_point(point)
    
    # Determine the ordering that sorts by Hilbert index.
    order = np.argsort(hilbert_indices)
    
    # Compute the inverse permutation.
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(n)
    
    # Use the computed ordering to rearrange values and coordinates.
    reordered_values = values[order]
    reordered_coords = coords[order]
    
    return reordered_values, reordered_coords, order, inverse_order


def exp_decay(d):
    """
    Exponentially decaying kernel which truncates at e^{-1}.

    Parameters
    ----------
    d : (N,) or (M,N) np.ndarray
        one- or two-dimensional array of distances

    Returns
    -------
    (N,) or (M,N) np.ndarray
        Exponential kernel weights

    Notes
    -----
    Characteristic length scale is set to d.max(axis=-1), i.e. the maximum
    distance within each row.

    Raises
    ------
    TypeError : `d` is not array_like

    """
    try:  # 2-dim
        return np.exp(-d / d.max(axis=-1)[:, np.newaxis])
    except IndexError:  # 1-dim
        return np.exp(-d/d.max())
    except AttributeError:
        raise TypeError("expected array_like, got {}".format(type(d)))


def rotate_nifti_axial(nifti_img_or_path, axial_degrees_rotation_clockwise):
    """
    Rotate a NIFTI image's axial plane (first two affine columns) about the SI axis by the specified
    clockwise angle (in degrees). The function accepts either a Nifti image object or a file path.
    Note: Clockwise means the anterior end of the axial plane will be deviated to the _right_. 
    This can be counterintuitive if using a viewer in radiological convention.
    
    Parameters:
      nifti_img_or_path: nibabel Nifti1Image object or str
         A NIFTI image or a path to one.
      axial_degrees_rotation_clockwise: float
         The rotation angle in degrees, where positive values rotate the image clockwise.
    
    Returns:
      new_img: nibabel Nifti1Image object
         The rotated image.
    """
    # Load the image if a file path was provided
    if isinstance(nifti_img_or_path, str):
        img = nib.load(nifti_img_or_path)
    else:
        img = nifti_img_or_path

    # Convert the provided clockwise angle to the appropriate angle in radians.
    # Under the right-hand rule, a clockwise rotation corresponds to a negative angle.
    angle = -np.deg2rad(axial_degrees_rotation_clockwise)

    # Get the SI axis from the affine (assumed to be the third column)
    si_axis = img.affine[:3, 2]
    si_unit = si_axis / np.linalg.norm(si_axis)

    # Build the 3D rotation matrix using Rodrigues’ rotation formula
    K = np.array([
        [    0,      -si_unit[2],  si_unit[1]],
        [ si_unit[2],      0,     -si_unit[0]],
        [-si_unit[1],  si_unit[0],      0    ]
    ])
    R_3d = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    # Create a new affine by rotating the first two columns (axial plane) and leaving the SI column unchanged
    new_affine = img.affine.copy()
    new_affine[:3, 0] = R_3d @ img.affine[:3, 0]
    new_affine[:3, 1] = R_3d @ img.affine[:3, 1]
    # new_affine[:3, 2] remains unchanged

    # Compute the center of the image in voxel space and its corresponding world coordinates.
    center_vox = np.array(img.shape) / 2.0
    center_world = img.affine.dot(np.hstack([center_vox, 1]))[:3]

    # Adjust the translation so that the center remains fixed after the rotation.
    new_affine[:3, 3] = center_world - (
        new_affine[:3, 0] * center_vox[0] +
        new_affine[:3, 1] * center_vox[1] +
        new_affine[:3, 2] * center_vox[2]
    )

    # Create the new NIFTI image with the updated affine
    new_img = nib.Nifti1Image(img.get_fdata(), new_affine, img.header)
    return new_img

def unravel_atlas(atlas, mask_img, background_value=0):

    atlas_img = nib.load(atlas) if isinstance(atlas, str) else atlas
    mask_img = nib.load(mask_img) if isinstance(mask_img, str) else mask_img
    vector, coords = get_data_vector_and_coord_matrix(atlas_img, mask_img)
    proxy_background_value = np.max(vector) + 1
    vector[vector == background_value] = proxy_background_value
    unique_ids = np.unique(vector)
    parcel_dictionary = {}

    for id in unique_ids:
        parcel_vector = np.where(vector == id, id, 0)
        parcel_vector = parcel_vector[parcel_vector != 0]
        parcel_coords = coords[vector == id, :]
        centroid_coords = np.round(np.mean(parcel_coords, axis=0), 0).astype(int)
        reordered_vector, reordered_coords, new_order, inverse_order = reorder_brain_data_hilbert(parcel_vector, parcel_coords, p=7, verbose=False)
        parcel_dictionary[id] = {'centroid': centroid_coords, 'ordered_indices': np.argwhere(vector == id)[new_order]}

    centroid_coords = {id: parcel['centroid'] for id, parcel in parcel_dictionary.items()}
    centroid_coords = np.array([centroid_coords[id] for id in unique_ids])
    dummy_vector = np.arange(len(unique_ids))
    reordered_vector, reordered_coords, new_order, inverse_order = reorder_brain_data_hilbert(dummy_vector, centroid_coords, p=8, verbose=False)
    ordered_parcel_dictionary = {unique_ids[i]: parcel_dictionary[unique_ids[j]] for i, j in enumerate(new_order)}
    order = np.concatenate([parcel['ordered_indices'] for parcel in ordered_parcel_dictionary.values()])
    order = np.squeeze(order)
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order))
    ordered_parcel_vector = vector[order]
    ordered_parcel_vector[ordered_parcel_vector == proxy_background_value] = background_value
    ordered_parcel_coords = coords[order]
    return ordered_parcel_vector, ordered_parcel_coords, order, inverse_order
