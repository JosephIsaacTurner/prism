import numpy as np
import warnings

def demean_glm_data(Y, X, C, f_contrast_indices=None):
    """
    Prepare data for GLM analysis by:
      1. Demeaning Y.
      2. Removing constant columns from X and centering continuous covariates.
      3. Adjusting contrast matrix C and optional F-contrast mask.

    Parameters
    ----------
    Y : array, shape (n_samples, n_features)
    X : array, shape (n_samples, n_regressors)
    C : array, shape (n_t_contrasts, n_regressors) or (n_regressors,)
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
    const_mask = np.ptp(X, axis=0) == 0

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
            warnings.warn(
                "Not enough F-contrast components; setting f_d=None", UserWarning
            )

    return Y_d, X_d, C_d, f_d


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
            print(
                f"Provided img was not Nifti1Image or path,str; instead got {type(img)}"
            )
        mask_img = img
    else:
        mask_img = nib.load(mask_img) if type(mask_img) == str else mask_img
        masker = NiftiMasker(mask_img=mask_img).fit()
        img = masker.inverse_transform(np.ravel(masker.fit_transform(img)))
    coords = np.indices(img.shape).reshape(3, -1).T
    coords = apply_affine(
        img.affine, coords
    )  # Convert voxel coordinates to world coordinates (usually MNI coordinate)
    coords = coords[mask_img.get_fdata().flatten() != 0]  # Remove data outside brain
    img_data_vector = img.get_fdata().flatten()[mask_img.get_fdata().flatten() != 0]
    return img_data_vector, coords


