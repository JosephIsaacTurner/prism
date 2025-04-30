from nilearn.maskers import NiftiMasker
import numpy as np
import jax.numpy as jnp
import os
import nibabel as nib
from sklearn.utils import Bunch
import pandas as pd

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
    if input.endswith(".csv"):
        data = pd.read_csv(input, header=None).values
        if data.shape[1] == 1:
            data = data[:, 0]  # Convert to 1D array if only one column
    elif input.endswith(".npy"):
        data = np.load(input)
    elif input.endswith(".txt"):
        data = pd.read_csv(input, sep="\t").values
    elif input.endswith(".nii") or input.endswith(".nii.gz"):
        data = nib.load(input)
    else:
        raise ValueError(
            "Unsupported file format. Please provide a .csv, .npy, .txt, .nii, or .nii.gz file."
        )
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
        raise ValueError(
            "Input must be a Nifti1Image or a file path. Got: {}".format(type(img))
        )
    return img


def is_nifti_like(data):
    """
    Check if the input data is Nifti-like.
    Parameters:
    - data: object to check

    Returns:
    - bool: True if data is Nifti-like (a Nifti1Image, a .nii/.nii.gz path,
            or a list/tuple of those), False otherwise
    """
    # single image
    if isinstance(data, nib.Nifti1Image):
        return True

    # single filename
    if isinstance(data, str) and data.endswith((".nii", ".nii.gz")):
        return True

    # list/tuple of images or filenames
    if isinstance(data, (list, tuple)):
        return all(is_nifti_like(item) for item in data)

    return False


class ResultSaver:
    def __init__(
        self,
        output_prefix="",
        variance_groups=None,
        stat_function="auto",
        f_stat_function="auto",
        zstat=False,
        save_permutations=False,
        mask_img=None,
        n_t_contrasts=1,
        permutation_output_dir=None,
    ):
        if not isinstance(n_t_contrasts, int) or n_t_contrasts < 1:
            raise ValueError("`n_t_contrasts` must be a positive integer")

        self.output_prefix = output_prefix
        self.variance_groups = variance_groups
        self.stat_fn = (
            stat_function
            if isinstance(stat_function, str)
            else getattr(stat_function, "__name__", "unknown")
        )
        self.f_stat_fn = (
            f_stat_function
            if isinstance(f_stat_function, str)
            else getattr(f_stat_function, "__name__", "unknown")
        )

        self.n_t_contrasts = n_t_contrasts
        self.permutation_output_dir = (
            (
                permutation_output_dir
                if permutation_output_dir
                else os.path.join(os.path.dirname(output_prefix), "permutations")
            )
            if save_permutations
            else None
        )
        self.zstat = zstat

        self.mask_img = mask_img
        self.masker = NiftiMasker(mask_img).fit() if mask_img is not None else None
        self.n_elements = getattr(self.masker, "n_elements_", None)

        self.processed = set()

        if self.output_prefix:
            out_dir = os.path.dirname(output_prefix) or "."
            os.makedirs(out_dir, exist_ok=True)
            self.output_dir = out_dir
            self.base = os.path.basename(output_prefix)
        else:
            self.output_dir = None
            self.base = None

        if save_permutations:
            os.makedirs(self.permutation_output_dir, exist_ok=True)

    def _rename(self, key):
        is_f = key.endswith(("_f", ".f"))
        fn = self.f_stat_fn if is_f else self.stat_fn
        use_groups = (self.variance_groups is not None) and (self.variance_groups is not False)
        tag = None
        if fn == "auto":
            if is_f:
                tag = "gstat" if use_groups else "fstat"
            else:
                tag = "vstat" if use_groups else "tstat"
        elif fn == "pearson":
            if is_f:
                tag = "rsqstat"
            else:
                tag = "rstat"
        elif isinstance(fn, str):
            tag = f"{fn.lower()}stat"

        if self.n_t_contrasts == 1:
            # If only one contrast was tested, remove the contrast suffix
            key = key.replace("_c1", "")

        if self.zstat:
            tag = f"z{tag}" if tag else None

        return key.replace("stat", tag) if tag and "stat" in key else key

    def save_results(self, results):
        if self.output_prefix is None:
            # No output_prefix provided, so we can't save results
            return

        if not hasattr(results, "items"):
            print("Warning: input not dict‑ or Bunch‑like; skipping")
            return

        for key, val in results.items():
            if not isinstance(key, str) or key in self.processed:
                continue

            renamed_key = self._rename(key)
            if (
                self.masker
                and (isinstance(val, np.ndarray) or isinstance(val, jnp.ndarray))
                and val.size == self.n_elements
            ):
                img = self.masker.inverse_transform(val)
                save_key = f"vox_{renamed_key}" if "tfce" not in key else renamed_key
                ext = ".nii.gz"
                to_save = img
            else:
                save_key = renamed_key
                ext = ".npy"
                to_save = val

            path = os.path.join(self.output_dir, f"{self.base}_{save_key}{ext}")

            try:
                if ext == ".nii.gz":
                    nib.save(to_save, path)
                else:
                    np.save(path, to_save)
                self.processed.add(key)

            except Exception as e:
                print(f"Error saving '{key}': {e}")

    def get_processed_keys(self):
        return set(self.processed)

    def clear_processed_keys(self):
        self.processed.clear()

    def save_permutation(self, permuted_stats, perm_idx, contrast_idx, *args, **kwargs):
        """Updates the manager with new permutation results."""
        if (contrast_idx is None or contrast_idx == -1) and self.n_t_contrasts > 1:
            contrast_label = "_f"
        else:
            contrast_label = f"_c{contrast_idx+1}" if self.n_t_contrasts > 1 else ""

        # pad to 5 digits, e.g. 00001, 01234, 12345
        perm_label = f"perm{perm_idx+1:05d}"

        if self.masker is not None:
            permuted_stats_img = self.masker.inverse_transform(np.ravel(permuted_stats))
            filename = os.path.join(
                self.permutation_output_dir,
                f"{os.path.basename(self.output_prefix)}_{perm_label}{contrast_label}.nii.gz",
            )
            permuted_stats_img.to_filename(filename)
        else:
            filename = os.path.join(
                self.permutation_output_dir,
                f"{os.path.basename(self.output_prefix)}_{perm_label}{contrast_label}.npy",
            )
            np.save(filename, permuted_stats)

    def finalize_results(self, results):
        if not hasattr(results, "items"):
            print("Warning: input not dict‑ or Bunch‑like; skipping")
            return

        updated_results = Bunch()

        for key, val in results.items():
            renamed_key = self._rename(key)
            if (
                self.masker
                and (isinstance(val, np.ndarray) or isinstance(val, jnp.ndarray))
                and val.size == self.n_elements
            ):
                val = self.masker.inverse_transform(val)
                save_key = f"vox_{renamed_key}" if "tfce" not in key else renamed_key
            else:
                save_key = renamed_key
            updated_results[save_key] = val
        return updated_results