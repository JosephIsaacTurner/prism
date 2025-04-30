# Permutation Analysis API

The `permutation_analysis()` and `permutation_analysis_nifti()` functions are the core statistical engines of PRISM. They perform permutation-based inference on mass univariate GLMs, with functionality modeled closely on Anderson Winkler's PALM.

For reference, see the original [PALM User Guide](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/PALM(2f)UserGuide.html).

---

## 🔁 Supported Features (Compared to PALM)

**Fully Supported in PRISM:**
- All standard GLM test statistics: `t`, `aspin_welch_v`, `F`, `G`, `pearson_r`, `r_squared`
- Freedman-Lane permutation algorithm
- Beckmann partitioning of model
- z-statistic conversions for all of the above (see [Statistical Functions documentation](statistical_functions.md))
- Exchangeability blocks with `within`, `whole`, and both modes
- Variance group logic (`vg_auto` or manual)
- Sign-flip (ISE), label permutation (EE), or combined
- TFCE (two-sided supported)
- Westfall–Young FWE correction (within and across contrasts)
- Benjamini–Hochberg FDR correction (within and across contrasts)
- GPD tail-fitting for p-value precision (`accel_tail`)
- Option to demean data and regressors
- Option to save raw permutations

**Not Yet Supported:**
- Non-parametric combination (NPC)
- Multimodal inputs
- MANOVA/MANCOVA/classical multivariate tests
- Surface-based TFCE
- Cluster extent/mass-based inference

---

## 🧾 Parameters

All parameters are optional unless marked required. For high-level usage, see the [Dataset documentation](dataset.md).

| Parameter | Type | Description |
|----------|------|-------------|
| `data` *(required)* | `np.ndarray` or `nib.Nifti1Image` | 2D matrix (samples × features) or masked NIfTI |
| `imgs` *(nifti only)* | `nib.Nifti1Image`, list of NIfTI | 4D NIfTI or list of 3D NIfTIs for dense inference |
| `design` *(required)* | `np.ndarray` | Full design matrix |
| `contrast` *(required)* | `np.ndarray` | Contrast vector or matrix |
| `output_prefix` | `str` | Prefix for output files |
| `f_contrast_indices` | `np.ndarray` | Indices or mask for F-test contrasts |
| `two_tailed` | `bool` | Whether to perform two-tailed test (default: True) |
| `exchangeability_matrix` | `np.ndarray` | Sample shuffling structure (blocks/groups) |
| `vg_auto` | `bool` | If True, infer variance groups from `exchangeability_matrix` |
| `variance_groups` | `np.ndarray` | Vector assigning variance group membership |
| `within` | `bool` | Permute within exchangeability blocks |
| `whole` | `bool` | Permute whole blocks |
| `flip_signs` | `bool` | Enable ISE (sign flips) for symmetric errors |
| `stat_function` | `str` or callable | Test function (e.g., "auto", "pearson", or custom) (see [Statistical Functions documentation](statistical_functions.md)) |
| `f_stat_function` | `str` or callable | Same, but for F-statistics |
| `f_only` | `bool` | Only compute F-test (skip t-tests) |
| `n_permutations` | `int` | Number of permutations to run |
| `accel_tail` | `bool` | Enable GPD tail-fitting for better p-values |
| `save_1minusp` | `bool` | Save 1 - p rather than p |
| `save_neglog10p` | `bool` | Save -log10(p) |
| `correct_across_contrasts` | `bool` | Apply FWE correction across contrasts |
| `random_state` | `int` | Seed for permutation reproducibility |
| `demean` | `bool` | Demean before GLM |
| `zstat` | `bool` | Convert stats to z |
| `mask_img` | `nib.Nifti1Image` | Optional mask for NIfTI input |
| `tfce` | `bool` | Enable TFCE (NIfTI only) |
| `save_fn`, `permute_fn` | `callable` | Optional hooks for saving/debugging |
| `save_permutations` | `bool` | Save full permutation distribution |

---

## 🧪 Example Usage

### In-Memory NumPy

```python
from prism.permutation_inference import permutation_analysis

results = permutation_analysis(
    data=Y,               # shape (n_samples, n_image_points)
    design=X,             # shape (n_samples, n_regressors)
    contrast=C,           # shape (n_regressors,) or (n_contrasts, n_regressors)
    f_contrast_indices=[0, 1],
    n_permutations=5000,
    correct_across_contrasts=True,
    accel_tail=True,
    output_prefix="/results/test"
)
```

### Dense NIfTI

```python
from prism.permutation_inference import permutation_analysis_nifti

results = permutation_analysis_nifti(
    imgs="/path/to/4d_data.nii.gz",
    design=X,
    contrast=C,
    mask_img="/path/to/mask.nii.gz",
    tfce=True,
    two_tailed=True,
    zstat=True,
    n_permutations=1000,
    output_prefix="/results/test_tfce"
)
```

### Using the Dataset Interface

You can also use the [Dataset](dataset.md) class to automatically load your files, infer data types (NIfTI vs. NumPy), and run the appropriate permutation analysis under the hood:

```python
from prism.datasets import Dataset

dataset = Dataset(
    data="brain_maps_4d.nii.gz",            # or .csv, .npy, np.ndarray, or Nifti1Image
    design="design_matrix.csv",            # or np.ndarray
    contrast="contrast.csv",               # 1D or 2D numpy array
    output_prefix="/output/dir/prefix",    # output path prefix
    n_permutations=1000
)

results = dataset.permutation_analysis()
```

See the [Dataset documentation](dataset.md) for more details on configuration and supported inputs.

---

## 🖥️ Command Line Interface (CLI)

The CLI is modeled after PALM and supports similar options:

```bash
prism \
  -i data.nii.gz \
  -d design.csv \
  -t contrast.csv \
  -f f_contrast_indices.csv \
  -o /results/test \
  -T \
  --tfce \
  --two-tailed \
  --flip_signs \
  --accel tail \
  --correct_across_contrasts \
  --f_only \
  --zstat
```

> ⚠️ Many PALM flags are **not yet implemented** in PRISM. Unrecognized arguments will be ignored with a warning. See full list in source code (`NON_IMPLEMENTED_ARGS`).

---
