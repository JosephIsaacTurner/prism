# Dataset Class Documentation

The `Dataset` class defines the structure and configuration for a single statistical dataset used in PRISM analyses. It supports a wide variety of data formats and allows for customized permutation testing workflows, including TFCE and F-statistics.

You can also use the `Dataset` class to run spatial similarity analyses, which compare the spatial similarity of statistical maps across multiple datasets. Take a look at the [spatial similarity documentation](spatial_similarity.md) for more details.

---

## Creating a Dataset

### Option 1: Using a Config File

```python
from prism.datasets import Dataset

# Load a previously saved dataset
dataset = Dataset(config_path="/path/to/prism_config.json")
```

### Option 2: Specifying Inputs Directly

```python
from prism.datasets import Dataset

dataset = Dataset(
    data="brain_maps_4d.nii.gz",            # or .csv, .npy, np.ndarray, or Nifti1Image
    design="design_matrix.csv",            # or np.ndarray
    contrast="contrast.csv",               # 1D or 2D numpy array
    output_prefix="/output/dir/prefix",    # output path prefix
    n_permutations=1000
)
```

You must either specify:
- A valid `config_path`, _or_
- All of the following: `data`, `design`, and `contrast`

---

## Supported Data Formats

The `data`, `design`, and `contrast` inputs can be provided in various formats:

- **4D NIfTI files** (`.nii`, `.nii.gz`)
- **CSV files** (with either filepaths to NIfTIs or raw data matrix)
- **NumPy files** (`.npy`)
- **In-memory**: `np.ndarray`, `nib.Nifti1Image`

`contrast` can be:
- **1D array**: A single contrast vector (t-test)
- **2D array**: Multiple contrast vectors (F-test support via `f_only=True` or `f_contrast_indices`)

Whenever working with CSVs, be aware that prism expects CSV inputs to be headerless.

---

## What is `output_prefix`?

The `output_prefix` determines how output files are named and saved. It is composed of:

```
output_prefix = output_dir + "/" + prefix
```

All outputs (e.g. maps, p-values, configuration) will use this prefix. Example:

```
output_prefix = "/results/prism/test1"
# -> Output files: test1_statmap.nii.gz, test1_config.json, etc.
```

---

## Running a Permutation Test

```python
results = dataset.permutation_analysis()
```
This runs the full permutation-based GLM analysis. Results are returned as an `sklearn.Bunch`.

For more information on permutation testing, see the [permutation analysis documentation](permutation_analysis.md).

---

## Parameters

All parameters below are optional unless marked required:

| Parameter              | Type                  | Description |
|------------------------|------------------------|-------------|
| `data` *(required)*    | str, np.ndarray, nib.Nifti1Image | Input data |
| `design` *(required)*  | str, np.ndarray        | Design matrix |
| `contrast` *(required)*| str, np.ndarray        | Contrast vector or matrix |
| `output_prefix`        | str                    | Output directory + filename prefix |
| `f_contrast_indices`   | str, np.ndarray        | Boolean array pointing to contrasts used in an F test |
| `two_tailed`           | bool                   | Whether to compute two-tailed p-values (default: True) |
| `exchangeability_matrix`| str, np.ndarray       | Exchangeability blocks for permutations |
| `vg_auto`              | bool                   | Automatically assign variance groups based on exchangeability matrix |
| `variance_groups`      | str, np.ndarray        | Predefined variance group vector |
| `within`               | bool                   | Permute within groups defined by exch. matrix |
| `whole`                | bool                   | Permute blocks as a whole; (needs `variance_groups` or `vg_auto`) |
| `flip_signs`           | bool                   | Enable sign-flip permutations |
| `stat_function`        | str, callable          | Custom or prebuilt t-statistic function (default: "auto") |
| `f_stat_function`      | str, callable          | Custom or prebuilt F-statistic function (default: "auto") |
| `f_only`               | bool                   | Use only F-test rather than t-tests |
| `n_permutations`       | int                    | Number of permutations to run |
| `accel_tail`           | bool                   | Use GPD tail-fitting for improved p-value precision |
| `save_1minusp`         | bool                   | Save 1 - p maps |
| `save_neglog10p`       | bool                   | Save -log10(p) maps |
| `correct_across_contrasts` | bool              | Correct across contrast columns |
| `random_state`         | int or np.random.RandomState | Seed or RNG |
| `demean`               | bool                   | Demean data and design before GLM |
| `zstat`                | bool                   | Convert test stats to z-statistics |
| `mask_img`             | str, nib.Nifti1Image   | Optional brain mask for NIfTI input |
| `tfce`                 | bool                   | Enable TFCE (only valid for NIfTI input) |
| `save_fn`, `permute_fn`| callable               | Optional callbacks for saving/intercepting logic |

---

## Notes on Contrast Behavior

- If using Dataset for regular permutation analysis:
    - All contrasts are used, and a separate statistical map is generated for each contrast.
    - If f_contrast_indices are specified, an F test is also performed across the specified contrasts.
    - If `f_only` is set, only the F test is performed.

- If using Dataset for spatial similarity analysis:
    - If using a **2D contrast matrix**, only the **first contrast** is used by default.
    - To perform an F-test across contrast vectors, either:
    - Set `f_only=True` (use all contrasts) and specify `f_contrast_indices` to indicate which contrasts to include.

---

## Saving & Loading Configs

You can save the current configuration for reproducibility:

```python
config_path = dataset.save_config()
# -> Creates a JSON config: /your/output_prefix_config.json
```

To reload it later:
```python
dataset = Dataset(config_path=config_path)
```

---

## Notes on Statistical Functions

For most analyses, use the default behavior of `stat_function="auto"` and `f_stat_function="auto"`. This will automatically select the appropriate statistical function based on your data and design matrix. For more details, see the [Statistical Functions documentation](statistical_functions.md).
