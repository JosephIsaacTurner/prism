# Spatial Similarity Analysis

The `spatial_similarity_permutation_analysis()` function allows you to compare the spatial similarity of statistical maps across multiple datasets, and optionally against fixed reference maps. It uses permutation testing to assess the significance of observed correlations.

---

### Datasets

You'll need to understand how to define and work with `Dataset` objects in PRISM. See the [Dataset documentation](dataset.md) for more details.

## Usage Example

```python
from prism.datasets import Dataset
from prism.spatial_similarity import spatial_similarity_permutation_analysis

# Load datasets from saved PRISM configs
dataset_one = Dataset(
    config_path = '/path/to/prism_output/prism_config.json'
)

dataset_two = Dataset(
    config_path = '/path/to/another_output/prism_config.json'
)

# Run spatial similarity analysis
spatial_similarity_results = spatial_similarity_permutation_analysis(
    datasets=[dataset_one, dataset_two]
)

# View results (returns an sklearn Bunch object)
spatial_similarity_results
```

---

## What is a Reference Map?

A `reference_map` is a static statistical map (either a NIfTI file or a NumPy vector) that is **not permuted**. Instead, it remains constant across all permutations. This is useful if you want to compare datasets to a canonical or literature-derived map while still performing permutation testing on your datasets.

You can pass in:
- A single reference map (`.nii`, `nib.Nifti1Image`, or `np.ndarray`)
- A list of reference maps (mix and match formats)

---

## Number of Permutations

The number of permutations used will be the **smallest number of permutations specified** across all datasets passed into the function. This ensures consistent and valid comparisons across datasets.

---

## Using Multiple Contrasts

If your dataset(s) include a **2D contrast matrix** (e.g., multiple contrast vectors), then:

- **By default**, we will only use the **first contrast vector** in that matrix when computing spatial similarity and permutations (i.e., t-tests).
- If you would like to run an **F-test across all contrast vectors** in the matrix instead, you must specify `f_only=True` when creating the `Dataset` object.
- If `f_only` is not set, **F-tests will not be used**, and only the first contrast vector is evaluated.

---

## Output Overview

The result is an `sklearn.utils.Bunch` object with the following keys:

- `corr_matrix_ds_ds`:  
  Correlation matrix of observed spatial similarity between datasets.  
  **Shape** = `(n_datasets, n_datasets)`

- `corr_matrix_ds_ref`:  
  Correlation matrix of similarity between each dataset and each reference map.  
  **Shape** = `(n_datasets, n_reference_maps)`

- `p_matrix_ds_ds`:  
  P-values for the dataset-to-dataset correlations. Diagonal = `NaN`.  
  **Shape** = `(n_datasets, n_datasets)`

- `p_matrix_ds_ref`:  
  P-values for the dataset-to-reference correlations.  
  **Shape** = `(n_datasets, n_reference_maps)`

- `corr_matrix_perm_ds_ds`:  
  Null distribution of dataset-to-dataset correlations across permutations.  
  **Shape** = `(n_permutations, n_datasets, n_datasets)`

- `corr_matrix_perm_ds_ref`:  
  Null distribution of dataset-to-reference correlations across permutations.  
  **Shape** = `(n_permutations, n_datasets, n_reference_maps)`

---

## Interpreting the Results

- **Correlation matrices (`corr_matrix_...`)**:  
  Show the observed spatial similarity (e.g., Pearson r) between datasets or between datasets and reference maps.

- **P-value matrices (`p_matrix_...`)**:  
  Indicate whether the observed correlations are statistically significant based on the null distribution from permutations.

- **Permutation matrices (`corr_matrix_perm_...`)**:  
  Contain the full distribution of correlation values under permutation. You can use these to visualize or compute empirical thresholds.

## ⚙️ Advanced Options

### Custom Similarity Metrics

You can override the default similarity metric (Pearson correlation) by providing your own function via the `compare_func` argument. This function should accept **two 1D NumPy arrays** (the flattened statistical maps being compared) and return a **single scalar similarity value**.

For example, to use **Spearman rank correlation** instead of Pearson:

```python
from prism.stats import spearman_r
from prism.spatial_similarity import spatial_similarity_permutation_analysis

spatial_similarity_results = spatial_similarity_permutation_analysis(
    datasets=[dataset_one, dataset_two],
    compare_func=spearman_r
)
```

Any valid function that follows the signature `func(x: np.ndarray, y: np.ndarray) -> float` can be used.

---

### One-Tailed or Two-Tailed Tests

By default, the permutation test uses **two-tailed p-values**, meaning it considers both positive and negative deviations from the null.  
If you want to perform a **one-tailed test** (i.e., test only for high similarity), specify `two_tailed=False`:

```python
spatial_similarity_results = spatial_similarity_permutation_analysis(
    datasets=[dataset_one, dataset_two],
    two_tailed=False
)
```

Use this when you're only interested in testing for unusually **strong** similarity, not dissimilarity.

---

### Accelerated P-Value Estimation (Tail Modeling)

To obtain more precise p-values with fewer permutations, you can enable **tail acceleration** by setting `accel_tail=True`. This fits a **generalized Pareto distribution (GPD)** to the extreme values (tail) of the permutation distribution, allowing accurate estimation of small p-values without requiring thousands of permutations.

This approach is described in:  
*A. Winkler et al., NeuroImage (2016),*  
["Faster permutation inference in brain imaging"](https://doi.org/10.1016/j.neuroimage.2016.05.068)

```python
spatial_similarity_results = spatial_similarity_permutation_analysis(
    datasets=[dataset_one, dataset_two],
    accel_tail=True
)
```

This is especially useful for exploratory analyses or when compute time is limited.
