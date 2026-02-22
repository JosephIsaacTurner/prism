# Quickstart

This guide provides quick examples of core PRISM features.

---

## 🔬 Permutation Analysis

PRISM allows you to run permutation tests on statistical maps using:
- **In-memory data** (NumPy arrays)
- **NIfTI files** (4D or list of 3D images)
- **Dataset objects** (automatically handles both input types)

---

### Example 1: With In-Memory NumPy Data

```python
import numpy as np
from prism.permutation_inference import permutation_analysis

n_samples, n_features, n_regressors = 100, 50, 2
Y = np.random.randn(n_samples, n_features)     # Your data
X = np.random.randn(n_samples, n_regressors)   # Design matrix
C = np.array([1, -1])                           # Contrast vector

results = permutation_analysis(
    data=Y,
    design=X,
    contrast=C,
    n_permutations=5000,
    output_prefix="prism_numpy_output"
)
```

---

### Example 2: With 4D NIfTI Image and TFCE

```python
import numpy as np
from prism.permutation_inference import permutation_analysis_nifti

X = np.random.randn(100, 2)
C = np.array([[1, 0], [0, 1]])

results = permutation_analysis_nifti(
    imgs="path/to/4d_data.nii.gz",
    design=X,
    contrast=C,
    mask_img="path/to/mask.nii.gz",
    tfce=True,
    two_tailed=True,
    zstat=True,
    n_permutations=1000,
    output_prefix="prism_nifti_output"
)
```

---

### Example 3: With the Dataset Class

```python
from prism.datasets.dataset import Dataset
import numpy as np

n_samples, n_features, n_regressors = 100, 50, 2
Y_mem = np.random.randn(n_samples, n_features)
X_mem = np.random.randn(n_samples, n_regressors)
C_mem = np.array([1, -1])

dataset_mem = Dataset(
    data=Y_mem,
    design=X_mem,
    contrast=C_mem,
    output_prefix="prism_dataset_output",
    n_permutations=1000
)

results_mem = dataset_mem.permutation_analysis()
```

---

## 🗺️ Spatial Similarity Analysis

Compare statistical maps across datasets.

```python
from prism.datasets.dataset import Dataset
from prism.spatial_similarity import spatial_similarity_permutation_analysis
import numpy as np

# Simulate two datasets
Y1 = np.random.randn(100, 50) + 0.5
X1 = np.random.randn(100, 2)
C1 = np.array([1, 0])

Y2 = np.random.randn(100, 50) + 0.5
X2 = np.random.randn(100, 2)
C2 = np.array([1, 0])

dataset_one = Dataset(data=Y1, design=X1, contrast=C1, n_permutations=100)
dataset_two = Dataset(data=Y2, design=X2, contrast=C2, n_permutations=100)

spatial_similarity_results = spatial_similarity_permutation_analysis(
    datasets=[dataset_one, dataset_two],
    accel_tail=True
)

print(spatial_similarity_results.corr_matrix_ds_ds)
```
