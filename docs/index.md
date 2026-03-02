# Welcome to Prism

**Fast, modular, and extensible permutation-based statistical inference for neuroimaging.**

Prism is a Python library for running fast, scalable, and fully nonparametric statistical analyses on brain imaging data. It replicates much of the core functionality of Anderson Winkler's [PALM](https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/palm/user_guide.html), but without MATLAB dependencies.

---

## 🚀 Features

- **Mass univariate GLM analysis** with flexible contrast modeling
- **Permutation-based testing** including sign flips and blockwise shuffling
- **Support for TFCE**, FDR, FWE (Westfall–Young), and GPD-based p-value tail estimation
- **Voxelwise map comparison tools** for assessing spatial similarity
- **CLI interface** modeled after PALM
- Works directly with **NIfTI files** or **NumPy arrays**

---

## Minimal Example

```python
import numpy as np
from prism.datasets import Dataset

Y = np.random.randn(100, 50)        # Brain data (samples x voxels)
X = np.random.randn(100, 2)         # Design matrix
C = np.array([1, -1])               # Contrast

dataset = Dataset(
    data=Y,
    design=X,
    contrast=C,
    output_prefix="prism_example",
    n_permutations=1000
)

results = dataset.permutation_analysis()
```

---

## License

Prism is released under the [MIT License](LICENSE).
