<p align="left">
    <img src="assets/logov2.svg" alt="prism Logo" width="400">
</p>

## Overview

Prism is a Python library designed for performing **fast, efficient, and scalable** statistical analysis on neuroimaging data using **permutation-based methods**. It provides tools for running mass univariate analyses using **General Linear Models (GLMs)** and comparing statistical map similarity in a pythonic manner without relying on external software.

It is designed to largely reproduce the methods Anderson Winkler's PALM (distributed with FSL), without requiring matlab. 

To read more about why this project is needed, see the [manuscript](manuscript/manuscript.md).

## Features

- **Permutation-based statistical testing** for robust inferences.
- **Efficient GLM analysis** tailored for neuroimaging datasets.
- **Statistical map similarity comparisons** for assessing voxelwise similarity of brain maps.
- **Support for neuroimaging-specific data structures** (e.g., NIfTI)
- **Modular and extensible** framework to integrate with existing workflows.

## Installation

You can install prism using pip and git:

```bash
git clone https://github.com/josephisaacturner/prism.git
cd prism
pip install -e .
```

If you are using MacOS with silicon, you may need to install the `jax` library separately:

```bash
pip install jax-metal
```

## Usage

### Example: Running a Second-Level GLM
```python
from prism.inference import permutation_analysis
from prism.stats import t
from nilearn.maskers import NiftiMasker

# Random seed for reproducibility
random_seed = 42

# Load neuroimaging data
masker = NiftiMasker(mask_img="mask.nii.gz").fit()
data = masker.transform("data.nii.gz") # 4d data, or pass in a list of filepaths to NIfTI files

design = np.load("design_matrix.npy") # Load design matrix (shape n_subjects x n_features)
contrast = np.array([1, 0, 0]) # Assuming on VOI, and two nuisance regressors/intercepts

# Run the ground-truth analysis
t_values = welchs_t_glm(data, design_matrix, contrast_matrix)

# Run permutation analysis
n_permutations = 1000
unc_p, fdr_p, fwe_p = permutation_analysis(data, design, contrast, welchs_t_glm, random_seed, n_permutations, two_tailed=True, accel_tail=True)

# Save results as NIfTI files
masker.inverse_transform(t_values).to_filename("t_values.nii.gz")
masker.inverse_transform(unc_p).to_filename("uncorrected_p_values.nii.gz")
masker.inverse_transform(fdr_p).to_filename("fdr_corrected_p_values.nii.gz")
masker.inverse_transform(fwe_p).to_filename("fwe_corrected_p_values.nii.gz")
```

### Example: Comparing Statistical Maps
```python
print("I'll add an example here soon!")
```

## Contributing

We welcome contributions! If you want to contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit: `git commit -m "Add new feature"`
4. Push to your branch: `git push origin feature-branch-name`
5. Open a pull request!

## License

Prism is open-source and available under the MIT License.

## Project Structure

The project is organized as follows:

```
prism/
├── prism/                            # Core Python package
│   ├── data/                         # Directory for data files (brain image templates, etc.)
│   ├── datasets/
│   │   ├── __init__.py               
│   │   ├── dataset.py                # Handles Dataset class/object
│   │   ├── utils.py                  # Utilities for fetching datasets/masks/atlases
│   ├── stats/
│   │   ├── __init__.py               
│   │   ├── glm.py                    # General Linear Model functions
│   │   ├── miscellaneous.py          # Miscellaneous functions
│   ├── permutation_inference.py      # Inference functions (hypothesis testing classes, etc.)
│   ├── permutation_logic.py          # Functions for implementing permutation logic
│   ├── preprocessing.py              # Functions to load data and preprocess it
│   ├── prism_cli.py                  # Command-line interface for prism
│   ├── spatial_similarity.py         # Functions for assessing similarity of statistical maps
│   ├── tfce.py                       # Functions for Threshold-Free Cluster Enhancement (TFCE)
├── notebooks/                        # Jupyter notebooks for examples and tutorials
├── tests/                            # Unit and integration tests
├── assets/                           # Directory for static assets (e.g., logo)
├── manuscript/                       # Manuscript outlining background and methodology
├── README.md                         # Introduction to the project for new users
├── requirements.txt                  # List of dependencies
├── LICENSE                           # License information
├── pyproject.toml                    # Project metadata and dependencies
└── .gitignore                        # Files and directories to be ignored by Git

```
