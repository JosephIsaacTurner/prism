# Command Line Interface (CLI)

Prism provides a powerful command-line interface designed to be a **drop-in replacement for FSL's PALM**. Most of the core arguments and flags match PALM's syntax, allowing you to easily port existing workflows to Prism.

For a more descriptive guide on many of these parameters, we recommend referring to the official [FSL PALM Documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/palm/user_guide.html).

---

## The `prism` Command

The `prism` command is the main entry point for running permutation-based GLM analyses.

### Basic Usage

```bash
prism -i data.nii.gz -d design.csv -t contrast.csv -n 1000 -o results/prefix
```

### Core Arguments

| Argument | Long Form | Description |
|----------|-----------|-------------|
| `-i` | `--input` | **(Required)** Input data file (`.nii.gz`, `.nii`, `.csv`, `.npy`, `.txt`). |
| `-d` | `--design` | **(Required)** Design matrix file (`.csv`, `.npy`). |
| `-t` | `--contrast` | **(Required)** Contrast file (`.csv`, `.npy`). |
| `-n` | `--n_permutations` | Number of permutations to perform (default: 1000). |
| `-o` | `--output` | Output prefix for all saved files (default: `palm`). |
| `-m` | `--mask` | Mask image file (`.nii`, `.nii.gz`). |

### Statistical & Permutation Flags

| Flag | Description |
|------|-------------|
| `-twotail` | Perform two-tailed tests (default is one-tailed). |
| `-zstat` | Save z-statistics instead of t-statistics. |
| `-T` | Enable **Threshold-Free Cluster Enhancement (TFCE)**. |
| `-fdr` | Produce FDR-adjusted p-values (enabled by default). |
| `-corrcon` | Perform FWE correction across contrasts. |
| `-ise` | Assume independent and symmetric errors (allow sign flipping). |
| `-ee` | Assume exchangeable errors (allow permutations; default). |
| `-accel tail` | Use GPD tail approximation for faster p-value estimation. |
| `-pearson` | Use Pearson correlation instead of GLM t-statistics. |
| `-demean` | Demean data and design before analysis. |

### Exchangeability & Variance Groups

| Argument | Description |
|----------|-------------|
| `-eb` | File defining exchangeability blocks (`.csv`, `.npy`). |
| `-vg auto` | Automatically detect variance groups from the exchangeability matrix. |
| `-within` | Permute within blocks (default for single-level EB). |
| `-whole` | Permute blocks as wholes. |

---

## The `prism_spatial_similarity` Command

This command allows you to compare statistical maps across different Prism analyses or against reference maps.

### Basic Usage

```bash
prism_spatial_similarity 
    --dataset-configs ds1_config.json ds2_config.json 
    --reference-maps canonical_map.nii.gz 
    --output-dir similarity_results 
    --n-permutations 1000
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--dataset-configs` | **(Required)** List of paths to dataset configuration JSON files. |
| `--reference-maps` | List of paths to static reference map files. |
| `--output-dir` | **(Required)** Directory to save similarity matrices and p-values. |
| `--n-permutations` | Number of permutations for the similarity test (default: 1000). |
| `--two-tailed` | Use two-tailed tests for similarity. |
| `--accel-tail` | Use GPD tail approximation. |

---

## PALM Compatibility Note

Prism aims to support the most commonly used PALM flags. If you use an argument from PALM that Prism doesn't yet support, it will print a warning and ignore it rather than crashing. If there's a specific PALM feature you'd like to see implemented, please [open an issue on GitHub](https://github.com/JosephIsaacTurner/prism/issues).
