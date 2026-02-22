# Installation

The easiest way to install Prism is via PyPI:

```bash
pip install prism-neuro
```

Note: While the package is named `prism-neuro` on PyPI to avoid conflicts, you still `import prism` in your Python code and use the `prism` command in your terminal.

---

## Development Installation

To install Prism from source (e.g., for development), clone the repository and install it in editable mode:

```bash
git clone https://github.com/josephisaacturner/prism.git
cd prism
pip install -e .
```

If you're on MacOS with Apple silicon, you may need to manually install a `jax` dependency:

```bash
pip install jax-metal
```

---

## Dependencies

Prism relies on the following key libraries:

- `numpy`
- `jax`
- `scipy`
- `nibabel`
- `nilearn`
- `scikit-learn`
- `statsmodels`
- `tqdm`
- `pandas`
