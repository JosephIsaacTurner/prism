# Installation

To install Prism, clone the repository and install it in editable mode:

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
