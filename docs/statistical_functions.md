## Statistical Functions

The statistical functions used for permutation testing in PRISM are determined by the `stat_function` and `f_stat_function` parameters. These can be set to `"auto"`, `"pearson"`, or a custom callable.

All of the default statistical functions are based on the GLM framework. The choice of function depends on whether you are performing a t-test or an F-test, and whether you are using variance groups for heteroscedastic data.

All default statistical funcitons use jax.numpy and are jitted for performance. This is one of the reasons why PRISM is so fast!

---

### 🔧 Default Behavior (`stat_function="auto"`, `f_stat_function="auto"`)

The selected function depends on two key flags:

- `zstat`: whether to convert test statistics to z-scores
- `vg_auto` or `variance_groups`: whether variance groups are used for heteroscedastic data

#### No Variance Groups (`vg_auto=False`, `variance_groups=None`):

| Test Type | `zstat=False`                   | `zstat=True`                  |
|-----------|----------------------------------|-------------------------------|
| **t-test**| `prism.stats.glm.t`              | `prism.stats.glm.t_z`         |
| **F-test**| `prism.stats.glm.F`              | `prism.stats.glm.F_z`         |

#### With Variance Groups (`vg_auto=True` or `variance_groups` is set):

| Test Type | `zstat=False`                        | `zstat=True`                         |
|-----------|--------------------------------------|--------------------------------------|
| **t-test**| `prism.stats.glm.aspin_welch_v`      | `prism.stats.glm.aspin_welch_v_z`    |
| **F-test**| `prism.stats.glm.G`                  | `prism.stats.glm.G_z`                |

---

### Pearson-Based Metrics (`stat_function="pearson"`, `f_stat_function="pearson"`)

This uses correlation-based tests instead of GLM t/F-statistics:

| Test Type | `zstat=False`                     | `zstat=True`                      |
|-----------|-----------------------------------|-----------------------------------|
| **t-test**| `prism.stats.glm.pearson_r`       | `prism.stats.glm.fisher_z`        |
| **F-test**| `prism.stats.glm.r_squared`       | `prism.stats.glm.r_squared_z`     |

---

### Using Custom Functions

You can pass your own functions for `stat_function` and `f_stat_function`. These must match the following signatures depending on whether variance groups are in use:

#### 🔹 Custom **t-statistic** function

- **Without variance groups** (`vg_auto=False`, no `variance_groups`):
  ```python
  def my_t_stat(data, design, contrast): ...
  ```

- **With variance groups** (`vg_auto=True` or `variance_groups` provided):
  ```python
  def my_t_stat(data, design, contrast, groups, n_groups): ...
  ```

#### 🔹 Custom **F-statistic** function

- **Without variance groups**:
  ```python
  def my_f_stat(data, design, contrast_matrix): ...
  ```

- **With variance groups**:
  ```python
  def my_f_stat(data, design, contrast_matrix, groups, n_groups): ...
  ```

Where:
- `data`: a 2D array (samples × features)
- `design`: design matrix for the GLM
- `contrast` or `contrast_matrix`: vector or matrix specifying contrasts
- `groups`: 1D array of group IDs (integers)
- `n_groups`: number of unique variance groups