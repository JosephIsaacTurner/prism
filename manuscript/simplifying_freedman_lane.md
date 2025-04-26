# Simplifying Freedman–Lane

Doing a permutation test with the [general linear model (GLM)](https://en.wikipedia.org/wiki/General_linear_model) in the presence of nuisance variables can be challenging. Let the model be:

$$
\mathbf{Y} = \mathbf{X} \boldsymbol{\beta} + \mathbf{Z} \boldsymbol{\gamma} + \boldsymbol{\epsilon}
$$

where **Y** is a matrix of observed variables, **X** is a matrix of predictors of interest, **Z** is a matrix of covariates (of no interest), and **ε** is a matrix of the same size as **Y** with the residuals.

Because the interest is in testing the relationship between **Y** and **X**, in principle it would be these that would need to be permuted, but doing so also breaks the relationship with **Z**, which would be undesirable. Over the years, many methods have been proposed. A review can be found in [Winkler et al. (2014)](https://doi.org/10.1016/j.neuroimage.2014.01.060); other previous work includes the papers by [Anderson and Legendre (1999)](https://doi.org/10.2307/2676808) and [Anderson and Robinson (2001)](https://doi.org/10.2307/2680923).

One of these various methods is the one published in [Freedman and Lane (1983)](https://www.jstor.org/stable/2531155), which consists of permuting data that has been residualised with respect to the covariates, then estimated covariate effects added back, then the full model fitted again. The procedure can be performed through the following steps:

1. Regress **Y** against the full model:
   $$
   \mathbf{Y} = \mathbf{X} \boldsymbol{\beta} + \mathbf{Z} \boldsymbol{\gamma} + \boldsymbol{\epsilon}
   $$

2. Regress **Y** on **Z** only:
   $$
   \mathbf{Y} = \mathbf{Z} \boldsymbol{\gamma} + \boldsymbol{\epsilon}_Z
   $$

3. Generate permuted datasets:
   $$
   \mathbf{Y}_j^* = \mathbf{P}_j \hat{\boldsymbol{\epsilon}}_Z + \mathbf{Z} \hat{\boldsymbol{\gamma}}
   $$

4. Regress each **Yⱼ*** on the full model and compute statistic **Tⱼ***.

5. Compare observed **T₀** to null distribution to get p-value.

Steps 1–4 can be written concisely as:

$$
(\mathbf{P}_j \mathbf{R}_Z + \mathbf{H}_Z) \mathbf{Y} = \mathbf{X} \boldsymbol{\beta} + \mathbf{Z} \boldsymbol{\gamma} + \boldsymbol{\epsilon}
$$

---

## Theoretical Justification

From [Winkler et al. (2014)](https://doi.org/10.1016/j.neuroimage.2014.01.060):

> "[...] the model can be expressed simply as  
> **Pⱼ R_Z Y = Xβ + Zγ + ε**,  
> implying permutations can be performed by permuting rows of the residual-forming matrix **R_Z**."

---

### Lemma 1:

$$
\mathbf{R}_Z \mathbf{H}_Z = \mathbf{H}_Z \mathbf{R}_Z = \mathbf{0}
$$

Proof:  
$$
\mathbf{R}_Z = \mathbf{I} - \mathbf{H}_Z \Rightarrow  
\mathbf{R}_Z \mathbf{H}_Z = (\mathbf{I} - \mathbf{H}_Z) \mathbf{H}_Z = \mathbf{H}_Z - \mathbf{H}_Z = \mathbf{0}
$$

---

### Lemma 2 (Frisch–Waugh–Lovell Theorem):

From the model:

$$
\mathbf{Y} = \mathbf{X} \boldsymbol{\beta} + \mathbf{Z} \boldsymbol{\gamma} + \boldsymbol{\epsilon}
$$

We have the equivalent:

$$
\mathbf{R}_Z \mathbf{Y} = \mathbf{R}_Z \mathbf{X} \boldsymbol{\beta} + \mathbf{R}_Z \boldsymbol{\epsilon}
$$

**Derivation:**

$$
\mathbf{R}_Z \mathbf{Y} = \mathbf{R}_Z (\mathbf{X} \boldsymbol{\beta} + \mathbf{Z} \boldsymbol{\gamma} + \boldsymbol{\epsilon}) \\
= \mathbf{R}_Z \mathbf{X} \boldsymbol{\beta} + \mathbf{R}_Z \mathbf{Z} \boldsymbol{\gamma} + \mathbf{R}_Z \boldsymbol{\epsilon} \\
= \mathbf{R}_Z \mathbf{X} \boldsymbol{\beta} + (\mathbf{I} - \mathbf{H}_Z)\mathbf{Z} \boldsymbol{\gamma} + \mathbf{R}_Z \boldsymbol{\epsilon} \\
= \mathbf{R}_Z \mathbf{X} \boldsymbol{\beta} + (\mathbf{Z} - \mathbf{Z} \mathbf{Z}^+ \mathbf{Z}) \boldsymbol{\gamma} + \mathbf{R}_Z \boldsymbol{\epsilon} \\
= \mathbf{R}_Z \mathbf{X} \boldsymbol{\beta} + \mathbf{0} \boldsymbol{\gamma} + \mathbf{R}_Z \boldsymbol{\epsilon} \\
= \mathbf{R}_Z \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}_Z
$$

---

## Main Result

Recall the Freedman–Lane form:

$$
(\mathbf{P}_j \mathbf{R}_Z + \mathbf{H}_Z) \mathbf{Y} = \mathbf{X} \boldsymbol{\beta} + \mathbf{Z} \boldsymbol{\gamma} + \boldsymbol{\epsilon}
$$

Apply **R_Z** to both sides (per Lemma 2):

$$
\mathbf{R}_Z (\mathbf{P}_j \mathbf{R}_Z + \mathbf{H}_Z) \mathbf{Y} = \mathbf{R}_Z \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}_Z
$$

Distribute:

$$
\mathbf{R}_Z \mathbf{P}_j \mathbf{R}_Z \mathbf{Y} + \mathbf{R}_Z \mathbf{H}_Z \mathbf{Y} = \mathbf{R}_Z \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}_Z
$$

Use Lemma 1:

$$
\mathbf{R}_Z \mathbf{P}_j \mathbf{R}_Z \mathbf{Y} + \mathbf{0} = \mathbf{R}_Z \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}_Z
$$

Therefore:

$$
\mathbf{R}_Z \mathbf{P}_j \mathbf{R}_Z \mathbf{Y} = \mathbf{R}_Z \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}_Z
$$

Thus, reversing it:

$$
\mathbf{P}_j \mathbf{R}_Z \mathbf{Y} = \mathbf{X} \boldsymbol{\beta} + \mathbf{Z} \boldsymbol{\gamma} + \boldsymbol{\epsilon}
$$

**Conclusion:**  
The hat matrix **H_Z** cancels out. It is not needed. Permuting **R_Z Y** suffices.
