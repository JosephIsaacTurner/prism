## PRISM: PeRmutation Inference for Statistical Mapping (library name)
### PRISM: A python library for permutation inference of linear models and assessing similarity of statistical maps

## Summary

PRISM is a Python library for permutation inference for neuroimaging analysis. It provides a fully Pythonic alternative to existing methods, enabling efficient second-level GLM analysis for statistical mapping of brain imaging data. Moreover, PRISM extends permutation inference to assessing the topographic similarity of statistical maps derived from different datasets or modalities, offering a solution to a long-standing challenge in neuroimaging. 

## Introduction

In recent years, the neuroimaging community has increasingly adopted Python as the primary language for data analysis, replacing standalone software suites and MATLAB. The shift toward Python has led to the development of powerful libraries like Nilearn and Neuromaps. However, Python still lacks a robust, fully implemented tool for nonparametric permutation-based inference.b

Another key challenge is assessing statistical map similarity. While spatial correlation is commonly used, traditional methods assume normality and ignore spatial autocorrelation. The spin test addresses this for surface data but does not generalize to volumetric neuroimaging.

PRISM fills these gaps by introducing a voxelwise permutation-based framework for second-level GLMs. It generates null statistical maps through permutation, enabling rigorous spatial correlation comparisons without parametric assumptions. By unifying nonparametric inference and spatial similarity testing, PSTN provides a necessary tool for modern neuroimaging analysis.

## Results

We replicated the core functionality of PALM, including its multi-level block permutation logic. Our Python implementation precisely reproduces PALM's statistical tests (t-test, Aspin–Welch v, F-test, G-statistic, Pearson correlation, and coefficient of determination), matching PALM's results within a tolerance of 1e-5.

Our implementation significantly improves upon PALM's computational performance, achieving a 2-3x speedup in permutation testing through the use of JAX. JAX enables efficient compilation and parallelization of matrix operations essential to mass univariate analyses.

PRISM can function as a drop-in replacement for PALM, the widely-used MATLAB-based permutation inference tool in neuroimaging. PRISM's command-line interface mirrors PALM's, enabling integration into existing workflows by simply substituting PRISM's path for PALM's executable. This allows users to benefit from Python's flexibility and eliminates dependency on MATLAB and its associated licensing constraints.

We have also extended the toolkit by including additional univariate tests such as Cohen's d and the Area Under the Curve (AUC) from Receiver Operating Characteristic (ROC) analyses.

## Methods

The t-test serves as the standard test, while Aspin–Welch v is applied when separate variance groups are present. Similarly, the F-test is the standard F-statistic, whereas the G-statistic provides a generalized version of the F-test suited to scenarios with separate variance groups.

Currently using the Draper-Stoneman method for permuting data (I learned later that people think Freedman and Lane is better, but Draper-Stoneman is more elegant from a software architecture perspective and I'm not sure if its worth refactoring the code to use Freedman and Lane).

We use the Westfall and Young method for FWE correction, and the Benjamini-Hochberg procedure for FDR correction. 

We've also implemented a faster permutation approach using GDP tail approximation, as described by Winkler and colleagues. This method allows for getting more precise p values with fewer permutations.

There are a few things from PALM that we actually haven't implemented, at least not yet. This includes non parametric combination (NPC), multivariate analyses, and spatial statistics for cluster extent and cluster mass (although TFCE is implemented).

## Discussion

...

## References 

Winkler et al. Permutation inference for the general linear model. Neuroimage. 2014 May 15;92(100):381-97. doi: 10.1016/j.neuroimage.2014.01.060

Winkler et al. Faster permutation inference in brain imaging. Neuroimage. 2016 Nov 1;141:502–516. doi: 10.1016/j.neuroimage.2016.05.068

Knijnenburg et al. Fewer permutations, more accurate P-values. Bioinformatics. 2009 May 27;25(12):i161–i168. doi: 10.1093/bioinformatics/btp211

Winkler et al. Multi-level block permutation. Neuroimage. 2015 Dec:123:253-68. doi: 10.1016/j.neuroimage.2015.05.092. Epub 2015 Jun 11.

## My chaotic thoughts

Do permuted maps have similar spatial autocorrelation to the original map? If so, that's fine. If not, that's even more interesting, because it undermines the belief that as long as a null model has equivalent autocorrelation it is valid.

This is an important point: Permuting group-level analyses to derive permuted maps works equally well for dense/parcellated surface, volumetric, and even tractography data. 

Is the method of using generated surrogate maps with equivalent spatial autocorrelation more or less conservative than the permutation approach? I suspect the permutation approach is more conservative.

How does a research project's null hypothesis determine the spatial similarity null model to use?

Does our permutation-based method take sample size of dataset used to inform the input maps into account? How about spin test (I doubt it) or the brainSMASH method (surrogate maps with similar spatial autocorrelation)?

Why do we get bimodal distributions?

Can we combine multiple methods? Let's say we have the underlying subject data for one group-level statistical map, but our reference map is unfixed. Could we permute the data we have access to to create one set of null maps, and then use the spatially autocorrelated surrogate maps on the other side? Does it even make sense to permute both sides or is that cheating?

Can we order our vector by following the gradient of slowest descent down from the peak? No because it wouldn't traverse local maxima. Unspooling algorithm. But if you started at inflection points? Parcellation-based unspooling. (I.e: How to flatten a 3d array while minimizing the euclidean distance between adjacent points in the new 1d vector?) plot_two_d_brain_map(map, parcellation). First, for the parcellation, you'd get the centroid at each parcel and then create a vector and coord matrix of that, where vector would be (n_parcels,) and contain the parcel indices, and coord_matrix would be (n_parcels, 3), where x, y, z is the coordinate of the centroid of the parcel. You'd find out how to order coord_matrix to minimize euclidean distance between adjacent points. Then you'd apply that order to the vector. That's the global, across-parcel organization.
Then, you'd want to get to the within-parcel structure. We'd apply a similar algorithm, parcel-by-parcel, within the parcels to linearly organize the data to minimize euclidean distance between adjacent points while still maintaining the global ordering. Finally you'd have an ordering to apply to the original map vector. The plot would probably have discrete jumps across parcels, but that is totally fine. X axis could have annotations for parcel labels. You could have horizontal lines for FWE/FDR significance. For maxima/minima we have a little label for MNI coord at that point. You could even make it a manhattan-esque plot.

Maintaining symmetry: If your input data is symmetric, your output data will likely be symmetric as well. Which means true null maps will preserve symmetry of input data. When you use the method of random rotations/spins, you likely will lose symmetry in surrogate null maps.