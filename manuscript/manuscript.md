## Statement of Need

In recent years, the neuroimaging community has increasingly adopted Python as the primary language for data analysis, gradually replacing standalone software suites and MATLAB. This shift has been driven by the emergence of robust Python-based libraries such as Nilearn, ENIGMA Toolbox, and Neuromaps, which now support a wide range of neuroimaging analyses. However, a key limitation remains: robust, non-parametric permutation-based inference has not been fully implemented in a pure Python framework.

A second major challenge in neuroimaging is quantifying the similarity between statistical maps. While spatial correlation has been widely used for this purpose, concerns over spatial autocorrelation undermine traditional assumptions of normality and statistical validity. The spin test has provided a solution for surface-based analyses, but this method does not generalize to volumetric neuroimaging data.

To address these gaps, PSTN introduces a permutation-based framework for second-level GLMs that enables the generation of null distributions in a voxelwise manner. By permuting the underlying data, we create null statistical maps that accurately reflect the structure of the null hypothesis. These null maps are then used to construct a distribution of spatial correlations, allowing for rigorous statistical comparison against the observed similarity between two statistical maps. This approach provides a robust, data-driven solution for assessing voxelwise similarity without relying on parametric assumptions.

By integrating non-parametric inference and map comparison into a unified Python-based workflow, PSTN fills a methodological gap in modern neuroimaging analysis.

## Bibliography 

Faster permutation inference in brain imaging. Neuroimage. 2016 Nov 1;141:502–516. doi: 10.1016/j.neuroimage.2016.05.068

Fewer permutations, more accurate P-values. Bioinformatics. 2009 May 27;25(12):i161–i168. doi: 10.1093/bioinformatics/btp211

