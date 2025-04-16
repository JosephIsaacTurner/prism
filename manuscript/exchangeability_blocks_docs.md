## Freely exchangeable data

For designs in which the data can be shuffled freely, there is no need to specify exchangeability blocks, and all observations are implicitly be assigned to the same large block. This also implies that the data is homoscedastic, i.e., same variance for all observations, and as a consequence, there is just one variance group (VG) for all data. The statistic that is computed by default is either t or F, depending on the rank of the contrast.

## Data exchangeable within block

If there are restrictions on exchangeability, blocks of data can be defined such that shuffling happens within block only. The exchangeability blocks file contains a single column vector with integer indices, each specifying a block. This file is supplied with the option `-eb`. An example is:

```
1
1
1
1
2
2
2
```

Permutations happen only between the observations that share the same index, i.e., within block only. For this kind of permutation, the blocks are not required to be of the same size. Moreover, it is no longer necessary to assume homoscedasticity, and by using the option `-vg auto`, each block is considered a variance group (VG). For the example above, this means that the automatic variance groups are:

```
1
1
1
1
2
2
2
```

Variances are estimated for each block, and statistics that are robust to heteroscedasticity are computed. In this case, instead of *t*, the Aspin-Welch *v* is calculated, and instead of *F*, the *G* statistic, which is a generalisation of all these common statistics.

However, if the user wants to group the observations differently based on other variance assumptions, irrespectively to the specification of exchangeability blocks, a file with variance groups indices can be supplied with the option `-vg`. The file format for the variance groups is always a column vector with integer indices, similar as the example above.

## Blocks of data exchangeable as a whole

If there are restrictions on exchangeability, and if instead of shuffling within block, the blocks as a whole can be shuffled while keeping the observations inside the block in the same order, the exchangeability blocks file is supplied with the option `-eb`, exactly as above, and the flag `-whole` (whole-block shuffling) is supplied. An example of such file is:

```
1
1
1
2
2
2
```

Permutations and/or sign-flips, as applicable, will rearrange the blocks as a whole, not each observation individually. All blocks must be of the same size. Moreover, it is no longer necessary to assume homoscedasticity, and by using the option `-vg auto`, variance groups are defined such that the first group includes all the first observations in each block, the second group all the second observations in each block, and so on. For the example above, the automatic variance groups are:

```
1
2
3
1
2
3
```

Variances are then estimated within group, and statistics that are robust to heteroscedasticity are computed, i.e., instead of *t*, *v* is calculated, and instead of *F*, the *G* statistic is calculated. As in the within block case, the default definition of variance groups can be overridden with the specification of a variance groups file with the option `-vg`.

## Simultaneous whole- and within-block permutation

If the file supplied with the exchangeability blocks has just 1 column, the options `-within` and `-whole` can be used together to indicate that blocks can be shuffled as a whole, and that the observations can be further shuffled within-block.

---

## Complex dependence between data

It is possible to have multiple levels of complexity, each encompassing within-block and whole-block shuffling, which are then nested into each other. This can be done by supplying an exchangeability blocks file not with one, but with multiple columns. Each column indicates a deeper level of dependence. Indices on one level indicate how the unique sub-indices of the next level should be shuffled:

- Positive indices at a given level indicate that the sub-indices of the next level should be shuffled as a whole, akin to whole-block shuffling described above.
- Negative indices at given level indicate that the corresponding sub-indices in the next level should remain fixed and their own sub-sub-indices should be shuffled, akin to within block permutation described above.

With this more complex structure, the option `-whole` and `-within`, even if supplied, are ignored, as this information is embedded, at the multiple levels, in the file with the block definitions. In fact, even in the simpler cases of no blocks, within and whole block permutation, discussed above, can be fully specified using a multi-column exchangeability blocks file, regardless of these options.

---

### Example 1: Freely exchangeable data.

This is equivalent to having all observations in a single block, with no restrictions.

```
1 1
1 2
1 3
1 4
1 5
1 6
```

The 1st column indicates that all indices in the next level (1,...,6) can be shuffled with each other. This also implies a single large variance group that encompasses all observations.

### Example 2: Three blocks of data that are exchangeable within block only.

```
-1 1
-1 1
-1 2
-1 2
-1 3
-1 3
```

The negative indices in the 1st column indicate that the sub-indices in the next column cannot be shuffled. The indices for each individual observation here were omitted, but could equivalently have been specified as:

```
-1 1 1
-1 1 2
-1 2 1
-1 2 2
-1 3 1
-1 3 2
```

The automatic VG looks like this:

```
1
1
2
2
3
3
```

The positive indices in the 2nd column indicate that the respective sub-indices in the next column can be shuffled.

### Example 3: Three blocks of data that are exchangeable as a whole.

```
1 1
1 1
1 2
1 2
1 3
1 3
```

The positive index in the 1st column indicates that the respective sub-indices in the next column (level) are exchangeable as a whole. The same could equivalently have been specified as:

```
1 -1 1
1 -1 2
1 -2 1
1 -2 2
1 -3 1
1 -3 2
```

The 1st column indicates that the sub-indices in the next level (2nd column) are exchangeable as a whole. The negative indices in the 2nd level indicate that each respective sub-index, now in the 3rd column, is exchangeable with each other, i.e., within the blocks specified in the previous level.

The automatic VG looks like this:

```
1
2
1
2
1
2
```

These same basic rules apply to hierarchies of blocks that can become far more complex.
