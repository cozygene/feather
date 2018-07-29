# Fast permutation testing for heritability and set-tests

FEATHER (Fast pErmutAtion Testing of HERitability) is a fast method for permutation-based testing of marker sets and of heritability. FEATHER is free of parametric and asymptotic assumptions, and is thus guaranteed to properly control for false positive results. Since standard permutation testing is computationally prohibitive, FEATHER combines several novel techniques to obtain speedups of up to eight orders of magnitude.

We currently offer a general Python implementation, and a more limited C++ implementation. 

Comments, questions, requests etc. are welcome at regevs@gmail.com.

## Simple example

The following reads phenotypes, eigenvalues and eigenvectors from a file, and calculates the permutation p-value based on 1000 permutations for each phenotype:

```python

```

The output looks like this:

```
```

For example, the parametric p-value for the first phenotype is xxx, while the permutation p-value is yyy.

## Python

### Installation

TBD

Dependencies: `numpy`, `scipy`. 
Install `tqdm` for progress bar (recommended, but not mandatory).
We also include a slightly modified version of `pylmm` by Nick Furlotte et al.

### Parametric testing



## C++
