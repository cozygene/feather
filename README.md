# Fast permutation testing for heritability and set-tests

FEATHER (Fast pErmutAtion Testing of HERitability) is a method for fast permutation-based testing of marker sets and of heritability. FEATHER is free of parametric and asymptotic assumptions, and is thus guaranteed to properly control for false positive results. Since standard permutation testing is computationally prohibitive, FEATHER combines several novel techniques to obtain speedups of up to eight orders of magnitude.

We currently offer a general Python implementation, and a more limited, but quite faster, C++ implementation. 

Comments, questions, requests etc. are welcome at regevs@gmail.com.

Citation: [Schweiger, Regev, Eyal Fisher, Omer Weissbrod, Elior Rahmani, Martina Müller-Nurasyid, Sonja Kunze, Christian Gieger, Melanie Waldenberger, Saharon Rosset, and Eran Halperin. "Detecting heritable phenotypes without a model using fast permutation testing for heritability and set-tests." Nature communications 9, no. 1 (2018): 4919](https://www.nature.com/articles/s41467-018-07276-w).

## Simple example

The following reads phenotypes, eigenvalues and eigenvectors from a file, and calculates the permutation p-value based on 1000 permutations for each phenotype:

```
python permutation_testing.py --kinship_eigenvalues data/example.eigenval   \
                              --kinship_eigenvectors data/example.eigenvec  \
                              --phenotypes data/phenotypes.txt              \
                              --permutation --fast                          \
                              --n_permutations 1000 
```

The output looks like this:

```
n_phen  h2_est  param_p perm_p                                                                                                                                                                              
0	0.00000	0.5	0.538
1	0.20404	0.1034	0.093
2	0.57383	0.0009423	0
3	0.33590	0.01151	0.015
4	0.54420	0.00035	0.002
5	0.54324	0.0004034	0.001
6	1.00000	4.422e-11	0
7	0.71340	3.888e-05	0
8	0.85968	9.523e-08	0
9	1.00000	1.64e-09	0
```

For example, the parametric p-value for the first phenotype is `0.5`, while the permutation p-value is `0.538`.

## Python

### Installation

Simply download or clone this directory.

Dependencies: `numpy`, `scipy`, [`attrs`](http://www.attrs.org/en/stable/).

Install [`tqdm`](https://github.com/tqdm/tqdm) for progress bar (recommended, but not mandatory).

We also include (no installation necessary) a slightly modified version of [`pylmm`](https://github.com/nickFurlotte/pylmm) by Nick Furlotte et al.

### Parametric testing

To perform only parametric testing (based of the likelihood ratio test), use:
```
python permutation_testing.py --kinship_eigenvalues filename
                              --kinship_eigenvectors filename
                              --phenotypes filename
                             [--cutoff 1e-5]
                             [--covariates filename]
                             [--no_intercept]
                              --parametric                              
                             
```
where:
* `parametric` (Shortcut: `-m`) - Perform only parametric testing.
* `kinship_eigenvalues` (`-k`) - A file containing the eigenvalues of the kinship matrix, one eigenvalue per line, in text format. No header or IDs supported - see example file. This could be created, for example, with GCTA's --pca flag, after removing the IDs from the output file (**required**).
* `kinship_eigenvectors` (`-v`) - A file containing the eigenvectors of the kinship matrix, one eigenvector per column, in text format. No header or IDs supported - see example file. This could be created, for example, with GCTA's --pca flag, after removing the IDs from the output file  (**required**).
* `phenotypes` (`-y`) - A file containing the phenotypes, one phenotype per column, in text format (**required**).
* `cutoff` (`-c`) - A threshold below which eigenvalues are considered to be effectively 0. **This is important to avoid numerical issues and get correct p-values.** Defaults to 1e-5.
* `covariates` (`-x`) - A file containing the covariates, one covariate per column, in text format. If omitted, only an intercept covariate will be used.
* `no_intercept` (`-i`) - Add to explicitly not use a constant 1 (intercept) covariate. Can be used without an additional covariates file. Is not enabled by default, i.e. an intercept is added.



We hope to soon add the option of having a file with pre-calculated heritability estimates.

### Permutation testing

You can perform 3 types of permutation testing:
* Naive - without any speedups, plain old permuting the phenotypes and re-estimating. This is slow but safe.
* Fast - using the derivative trick, without SAMC (see paper).
* Very fast - using SAMC; is faster with >100,000 permutations, but approximate.

#### Naive

```
python permutation_testing.py --kinship_eigenvalues filename
                              --kinship_eigenvectors filename
                              --phenotypes filename
                             [--cutoff 1e-5]
                             [--covariates filename]
                             [--no_intercept]
                             --permutation --naive
                             --n_permutations 1000
```
where flags are as before, and:
* `permutation` (Shortcut: `-p`) - Perform permutation testing.
* `naive` (`-a`) - Perform naive permutation testing.
* `n_permutations` (`-n`) - Number of permutations per phenotype.

#### Fast (no SAMC)

```
python permutation_testing.py --kinship_eigenvalues filename
                              --kinship_eigenvectors filename
                              --phenotypes filename
                             [--cutoff 1e-5]
                             [--covariates filename]
                             [--no_intercept]
                             --permutation --fast
                             --n_permutations 1000
```
where flags are as before, and:
* `fast` (`-f`) - Perform fast permutation testing (without SAMC).

#### Very fast (with SAMC, approximate)
Running SAMC requires a few more parameters, and may generate additional output.

```
python permutation_testing.py --kinship_eigenvalues filename
                              --kinship_eigenvectors filename
                              --phenotypes filename
                             [--cutoff 1e-5]
                             [--covariates filename]
                             [--no_intercept]
                             --permutation --samc
                             --n_permutations 100000
```
where flags are as before, and:
* `samc` (`-s`) - Perform very fast permutation testing (with SAMC).
* `n_permutations` (`-n`) - Here, this refers to the number of iterations. As described in the main paper, SAMC can estimate p-values much smaller than `1/n_permutations`.

Additional flags that have to do with the calibration of SAMC (see main text):
* `n_partitions`
* `replace_proportion`
* `relative_sampling_error_threshold`
* `t0`

Note that you have to have a sufficient number of permutations (e.g. 100,000) for convergence. This is tracked by the relative sampling error, which is printed (`RSE` column), and should be lower than `--relative_sampling_error_threshold` (defaults to `0.01`).

## C++

**Important**: The C++ implementation currently does not support covariates.

The C++ implementation is more limited at this stage. If you want to use some yet-unimplemented feature, let me know!

### Installation

Make sure boost 1.66+ is installed on your system.
Make sure Eigen (tested with 3.3.5, older version would probably work) is installed on your system.

Compile the cpp file with Makefile (`cd` to the directory, then `make`). You may need to modify the path to `boost` and `eigen`, e.g.
```
CXXFLAGS=-std=c++11 -Wall -pedantic -O3 -DNDEBUG -pthread -I/usr/local/Cellar/boost/1.67.0_1/include/ -I/usr/local/Cellar/eigen/3.3.5/include/eigen3/
```

### Running

To see all flags:
```
./permutation_testing_samc --help
```

You can perform fast permutation testing (using the derivative-trick, no SAMC), or SAMC-based permutation testing.

#### Fast permutation testing (no SAMC)

The syntax is:

```
./permutation_testing_samc --eigenvectors_filename filename 
                           --eigenvalues_filename filename
                           --phenotypes_filename filename
                           --heritabilities_filename filename
                          [--n_permutations 10000]    
                          [--n_repetitions 1]
                          [--output_filename filename]
                          [--phenotype_indices 0-3,5-7]
                          [--n_chunks 10]
                          [--n_threads -1]
```
where:
* `eigenvectors_filename` - A file containing the eigenvalues of the kinship matrix, one eigenvalue per line, in text format. No header or IDs supported - see example file. This could be created, for example, with GCTA's --pca flag, after removing the IDs from the output file (**required**).
* `eigenvalues_filename` - A file containing the eigenvectors of the kinship matrix, one eigenvector per column, in text format. No header or IDs supported - see example file. This could be created, for example, with GCTA's --pca flag, after removing the IDs from the output file  (**required**).
* `phenotypes_filename` - A file containing the phenotypes, one phenotype per column, in text format (**required**). No header or IDs supported - see example file.
* `heritabilities_filename` - A file containing the estimated heritabilities, one estimated value per row, in text format (**required**). This can be obtained from the Python version by running, e.g.:
```
python2 permutation_testing.py -k data/example.eigenval -v data/example.eigenvec -y data/phenotypes.txt --parametric | tail -n +2 | cut -f 2 > data/heritability_estimates.txt
```

* `n_permutations` - How many permutations to use?
* `n_repetitions` - If you want to repeat the analysis; how many times to do that?
* `output_filename` - Output filename; if not specified, `[phenotypes_filename].out` will be used. Results are printed to the screen as well.
* `phenotype_indices` - If you want to run only on some of the phenotypes, you can specify their indices here. You can use comma-separated ranges (e.g., `0,3,4` or `0-3,5-7`).
* `n_chunks` - Divide the permutations into this number of chunks. Make this larger if you run into memory issues.
* `n_threads` - How many threads to use? Use -1 for the number of processors on your computer.

The output is simply a list of p-values, one per phenotype, e.g.:
```
$ ./permutation_testing_samc --eigenvectors_filename ./data/example.eigenvec --eigenvalues_filename ./data/example.eigenval --phenotypes_filename ./data/phenotypes.txt --heritabilities_filename ./data/heritability_estimates.txt 

$ cat ./data/phenotypes.txt.out 
     1
0.1042
0.0005
 0.019
0.0006
0.0006
     0
     0
     0
     0
```

#### Permutation testing with SAMC
The syntax is:

```
./permutation_testing_samc --eigenvectors_filename filename
                           --eigenvalues_filename filename
                           --phenotypes_filename filename
                           --heritabilities_filename filename
                           --samc true
                          [--n_permutations 10000]                          
                          [--output_filename filename]
                          [--phenotype_indices 0-3,5-7]
                          [--n_chunks 10]
                          [--n_threads -1]                          
                          [--debug false]
                          [--n_partitions 50]
                          [--replace_proportion 0.05]
                          [--relative_sampling_error_threshold 0.0001]                          
                          [--t0 10000]
```
Options are as before, with the addition of:
* `samc` - Set to `true` to run SAMC (`false` is the default, meaning permutation testing as before).
* `debug` - Set to `true`, to generate additional files with RSE and bias estimates, for parameter calibration.

Additional flags that have to do with the calibration of SAMC (see main text):
* `n_partitions`
* `replace_proportion`
* `relative_sampling_error_threshold`
* `t0`

## How to cite?

The relevant paper is [this](https://www.nature.com/articles/s41467-018-07276-w).
If you use the parametric option, based on `pylmm`, please refer to `https://github.com/nickFurlotte/pylmm`.
