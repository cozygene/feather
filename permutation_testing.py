from __future__ import print_function

from numpy import * 
import numpy.linalg
import numpy as np
import albi_lib
import importlib
import sys
import scipy.stats
import argparse
import itertools

import os.path
curdir = os.path.dirname(os.path.realpath(__file__))

import sys
if os.path.join(curdir, "pylmm") not in sys.path:
    sys.path.insert(0, os.path.join(curdir, "pylmm"))
import pylmm.lmm_unbounded


import albi_lib
import samc

if sys.version_info.major == 3:
    importlib.reload(pylmm.lmm_unbounded)
    #importlib.reload(samc)
    importlib.reload(albi_lib)

try:
    from tqdm import *
except:
    trange = lambda *args, **kw: range(*args, **kws)
    tqdm = lambda x, *args, **kw: x


#from memory_profiler import profile


################################################################################################
# Parametric testing
#
def parametric_testing(y, kinship_eigenvectors, kinship_eigenvalues, covariates=None, pylmm_resolution=100, cutoff=1e-5):
    if covariates is None:
        covariates = ones_like(y[:,newaxis])

    # Make sure the eigenvalues are in decreasing order and nonzero
    reverse_sorted_indices = argsort(kinship_eigenvalues)[::-1]
    kinship_eigenvalues = array(kinship_eigenvalues)[reverse_sorted_indices]
    kinship_eigenvectors = array(kinship_eigenvectors)[:, reverse_sorted_indices]

    n_effective = len(where(kinship_eigenvalues >= cutoff)[0])
    kinship_eigenvalues = kinship_eigenvalues[:n_effective]

    rotated_covariates = dot(kinship_eigenvectors.T, covariates)[:n_effective,:]
    rotated_y = dot(kinship_eigenvectors.T, y)[:n_effective]

    obj = pylmm.lmm_unbounded.LMM(rotated_y[:,newaxis], identity(n_effective), kinship_eigenvalues, identity(n_effective), X0=rotated_covariates)
    res = obj.fit(REML=True, explicit_H=linspace(0,1,pylmm_resolution+1))
    return res[0], 0.5*scipy.stats.distributions.chi2.sf(2*(res[3]-obj.LLs[0]), 1)


####

def permute_columns_block(x):
    ix_i = np.random.sample(x.shape).argsort(axis=0)
    ix_j = tile(arange(x.shape[1]), (x.shape[0], 1))
    return x[ix_i, ix_j]

def permute_columns(x, permutation_blocks, seed):
    np.random.seed(seed)

    if permutation_blocks == None:
        permutation_blocks = [shape(x)[0]]
    ends = add.accumulate(permutation_blocks)
    starts = concatenate([[0], ends[:-1]])
    for i in range(len(permutation_blocks)):
        start, end = starts[i], ends[i]
        x[start:end, :] = permute_columns_block(x[start:end, :])

    return x    

################################################################################################
# Derivative-based permutation testing
#

def permutation_testing_only_eigenvectors(y, h2_estimate, kinship_eigenvectors, kinship_eigenvalues, n_permutations, permutation_blocks=None, chunk_size=None, seed=None, cutoff=1e-5, verbose=False):
    n = len(kinship_eigenvalues)
    der = albi_lib.OnlyEigenvectorsDerivativeSignCalculator([0], [h2_estimate], kinship_eigenvalues, kinship_eigenvectors, eigenvectors_as_X=[-1], cutoff=cutoff)

    if chunk_size is None:
        chunk_size = n_permutations

    if seed is None:
        seed = np.random.randint(0, 2**32) 


    p = []
    disable_chunk = (chunk_size == n_permutations)

    for n_chunk in trange(0, n_permutations, chunk_size, disable=disable_chunk, leave=False):
        n_permutations_in_chunk = min(n_permutations, n_chunk + chunk_size) - n_chunk
        permuted = permute_columns(repeat(y[:,newaxis], n_permutations_in_chunk, 1), permutation_blocks, (seed + n_chunk if seed is not None  else None))

        rotated = np.dot(kinship_eigenvectors.T, permuted)

        p.append(der.get_derivative_signs(rotated.T[:,newaxis,:])[:,0,0] >= 0)
        
          
    return sum(concatenate(p))

def permutation_testing(y, h2_estimate, kinship_eigenvectors, kinship_eigenvalues, covariates, n_permutations, permutation_blocks=None, chunk_size=None, seed=None, cutoff=1e-5, verbose=False):
    n = len(kinship_eigenvalues)

    if chunk_size is None:
        chunk_size = n_permutations

    if seed is None:
        seed = np.random.randint(0, 2**32)

    p = []
    disable_chunk = (chunk_size == n_permutations)
    for n_chunk in trange(0, n_permutations, chunk_size, disable=disable_chunk, leave=False):
        n_permutations_in_chunk = min(n_permutations, n_chunk + chunk_size) - n_chunk
        permuted_y = permute_columns(repeat(y[:,newaxis], n_permutations_in_chunk, 1), permutation_blocks,  seed + n_chunk)
        
        l = []
        for c in trange(shape(covariates)[1]):
            a = permute_columns(repeat(covariates[:,c:(c+1)], n_permutations_in_chunk, 1), permutation_blocks, seed + n_chunk) 
            l.append(a)

        permuted_X = array(l)
        
        for i in trange(n_permutations_in_chunk, leave=False):
            der = albi_lib.GeneralDerivativeSignCalculator([0], [h2_estimate], kinship_eigenvalues, kinship_eigenvectors, permuted_X[:, :, i].T, cutoff=cutoff)
            p.append(der.get_derivative_signs(permuted_y[newaxis, :, i]) >= 0)
          
    return mean(concatenate(p))

def naive_permutation_testing(y, h2_estimate, kinship_eigenvectors, kinship_eigenvalues, covariates, n_permutations, permutation_blocks=None, chunk_size=None, seed=0, verbose=False):
    n = len(kinship_eigenvalues)
    
    if chunk_size is None:
        chunk_size = n_permutations

    if seed is None:
        seed = np.random.randint(0, 2**32)

    p = []
    pne = []
    h2s = []
    disable_chunk = (chunk_size == n_permutations)
    for n_chunk in trange(0, n_permutations, chunk_size, disable=disable_chunk, leave=False):
        n_permutations_in_chunk = min(n_permutations, n_chunk + chunk_size) - n_chunk
        permuted_y = permute_columns(repeat(y[:,newaxis], n_permutations_in_chunk, 1), permutation_blocks,  seed + n_chunk)
        
        l = []
        for c in range(shape(covariates)[1]):
            a = permute_columns(repeat(covariates[:,c:(c+1)], n_permutations_in_chunk, 1), permutation_blocks, seed + n_chunk) 
            l.append(a)

        permuted_X = array(l)
        
        for i in trange(n_permutations_in_chunk, disable=(not verbose), leave=False):
            h2, param_p = parametric_testing(permuted_y[:, i], kinship_eigenvectors, kinship_eigenvalues, permuted_X[:, :, i].T)            
            p.append(h2 >= h2_estimate)
            pne.append(h2 > h2_estimate)
            h2s.append(h2)
          
    return mean(p), mean(pne), np.array(h2s)

################################################################################################
# SAMC with ALBI
#
import scipy.optimize, scipy.interpolate
import sys

def samc_heritability_only_eigenvectors(x0, n_partitions, n_iterations, kinship_eigenvalues, kinship_eigenvectors, 
                                        replace_proportion=0.05, relative_sampling_error_threshold=0.2, 
                                        t0=1000):

    def heritability_test_statistic(x, current_partition, kinship_eigenvectors, partitions, derivative_calculator):
        rotated_x = np.dot(kinship_eigenvectors.T, x)
        ds = derivative_calculator.get_derivative_signs(rotated_x[newaxis, newaxis, :])[0,0,:]

        # First check if we are in the same partition
        if current_partition == 0 and ds[0] <= 0:
            part = current_partition      
            pp = [0, partitions[0]]
        elif current_partition == len(partitions) and ds[-1] >= 0:
            part = current_partition
            pp = [partitions[-1], 1]
        elif ds[current_partition-1] >= 0 and ds[current_partition] <= 0:
            part = current_partition
            pp = partitions[part-1:part+1].tolist()

        # If not, find the new partition
        else:
            if ds[0] <= 0:
                part = 0
                pp = [0, partitions[0]]
            elif ds[-1] >= 0:
                part = len(ds)
                pp = [partitions[-1], 1]
            else:
                w = where((ds[:-1] >= 0) & (ds[1:] <= 0))[0]
                if len(w):
                    part = w[0]+1
                    pp = partitions[w[0]:w[0]+2].tolist()
                else:
                    assert "Should not happen"

        return pp, part

    def heritability_generate_sample(x):
        L = int(replace_proportion*len(x))
        indices = np.random.choice(range(len(x)), L, replace=False)
        permuted_indices = np.random.permutation(indices)
        y = x.copy()
        y[indices] = y[permuted_indices]
        return y, 1.0

    K = np.linalg.multi_dot([kinship_eigenvectors, diag(kinship_eigenvalues), kinship_eigenvectors.T])

    res0 = parametric_testing(x0, K, kinship_eigenvectors, kinship_eigenvalues)
    print("Estimated h^2:", res0[0])
    
    observed_statistic = res0[0]


    partitions = concatenate([arange(0, observed_statistic, observed_statistic/n_partitions)[1:], [observed_statistic]])

    #weights_at_partitions = albi_lib.weights_zero_derivative([0], partitions, kinship_eigenvalues)[0, :, :]

    derivative_calculator = albi_lib.OnlyEigenvectorsDerivativeSignCalculator([0], partitions, kinship_eigenvalues, kinship_eigenvectors, eigenvectors_as_X=[-1])

    n_partitions_total = n_partitions + 1

    theta0 = log([0.5] + [0.5/(n_partitions_total-1)]*(n_partitions_total-1))
    theta0 -= mean(theta0)

    thetas, observed_sampling_distribution, statistics = samc.SAMC_simple(samc.SAMCSimpleParameters(
        x0=x0, 
        test_statistic_func=lambda x, current_partition: heritability_test_statistic(x, current_partition, kinship_eigenvectors, 
                                                                                     partitions, derivative_calculator), 
        generate_sample_func=heritability_generate_sample, 
        n_partitions=n_partitions_total, 
        n_iterations=n_iterations, 
        relative_sampling_error_threshold=relative_sampling_error_threshold,
        t0=t0))

    #return thetas, observed_sampling_distribution, statistics
    return exp(thetas[-1])/sum(exp(thetas))



def samc_heritability(x0, h2_estimate, n_partitions, n_iterations, kinship_eigenvalues, kinship_eigenvectors,
                      covariates, cutoff,
                      replace_proportion=0.05, relative_sampling_error_threshold=0.2, 
                      t0=1000):

    def heritability_test_statistic(x, current_partition, kinship_eigenvectors, partitions, derivative_calculator):
        ds = derivative_calculator.get_derivative_signs(x[newaxis, :])[0,:]

       # First check if we are in the same partition
        if current_partition == 0 and ds[0] <= 0:
            part = current_partition      
            pp = [0, partitions[0]]
        elif current_partition == len(partitions) and ds[-1] >= 0:
            part = current_partition
            pp = [partitions[-1], 1]
        elif ds[current_partition-1] >= 0 and ds[current_partition] <= 0:
            part = current_partition
            pp = partitions[part-1:part+1].tolist()

        # If not, find the new partition
        else:
            if ds[0] <= 0:
                part = 0
                pp = [0, partitions[0]]
            elif ds[-1] >= 0:
                part = len(ds)
                pp = [partitions[-1], 1]
            else:
                w = where((ds[:-1] >= 0) & (ds[1:] <= 0))[0]
                if len(w):
                    part = w[0]+1
                    pp = partitions[w[0]:w[0]+2].tolist()
                else:
                    assert "Should not happen"

        return pp, part

    def heritability_generate_sample(x):
        L = int(replace_proportion*len(x))
        indices = np.random.choice(range(len(x)), L, replace=False)
        permuted_indices = np.random.permutation(indices)
        y = x.copy()
        y[indices] = y[permuted_indices]
        return y, 1.0
    
    if h2_estimate == 0:
        return 0.5, 0

    observed_statistic = h2_estimate #res0[0]

    partitions = linspace(0, observed_statistic, n_partitions+1)[1:]

    derivative_calculator = albi_lib.GeneralDerivativeSignCalculator(
        h2_values=[0], 
        H2_values=partitions, 
        kinship_eigenvalues=kinship_eigenvalues, 
        kinship_eigenvectors=kinship_eigenvectors, 
        covariates=covariates,
        REML=True,
        cutoff=cutoff)

    n_partitions_total = n_partitions + 1

#    theta0 = log([0.5] + [0.5/(n_partitions_total-1)]*(n_partitions_total-1))
#    theta0 -= mean(theta0)

    thetas, observed_sampling_distribution, statistics, relative_sampling_error = \
        samc.SAMC_simple(samc.SAMCSimpleParameters(
            x0=x0, 
            test_statistic_func=lambda x, current_partition: heritability_test_statistic(x, current_partition, kinship_eigenvectors, 
                                                                                         partitions, derivative_calculator), 
            generate_sample_func=heritability_generate_sample, 
            n_partitions=n_partitions_total, 
            n_iterations=n_iterations, 
            relative_sampling_error_threshold=relative_sampling_error_threshold,
            t0=t0))

    #return thetas, observed_sampling_distribution, statistics
    return exp(thetas[-1])/sum(exp(thetas)), relative_sampling_error



def draw_multivariate(X, times=1, random_seed=None):
    return dot(X, numpy.random.RandomState(random_seed).randn(shape(X)[0], times))

def draw_multivariate_from_eigen(U, eigvals, times=1, random_seed=None):
    return draw_multivariate(U * (maximum(eigvals, 0)**0.5)[newaxis, :], times, random_seed)


def samc_heritability_sim(n_partitions, n_iterations, h2, kinship_eigenvalues, kinship_eigenvectors, replace_proportion=0.05, relative_sampling_error_threshold=0.2):
    y = draw_multivariate_from_eigen(kinship_eigenvectors, h2*kinship_eigenvalues + (1-h2))[:, 0]
    #print(samc_heritability(y, n_partitions, n_iterations, kinship_eigenvalues, kinship_eigenvectors, replace_proportion, relative_sampling_error_threshold))
    print(samc_heritability_only_eigenvectors(y, n_partitions, n_iterations, kinship_eigenvalues, kinship_eigenvectors, replace_proportion, relative_sampling_error_threshold))
    K = np.linalg.multi_dot([kinship_eigenvectors, diag(kinship_eigenvalues), kinship_eigenvectors.T])
    res = parametric_testing(y, K, kinship_eigenvectors, kinship_eigenvalues)
    print(res)
    print(permutation_testing_only_eigenvectors(y, res[0], kinship_eigenvectors, kinship_eigenvalues, 10000, verbose=True)/10000.0)


################################################################################################
# Main 

class FeatherArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('Error: %s\n' % message)
        print("To see the full help: %s -h/--help" % self.prog)
        sys.exit(2)

FEATHER_USAGE = """
See https://github.com/cozygene/permutation_testing for full documentation about usage.
"""

if __name__ == '__main__':
    #
    # Parse arguments
    #
    parser = FeatherArgumentParser(prog=os.path.basename(sys.argv[0]), usage=FEATHER_USAGE)
      
    parser.add_argument('-k', '--kinship_eigenvalues', type=str, help="A file containing the eigenvalues of the kinship matrix, one eigenvalue per line, in text format.") 
    parser.add_argument('-v', '--kinship_eigenvectors', type=str, help="A file containing the eigenvectors of the kinship matrix, one eigenvector per column, in text format.")
    parser.add_argument('-c', '--cutoff', type=float, default=1e-5, help="A threshold below which eigenvalues are considered to be effectively 0.")
    
    parser.add_argument('-x', '--covariates', type=str, help="A file containing the covariates, one covariate per column, in text format.")
    parser.add_argument('-y', '--phenotypes', type=str, help="A file containing the phenotypes, one phenotypes per column, in text format.")
    
    parser.add_argument('-i', '--no_intercept', action='store_true', help="If using covariates, don't add an intercept covariate.")

#    parser.add_argument('-u', '--use_eigenvectors_as_covariates', type=str, help="A comma-separated list detailing which eigenvectors should be used as covariates.")

    which_testing = parser.add_mutually_exclusive_group(required=True)
    which_testing.add_argument('-m', '--parametric', action='store_true', help="Perform parametric testing.")
    which_testing.add_argument('-p', '--permutation', action='store_true', help="Perform permutation testing.")

    which_perm = parser.add_mutually_exclusive_group(required=False)
    which_perm.add_argument('-a', '--naive', action='store_true', help='Naive permutation testing.')
    which_perm.add_argument('-f', '--fast', action='store_true', help='Fast permutation testing (no SAMC).')
    which_perm.add_argument('-s', '--samc', action='store_true', help='Very fast permutation testing (no SAMC).')

    parser.add_argument('-n', '--n_permutations', type=int, default=1000, help="Number of permutations/iterations to use for estimation.")

    # Fancy stuff
    parser.add_argument('--chunk_size', type=int, help="Chunk size")
    parser.add_argument('--seed', type=int, help="Random seed")
    parser.add_argument('--n_partitions', type=int, default=50, help="Number of partitions in SAMC")
    parser.add_argument('--replace_proportion', type=float, default=0.05, help="Proportion of entries to swap in each SAMC iteration")
    parser.add_argument('--relative_sampling_error_threshold', type=float, default=0.01, help="Relative sampling error threshold")
    parser.add_argument('--t0', type=int, default=1000, help="t0 in SAMC")
    
    
    args = parser.parse_args()

    #
    # Validate arguments
    #
    for filename in [args.kinship_eigenvalues,
                     args.kinship_eigenvectors,
                     args.covariates,                     
                     args.phenotypes]:
        if filename and not os.path.exists(filename):
            print("File %s does not exist." % filename, file=sys.stderr); sys.exit(2)

    if args.n_permutations <= 0:
        print("Number of iterations should be a positive integer.", file=sys.stderr); sys.exit(2)

    if args.kinship_eigenvalues is None:
        print("Kinship matrix eigenvalues file is required.", file=sys.stderr); sys.exit(2)
    
    try:
        kinship_eigenvalues = loadtxt(args.kinship_eigenvalues)
    except:
        print("Failed reading eigenvalues file.", file=sys.stderr); raise

    if args.kinship_eigenvectors is None:
        print("Kinship matrix eigenvectors file is required.", file=sys.stderr); sys.exit(2)
    
    try:
        kinship_eigenvectors = loadtxt(args.kinship_eigenvectors)
    except:
        print("Failed reading eigenvectors file.", file=sys.stderr); raise

    if args.covariates is not None: 
        try:
            covariates = loadtxt(args.covariates)
        except:
            print("Failed reading covariates file.", file=sys.stderr); raise

        if not any(mean(covariates == 1, axis=0) == 1) and not args.no_intercept:
            print("Adding a constant intercept covariate (-i to turn off).", file=sys.stderr)
            covariates = hstack([ones((len(kinship_eigenvalues), 1)), covariates])
    else:
        print("Note: No covariates supplied, using a constant intercept covariate.", file=sys.stderr)
        covariates = ones((len(kinship_eigenvalues), 1))

    if args.phenotypes is None:
        print("Phenotypes file is required.", file=sys.stderr); sys.exit(2)
    
    try:
        phenotypes = loadtxt(args.phenotypes)
    except:
        print("Failed reading phenotypes file.", file=sys.stderr); raise

    assert shape(phenotypes)[0] == len(kinship_eigenvalues), "Bad shape for phenotypes."



    #
    # Parametric testing
    #
    print("Calculating heritability estimates...", file=sys.stderr)
    h2_estimates = []
    param_ps = []
    for i in trange(phenotypes.shape[1]):
        y = phenotypes[:,i]
        h2, param_p = parametric_testing(
            y, 
            kinship_eigenvectors, 
            kinship_eigenvalues, 
            covariates=None, 
            pylmm_resolution=100, 
            cutoff=args.cutoff)

        h2_estimates.append(h2)
        param_ps.append(param_p)

    if args.parametric:
        print("n_phen\th2_est\tparam_p")
        for i, (h2, param_p) in enumerate(zip(h2_estimates, param_ps)):
            if h2 == 0:
                param_p = 1
            print("%d\t%1.5f\t%.4g" % (i, h2, param_p))

    #
    # Permutation testing
    #
    elif args.permutation:
        print(file=sys.stderr)
        print("Calculating permutation p-values...", file=sys.stderr)
        perm_ps = []
        rses = []

        #
        # Naive perm testing
        #
        if args.naive:            
            for i in trange(phenotypes.shape[1]):
                y = phenotypes[:,i]
                h2_estimate = h2_estimates[i]
                p, _, _ = naive_permutation_testing(
                    y, 
                    h2_estimate, 
                    kinship_eigenvectors, 
                    kinship_eigenvalues, 
                    covariates, 
                    n_permutations = args.n_permutations, 
                    permutation_blocks=None, 
                    chunk_size=None, 
                    seed=args.seed, 
                    verbose=True)
                perm_ps.append(p)

        elif args.fast:
            for i in trange(phenotypes.shape[1]):
                y = phenotypes[:,i]
                h2_estimate = h2_estimates[i]                
                p = permutation_testing(
                    y, 
                    h2_estimate, 
                    kinship_eigenvectors, 
                    kinship_eigenvalues, 
                    covariates, 
                    args.n_permutations, 
                    permutation_blocks=None, 
                    chunk_size=None, 
                    seed=args.seed, 
                    cutoff=args.cutoff, 
                    verbose=True)
                perm_ps.append(p)

        elif args.samc:
            for i in trange(phenotypes.shape[1]):
                y = phenotypes[:,i]
                h2_estimate = h2_estimates[i]                
                p, relative_sampling_error = samc_heritability(
                    x0=y, 
                    h2_estimate=h2_estimate,
                    n_partitions=args.n_partitions, 
                    n_iterations=args.n_permutations, 
                    kinship_eigenvalues=kinship_eigenvalues, 
                    kinship_eigenvectors=kinship_eigenvectors,
                    covariates=covariates,
                    cutoff=args.cutoff, 
                    replace_proportion=args.replace_proportion, 
                    relative_sampling_error_threshold=args.relative_sampling_error_threshold, 
                    t0=args.t0)
                perm_ps.append(p)
                rses.append(relative_sampling_error)

        if len(rses):
            print("n_phen\th2_est\tparam_p\tperm_p\tRSE")
            for i, (h2, param_p, perm_p, rse) in enumerate(zip(h2_estimates, param_ps, perm_ps, rses)):
                if h2 == 0:
                    param_p = 1
                    perm_p = 1
                
                print("%d\t%1.5f\t%.4g\t%.4g\t%.4g" % (i, h2, param_p, perm_p, rse))
        else:
            print("n_phen\th2_est\tparam_p\tperm_p")
            for i, (h2, param_p, perm_p) in enumerate(zip(h2_estimates, param_ps, perm_ps)):
                if h2 == 0:
                    param_p = 1
                    perm_p = 1
                
                print("%d\t%1.5f\t%.4g\t%.4g" % (i, h2, param_p, perm_p))





