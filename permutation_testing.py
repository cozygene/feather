from numpy import * 
import numpy.linalg
import numpy as np
import albi_lib

import os.path
curdir = os.path.dirname(os.path.realpath(__file__))

import sys
if os.path.join(curdir, "pylmm") not in sys.path:
    sys.path.insert(0, os.path.join(curdir, "pylmm"))
import pylmm.lmm_unbounded
reload(pylmm.lmm_unbounded)

from tqdm import *


################################################################################################
# Parametric testing
#
def parametric_testing(y, kinship_matrix, kinship_eigenvectors, kinship_eigenvalues, covariates=None):
    if covariates is None:
        covariates = ones_like(y[:,newaxis])
    obj = pylmm.lmm_unbounded.LMM(y[:,newaxis], kinship_eigenvectors, kinship_eigenvalues, kinship_eigenvectors, X0=covariates)
    res = obj.fit(REML=True, explicit_H=arange(0,1.01,0.01))
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

def permutation_testing_only_eigenvectors(y, h2_estimate, kinship_eigenvectors, kinship_eigenvalues, n_permutations, permutation_blocks=None, max_memory_in_gb=None, seed=None, verbose=False):
    n = len(kinship_eigenvalues)
    der = albi_lib.OnlyEigenvectorsDerivativeSignCalculator([0], [h2_estimate], kinship_eigenvalues, eigenvectors_as_X=[-1])

    if max_memory_in_gb is not None:
        chunk_size = max(1, int((max_memory_in_gb * 2.0**30)/ (n * 64)))
    else:
        chunk_size = n_permutations
    p = []
    for n_chunk in tqdm(range(0, n_permutations, chunk_size), disable=(not verbose)):
        n_permutations_in_chunk = min(n_permutations, n_chunk + chunk_size) - n_chunk
        permuted = permute_columns(repeat(y[:,newaxis], n_permutations_in_chunk, 1), permutation_blocks, (seed + n_chunk if seed is not None  else None))

        p.append(der.get_derivative_signs(permuted.T[:,newaxis,:])[:,0,0] >= 0)
          
    return sum(concatenate(p))

def permutation_testing(y, h2_estimate, kinship_eigenvectors, kinship_eigenvalues, covariates, n_permutations, permutation_blocks=None, max_memory_in_gb=None, seed=0, verbose=False):
    n = len(kinship_eigenvalues)
    
    if max_memory_in_gb is not None:
        chunk_size = max(1, int((max_memory_in_gb * 2.0**30)/ (n * 64)))
    else:
        chunk_size = n_permutations

    p = []
    for n_chunk in tqdm(range(0, n_permutations, chunk_size), disable=(not verbose)):
        n_permutations_in_chunk = min(n_permutations, n_chunk + chunk_size) - n_chunk
        permuted_y = permute_columns(repeat(y[:,newaxis], n_permutations_in_chunk, 1), permutation_blocks,  seed + n_chunk)
        
        l = []
        for c in range(shape(covariates)[1]):
            a = permute_columns(repeat(covariates[:,c:(c+1)], n_permutations_in_chunk, 1), permutation_blocks, seed + n_chunk) 
            l.append(a)

        permuted_X = array(l)
        
        for i in range(n_permutations_in_chunk):
            der = albi_lib.GeneralDerivativeSignCalculator([0], [h2_estimate], kinship_eigenvalues, kinship_eigenvectors, permuted_X[:, :, i].T)
            p.append(der.get_derivative_signs(permuted_y[newaxis, :, i]) >= 0)
          
    return mean(concatenate(p))

################################################################################################
# SAMC with ALBI
#
import scipy.optimize, scipy.interpolate
import sys


def samc_heritability(x0, n_partitions, n_iterations, kinship_eigenvalues, kinship_eigenvectors, 
                      replace_proportion=0.05, n_permutations=100000, relative_sampling_error_threshold=0.2, 
                      random_x0=False, t0=1000, just_return_partitions=False, theta0=None, use_theta=None, init_run_iterations=0, 
                      second_run_partitions=0, polyak_ruppert=False):
    def heritability_test_statistic(x, current_partition, kinship_eigenvectors, partitions, weights_at_partitions):
        rotated = dot(kinship_eigenvectors.T, x)
        

        # First check if we are in the same partition
        if current_partition == 0 and dot(weights_at_partitions[0,:], rotated**2) <= 0:
            part = current_partition      
            pp = [0, partitions[0]]
        elif current_partition == len(partitions) and dot(weights_at_partitions[-1,:], rotated**2) >= 0:
            part = current_partition
            pp = [partitions[-1], 1]
        elif dot(weights_at_partitions[current_partition-1,:], rotated**2) >= 0 and \
             dot(weights_at_partitions[current_partition,:], rotated**2) <= 0:
            part = current_partition
            pp = partitions[part-1:part+1].tolist()

        # If not, find the new partition
        else:
            ds = dot(weights_at_partitions, rotated**2)
            
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

    res0 = pylmm_estimate(x0, kinship_eigenvalues, kinship_eigenvectors)
    print "Estimated h^2:", res0[0]
    #ps = albi_permutation_testing(x0, res0[0], kinship_eigenvectors, kinship_eigenvalues, n_permutations, max_memory_in_gb=0.1, verbose=True)
    #print "Permutation testing p:", mean(ps)

    observed_statistic = res0[0]


    partitions = concatenate([arange(0, observed_statistic, observed_statistic/n_partitions)[1:], [observed_statistic]])

    if just_return_partitions:
        return partitions


    weights_at_partitions = albi_lib.albi_lib.weights_zero_derivative([0], partitions, kinship_eigenvalues)[0, :, :]

    n_partitions_total = n_partitions + 1

    if theta0 is None:
        theta0 = log([0.5] + [0.5/(n_partitions_total-1)]*(n_partitions_total-1))
        theta0 -= mean(theta0)

    if random_x0:
        x0 = np.random.permutation(x0)

    thetas, observed_sampling_distribution, statistics = SAMC_simple(SAMCSimpleParameters(
        x0=x0, 
        test_statistic_func=lambda x, current_partition: heritability_test_statistic(x, current_partition, kinship_eigenvectors, 
                                                                                     partitions, weights_at_partitions), 
        generate_sample_func=heritability_generate_sample, 
        n_partitions=n_partitions_total, 
        n_iterations=(init_run_iterations if init_run_iterations > 0 else n_iterations), 
        relative_sampling_error_threshold=relative_sampling_error_threshold,
        t0=t0,
        theta0=theta0,
        fix_theta=(use_theta is not None),
        step_size_power=(0.75 if polyak_ruppert else 1.0)))

    if init_run_iterations == 0:
        return thetas, observed_sampling_distribution, statistics

    n_partitions_total = second_run_partitions+1

    fhat = exp(thetas)/sum(exp(thetas))
    Fhat = add.accumulate(fhat)
    delta = (1-Fhat[-2])
    bestr = scipy.optimize.brentq(lambda r: delta*sum([r**(n_partitions_total-1-k) for k in range(1,n_partitions_total)]) - Fhat[-2], 0, int(sys.float_info.max) ** (1.0/(n_partitions_total-1)))
    Fhat_interpolated = scipy.interpolate.interp1d([0]+list(partitions), [0]+list(Fhat[:-1]))
    accr = delta*add.accumulate([bestr**(n_partitions_total-1-k) for k in range(1, n_partitions_total)])
    newpartitions = array([scipy.optimize.brentq(lambda x: Fhat_interpolated(x) - p, 0, partitions[-1]) for p in accr[:-1]] + [partitions[-1]])

    theta0 = concatenate([[log(delta*bestr**(n_partitions_total-1-k)) for k in range(1, n_partitions_total)], [log(delta)]])
    theta0 -= mean(theta0)
 
    weights_at_new_partitions = albi_lib.albi_lib.weights_zero_derivative([0], newpartitions, kinship_eigenvalues)[0, :, :]

    return SAMC_simple(SAMCSimpleParameters(
        x0=x0, 
        test_statistic_func=lambda x, current_partition: heritability_test_statistic(x, current_partition, kinship_eigenvectors, newpartitions, weights_at_new_partitions), 
        generate_sample_func=heritability_generate_sample, 
        n_partitions=n_partitions_total, 
        n_iterations=n_iterations, 
        relative_sampling_error_threshold=relative_sampling_error_threshold,
        t0=t0,
        theta0=theta0,
        fix_theta=(use_theta is not None),
        step_size_power=(0.75 if polyak_ruppert else 1.0)))

def samc_heritability_sim(n_partitions, n_iterations, h2, kinship_eigenvalues, kinship_eigenvectors, replace_proportion=0.05, relative_sampling_error_threshold=0.2):
    y = draw_multivariate_from_eigen(kinship_eigenvectors, h2*kinship_eigenvalues + (1-h2))[:, 0]
    return samc_heritability(y, n_partitions, n_iterations, kinship_eigenvalues, kinship_eigenvectors, replace_proportion, relative_sampling_error_threshold)
