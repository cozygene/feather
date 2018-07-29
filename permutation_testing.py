from __future__ import print_function

from numpy import * 
import numpy.linalg
import numpy as np
import albi_lib
import importlib
import sys
import scipy.stats


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
    importlib.reload(samc)
    importlib.reload(albi_lib)


from tqdm import *
#from memory_profiler import profile


################################################################################################
# Parametric testing
#

def parametric_testing(y, kinship_matrix, kinship_eigenvectors, kinship_eigenvalues, covariates=None, pylmm_resolution=100, cutoff=1e-5):
    if covariates is None:
        covariates = ones_like(y[:,newaxis])

    # Make sure the eigenvalues are in decreasing order and nonzero
    reverse_sorted_indices = argsort(kinship_eigenvalues)[::-1]
    kinship_eigenvalues = array(kinship_eigenvalues)[reverse_sorted_indices]
    kinship_eigenvectors = array(kinship_eigenvectors)[:, reverse_sorted_indices]

    n_effective = len(where(kinship_eigenvalues >= cutoff)[0])
    kinship_eigenvalues =kinship_eigenvalues[:n_effective]

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

def permutation_testing_only_eigenvectors(y, h2_estimate, kinship_eigenvectors, kinship_eigenvalues, n_permutations, permutation_blocks=None, max_memory_in_gb=None, seed=None, cutoff=1e-5, verbose=False):
    n = len(kinship_eigenvalues)
    der = albi_lib.OnlyEigenvectorsDerivativeSignCalculator([0], [h2_estimate], kinship_eigenvalues, kinship_eigenvectors, eigenvectors_as_X=[-1], cutoff=cutoff)

    if max_memory_in_gb is not None:
        chunk_size = max(1, int((max_memory_in_gb * 2.0**30)/ (n * 64)))
    else:
        chunk_size = n_permutations
    p = []
    for n_chunk in tqdm(range(0, n_permutations, chunk_size), disable=(not verbose)):
        n_permutations_in_chunk = min(n_permutations, n_chunk + chunk_size) - n_chunk
        permuted = permute_columns(repeat(y[:,newaxis], n_permutations_in_chunk, 1), permutation_blocks, (seed + n_chunk if seed is not None  else None))

        rotated = np.dot(kinship_eigenvectors.T, permuted)

        p.append(der.get_derivative_signs(rotated.T[:,newaxis,:])[:,0,0] >= 0)
        
          
    return sum(concatenate(p))

def permutation_testing(y, h2_estimate, kinship_eigenvectors, kinship_eigenvalues, covariates, n_permutations, permutation_blocks=None, max_memory_in_gb=None, seed=None, cutoff=1e-5, verbose=False):
    n = len(kinship_eigenvalues)

    if seed is None:
        seed = np.random.randint(0, 2**32)
    
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
            der = albi_lib.GeneralDerivativeSignCalculator([0], [h2_estimate], kinship_eigenvalues, kinship_eigenvectors, permuted_X[:, :, i].T, cutoff=cutoff)
            p.append(der.get_derivative_signs(permuted_y[newaxis, :, i]) >= 0)
          
    return mean(concatenate(p))

def naive_permutation_testing(y, h2_estimate, kinship_matrix, kinship_eigenvectors, kinship_eigenvalues, covariates, n_permutations, permutation_blocks=None, max_memory_in_gb=None, seed=0, verbose=False):
    n = len(kinship_eigenvalues)
    
    if max_memory_in_gb is not None:
        chunk_size = max(1, int((max_memory_in_gb * 2.0**30)/ (n * 64)))
    else:
        chunk_size = n_permutations

    p = []
    pne = []
    h2s = []
    for n_chunk in (range(0, n_permutations, chunk_size)):
        n_permutations_in_chunk = min(n_permutations, n_chunk + chunk_size) - n_chunk
        permuted_y = permute_columns(repeat(y[:,newaxis], n_permutations_in_chunk, 1), permutation_blocks,  seed + n_chunk)
        
        l = []
        for c in range(shape(covariates)[1]):
            a = permute_columns(repeat(covariates[:,c:(c+1)], n_permutations_in_chunk, 1), permutation_blocks, seed + n_chunk) 
            l.append(a)

        permuted_X = array(l)
        
        for i in trange(n_permutations_in_chunk, disable=(not verbose)):
            h2, param_p = parametric_testing(permuted_y[:, i], kinship_matrix, kinship_eigenvectors, kinship_eigenvalues, permuted_X[:, :, i].T)            
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



def samc_heritability(x0, n_partitions, n_iterations, kinship_eigenvalues, kinship_eigenvectors, 
                      replace_proportion=0.05, relative_sampling_error_threshold=0.2, 
                      t0=1000):

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

    K = np.linalg.multi_dot([kinship_eigenvectors, diag(kinship_eigenvalues), kinship_eigenvectors.T])

    res0 = parametric_testing(x0, K, kinship_eigenvectors, kinship_eigenvalues)
    print("Estimated h^2:", res0[0])
    
    observed_statistic = res0[0]


    partitions = concatenate([arange(0, observed_statistic, observed_statistic/n_partitions)[1:], [observed_statistic]])

    weights_at_partitions = albi_lib.weights_zero_derivative([0], partitions, kinship_eigenvalues)[0, :, :]

    n_partitions_total = n_partitions + 1

    theta0 = log([0.5] + [0.5/(n_partitions_total-1)]*(n_partitions_total-1))
    theta0 -= mean(theta0)

    thetas, observed_sampling_distribution, statistics = samc.SAMC_simple(samc.SAMCSimpleParameters(
        x0=x0, 
        test_statistic_func=lambda x, current_partition: heritability_test_statistic(x, current_partition, kinship_eigenvectors, 
                                                                                     partitions, weights_at_partitions), 
        generate_sample_func=heritability_generate_sample, 
        n_partitions=n_partitions_total, 
        n_iterations=n_iterations, 
        relative_sampling_error_threshold=relative_sampling_error_threshold,
        t0=t0))

    #return thetas, observed_sampling_distribution, statistics
    return exp(thetas[-1])/sum(exp(thetas))



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

