import unittest

import albi_lib
import pylmm.lmm_unbounded
import permutation_testing

import numpy as np
import pandas as pd

def draw_multivariate(X, times=1, random_seed=None):
    return np.dot(X, np.random.RandomState(random_seed).randn(np.shape(X)[0], times))

def draw_multivariate_from_eigen(U, eigvals, times=1, random_seed=None):
    return draw_multivariate(U * (np.maximum(eigvals, 0)**0.5)[np.newaxis, :], times, random_seed)


class TestAlbiMethods(unittest.TestCase):

    def setUp(self, n_covariates=3, n_grid_points = 11):
        self.eigenvalues = np.loadtxt("/home/cozygene/schweiger/backup/albi-testing/data/KORA/HW_filtered_SNP_list_F4_maf_005.eigenval")
        self.eigenvectors = pd.read_csv("/home/cozygene/schweiger/backup/albi-testing/data/KORA/HW_filtered_SNP_list_F4_maf_005.eigenvec", sep=' ', header=None, usecols=range(1,1801), index_col=0, names=["ID"]*2+["PC%d"%i for i in range(1,1800)]).values
        self.kinship = np.linalg.multi_dot([self.eigenvectors, np.diag(self.eigenvalues), self.eigenvectors.T])
        self.covariates = np.random.RandomState(777).randn(len(self.eigenvalues), n_covariates)
        self.grid = np.linspace(0, 1, n_grid_points)

    def test_consistency_intercept_only(self, n_random_vectors=500, h2=0.5):
        der = albi_lib.OnlyEigenvectorsDerivativeSignCalculator([0], self.grid, self.eigenvalues, self.eigenvectors, eigenvectors_as_X=[-1])

        ys = draw_multivariate_from_eigen(self.eigenvectors, h2*self.eigenvalues + (1-h2), n_random_vectors, random_seed=0)
        rotated_ys = np.dot(self.eigenvectors.T, ys)

        ests = np.array([permutation_testing.parametric_testing(y, self.kinship, self.eigenvectors, self.eigenvalues, covariates=None, pylmm_resolution=1000)[0] for y in ys.T])
        true_signs = np.greater.outer(ests, self.grid)

        true_signs[(ests == 0), 0] = False
        if self.grid[-1] == 1:
            true_signs[(ests == 1), -1] = True

        # Check the rotated version
        albi_signs_rotated = der.get_derivative_signs_rotated(rotated_ys.T[:, np.newaxis, :-1])[:,0,:]

        if not np.alltrue((albi_signs_rotated == 1) == true_signs):
            for l in np.where(np.any((albi_signs_rotated == 1) != true_signs, axis=1))[0]:
                print("ID {}, estimated h^2 = {}, {}, {}".format(l, ests[l], true_signs[l,:], albi_signs_rotated[l,:]))

        self.assertTrue(np.alltrue((albi_signs_rotated == 1) == true_signs))

        # Check the unrotated version
        albi_signs = der.get_derivative_signs(ys.T[:, np.newaxis, :])[:,0,:]

        if not np.alltrue((albi_signs == 1) == true_signs):
            for l in np.where(np.any((albi_signs == 1) != true_signs, axis=1))[0]:
                print("ID {}, estimated h^2 = {}, {}, {}".format(l, ests[l], true_signs[l,:], albi_signs[l,:]))

        self.assertTrue(np.alltrue((albi_signs == 1) == true_signs))


    def test_consistency_covariates(self, n_random_vectors=500, n_grid_points = 11, h2=0.5):
        der = albi_lib.GeneralDerivativeSignCalculator([0], self.grid, self.eigenvalues, self.eigenvectors, self.covariates)

        ys = draw_multivariate_from_eigen(self.eigenvectors, h2*self.eigenvalues + (1-h2), n_random_vectors, random_seed=0)
        rotated_ys = np.dot(self.eigenvectors.T, ys)

        #ests = np.array([permutation_testing.parametric_testing(y, self.kinship, self.eigenvectors, self.eigenvalues, covariates=self.covariates, pylmm_resolution=1000)[0] \
        #    for y in ys.T])
        ests = np.array([permutation_testing.parametric_testing(ry[:-1], np.diag(self.eigenvalues[:-1]), np.identity(1798), self.eigenvalues[:-1], 
                                                                covariates=np.dot(self.eigenvectors.T, self.covariates)[:-1,:], pylmm_resolution=1000)[0] \
                for ry in rotated_ys.T])
        true_signs = np.greater.outer(ests, self.grid)

        true_signs[(ests == 0), 0] = False
        if self.grid[-1] == 1:
            true_signs[(ests == 1), -1] = True

        np.savetxt("/tmp/ys", ys)
        np.savetxt("/tmp/rotated_ys", rotated_ys)           

        # Check the rotated version
        albi_signs_rotated = np.array([der.get_derivative_signs_rotated(ry[np.newaxis, :-1])[0,:] for ry in rotated_ys.T])

        if not np.alltrue((albi_signs_rotated == 1) == true_signs):
            for l in np.where(np.any((albi_signs_rotated == 1) != true_signs, axis=1))[0]:
                print("ID {}, estimated h^2 = {}, {}, {}".format(l, ests[l], true_signs[l,:], albi_signs_rotated[l,:]))

        self.assertTrue(np.alltrue((albi_signs_rotated == 1) == true_signs))

        # Check the unrotated version
        albi_signs = np.array([der.get_derivative_signs(y[np.newaxis, :])[0,:] for y in ys.T])

        if not np.alltrue((albi_signs == 1) == true_signs):
            for l in np.where(np.any((albi_signs == 1) != true_signs, axis=1))[0]:
                print("ID {}, estimated h^2 = {}, {}, {}".format(l, ests[l], true_signs[l,:], albi_signs[l,:]))

        self.assertTrue(np.alltrue((albi_signs == 1) == true_signs))


    
if __name__ == '__main__':
    unittest.main()