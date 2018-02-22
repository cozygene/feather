import numpy as np
import scipy.stats
import statsmodels.api as sm
import attr

from tqdm import *

###############################################################################################
# General SAMC implementation (Yu. et al, Biostatistics 2011)
#

@attr.s
class SAMCParameters(object):
	# x0 - Initial data point to start from
	x0 = attr.ib()

	# test_statistic_func - Function that returns the test statistic (used only for debugging) and identity of the partition of a data point x	
	test_statistic_func = attr.ib()

	# generate_sample_func - Function that accepts a data point x, and returns a new data point y, and the ratio q(x|y)/q(y|x)
	generate_sample_func = attr.ib()

	# required_sampling_distribution - The required sampling distribution across partitions
	required_sampling_distribution = attr.ib(default=None)
	
	# theta0 - Inital estimate for the log probability of the different partitions
	theta0 = attr.ib(default=None)

	# psi - Relative weights of sampled points (in log)
	log_psi = attr.ib(default=lambda x: 0)
	
	# t0 - Value used in gain factor sequence
	t0 = attr.ib(default=1000)
	
	# n_iterations - Number of iterations after which to stop
	n_iterations = attr.ib(default=100000)

	# relative_sampling_error_threshold - Relative sampling error threshold below which to stop
	relative_sampling_error_threshold = attr.ib(default=0.2)

	# fix_theta - For debugging, just use theta0 and don't change it
	fix_theta = attr.ib(default=False)

	# step_size_power - 1 for SAMC, <1 for Polyak-Ruppert
	step_size_power = attr.ib(default=1.0)
	

def SAMC(samc_parameters):
	"""
	samc_parameters - An SAMCParameters instance
	"""
	_p = samc_parameters
	
	# Initialize values for algorithm
	current_x = _p.x0
	current_theta = np.array(_p.theta0)
	current_statistic, current_partition = _p.test_statistic_func(current_x, 0)

	observed_sampling_distribution = np.zeros_like(_p.required_sampling_distribution)
	m = len(_p.required_sampling_distribution)

	# Statistics to aggregate
	statistics = []

	with trange(_p.n_iterations) as T:
		for n_iter in T:
			# Draw a new data point
			y, proposal_ratio = _p.generate_sample_func(current_x)

			# Calculate the ratio for MH
			suggested_statistic, suggested_partition = _p.test_statistic_func(y, current_partition)
			if _p.fix_theta:
				r = np.exp(_p.theta0[current_partition] - _p.theta0[suggested_partition] + _p.log_psi(y) - _p.log_psi(current_x)) * proposal_ratio
			else:
				r = np.exp(current_theta[current_partition] - current_theta[suggested_partition] + _p.log_psi(y) - _p.log_psi(current_x)) * proposal_ratio

			# Accept in probability min(1, r)
			updated = False
			if np.random.uniform(size=1)[0] <= min(1, r):
				updated = True
				current_x = y
				current_statistic, current_partition = suggested_statistic, suggested_partition

			# Update observed sampling distribution
			statistics.append([updated, current_partition, current_statistic, current_theta.copy(), r])
			#print r, samples[-1][0], samples[-1][2:], current_theta
			observed_sampling_distribution[current_partition] += 1

			# Update weights
			d = -np.array(_p.required_sampling_distribution)
			d[current_partition] += 1
			gain_factor = (float(_p.t0) / max(_p.t0, n_iter)) ** _p.step_size_power
			current_theta += gain_factor * d

			# Decide if we should stop
			m0 = sum(observed_sampling_distribution == 0)
			relative_sampling_error =  max(abs(observed_sampling_distribution/float(n_iter+1) - 1.0/(m-m0)) / (1.0/(m-m0)))

			# chisquare = scipy.stats.chisquare(observed_sampling_distribution)
			# regp = sm.OLS(observed_sampling_distribution, sm.add_constant(np.arange(len(observed_sampling_distribution)))).fit().pvalues[-1]
			# if n_iter % (_p.n_iterations/10) == 1:
			# 	T.write("RSE: %f\t Chi^2 p, log10(p): %f %f\t LinReg p, log10(p): %f %f" % (relative_sampling_error, chisquare.pvalue, np.log10(chisquare.pvalue), regp, np.log10(regp)))

			if relative_sampling_error < _p.relative_sampling_error_threshold:
				break

	return current_theta, observed_sampling_distribution, statistics


@attr.s
class SAMCSimpleParameters(SAMCParameters):
	n_partitions = attr.ib(default=None)

def SAMC_simple(samc_simple_parameters):
	"""
	samc_simple_parameters - An SAMCSimpleParameters instance
	"""
	_p = samc_simple_parameters
	_p.required_sampling_distribution = np.ones(_p.n_partitions) / float(_p.n_partitions)
	_p.log_psi = lambda x: 0.0
	_p.theta0 = np.ones(_p.n_partitions) / float(_p.n_partitions)

	return SAMC(_p)

###############################################################################################
# Several simple SAMC cases
#

#
#  Two-sample t-test (Yu. et al, Biostatistics 2011)
#
import bisect

def SAMC_two_sample_t_test(n_partitions, n_iterations, n=100, replace_proportion=0.05):
	def two_sample_test_statistic(x, partitions):
		t, p = scipy.stats.ttest_ind(x[0], x[1])
		j = bisect.bisect(partitions, t)
		return (t,p), j

	def two_sample_generate_sample(x):
		L = int(replace_proportion*n)
		inds0 = np.random.choice(range(n), L, replace=True)
		inds1 = np.random.choice(range(n), L, replace=True)
		
		y = x.copy()
		for l in range(L):
			y[0, inds0[l]], y[1, inds1[l]] = y[1, inds1[l]], y[0, inds0[l]]

		return y, 1.0

	x0 = vstack([np.random.normal(1, 1, size=n), np.random.normal(0, 1, size=n)])
	
	observed_statistic, p = scipy.stats.ttest_ind(x0[0], x0[1])
	#print observed_statistic, p
	partitions = np.concatenate([np.arange(0, observed_statistic, observed_statistic/n_partitions)[1:], [observed_statistic]])

	return SAMC_simple(SAMCSimpleParameters(
		x0=x0, 
		test_statistic_func=lambda x, cp: two_sample_test_statistic(x, partitions), 
		generate_sample_func=two_sample_generate_sample, 
		n_iterations=n_iterations, 
		n_partitions=1000,
		relative_sampling_error_threshold=0))






