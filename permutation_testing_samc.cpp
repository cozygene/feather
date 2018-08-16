#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <algorithm>
#include <iterator>
#include <thread>
#include <mutex>
#include <future>
#include <chrono>
#include <cmath>
#include <memory>

#include <boost/progress.hpp>
#include <boost/program_options.hpp>
#include <boost/bind.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/ref.hpp>

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/filesystem.hpp>


//#include "boost/thread/thread_pool.hpp"
#include "boost/asio/thread_pool.hpp"
#include "boost/asio.hpp"

#include "Eigen/Dense"
#include "Eigen/Core"
#include "Eigen/Sparse"

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

mutex progress_bar_lock;
boost::progress_display* progress_bar;


MatrixXd load_csv(
	const string& path, 
	const string& param_name, 
	const char delimiter = ' ') {

	if (!boost::filesystem::exists(path)) {
		cout << param_name << " - cannot open filename \"" << path << "\"" << endl;
		exit(-1);
	}	

	ifstream indata;
	indata.open(path);
	string line;
	vector<double> values;
	uint rows = 0;
	while (getline(indata, line)) {
	stringstream lineStream(line);
	string cell;
	while (getline(lineStream, cell, delimiter)) {
	    values.push_back(stod(cell));
	}
	++rows;
	}
	return Map<const Matrix<double, Dynamic, Dynamic, RowMajor>>(values.data(), rows, values.size()/rows);
}

// TODO: Make general X/PCs
void generate_derivative_weights(double H2, const VectorXd& eigenvalues, ArrayXd& weights) {
	int n = eigenvalues.rows();
	
	ArrayXd partial = (eigenvalues.array() - 1) / (H2 * (eigenvalues.array() - 1) + 1);
	partial(n-1) = 0;

	double mean_partial = partial.sum() / (n-1);	
	weights = (1 / (H2 * (eigenvalues.array() - 1) + 1)) * (partial - mean_partial);
	weights(n-1) = 0;
}

int naive_permutation_testing(const VectorXd& phenotype, const ArrayXd& weights, const MatrixXd& eigenvectors_T, int n_permutations, int n_chunks) {
	int n = phenotype.rows();
	int chunk_size = ceil(n_permutations / n_chunks);

	random_device rd;
  mt19937 g(rd());

  vector<double> permuted_phenotype(phenotype.data(), phenotype.data() + n);
  Map<VectorXd> permuted_phenotype_vec(permuted_phenotype.data(), n);
  MatrixXd permuted_phenotypes(n, chunk_size);

  int count = 0;
  int effective_chunk_size = chunk_size;
  for (int t = 0; t < n_permutations; t += chunk_size) {
  	if ((n_permutations - t) < chunk_size) {
			effective_chunk_size = n_permutations - t;
			permuted_phenotypes.conservativeResize(NoChange_t::NoChange, effective_chunk_size);
		}

	  for (int n_perm = 0; n_perm < effective_chunk_size; n_perm++) {
			shuffle(permuted_phenotype.begin(), permuted_phenotype.end(), g);		
			permuted_phenotypes.col(n_perm) = permuted_phenotype_vec;
		}		
		
		MatrixXd rotated = eigenvectors_T * permuted_phenotypes;  // TODO: Maybe avoid this variable?
		RowVectorXd albi = weights.matrix().transpose() * rotated.array().square().matrix();
		count += (albi.array() >= 0).count();

		// Update progress bar
		{
			progress_bar_lock.lock();
			(*progress_bar) += effective_chunk_size;
			progress_bar_lock.unlock();
		}
	}

	return count;
}

int multithreaded_permutation_testing(const VectorXd& phenotype, const ArrayXd& weights, const MatrixXd& eigenvectors_T, int n_permutations, int n_chunks, int n_threads) {
	int count = 0;
	int sent_perms = 0;
	int n_permutations_per_thread = ceil(n_permutations / n_threads);

	vector<future<int>> futures;
	for (int n_thread = 0; n_thread < n_threads; n_thread++) {
		int n_permutations_this_thread = ((n_permutations - sent_perms) < n_permutations_per_thread) ? (n_permutations - sent_perms) : n_permutations_per_thread;
		sent_perms += n_permutations_this_thread;

		futures.push_back(async(std::launch::async, 
								naive_permutation_testing, 
								phenotype, weights, eigenvectors_T, n_permutations_this_thread, n_chunks));
	}

	for (auto &e : futures) {		
		count += e.get();
	}

	return count;
}


//
// SAMC
//
class SAMC {
  public: 
	//
	// Algorithm parameters that do not vary between runs
	//
	int _n_partitions;  // Not including the last partition
	int _n_partitions_total;
	int _n_iterations;
	const VectorXd& _eigenvalues;
	const MatrixXd& _eigenvectors_T;
	double _replace_proportion;
	double _relative_sampling_error_threshold;
	int _t0;

	default_random_engine _rd;
  	mt19937 _g;
  	uniform_real_distribution<double> _uniform_sampler;

	//
	// Internal data structures
	//
	int _n_individuals;
	int _n_replacements;
	int _n_iter;

	vector<double> _partition_boundaries;
	MatrixXd _weights_at_partition_boundaries;

	VectorXd _current_x;
	VectorXd _current_rotated;
	ArrayXd _current_theta;
	int _current_partition_index;

	VectorXd _proposed_x;
	VectorXd _proposed_rotated;
	int _proposed_partition_index;

	ArrayXd _observed_sampling_distribution;

	vector<int> _random_permutation;
	vector<int> _random_subset;

	// Constructor
	SAMC(int n_partitions, 
		 int n_iterations, 
		 const VectorXd& eigenvalues,
		 const MatrixXd& eigenvectors_T,
		 double replace_proportion,
		 double relative_sampling_error_threshold,
		 int t0
		 ) : _n_partitions(n_partitions),
		 	 _n_partitions_total(n_partitions + 1),
			 _n_iterations(n_iterations),
			 _eigenvalues(eigenvalues),
			 _eigenvectors_T(eigenvectors_T),
			 _replace_proportion(replace_proportion),
			 _relative_sampling_error_threshold(relative_sampling_error_threshold),
			 _t0(t0),
			 _rd(std::chrono::system_clock::now().time_since_epoch().count()),
			 _g(_rd()) { }

	void propose_new_permutation() {
		_proposed_x = _current_x;
		_proposed_rotated = _current_rotated;

		// Get a random subset from a random permutation
		// _random_subset maps to the prefix of _random_permutation
		shuffle(_random_permutation.begin(), _random_permutation.end(), _g);
		_random_subset.assign(_random_permutation.begin(), _random_permutation.begin() + _n_replacements);	
		sort(_random_subset.begin(), _random_subset.begin() + _n_replacements);

		// Shuffle subset
		for (int i = 0; i < _n_replacements; i++) {
			_proposed_x[_random_subset[i]] = _current_x[_random_permutation[i]];
		}

		// Update the rotated x
		for (int i = 0; i < _n_replacements; i++) {
			_proposed_rotated += _eigenvectors_T.col(_random_subset[i]) * (_current_x[_random_permutation[i]] - _current_x[_random_subset[i]]);
		}

		/*
		SparseVector<double> update_coefficients(_n_individuals);		
		for (int i = 0; i < _n_replacements; i++) {
			update_coefficients.coeffRef(_random_subset[i]) = _current_x[_random_permutation[i]] - _current_x[_random_subset[i]];
		}
		_proposed_rotated = _eigenvectors_T * update_coefficients;
		*/
	}

	int find_partition_proposed_index() {
		ArrayXd derivative_signs = (_weights_at_partition_boundaries * _proposed_rotated.array().square().matrix()).array();

		if (derivative_signs[0] <= 0) {
			return 0;
		} else if (derivative_signs[_n_partitions-1] >= 0) {
			return _n_partitions_total-1;
		} else {
			for (int i = 0; i < _n_partitions-1; i++) {
				if ((derivative_signs[i] >= 0) && (derivative_signs[i+1] <= 0)) {
					return i+1;
				}
			}
			cout << "Error in partition finding" << endl;
			return 0; // Should not happen
		}
	}

	double calculate_relative_sampling_error(int current_total) {
		int m0 = (_observed_sampling_distribution == 0).count();
		int m = _n_partitions_total;
		double relative_sampling_error =  ((_observed_sampling_distribution / float(current_total+1) - 1.0/(m-m0)).abs() / (1.0/(m-m0))).maxCoeff();
		return relative_sampling_error;
	}

	double estimated_p_value() {
		return exp(_current_theta[_n_partitions_total-1]) / _current_theta.exp().sum();
	}

	void run(const VectorXd& phenotype, double H2, double* return_p) {
		//
		// Init structures
		//
		_n_individuals = phenotype.rows();		
		for (int i = 0; i < _n_individuals; i++) {
			_random_permutation.push_back(i);
		}
		_n_replacements = int(_n_individuals * _replace_proportion);

		// Partitions
		for (double b = H2/_n_partitions; b < H2; b += H2/_n_partitions) {
			_partition_boundaries.push_back(b);			
		}
		_partition_boundaries.push_back(H2);

		// Weights at partitions
		_weights_at_partition_boundaries = MatrixXd(_n_partitions, _n_individuals);
		for (int i = 0; i < _n_partitions; i++) {
			ArrayXd weights(_n_individuals);
			generate_derivative_weights(_partition_boundaries[i], 
										_eigenvalues, 
										weights);
			_weights_at_partition_boundaries.row(i) = weights;
		}

		// Initial points
		_current_x = phenotype;
		_current_rotated = _eigenvectors_T * _current_x; 
		_current_partition_index = _n_partitions_total - 1;
		_current_theta = VectorXd::Zero(_n_partitions_total);
		_observed_sampling_distribution = ArrayXd::Zero(_n_partitions_total);

		// Iterate!
		for (_n_iter = 0; _n_iter < _n_iterations; _n_iter++) {
			//
			// Draw a new data point from MH
			//
			propose_new_permutation();  // New proposal in _proposed_x
			_proposed_partition_index = find_partition_proposed_index();
			
			//
			// Calculate the ratio, and accept if necessary
			//
			double r = exp(_current_theta[_current_partition_index] - _current_theta[_proposed_partition_index]);
			if (_uniform_sampler(_rd) < r) {
				_current_x = _proposed_x;
				_current_rotated = _proposed_rotated;
				_current_partition_index = _proposed_partition_index;
			}

			//
			// Update estimates
			//
			_observed_sampling_distribution[_current_partition_index]++;
			double gain_factor = float(_t0) / max(_t0, _n_iter);
			_current_theta -= (gain_factor / _n_partitions_total);
			_current_theta[_current_partition_index] += gain_factor;

			//
			// Decide if we should stop
			//
			double relative_sampling_error = calculate_relative_sampling_error(_n_iter);
			
			if (relative_sampling_error <= _relative_sampling_error_threshold) {
				{
					progress_bar_lock.lock();
					(*progress_bar) += (_n_iterations - _n_iter);
					progress_bar_lock.unlock();
				}
				break;
			}

			{
				progress_bar_lock.lock();
				(*progress_bar) += 1;
				progress_bar_lock.unlock();
			}			
		}

		double p = estimated_p_value();

		/*
		cout << _current_theta.transpose() << endl;
		cout << _observed_sampling_distribution.transpose() << endl;
		cout << p << endl;
		*/

		*return_p = p;
	}

};





// ========================================================================
//
// MAIN
//
// ========================================================================

int main(int argc, char** argv) {
	Eigen::initParallel();

	//
	// Parse flags
	//
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "Produce help message")
	    ("eigenvectors_filename", po::value<string>()->default_value(""), "Eigenvectors filename")
	    ("eigenvalues_filename", po::value<string>()->default_value(""), "Eigenvalues filename")
	    ("phenotypes_filename", po::value<string>()->default_value(""), "Phenotypes filename")
	    ("heritabilities_filename", po::value<string>()->default_value(""), "Heritabilities filename")
        ("output_filename", po::value<string>()->default_value(""), "Output filename")    
	    ("phenotype_indices", po::value<string>()->default_value(""), "Which phenotype indices?")
	    ("n_permutations", po::value<int>()->default_value(10000), "# of permutations")
	    ("n_chunks", po::value<int>()->default_value(10), "# of chunks")
	    ("n_repetitions", po::value<int>()->default_value(1), "# of repetitions per phenotype")
	    ("n_threads", po::value<int>()->default_value(-1), "# of threads to use (-1 for # of cpus)")
	    ("samc", po::value<bool>()->default_value(false), "Use SAMC (or regular permutation testing)")
	    ("debug", po::value<bool>()->default_value(false), "Print debug files")
	    ("n_partitions", po::value<int>()->default_value(50), "Number of partitions")
	    ("t0", po::value<int>()->default_value(10000), "t0 parameter of SAMC")
	    ("replace_proportion", po::value<double>()->default_value(0.05), "Replace proportion")
	    ("relative_sampling_error_threshold", po::value<double>()->default_value(0.0001), "relative_sampling_error_threshold")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);    

	if (vm.count("help")) {
	    cout << desc << "\n";
	    return 1;
	}

	//
	// Load data
	//
	
	MatrixXd eigenvectors = load_csv(vm["eigenvectors_filename"].as<string>(), "eigenvectors_filename");
	MatrixXd eigenvectors_T = eigenvectors.transpose();
	VectorXd eigenvalues = load_csv(vm["eigenvalues_filename"].as<string>(), "eigenvalues_filename");
	MatrixXd phenotypes = load_csv(vm["phenotypes_filename"].as<string>(), "phenotypes_filename");
	VectorXd H2s = load_csv(vm["heritabilities_filename"].as<string>(), "heritabilities_filename");

	//
	// Run
	//
	int n_permutations = vm["n_permutations"].as<int>();
	int n_repetitions = vm["n_repetitions"].as<int>();
	int n_chunks = vm["n_chunks"].as<int>();

	// Parse which phenotype to run on
	vector<string> phenotype_indices_split;
	vector<int> phenotype_indices;
	if (vm["phenotype_indices"].as<string>().size() == 0) {
		for (int i = 0; i < phenotypes.cols(); i++) {
			phenotype_indices.push_back(i);
		}
	} else {
		boost::split(phenotype_indices_split, vm["phenotype_indices"].as<string>(), boost::is_any_of(","));
		for (auto subrange : phenotype_indices_split) {
			vector<string> hyphen_split;
			boost::split(hyphen_split, subrange, boost::is_any_of("-"));
			if (hyphen_split.size() == 1) {
				phenotype_indices.push_back(boost::lexical_cast<int>(hyphen_split[0]));
			} else if (hyphen_split.size() == 2) {
				int start = boost::lexical_cast<int>(hyphen_split[0]);
				int end = boost::lexical_cast<int>(hyphen_split[1]);
				for (int i = start; i < end+1; i++) {
					phenotype_indices.push_back(i);
				}
			} else {
				cerr << "Error parsing phenotype indices";
				exit(-1);
			}
		}
	}

	int n_phenotypes = phenotype_indices.size();

	bool use_samc = vm["samc"].as<bool>();


	progress_bar = new boost::progress_display(n_phenotypes * n_permutations * n_repetitions);

	MatrixXd results(n_phenotypes, n_repetitions);
	vector<shared_ptr<SAMC>> samc_objects;

	// Figure out number of threads
	int n_threads = 1;
	int concurrentThreadsSupported = std::thread::hardware_concurrency();
	if (concurrentThreadsSupported > 0) {
		n_threads = concurrentThreadsSupported ;
	}
	if (vm["n_threads"].as<int>() > 0) {
		n_threads = vm["n_threads"].as<int>();
	}	

	// Either to simple permutation testing, or SAMC. The multithreading model is different, because simple
	// permutation testing can be parallelized even for one phenotype, while SAMC cannot.
	if (use_samc) {
		//
		// SAMC
		//


		{ // Beginning of thread pool scope
			// Create a thread pool			
			//boost::threadpool::pool thread_pool(n_threads);
			boost::asio::thread_pool thread_pool(n_threads);

			int result_index = 0;
			for (int n_phenotype : phenotype_indices) {
				for (int n_repeat = 0; n_repeat < n_repetitions; n_repeat++) {
					if (H2s(n_phenotype) == 0) {
						results(result_index, n_repeat) = 1.0;

						progress_bar_lock.lock();
						(*progress_bar) += n_permutations;
						progress_bar_lock.unlock();
						continue;
					}
				

					// Construct SAMC object				
					samc_objects.push_back(shared_ptr<SAMC>(new SAMC(vm["n_partitions"].as<int>(), n_permutations, eigenvalues, eigenvectors_T, 
							  vm["replace_proportion"].as<double>(), vm["relative_sampling_error_threshold"].as<double>(), vm["t0"].as<int>())));

					// samc_objects.back()->run(phenotypes.col(n_phenotype), H2s(n_phenotype), &results(result_index, n_repeat));
					//thread_pool.schedule(boost::bind(&SAMC::run, boost::ref(*samc_objects.back()), 
					//								 phenotypes.col(n_phenotype), H2s(n_phenotype), &results(result_index, n_repeat)));
					boost::asio::post(thread_pool, boost::bind(&SAMC::run, boost::ref(*samc_objects.back()), 
													 phenotypes.col(n_phenotype), H2s(n_phenotype), &results(result_index, n_repeat)));

				}
				result_index++;
			}
		} // End of thread pool scope - all threads should finish here

	} else {
		//
		// Simple permutation testing
		//
		int result_index = 0;		
		for (int n_phenotype : phenotype_indices) {
			for (int n_repeat = 0; n_repeat < n_repetitions; n_repeat++) {
				if (H2s(n_phenotype) == 0) {
					results(result_index, n_repeat) = 1.0;

					progress_bar_lock.lock();
					(*progress_bar) += n_permutations;
					progress_bar_lock.unlock();
					continue;
				}
			
			
				ArrayXd weights(eigenvalues.rows());
				generate_derivative_weights(H2s(n_phenotype), eigenvalues, weights);

				results(result_index, n_repeat) = multithreaded_permutation_testing(phenotypes.col(n_phenotype), weights, eigenvectors_T, 
																				   n_permutations, n_chunks, n_threads) / float(n_permutations);		
				
			}
			result_index++;			
		}
	}

	cout << results << endl;

    //
    // Write output
    //
    string output_filename = vm["output_filename"].as<string>();
    if (output_filename.length() == 0) {
        output_filename = vm["phenotypes_filename"].as<string>() + ".out";
    }
    ofstream output_file;
    output_file.open(output_filename);
    output_file << results << endl; 
    output_file.close();

    if (use_samc && vm["debug"].as<bool>()) {
    	// If this is SAMC and we were asked to print debug info, print the bias estimate and the last rle
    	string debug_filename_bias = output_filename + ".debug.bias";
    	string debug_filename_rse = output_filename + ".debug.rse";

    	MatrixXf bias(n_phenotypes, n_repetitions);
    	MatrixXf rse(n_phenotypes, n_repetitions);

    	int cnt = 0;
		for (int result_index = 0; result_index < phenotype_indices.size(); result_index++) {
			for (int n_repeat = 0; n_repeat < n_repetitions; n_repeat++) {
				bias(result_index, n_repeat) = ((samc_objects[cnt]->_observed_sampling_distribution[samc_objects[cnt]->_n_partitions_total-1]) / float(n_permutations)) * (samc_objects[cnt]->_n_partitions_total);
				rse(result_index, n_repeat) = samc_objects[cnt]->calculate_relative_sampling_error(samc_objects[cnt]->_n_iter);
				cnt++;
			}

		}
	    ofstream debug_file_bias;
	    debug_file_bias.open(debug_filename_bias);
	    debug_file_bias << bias << endl; 
	    debug_file_bias.close();

	    ofstream debug_file_rse;
	    debug_file_rse.open(debug_filename_rse);
	    debug_file_rse << rse << endl; 
	    debug_file_rse.close();
    }

  return 0;
}