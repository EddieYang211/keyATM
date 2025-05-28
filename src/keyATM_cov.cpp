#include "keyATM_cov.h"

// OpenMP support
#ifdef _OPENMP
#include <omp.h>
#endif

// Additional includes for safer parallel implementation
#include <ctime>
#include <stdexcept>

using namespace Eigen;
using namespace Rcpp;
using namespace std;

# define PI_V   3.14159265358979323846  /* pi */


void keyATMcov::setup_openmp()
{
#ifdef _OPENMP
    // Be very conservative with thread count for safety
    int max_threads = omp_get_max_threads();
    
    // Start with conservative threading based on dataset size
    if (num_doc < 10000) {
        num_threads = 1; // Single thread for small datasets
    } else if (num_doc < 100000) {
        num_threads = std::min(max_threads, 2); // Max 2 threads for medium datasets
    } else if (num_doc < 500000) {
        num_threads = std::min(max_threads, 3); // Max 3 threads for large datasets
    } else {
        num_threads = std::min(max_threads, 4); // Max 4 threads for very large datasets
    }
    
    // Additional checks for memory-intensive operations
    if (num_cov > 100) {
        num_threads = std::min(num_threads, 4); // Reduce threads for high-dimensional problems
    }
    
    use_openmp = (num_threads > 1);
    
    if (verbose) {
        if (use_openmp) {
            Rcpp::Rcout << "Using OpenMP with " << num_threads << " threads for optimization." << std::endl;
            Rcpp::Rcout << "Dataset size: " << num_doc << " documents, " << num_cov << " covariates, " 
                        << num_topics << " topics." << std::endl;
        } else {
            Rcpp::Rcout << "Using single-threaded computation (dataset size: " << num_doc << " docs)." << std::endl;
        }
    }
#else
    num_threads = 1;
    use_openmp = false;
    if (verbose) {
        Rcpp::Rcout << "OpenMP not available. Using single-threaded computation." << std::endl;
    }
#endif
}


void keyATMcov::init_thread_storage()
{
    thread_storage.clear();
    
    if (!use_openmp || num_threads <= 1) {
        return; // No need to initialize thread storage for single-threaded execution
    }
    
    try {
        thread_storage.reserve(num_threads);
        thread_storage.resize(num_threads);
        
        for (int t = 0; t < num_threads; ++t) {
            // Check memory availability before allocating large vectors
            if (num_doc > 1000000) {
                size_t estimated_memory = num_doc * 10 * sizeof(double); // Rough estimate
                if (verbose) {
                    Rcpp::Rcout << "Estimated memory per thread: " << (estimated_memory / 1024 / 1024) << " MB" << std::endl;
                }
            }
            
            thread_storage[t].alpha_sum_new_overall_vec.resize(num_doc);
            thread_storage[t].term_weighted_sum_new.resize(num_doc);
            thread_storage[t].term_weighted_sum_old.resize(num_doc);
            thread_storage[t].term_ndk_new.resize(num_doc);
            thread_storage[t].term_ndk_old.resize(num_doc);
            thread_storage[t].log_alpha_k_topic_base.resize(num_doc);
            thread_storage[t].alpha_k_topic_base_vec.resize(num_doc);
            thread_storage[t].proposed_alpha_k_vec.resize(num_doc);
            thread_storage[t].X_k_proposal.resize(num_doc);
            thread_storage[t].C_col_t_times_delta.resize(num_doc);
            
            // Initialize thread-specific RNG with high-quality seed
            std::random_device rd;
            thread_storage[t].rng.seed(rd() + t * 98765 + static_cast<unsigned>(std::time(nullptr)));
            
            // Test the RNG to make sure it's working
            double test_val = thread_storage[t].unif_dist(thread_storage[t].rng);
            if (test_val < 0.0 || test_val > 1.0) {
                throw std::runtime_error("RNG initialization failed for thread " + std::to_string(t));
            }
        }
        
        if (verbose) {
            Rcpp::Rcout << "Successfully initialized thread-local storage for " << num_threads << " threads." << std::endl;
        }
        
    } catch (const std::exception& e) {
        Rcpp::Rcerr << "Error initializing thread storage: " << e.what() << std::endl;
        Rcpp::Rcerr << "Falling back to single-threaded execution." << std::endl;
        
        // Clean up any partially initialized storage
        thread_storage.clear();
        use_openmp = false;
        num_threads = 1;
        
        // Try to free memory
        std::vector<ThreadLocalStorage>().swap(thread_storage);
    } catch (...) {
        Rcpp::Rcerr << "Unknown error during thread storage initialization. Using single-threaded execution." << std::endl;
        thread_storage.clear();
        use_openmp = false;
        num_threads = 1;
    }
}


void keyATMcov::read_data_specific()
{
  // Covariate
  model_settings = model["model_settings"];
  NumericMatrix C_r = model_settings["covariates_data_use"];
  C = Rcpp::as<Eigen::MatrixXd>(C_r);
  num_cov = C.cols();

  // Pre-compute C transpose for efficiency
  C_transpose = C.transpose();

  // Slice Sampling
  val_min = model_settings["slice_min"];
  val_min = shrink(val_min, slice_A);

  val_max = model_settings["slice_max"];
  val_max = shrink(val_max, slice_A);

  // Metropolis Hastings
  mh_use = model_settings["mh_use"];
  
  // Pre-allocate likelihood computation vectors
  doc_alpha_sums.resize(num_doc);
  doc_alpha_weighted_sums.resize(num_doc);
  
  // Setup OpenMP - but be conservative
  setup_openmp();
  init_thread_storage();
}


void keyATMcov::initialize_specific()
{
  // Alpha
  Alpha = MatrixXd::Zero(num_doc, num_topics);  // used in iteration
  alpha = VectorXd::Zero(num_topics);  // used in iteration

  // Lambda
  mu = 0.0;
  sigma = 1.0;
  Lambda = MatrixXd::Zero(num_topics, num_cov);
  for (int k = 0; k < num_topics; ++k) {
    // Initialize with R random
    for (int i = 0; i < num_cov; ++i) {
      Lambda(k, i) = R::rnorm(0.0, 0.3);
    }
  }
  
  // Pre-compute constants
  sigma_squared = sigma * sigma;
  inv_2sigma_squared = 1.0 / (2.0 * sigma_squared);
  log_prior_const = -0.5 * log(2.0 * PI_V * sigma_squared);
}


void keyATMcov::resume_initialize_specific()
{
  // Alpha
  Alpha = MatrixXd::Zero(num_doc, num_topics);
  alpha = VectorXd::Zero(num_topics); // used in iteration

  // Lambda
  mu = 0.0;
  sigma = 1.0;
  Lambda = MatrixXd::Zero(num_topics, num_cov);
  List Lambda_iter = stored_values["Lambda_iter"];
  NumericMatrix Lambda_r = Lambda_iter[Lambda_iter.size() - 1];
  Lambda = Rcpp::as<Eigen::MatrixXd>(Lambda_r);
  
  // Pre-compute constants
  sigma_squared = sigma * sigma;
  inv_2sigma_squared = 1.0 / (2.0 * sigma_squared);
  log_prior_const = -0.5 * log(2.0 * PI_V * sigma_squared);
  
  // Setup OpenMP
  setup_openmp();
  init_thread_storage();
}


// Thread-safe version of likelihood computation
double keyATMcov::compute_likelihood_terms_threadlocal(int k, int t, double current_lambda_kt_val,
                                                      const Eigen::VectorXd& current_alpha_k_vec,
                                                      ThreadLocalStorage& tls)
{
  double loglik = 0.0;

  if (num_doc == 0) {
    // Prior for Lambda(k,t)
    loglik += log_prior_const;
    double lambda_diff = current_lambda_kt_val - mu;
    loglik -= lambda_diff * lambda_diff * inv_2sigma_squared;
    return loglik;
  }

  // Use thread-local pre-allocated vectors
  Eigen::Map<const Eigen::VectorXd> doc_each_len_weighted_eigen(doc_each_len_weighted.data(), doc_each_len_weighted.size());

  // Existing Alpha values for topic k across all documents
  Eigen::VectorXd alpha_k_old_from_global_Alpha_vec = Alpha.col(k);

  // Sum of Alpha values for each document, based on global Alpha
  const Eigen::VectorXd& alpha_sum_old_overall_vec = doc_alpha_sums;

  // New sum of Alpha values if current_alpha_k_vec is used for topic k
  tls.alpha_sum_new_overall_vec.noalias() = alpha_sum_old_overall_vec - alpha_k_old_from_global_Alpha_vec + current_alpha_k_vec;
  
  // Counts for topic k across all documents
  Eigen::VectorXd n_dk_k_vec = n_dk.col(k);

  // Define a lambda for applying mylgamma element-wise
  auto mylgamma_unary_op = [this](double x) { return this->mylgamma(x); };

  // Perform vectorized calculations with thread-local memory
  loglik += (tls.alpha_sum_new_overall_vec.unaryExpr(mylgamma_unary_op) - alpha_sum_old_overall_vec.unaryExpr(mylgamma_unary_op)).sum();
  
  tls.term_weighted_sum_new.noalias() = doc_each_len_weighted_eigen + tls.alpha_sum_new_overall_vec;
  tls.term_weighted_sum_old.noalias() = doc_each_len_weighted_eigen + alpha_sum_old_overall_vec;
  loglik -= (tls.term_weighted_sum_new.unaryExpr(mylgamma_unary_op) - tls.term_weighted_sum_old.unaryExpr(mylgamma_unary_op)).sum();
  
  loglik -= (current_alpha_k_vec.unaryExpr(mylgamma_unary_op) - alpha_k_old_from_global_Alpha_vec.unaryExpr(mylgamma_unary_op)).sum();
  
  tls.term_ndk_new.noalias() = n_dk_k_vec + current_alpha_k_vec;
  tls.term_ndk_old.noalias() = n_dk_k_vec + alpha_k_old_from_global_Alpha_vec;
  loglik += (tls.term_ndk_new.unaryExpr(mylgamma_unary_op) - tls.term_ndk_old.unaryExpr(mylgamma_unary_op)).sum();

  // Prior for Lambda(k,t)
  loglik += log_prior_const;
  double lambda_diff = current_lambda_kt_val - mu;
  loglik -= lambda_diff * lambda_diff * inv_2sigma_squared;

  return loglik;
}


// OPTIMIZATION 1: Vectorized likelihood computation with pre-allocated memory
double keyATMcov::compute_likelihood_terms(int k, int t, double current_lambda_kt_val,
                                           const Eigen::VectorXd& current_alpha_k_vec)
{
  // Use thread-local storage if OpenMP is available
#ifdef _OPENMP
  if (use_openmp && omp_in_parallel()) {
    int thread_id = omp_get_thread_num();
    if (thread_id < thread_storage.size()) {
      return compute_likelihood_terms_threadlocal(k, t, current_lambda_kt_val, current_alpha_k_vec, thread_storage[thread_id]);
    }
  }
#endif
  
  // Fallback to original static thread_local implementation for single-threaded
  double loglik = 0.0;

  if (num_doc == 0) { // Handle case with no documents
    // Prior for Lambda(k,t)
    loglik += log_prior_const;
    double lambda_diff = current_lambda_kt_val - mu;
    loglik -= lambda_diff * lambda_diff * inv_2sigma_squared;
    return loglik;
  }

  // Use pre-allocated vectors to avoid memory allocation overhead
  static thread_local Eigen::VectorXd alpha_sum_new_overall_vec;
  static thread_local Eigen::VectorXd term_weighted_sum_new;
  static thread_local Eigen::VectorXd term_weighted_sum_old;
  static thread_local Eigen::VectorXd term_ndk_new;
  static thread_local Eigen::VectorXd term_ndk_old;
  
  // Resize once if needed
  if (alpha_sum_new_overall_vec.size() != num_doc) {
    alpha_sum_new_overall_vec.resize(num_doc);
    term_weighted_sum_new.resize(num_doc);
    term_weighted_sum_old.resize(num_doc);
    term_ndk_new.resize(num_doc);
    term_ndk_old.resize(num_doc);
  }

  // Ensure doc_each_len_weighted is available as Eigen::VectorXd
  Eigen::Map<const Eigen::VectorXd> doc_each_len_weighted_eigen(doc_each_len_weighted.data(), doc_each_len_weighted.size());

  // Existing Alpha values for topic k across all documents (from global Alpha matrix)
  Eigen::VectorXd alpha_k_old_from_global_Alpha_vec = Alpha.col(k);

  // Sum of Alpha values for each document, based on global Alpha
  const Eigen::VectorXd& alpha_sum_old_overall_vec = doc_alpha_sums;

  // New sum of Alpha values if current_alpha_k_vec is used for topic k
  alpha_sum_new_overall_vec.noalias() = alpha_sum_old_overall_vec - alpha_k_old_from_global_Alpha_vec + current_alpha_k_vec;
  
  // Counts for topic k across all documents
  Eigen::VectorXd n_dk_k_vec = n_dk.col(k);

  // Define a lambda for applying mylgamma element-wise
  auto mylgamma_unary_op = [this](double x) { return this->mylgamma(x); };

  // Perform vectorized calculations with pre-allocated memory
  loglik += (alpha_sum_new_overall_vec.unaryExpr(mylgamma_unary_op) - alpha_sum_old_overall_vec.unaryExpr(mylgamma_unary_op)).sum();
  
  term_weighted_sum_new.noalias() = doc_each_len_weighted_eigen + alpha_sum_new_overall_vec;
  term_weighted_sum_old.noalias() = doc_each_len_weighted_eigen + alpha_sum_old_overall_vec;
  loglik -= (term_weighted_sum_new.unaryExpr(mylgamma_unary_op) - term_weighted_sum_old.unaryExpr(mylgamma_unary_op)).sum();
  
  loglik -= (current_alpha_k_vec.unaryExpr(mylgamma_unary_op) - alpha_k_old_from_global_Alpha_vec.unaryExpr(mylgamma_unary_op)).sum();
  
  term_ndk_new.noalias() = n_dk_k_vec + current_alpha_k_vec;
  term_ndk_old.noalias() = n_dk_k_vec + alpha_k_old_from_global_Alpha_vec;
  loglik += (term_ndk_new.unaryExpr(mylgamma_unary_op) - term_ndk_old.unaryExpr(mylgamma_unary_op)).sum();

  // Prior for Lambda(k,t)
  loglik += log_prior_const;
  double lambda_diff = current_lambda_kt_val - mu;
  loglik -= lambda_diff * lambda_diff * inv_2sigma_squared;

  return loglik;
}


void keyATMcov::iteration_single(int it)
{ 
  int doc_id_;
  int doc_length;
  int w_, z_, s_;
  int new_z, new_s;
  int w_position;

  doc_indexes = sampler::shuffled_indexes(num_doc); // shuffle

  // Create Alpha for this iteration - vectorized computation with safe parallelization
  update_alpha_efficient();

  // Document-level sampling remains sequential for thread safety
  // The R sampling functions (sample_z, sample_s) are not thread-safe
  // Future optimization: implement thread-safe sampling functions
  for (int ii = 0; ii < num_doc; ++ii) {
    doc_id_ = doc_indexes[ii];
    doc_s = S[doc_id_], doc_z = Z[doc_id_], doc_w = W[doc_id_];
    doc_length = doc_each_len[doc_id_];

    token_indexes = sampler::shuffled_indexes(doc_length); //shuffle

    // Prepare Alpha for the doc
    alpha = Alpha.row(doc_id_).transpose(); // take out alpha

    // Iterate each word in the document
    for (int jj = 0; jj < doc_length; ++jj) {
      w_position = token_indexes[jj];
      s_ = doc_s[w_position], z_ = doc_z[w_position], w_ = doc_w[w_position];

      new_z = sample_z(alpha, z_, s_, w_, doc_id_);
      doc_z[w_position] = new_z;

      if (keywords[new_z].find(w_) == keywords[new_z].end())
        continue;

      z_ = doc_z[w_position]; // use updated z
      new_s = sample_s(z_, s_, w_, doc_id_);
      doc_s[w_position] = new_s;
    }

    Z[doc_id_] = doc_z;
    S[doc_id_] = doc_s;
    
    // Progress indicator for long iterations
    if (verbose && num_doc > 50000 && ii % (num_doc / 10) == 0) {
      Rcpp::Rcout << "  Processed " << ii << "/" << num_doc << " documents (" 
                  << (100 * ii / num_doc) << "%)" << std::endl;
    }
  }
  
  sample_parameters(it);
}


void keyATMcov::update_alpha_efficient()
{
  // Vectorized computation: Alpha = exp(C * Lambda^T)
  // This matrix operation is thread-safe when done as a single operation
  Alpha = (C * Lambda.transpose()).array().exp();
  
  // Pre-compute sums for likelihood computation with safe parallelization
  if (num_doc > 0 && Alpha.rows() == num_doc && Alpha.cols() == num_topics) {
    
#ifdef _OPENMP
    if (use_openmp && num_doc > 5000) {
      // Parallel computation of row sums
      #pragma omp parallel for num_threads(num_threads) schedule(static)
      for (int d = 0; d < num_doc; ++d) {
        doc_alpha_sums(d) = Alpha.row(d).sum();
      }
      
      // Parallel computation of weighted sums
      if (doc_each_len_weighted.size() == num_doc) {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int d = 0; d < num_doc; ++d) {
          doc_alpha_weighted_sums(d) = doc_each_len_weighted[d] + doc_alpha_sums(d);
        }
      } else {
        // Fallback to sequential if size mismatch
        if (verbose) {
          Rcpp::Rcerr << "Dimension mismatch in update_alpha_efficient, using sequential computation" << std::endl;
        }
        for (int d = 0; d < num_doc; ++d) {
          if (d < doc_alpha_weighted_sums.size() && d < doc_each_len_weighted.size() && d < doc_alpha_sums.size()) {
             doc_alpha_weighted_sums(d) = doc_each_len_weighted[d] + doc_alpha_sums(d);
          }
        }
      }
    } else {
#endif
      // Sequential computation for smaller datasets or when OpenMP unavailable
      doc_alpha_sums = Alpha.rowwise().sum();
      if (doc_each_len_weighted.size() == num_doc && doc_alpha_sums.size() == num_doc) {
          Eigen::VectorXd doc_each_len_weighted_eigen = Eigen::Map<const Eigen::VectorXd>(doc_each_len_weighted.data(), doc_each_len_weighted.size());
          doc_alpha_weighted_sums = doc_each_len_weighted_eigen.array() + doc_alpha_sums.array();
      } else {
          // Handle size mismatch safely
          if (doc_each_len_weighted.size() != num_doc) {
               Rcpp::Rcerr << "Dimension mismatch: doc_each_len_weighted.size() (" << doc_each_len_weighted.size() 
                           << ") != num_doc (" << num_doc << ")" << std::endl;
          }
          if (doc_alpha_sums.size() != num_doc) {
              Rcpp::Rcerr << "Dimension mismatch: doc_alpha_sums.size() (" << doc_alpha_sums.size() 
                          << ") != num_doc (" << num_doc << ")" << std::endl;
          }
      }
#ifdef _OPENMP
    }
#endif
    
  } else { // Fallback to loop if dimensions are unexpected
      if (num_doc > 0 && Alpha.rows() > 0 && Alpha.cols() > 0) {
        // Safe fallback with bounds checking
        int max_docs = std::min(num_doc, (int)Alpha.rows());
        for (int d = 0; d < max_docs; ++d) {
          if (d < doc_alpha_sums.size()) {
            doc_alpha_sums(d) = Alpha.row(d).sum();
          }
          if (d < doc_alpha_weighted_sums.size() && d < doc_each_len_weighted.size() && d < doc_alpha_sums.size()) {
             doc_alpha_weighted_sums(d) = doc_each_len_weighted[d] + doc_alpha_sums(d);
          }
        }
      }
  }
}


void keyATMcov::update_alpha_row_efficient(int k)
{
  // Bounds checking
  if (k < 0 || k >= num_topics || k >= Lambda.rows()) {
    Rcpp::Rcerr << "Invalid topic index in update_alpha_row_efficient: " << k << std::endl;
    return;
  }
  
  // Update only row k of Alpha after Lambda(k, :) changes
  VectorXd lambda_k = Lambda.row(k).transpose();
  VectorXd new_alpha_k_col = (C * lambda_k).array().exp();
  
  // Bounds checking for new_alpha_k_col
  if (new_alpha_k_col.size() != num_doc) {
    Rcpp::Rcerr << "Size mismatch in update_alpha_row_efficient" << std::endl;
    return;
  }
  
#ifdef _OPENMP
  if (use_openmp && num_doc > 10000) {
    // Parallel update with atomic operations for shared data
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int d = 0; d < num_doc; ++d) {
      if (d < Alpha.rows() && k < Alpha.cols()) {
        double old_alpha_dk = Alpha(d, k);
        double new_alpha_dk = new_alpha_k_col(d);
        
        // Update Alpha matrix
        Alpha(d, k) = new_alpha_dk;
        
        // Thread-safe updates of sums
        if (d < doc_alpha_sums.size()) {
          #pragma omp atomic
          doc_alpha_sums(d) += (new_alpha_dk - old_alpha_dk);
        }
        
        // Update weighted sums (can be done without atomic since it's per-thread)
        if (d < doc_alpha_weighted_sums.size() && d < doc_each_len_weighted.size()) {
          doc_alpha_weighted_sums(d) = doc_each_len_weighted[d] + doc_alpha_sums(d);
        }
      }
    }
  } else {
#endif
    // Sequential version for smaller datasets
    for (int d = 0; d < num_doc; ++d) {
      if (d < Alpha.rows() && k < Alpha.cols() && d < doc_alpha_sums.size()) {
        doc_alpha_sums(d) = doc_alpha_sums(d) - Alpha(d, k) + new_alpha_k_col(d);
        Alpha(d, k) = new_alpha_k_col(d);
        
        if (d < doc_alpha_weighted_sums.size() && d < doc_each_len_weighted.size()) {
          doc_alpha_weighted_sums(d) = doc_each_len_weighted[d] + doc_alpha_sums(d);
        }
      }
    }
#ifdef _OPENMP
  }
#endif
}


void keyATMcov::sample_parameters(int it)
{
  sample_lambda();

  // Store lambda
  int r_index = it + 1;
  if (r_index % thinning == 0 || r_index == 1 || r_index == iter) {
    Rcpp::NumericMatrix Lambda_R = Rcpp::wrap(Lambda);
    List Lambda_iter = stored_values["Lambda_iter"];
    Lambda_iter.push_back(Lambda_R);
    stored_values["Lambda_iter"] = Lambda_iter;
  }
}


double keyATMcov::likelihood_lambda_efficient(int k, int t, const Eigen::VectorXd* precomputed_alpha_k)
{
  double loglik = 0.0;
  Eigen::VectorXd current_alpha_k_for_eval;

  if (precomputed_alpha_k) {
    current_alpha_k_for_eval = *precomputed_alpha_k;
  } else {
    // Compute alpha values for topic k based on current Lambda.row(k)
    // Lambda.row(k) reflects the value of Lambda(k,t) for which likelihood is being computed.
    Eigen::VectorXd current_lambda_k_row = Lambda.row(k).transpose(); 
    current_alpha_k_for_eval = (C * current_lambda_k_row).array().exp(); 
  }
  
  // Lambda(k,t) from the global Lambda matrix is used as current_lambda_kt_val
  loglik = compute_likelihood_terms(k, t, Lambda(k,t), current_alpha_k_for_eval);

  return loglik;
}


double keyATMcov::likelihood_lambda(int k, int t)
{
  // Use the efficient version
  return likelihood_lambda_efficient(k, t);
}


void keyATMcov::sample_lambda()
{
  // Start with safer parallelization - only for likelihood computation heavy operations
#ifdef _OPENMP
  if (use_openmp && (num_topics > 5 && num_cov > 20)) {
    // Use parallel versions only for larger problems where it's worth the overhead
    mh_use ? sample_lambda_mh_parallel() : sample_lambda_slice_parallel();
  } else {
#endif
    mh_use ? sample_lambda_mh_efficient() : sample_lambda_slice();
#ifdef _OPENMP
  }
#endif
}


void keyATMcov::sample_lambda_mh_parallel()
{
#ifdef _OPENMP
  if (!use_openmp || thread_storage.size() == 0) {
    sample_lambda_mh_efficient();
    return;
  }

  std::vector<double> adapter_step(num_cov, 0.1);
  std::vector<std::vector<bool>> accept_flags(num_topics, std::vector<bool>(num_cov, false));
  std::vector<std::vector<double>> mh_step_size = model_settings["mh_step_size"];

  topic_ids = sampler::shuffled_indexes(num_topics);
  cov_ids = sampler::shuffled_indexes(num_cov);

  // Process topics in parallel, but with careful synchronization
  #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads) shared(accept_flags)
  for (int k_idx = 0; k_idx < num_topics; ++k_idx) {
    int k = topic_ids[k_idx];
    int thread_id = omp_get_thread_num();
    
    // Ensure thread_id is valid
    if (thread_id >= thread_storage.size()) continue;
    
    ThreadLocalStorage& tls = thread_storage[thread_id];
    
    // Read Lambda.row(k) once at the start (read-only, thread-safe)
    Eigen::VectorXd current_lambda_k_row;
    #pragma omp critical(lambda_read)
    {
      current_lambda_k_row = Lambda.row(k).transpose();
    }
    
    Eigen::VectorXd current_X_k = C * current_lambda_k_row;

    for (int t_idx = 0; t_idx < num_cov; ++t_idx) {
      int t = cov_ids[t_idx];
      double lambda_kt_current = current_lambda_k_row(t);

      // Alpha vector for current Lambda(k,t) using the current_X_k state
      tls.alpha_k_topic_base_vec = current_X_k.array().exp();
      double current_L = compute_likelihood_terms_threadlocal(k, t, lambda_kt_current, 
                                                            tls.alpha_k_topic_base_vec, tls);

      // Proposal using thread-safe RNG
      double U_var = mh_step_size[k][t];
      double step = tls.norm_dist(tls.rng) * U_var;
      double lambda_kt_new = lambda_kt_current + step;

      // Calculate the new X_k for the proposal
      tls.X_k_proposal = current_X_k + C.col(t) * step;
      tls.proposed_alpha_k_vec = tls.X_k_proposal.array().exp();

      // Calculate new likelihood with thread-local storage
      double new_L = compute_likelihood_terms_threadlocal(k, t, lambda_kt_new, 
                                                         tls.proposed_alpha_k_vec, tls);

      // Acceptance calculation
      double log_acceptance_ratio = new_L - current_L;
      
      if (std::log(tls.unif_dist(tls.rng)) < log_acceptance_ratio) {
        // Accept: update Lambda(k,t) in a thread-safe way
        #pragma omp critical(lambda_update)
        {
          Lambda(k,t) = lambda_kt_new;
        }
        accept_flags[k][t] = true;
        current_X_k = tls.X_k_proposal; // Update local X_k
        current_lambda_k_row(t) = lambda_kt_new; // Update local copy
      }
      // If rejected, no updates needed
    }
  }

  // Store acceptance rates
  model_settings["accept_Lambda"] = accept_flags;
#else
  sample_lambda_mh_efficient();
#endif
}


void keyATMcov::sample_lambda_slice_parallel()
{
#ifdef _OPENMP
  if (!use_openmp || thread_storage.size() == 0) {
    sample_lambda_slice();
    return;
  }

  topic_ids = sampler::shuffled_indexes(num_topics);
  cov_ids = sampler::shuffled_indexes(num_cov);
  const double A = slice_A;

  // Process topics in parallel with careful synchronization
  #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
  for (int kk = 0; kk < num_topics; ++kk) {
    int k = topic_ids[kk];
    int thread_id = omp_get_thread_num();
    
    if (thread_id >= thread_storage.size()) continue;
    
    ThreadLocalStorage& tls = thread_storage[thread_id];

    // Read current Lambda.row(k) in thread-safe manner
    Eigen::VectorXd current_lambda_k_row;
    #pragma omp critical(lambda_read)
    {
      current_lambda_k_row = Lambda.row(k).transpose();
    }
    
    // Current log(Alpha.col(k)) based on accepted Lambda.row(k) values for this topic
    tls.log_alpha_k_topic_base.noalias() = C * current_lambda_k_row;
    tls.alpha_k_topic_base_vec = tls.log_alpha_k_topic_base.array().exp();

    for (int tt = 0; tt < num_cov; ++tt) {
      int t = cov_ids[tt];
      
      double original_Lambda_kt = current_lambda_k_row(t);
      
      // Calculate log f(x_0) + log |dx_0/dp_0| for slice definition
      double previous_p_val = shrink(original_Lambda_kt, A);
      double log_f_x0 = compute_likelihood_terms_threadlocal(k, t, original_Lambda_kt, 
                                                            tls.alpha_k_topic_base_vec, tls);
      double transformed_log_f_x0 = log_f_x0 - std::log(A * previous_p_val * (1.0 - previous_p_val));
      double slice_level = transformed_log_f_x0 + std::log(tls.unif_dist(tls.rng));

      double start_p = val_min;
      double end_p = val_max;

      // Pre-compute C.col(t) to avoid repeated column access
      const Eigen::VectorXd& C_col_t = C.col(t);

      for (int shrink_time = 0; shrink_time < max_shrink_time; ++shrink_time) {
        double new_p_val = start_p + (end_p - start_p) * tls.unif_dist(tls.rng); 
        double proposed_Lambda_kt = expand(new_p_val, A);
        
        // Efficiently compute alpha_k vector for the proposed_Lambda_kt
        double delta_lambda = proposed_Lambda_kt - original_Lambda_kt; 
        
        // Vectorized computation with thread-local memory
        tls.C_col_t_times_delta.noalias() = C_col_t * delta_lambda;
        tls.proposed_alpha_k_vec = tls.alpha_k_topic_base_vec.array() * tls.C_col_t_times_delta.array().exp();

        double log_f_proposed = compute_likelihood_terms_threadlocal(k, t, proposed_Lambda_kt, 
                                                                   tls.proposed_alpha_k_vec, tls);
        double transformed_log_f_proposed = log_f_proposed - std::log(A * new_p_val * (1.0 - new_p_val));

        if (slice_level < transformed_log_f_proposed) { // Accept proposal
          #pragma omp critical(lambda_update)
          {
            Lambda(k,t) = proposed_Lambda_kt;
          }
          // Update the topic's base log_alpha and alpha_vector due to accepted change
          tls.log_alpha_k_topic_base += tls.C_col_t_times_delta; 
          tls.alpha_k_topic_base_vec = tls.log_alpha_k_topic_base.array().exp();
          current_lambda_k_row(t) = proposed_Lambda_kt; // Update local copy
          break; // Exit shrink_time loop
        } else { // Shrink interval
          if (std::abs(end_p - start_p) < 1e-9) {
            if (verbose) {
              #pragma omp critical(error_output)
              {
                Rcpp::Rcerr << "Slice sampler interval shrunk too much for Lambda(" << k << "," << t 
                            << "). Keeping current value: " << original_Lambda_kt << std::endl;
              }
            }
            break;
          }
          if (previous_p_val < new_p_val) {
             end_p = new_p_val;
          } else if (new_p_val < previous_p_val) {
             start_p = new_p_val;
          } else {
            if (verbose) {
              #pragma omp critical(error_output)
              {
                Rcpp::Rcerr << "Slice sampler new_p equals previous_p for Lambda(" << k << "," << t 
                            << "). Keeping current value." << std::endl;
              }
            }
            break;
          }
        }
      } // End shrink_time loop
    } // End tt loop (covariates)
  } // End kk loop (topics)

  // Update global Alpha matrix and sums after all parallel work is done
  update_alpha_efficient();
#else
  sample_lambda_slice();
#endif
}


double keyATMcov::loglik_total()
{
  double loglik = 0.0;
  
#ifdef _OPENMP
  if (use_openmp && num_topics > 3) {
    // Safe parallelization of likelihood computation (read-only operations)
    double loglik_topics = 0.0;
    
    #pragma omp parallel for reduction(+:loglik_topics) num_threads(num_threads) schedule(static)
    for (int k = 0; k < num_topics; ++k) {
      double topic_loglik = 0.0;
      
      // Read-only operations on shared data - thread safe
      for (int v = 0; v < num_vocab; ++v) {
        topic_loglik += mylgamma(beta + n_s0_kv(k, v)) - mylgamma(beta);
      }

      // word normalization
      topic_loglik += mylgamma( beta * (double)num_vocab ) - mylgamma(beta * (double)num_vocab + n_s0_k(k) );

      if (k < keyword_k) {
        // For keyword topics - read-only operations
        for (SparseMatrix<double,RowMajor>::InnerIterator it(n_s1_kv, k); it; ++it) {
          topic_loglik += mylgamma(beta_s + it.value()) - mylgamma(beta_s);
        }
        topic_loglik += mylgamma( beta_s * (double)keywords_num[k] ) - mylgamma(beta_s * (double)keywords_num[k] + n_s1_k(k) );

        // Normalization
        topic_loglik += mylgamma( prior_gamma(k, 0) + prior_gamma(k, 1)) - mylgamma( prior_gamma(k, 0)) - mylgamma( prior_gamma(k, 1));

        // s
        topic_loglik += mylgamma( n_s0_k(k) + prior_gamma(k, 1) )
                      - mylgamma(n_s1_k(k) + prior_gamma(k, 0) + n_s0_k(k) + prior_gamma(k, 1))
                      + mylgamma( n_s1_k(k) + prior_gamma(k, 0) );
      }
      
      loglik_topics += topic_loglik;
    }
    
    loglik += loglik_topics;
    
    // Document-level likelihood computation in parallel (read-only)
    if (num_doc > 1000) {
      double loglik_docs = 0.0;
      #pragma omp parallel for reduction(+:loglik_docs) num_threads(num_threads) schedule(static)
      for (int d = 0; d < num_doc; ++d) {
        double doc_loglik = mylgamma(doc_alpha_sums(d)) - mylgamma(doc_alpha_weighted_sums(d));
        for (int k = 0; k < num_topics; ++k) {
          doc_loglik += mylgamma(n_dk(d,k) + Alpha(d, k)) - mylgamma(Alpha(d, k));
        }
        loglik_docs += doc_loglik;
      }
      loglik += loglik_docs;
    } else {
      // Sequential for small datasets
      for (int d = 0; d < num_doc; ++d) {
        loglik += mylgamma(doc_alpha_sums(d)) - mylgamma(doc_alpha_weighted_sums(d));
        for (int k = 0; k < num_topics; ++k) {
          loglik += mylgamma(n_dk(d,k) + Alpha(d, k)) - mylgamma(Alpha(d, k));
        }
      }
    }
    
  } else {
#endif
    // Sequential computation for smaller datasets
    for (int k = 0; k < num_topics; ++k) {
      for (int v = 0; v < num_vocab; ++v) { // word
        loglik += mylgamma(beta + n_s0_kv(k, v)) - mylgamma(beta);
      }

      // word normalization
      loglik += mylgamma( beta * (double)num_vocab ) - mylgamma(beta * (double)num_vocab + n_s0_k(k) );

      if (k < keyword_k) {
        // For keyword topics

        // n_s1_kv
        for (SparseMatrix<double,RowMajor>::InnerIterator it(n_s1_kv, k); it; ++it) {
          loglik += mylgamma(beta_s + it.value()) - mylgamma(beta_s);
        }
        loglik += mylgamma( beta_s * (double)keywords_num[k] ) - mylgamma(beta_s * (double)keywords_num[k] + n_s1_k(k) );

        // Normalization
        loglik += mylgamma( prior_gamma(k, 0) + prior_gamma(k, 1)) - mylgamma( prior_gamma(k, 0)) - mylgamma( prior_gamma(k, 1));

        // s
        loglik += mylgamma( n_s0_k(k) + prior_gamma(k, 1) )
                  - mylgamma(n_s1_k(k) + prior_gamma(k, 0) + n_s0_k(k) + prior_gamma(k, 1))
                  + mylgamma( n_s1_k(k) + prior_gamma(k, 0) );
      }
    }

    // z - use pre-computed values
    for (int d = 0; d < num_doc; ++d) {
      loglik += mylgamma(doc_alpha_sums(d)) - mylgamma(doc_alpha_weighted_sums(d));
      for (int k = 0; k < num_topics; ++k) {
        loglik += mylgamma(n_dk(d,k) + Alpha(d, k)) - mylgamma(Alpha(d, k));
      }
    }
#ifdef _OPENMP
  }
#endif

  // Lambda loglik - can be parallelized safely (read-only)
#ifdef _OPENMP
  if (use_openmp && num_topics > 2) {
    double lambda_loglik = 0.0;
    #pragma omp parallel for reduction(+:lambda_loglik) num_threads(num_threads) collapse(2)
    for (int k = 0; k < num_topics; ++k) {
      for (int t = 0; t < num_cov; ++t) {
        double lambda_diff = Lambda(k,t) - mu;
        lambda_loglik += log_prior_const - lambda_diff * lambda_diff * inv_2sigma_squared;
      }
    }
    loglik += lambda_loglik;
  } else {
#endif
    for (int k = 0; k < num_topics; ++k) {
      for (int t = 0; t < num_cov; ++t) {
        loglik += log_prior_const;
        double lambda_diff = Lambda(k,t) - mu;
        loglik -= lambda_diff * lambda_diff * inv_2sigma_squared;
      }
    }
#ifdef _OPENMP
  }
#endif

  return loglik;
}