#include "keyATM_cov.h"

using namespace Eigen;
using namespace Rcpp;
using namespace std;

# define PI_V   3.14159265358979323846  /* pi */


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
  
  // Pre-allocate temporary vectors for likelihood computation
  temp_alpha_sums.resize(num_doc);
  temp_lgamma_cache.resize(num_doc * 2); // Cache for lgamma computations
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
  
  // Initialize lgamma cache
  lgamma_cache.clear();
  lgamma_cache.reserve(10000); // Reserve space for frequently computed values
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
  
  // Initialize lgamma cache
  lgamma_cache.clear();
  lgamma_cache.reserve(10000);
}


void keyATMcov::iteration_single(int it)
{ 
  int doc_id_;
  int doc_length;
  int w_, z_, s_;
  int new_z, new_s;
  int w_position;

  doc_indexes = sampler::shuffled_indexes(num_doc); // shuffle

  // Create Alpha for this iteration - vectorized computation
  update_alpha_vectorized();

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
  }
  sample_parameters(it);
}


void keyATMcov::update_alpha_vectorized()
{
  // Vectorized computation: Alpha = exp(C * Lambda^T)
  // Use matrix multiplication for maximum efficiency
  Alpha.noalias() = (C * Lambda.transpose()).array().exp();
  
  // Pre-compute sums using vectorized operations
  for (int d = 0; d < num_doc; ++d) {
    doc_alpha_sums[d] = Alpha.row(d).sum();
    doc_alpha_weighted_sums[d] = doc_each_len_weighted[d] + doc_alpha_sums[d];
  }
}


void keyATMcov::sample_parameters(int it)
{
  sample_lambda_optimized();

  // Store lambda
  int r_index = it + 1;
  if (r_index % thinning == 0 || r_index == 1 || r_index == iter) {
    Rcpp::NumericMatrix Lambda_R = Rcpp::wrap(Lambda);
    List Lambda_iter = stored_values["Lambda_iter"];
    Lambda_iter.push_back(Lambda_R);
    stored_values["Lambda_iter"] = Lambda_iter;
  }
}


// Cached lgamma function for frequently computed values
inline double keyATMcov::cached_mylgamma(double x)
{
  // For small integer values, use cache
  if (x > 0 && x <= 20 && std::floor(x) == x) {
    int key = static_cast<int>(x);
    auto it = lgamma_cache.find(key);
    if (it != lgamma_cache.end()) {
      return it->second;
    } else {
      double val = mylgamma(x);
      lgamma_cache[key] = val;
      return val;
    }
  }
  return mylgamma(x);
}


double keyATMcov::likelihood_lambda_vectorized(int k, int t)
{
  double loglik = 0.0;
  
  // Compute new lambda_k vector
  VectorXd lambda_k_new = Lambda.row(k).transpose();
  
  // Vectorized computation of new alpha values for topic k
  VectorXd C_times_lambda = C * lambda_k_new;
  
  // Batch compute likelihood contributions
  const double* alpha_ptr = Alpha.col(k).data();
  const double* c_lambda_ptr = C_times_lambda.data();
  const double* weighted_len_ptr = doc_each_len_weighted.data();
  
  // Use SIMD-friendly loop
  for (int d = 0; d < num_doc; ++d) {
    double alpha_k_old = alpha_ptr[d];
    double alpha_k_new_val = std::exp(c_lambda_ptr[d]);
    double alpha_sum_old = doc_alpha_sums[d];
    double alpha_sum_new = alpha_sum_old - alpha_k_old + alpha_k_new_val;
    double weighted_len = weighted_len_ptr[d];
    
    // Likelihood terms using cached lgamma when possible
    loglik += cached_mylgamma(alpha_sum_new) - cached_mylgamma(weighted_len + alpha_sum_new);
    loglik -= cached_mylgamma(alpha_sum_old) - cached_mylgamma(weighted_len + alpha_sum_old);
    
    loglik -= cached_mylgamma(alpha_k_new_val) - cached_mylgamma(alpha_k_old);
    
    double n_dk_val = n_dk(d, k);
    loglik += cached_mylgamma(n_dk_val + alpha_k_new_val) - cached_mylgamma(n_dk_val + alpha_k_old);
  }

  // Prior (vectorized)
  loglik += log_prior_const;
  double lambda_diff = Lambda(k, t) - mu;
  loglik -= lambda_diff * lambda_diff * inv_2sigma_squared;

  return loglik;
}


void keyATMcov::sample_lambda_optimized()
{
  mh_use ? sample_lambda_mh_vectorized() : sample_lambda_slice_optimized();
}


void keyATMcov::sample_lambda_mh_vectorized()
{
  topic_ids = sampler::shuffled_indexes(num_topics);
  cov_ids = sampler::shuffled_indexes(num_cov);
  
  const double mh_sigma = 0.4;
  const double log_mh_sigma = std::log(mh_sigma);
  
  // Pre-allocate proposal matrix for batch processing
  MatrixXd Lambda_backup = Lambda;
  
  for(int kk = 0; kk < num_topics; ++kk) {
    int k = topic_ids[kk];

    for(int tt = 0; tt < num_cov; ++tt) {
      int t = cov_ids[tt];

      double Lambda_current = Lambda(k, t);

      // Current llk - use vectorized computation
      double llk_current = likelihood_lambda_vectorized(k, t);

      // Proposal
      double proposal_value = Lambda_current + R::rnorm(0.0, mh_sigma);
      Lambda(k, t) = proposal_value;
      
      // Update Alpha for this change only - vectorized
      update_alpha_row_vectorized(k);
      
      double llk_proposal = likelihood_lambda_vectorized(k, t);

      double diffllk = llk_proposal - llk_current;
      double r = std::min(0.0, diffllk);
      double u = std::log(unif_rand());

      if (u < r) {
        // accepted - Alpha is already updated
        // Update backup
        Lambda_backup(k, t) = proposal_value;
      } else {
        // Put back original values
        Lambda(k, t) = Lambda_current;
        update_alpha_row_vectorized(k);
      }
    }
  }
}


void keyATMcov::update_alpha_row_vectorized(int k)
{
  // Vectorized update of row k of Alpha after Lambda(k, :) changes
  VectorXd lambda_k = Lambda.row(k).transpose();
  VectorXd C_times_lambda = C * lambda_k;
  
  // Vectorized exponential and update
  for (int d = 0; d < num_doc; ++d) {
    double old_alpha = Alpha(d, k);
    double new_alpha = std::exp(C_times_lambda(d));
    
    doc_alpha_sums[d] += (new_alpha - old_alpha);
    doc_alpha_weighted_sums[d] = doc_each_len_weighted[d] + doc_alpha_sums[d];
    Alpha(d, k) = new_alpha;
  }
}


void keyATMcov::sample_lambda_slice_optimized()
{
  double start = 0.0;
  double end = 0.0;

  double previous_p = 0.0;
  double new_p = 0.0;

  double newlikelihood = 0.0;
  double slice_ = 0.0;
  double current_lambda = 0.0;

  double store_loglik;
  double newlambdallk;

  topic_ids = sampler::shuffled_indexes(num_topics);
  cov_ids = sampler::shuffled_indexes(num_cov);
  int k, t;
  const double A = slice_A;

  // Pre-allocate for batch processing
  std::vector<double> likelihood_cache(max_shrink_time);

  for (int kk = 0; kk < num_topics; ++kk) {
    k = topic_ids[kk];

    for (int tt = 0; tt < num_cov; ++tt) {
      t = cov_ids[tt];
      store_loglik = likelihood_lambda_vectorized(k, t);

      start = val_min; // shrinked value
      end = val_max; // shrinked value

      current_lambda = Lambda(k,t);
      previous_p = shrink(current_lambda, A);
      slice_ = store_loglik - std::log(A * previous_p * (1.0 - previous_p))
              + std::log(unif_rand());

      for (int shrink_time = 0; shrink_time < max_shrink_time; ++shrink_time) {
        new_p = sampler::slice_uniform(start, end);
        Lambda(k,t) = expand(new_p, A);
        
        // Update Alpha for this change - vectorized
        update_alpha_row_vectorized(k);

        newlambdallk = likelihood_lambda_vectorized(k, t);
        newlikelihood = newlambdallk - std::log(A * new_p * (1.0 - new_p));

        if (slice_ < newlikelihood) {
          break;
        } else if (std::abs(end - start) < 1e-9) {
          Rcerr << "Shrinked too much. Using a current value." << std::endl;
          Lambda(k,t) = current_lambda;
          update_alpha_row_vectorized(k);
          break;
        } else if (previous_p < new_p) {
          end = new_p;
        } else if (new_p < previous_p) {
          start = new_p;
        } else {
          Rcpp::stop("Something goes wrong in sample_lambda_slice(). Adjust `A_slice`.");
        }
      }
    }
  }
}


double keyATMcov::loglik_total_optimized()
{
  double loglik = 0.0;
  
  // Vectorized word likelihood computation
  for (int k = 0; k < num_topics; ++k) {
    // Vectorized computation for all vocabulary
    for (int v = 0; v < num_vocab; ++v) {
      double n_s0_kv_val = n_s0_kv(k, v);
      if (n_s0_kv_val > 0) {
        loglik += cached_mylgamma(beta + n_s0_kv_val) - cached_mylgamma(beta);
      }
    }

    // word normalization
    loglik += cached_mylgamma(beta * (double)num_vocab) - cached_mylgamma(beta * (double)num_vocab + n_s0_k(k));

    if (k < keyword_k) {
      // For keyword topics - sparse matrix iteration
      for (SparseMatrix<double,RowMajor>::InnerIterator it(n_s1_kv, k); it; ++it) {
        loglik += cached_mylgamma(beta_s + it.value()) - cached_mylgamma(beta_s);
      }
      loglik += cached_mylgamma(beta_s * (double)keywords_num[k]) - cached_mylgamma(beta_s * (double)keywords_num[k] + n_s1_k(k));

      // Normalization
      loglik += cached_mylgamma(prior_gamma(k, 0) + prior_gamma(k, 1)) - cached_mylgamma(prior_gamma(k, 0)) - cached_mylgamma(prior_gamma(k, 1));

      // s
      loglik += cached_mylgamma(n_s0_k(k) + prior_gamma(k, 1))
                - cached_mylgamma(n_s1_k(k) + prior_gamma(k, 0) + n_s0_k(k) + prior_gamma(k, 1))
                + cached_mylgamma(n_s1_k(k) + prior_gamma(k, 0));
    }
  }

  // z - use pre-computed values (vectorized)
  const double* alpha_sums_ptr = doc_alpha_sums.data();
  const double* alpha_weighted_sums_ptr = doc_alpha_weighted_sums.data();
  
  for (int d = 0; d < num_doc; ++d) {
    loglik += cached_mylgamma(alpha_sums_ptr[d]) - cached_mylgamma(alpha_weighted_sums_ptr[d]);
    
    // Vectorized computation for all topics in document d
    for (int k = 0; k < num_topics; ++k) {
      double n_dk_val = n_dk(d, k);
      double alpha_val = Alpha(d, k);
      if (n_dk_val > 0 || alpha_val > 0) {
        loglik += cached_mylgamma(n_dk_val + alpha_val) - cached_mylgamma(alpha_val);
      }
    }
  }

  // Lambda loglik - vectorized computation
  double lambda_loglik = 0.0;
  for (int k = 0; k < num_topics; ++k) {
    for (int t = 0; t < num_cov; ++t) {
      double lambda_diff = Lambda(k,t) - mu;
      lambda_loglik += log_prior_const - lambda_diff * lambda_diff * inv_2sigma_squared;
    }
  }
  loglik += lambda_loglik;

  return loglik;
}


double keyATMcov::loglik_total()
{
  return loglik_total_optimized();
}


// Legacy function wrappers for compatibility
double keyATMcov::likelihood_lambda(int k, int t)
{
  return likelihood_lambda_vectorized(k, t);
}

void keyATMcov::sample_lambda()
{
  sample_lambda_optimized();
}

void keyATMcov::sample_lambda_mh()
{
  sample_lambda_mh_vectorized();
}

void keyATMcov::update_alpha_efficient()
{
  update_alpha_vectorized();
}

void keyATMcov::update_alpha_row_efficient(int k)
{
  update_alpha_row_vectorized(k);
}

double keyATMcov::likelihood_lambda_efficient(int k, int t)
{
  return likelihood_lambda_vectorized(k, t);
}

void keyATMcov::sample_lambda_mh_efficient()
{
  sample_lambda_mh_vectorized();
}