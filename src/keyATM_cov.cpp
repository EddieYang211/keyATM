#include "keyATM_cov.h"
#ifdef _OPENMP
#include <omp.h>
#endif

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
  
  // Pre-allocate thread-local storage for OpenMP
  #ifdef _OPENMP
  int max_threads = omp_get_max_threads();
  thread_alpha_cache.resize(max_threads);
  thread_prob_cache.resize(max_threads);
  for (int t = 0; t < max_threads; ++t) {
    thread_alpha_cache[t].resize(num_topics);
    thread_prob_cache[t].resize(num_topics);
  }
  #endif
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
  
  // Initialize with better random values
  #ifdef _OPENMP
  #pragma omp parallel for collapse(2) schedule(static)
  #endif
  for (int k = 0; k < num_topics; ++k) {
    for (int i = 0; i < num_cov; ++i) {
      Lambda(k, i) = R::rnorm(0.0, 0.3);
    }
  }
  
  // Pre-compute constants
  sigma_squared = sigma * sigma;
  inv_2sigma_squared = 1.0 / (2.0 * sigma_squared);
  log_prior_const = -0.5 * log(2.0 * PI_V * sigma_squared);
  
  // Pre-allocate matrices for batch operations
  exp_buffer = MatrixXd::Zero(num_doc, num_topics);
  likelihood_cache.resize(num_topics * num_cov);
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
  
  // Pre-allocate matrices for batch operations
  exp_buffer = MatrixXd::Zero(num_doc, num_topics);
  likelihood_cache.resize(num_topics * num_cov);
}

void keyATMcov::iteration_single(int it)
{ 
  doc_indexes = sampler::shuffled_indexes(num_doc); // shuffle

  // Create Alpha for this iteration - vectorized computation
  update_alpha_batch();

  // Parallel document processing
  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 1)
  #endif
  for (int ii = 0; ii < num_doc; ++ii) {
    process_document_parallel(ii);
  }
  
  sample_parameters(it);
}

void keyATMcov::process_document_parallel(int ii)
{
  int doc_id_ = doc_indexes[ii];
  int doc_length = doc_each_len[doc_id_];
  
  // Get thread-local storage
  #ifdef _OPENMP
  int thread_id = omp_get_thread_num();
  VectorXd& local_alpha = thread_alpha_cache[thread_id];
  VectorXd& local_prob = thread_prob_cache[thread_id];
  #else
  VectorXd local_alpha = VectorXd::Zero(num_topics);
  VectorXd local_prob = VectorXd::Zero(num_topics);
  #endif
  
  // Copy alpha for this document
  local_alpha = Alpha.row(doc_id_).transpose();
  
  IntegerVector doc_s = S[doc_id_];
  IntegerVector doc_z = Z[doc_id_];
  IntegerVector doc_w = W[doc_id_];
  
  std::vector<int> token_indexes = sampler::shuffled_indexes(doc_length);

  // Iterate each word in the document
  for (int jj = 0; jj < doc_length; ++jj) {
    int w_position = token_indexes[jj];
    int s_ = doc_s[w_position];
    int z_ = doc_z[w_position]; 
    int w_ = doc_w[w_position];

    int new_z = sample_z_optimized(local_alpha, local_prob, z_, s_, w_, doc_id_);
    doc_z[w_position] = new_z;

    if (keywords[new_z].find(w_) == keywords[new_z].end())
      continue;

    z_ = doc_z[w_position]; // use updated z
    int new_s = sample_s(z_, s_, w_, doc_id_);
    doc_s[w_position] = new_s;
  }

  // Update the global arrays (need synchronization)
  #ifdef _OPENMP
  #pragma omp critical
  #endif
  {
    Z[doc_id_] = doc_z;
    S[doc_id_] = doc_s;
  }
}

int keyATMcov::sample_z_optimized(VectorXd& alpha, VectorXd& prob_vec, 
                                  int z, int s, int w, int doc_id)
{
  double numerator, denominator;
  double sum = 0.0;

  // Remove data (thread-safe operations on local copies)
  // Note: This requires careful handling of shared data structures
  // For now, maintaining original logic but with optimized probability computation
  
  if (s == 0) {
    #ifdef _OPENMP
    #pragma omp atomic
    #endif
    n_s0_kv(z, w) -= vocab_weights(w);
    
    #ifdef _OPENMP
    #pragma omp atomic
    #endif
    n_s0_k(z) -= vocab_weights(w);
  } else if (s == 1) {
    #ifdef _OPENMP
    #pragma omp critical
    #endif
    {
      n_s1_kv.coeffRef(z, w) -= vocab_weights(w);
      n_s1_k(z) -= vocab_weights(w);
    }
  }

  #ifdef _OPENMP
  #pragma omp atomic
  #endif
  n_dk(doc_id, z) -= vocab_weights(w);
  
  #ifdef _OPENMP
  #pragma omp atomic
  #endif
  n_dk_noWeight(doc_id, z) -= 1.0;

  // Compute probabilities - vectorized where possible
  if (s == 0) {
    for (int k = 0; k < num_topics; ++k) {
      numerator = (beta + n_s0_kv(k, w)) *
        (n_s0_k(k) + prior_gamma(k, 1)) *
        (n_dk(doc_id, k) + alpha(k));

      denominator = (Vbeta + n_s0_k(k)) *
        (n_s1_k(k) + prior_gamma(k, 0) + n_s0_k(k) + prior_gamma(k, 1));

      prob_vec(k) = numerator / denominator;
      sum += prob_vec(k);
    }
  } else {
    for (int k = 0; k < num_topics; ++k) {
      if (keywords[k].find(w) == keywords[k].end()) {
        prob_vec(k) = 0.0;
        continue;
      } else {
        numerator = (beta_s + n_s1_kv.coeffRef(k, w)) *
          (n_s1_k(k) + prior_gamma(k, 0)) *
          (n_dk(doc_id, k) + alpha(k));
        denominator = (Lbeta_sk(k) + n_s1_k(k) ) *
          (n_s1_k(k) + prior_gamma(k, 0) + n_s0_k(k) + prior_gamma(k, 1));

        prob_vec(k) = numerator / denominator;
        sum += prob_vec(k);
      }
    }
  }

  int new_z = sampler::rcat_without_normalize(prob_vec, sum, num_topics);

  // Add back data counts
  if (s == 0) {
    #ifdef _OPENMP
    #pragma omp atomic
    #endif
    n_s0_kv(new_z, w) += vocab_weights(w);
    
    #ifdef _OPENMP
    #pragma omp atomic
    #endif
    n_s0_k(new_z) += vocab_weights(w);
  } else if (s == 1) {
    #ifdef _OPENMP
    #pragma omp critical
    #endif
    {
      n_s1_kv.coeffRef(new_z, w) += vocab_weights(w);
      n_s1_k(new_z) += vocab_weights(w);
    }
  }
  
  #ifdef _OPENMP
  #pragma omp atomic
  #endif
  n_dk(doc_id, new_z) += vocab_weights(w);
  
  #ifdef _OPENMP
  #pragma omp atomic
  #endif
  n_dk_noWeight(doc_id, new_z) += 1.0;

  return new_z;
}

void keyATMcov::update_alpha_batch()
{
  // Batch computation: Alpha = exp(C * Lambda^T)
  // Use Eigen's optimized matrix multiplication
  exp_buffer.noalias() = C * Lambda.transpose();
  
  // Vectorized exponential
  #ifdef _OPENMP
  #pragma omp parallel for collapse(2) schedule(static)
  #endif
  for (int d = 0; d < num_doc; ++d) {
    for (int k = 0; k < num_topics; ++k) {
      Alpha(d, k) = std::exp(exp_buffer(d, k));
    }
  }
  
  // Pre-compute sums for likelihood computation
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int d = 0; d < num_doc; ++d) {
    doc_alpha_sums[d] = Alpha.row(d).sum();
    doc_alpha_weighted_sums[d] = doc_each_len_weighted[d] + doc_alpha_sums[d];
  }
}

void keyATMcov::sample_parameters(int it)
{
  sample_lambda_parallel();

  // Store lambda
  int r_index = it + 1;
  if (r_index % thinning == 0 || r_index == 1 || r_index == iter) {
    Rcpp::NumericMatrix Lambda_R = Rcpp::wrap(Lambda);
    List Lambda_iter = stored_values["Lambda_iter"];
    Lambda_iter.push_back(Lambda_R);
    stored_values["Lambda_iter"] = Lambda_iter;
  }
}

void keyATMcov::sample_lambda_parallel()
{
  if (mh_use) {
    sample_lambda_mh_parallel();
  } else {
    sample_lambda_slice_parallel();
  }
}

void keyATMcov::sample_lambda_mh_parallel()
{
  std::vector<int> topic_ids = sampler::shuffled_indexes(num_topics);
  std::vector<int> cov_ids = sampler::shuffled_indexes(num_cov);
  double mh_sigma = 0.4;

  // Pre-compute all likelihood values in parallel
  #ifdef _OPENMP
  #pragma omp parallel for collapse(2) schedule(static)
  #endif
  for (int kk = 0; kk < num_topics; ++kk) {
    for (int tt = 0; tt < num_cov; ++tt) {
      int k = topic_ids[kk];
      int t = cov_ids[tt];
      int idx = k * num_cov + t;
      likelihood_cache[idx] = likelihood_lambda_vectorized(k, t);
    }
  }

  // Sequential update (MH requires sequential updates)
  for (int kk = 0; kk < num_topics; ++kk) {
    int k = topic_ids[kk];
    
    for (int tt = 0; tt < num_cov; ++tt) {
      int t = cov_ids[tt];
      int idx = k * num_cov + t;
      
      double Lambda_current = Lambda(k, t);
      double llk_current = likelihood_cache[idx];

      // Proposal
      double proposal_value = Lambda_current + R::rnorm(0.0, mh_sigma);
      Lambda(k, t) = proposal_value;
      
      // Update Alpha for this change only
      update_alpha_row_vectorized(k);
      
      double llk_proposal = likelihood_lambda_vectorized(k, t);
      double diffllk = llk_proposal - llk_current;
      double r = std::min(0.0, diffllk);
      double u = log(unif_rand());

      if (u < r) {
        // accepted - Alpha is already updated
        likelihood_cache[idx] = llk_proposal;
      } else {
        // Put back original values
        Lambda(k, t) = Lambda_current;
        update_alpha_row_vectorized(k);
      }
    }
  }
}

void keyATMcov::sample_lambda_slice_parallel()
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

  std::vector<int> topic_ids = sampler::shuffled_indexes(num_topics);
  std::vector<int> cov_ids = sampler::shuffled_indexes(num_cov);
  const double A = slice_A;

  for (int kk = 0; kk < num_topics; ++kk) {
    int k = topic_ids[kk];

    for (int tt = 0; tt < num_cov; ++tt) {
      int t = cov_ids[tt];
      store_loglik = likelihood_lambda_vectorized(k, t);

      start = val_min;
      end = val_max;

      current_lambda = Lambda(k,t);
      previous_p = shrink(current_lambda, A);
      slice_ = store_loglik - std::log(A * previous_p * (1.0 - previous_p))
              + log(unif_rand());

      for (int shrink_time = 0; shrink_time < max_shrink_time; ++shrink_time) {
        new_p = sampler::slice_uniform(start, end);
        Lambda(k,t) = expand(new_p, A);
        
        // Update Alpha for this change
        update_alpha_row_vectorized(k);

        newlambdallk = likelihood_lambda_vectorized(k, t);
        newlikelihood = newlambdallk - std::log(A * new_p * (1.0 - new_p));

        if (slice_ < newlikelihood) {
          break;
        } else if (abs(end - start) < 1e-9) {
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

double keyATMcov::likelihood_lambda_vectorized(int k, int t)
{
  double loglik = 0.0;
  
  // Get current lambda for topic k
  VectorXd lambda_k = Lambda.row(k).transpose();
  
  // Compute new alpha values for topic k across all documents
  VectorXd alpha_k_new = (C * lambda_k).array().exp();
  
  // Vectorized likelihood computation
  #ifdef _OPENMP
  #pragma omp parallel for reduction(+:loglik) schedule(static)
  #endif
  for (int d = 0; d < num_doc; ++d) {
    double alpha_sum_old = doc_alpha_sums[d];
    double alpha_k_old = Alpha(d, k);
    double alpha_k_new_val = alpha_k_new(d);
    double alpha_sum_new = alpha_sum_old - alpha_k_old + alpha_k_new_val;
    
    // Likelihood terms
    double doc_contrib = mylgamma(alpha_sum_new) - mylgamma(doc_each_len_weighted[d] + alpha_sum_new);
    doc_contrib -= mylgamma(alpha_sum_old) - mylgamma(doc_each_len_weighted[d] + alpha_sum_old);
    doc_contrib -= mylgamma(alpha_k_new_val) - mylgamma(alpha_k_old);
    doc_contrib += mylgamma(n_dk(d, k) + alpha_k_new_val) - mylgamma(n_dk(d, k) + alpha_k_old);
    
    loglik += doc_contrib;
  }

  // Prior
  loglik += log_prior_const;
  double lambda_diff = Lambda(k, t) - mu;
  loglik -= lambda_diff * lambda_diff * inv_2sigma_squared;

  return loglik;
}

void keyATMcov::update_alpha_row_vectorized(int k)
{
  // Update only row k of Alpha after Lambda(k, :) changes
  VectorXd lambda_k = Lambda.row(k).transpose();
  VectorXd C_lambda_k = C * lambda_k;
  
  // Vectorized exponential and update
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int d = 0; d < num_doc; ++d) {
    double old_alpha_k = Alpha(d, k);
    double new_alpha_k = std::exp(C_lambda_k(d));
    
    doc_alpha_sums[d] = doc_alpha_sums[d] - old_alpha_k + new_alpha_k;
    doc_alpha_weighted_sums[d] = doc_each_len_weighted[d] + doc_alpha_sums[d];
    Alpha(d, k) = new_alpha_k;
  }
}

double keyATMcov::loglik_total()
{
  double loglik = 0.0;
  
  // Word-topic distributions - parallelizable
  #ifdef _OPENMP
  #pragma omp parallel for reduction(+:loglik) schedule(static)
  #endif
  for (int k = 0; k < num_topics; ++k) {
    double k_contrib = 0.0;
    
    for (int v = 0; v < num_vocab; ++v) {
      k_contrib += mylgamma(beta + n_s0_kv(k, v)) - mylgamma(beta);
    }

    // word normalization
    k_contrib += mylgamma( beta * (double)num_vocab ) - mylgamma(beta * (double)num_vocab + n_s0_k(k) );

    if (k < keyword_k) {
      // For keyword topics
      for (SparseMatrix<double,RowMajor>::InnerIterator it(n_s1_kv, k); it; ++it) {
        k_contrib += mylgamma(beta_s + it.value()) - mylgamma(beta_s);
      }
      k_contrib += mylgamma( beta_s * (double)keywords_num[k] ) - mylgamma(beta_s * (double)keywords_num[k] + n_s1_k(k) );

      // Normalization
      k_contrib += mylgamma( prior_gamma(k, 0) + prior_gamma(k, 1)) - mylgamma( prior_gamma(k, 0)) - mylgamma( prior_gamma(k, 1));

      // s
      k_contrib += mylgamma( n_s0_k(k) + prior_gamma(k, 1) )
                - mylgamma(n_s1_k(k) + prior_gamma(k, 0) + n_s0_k(k) + prior_gamma(k, 1))
                + mylgamma( n_s1_k(k) + prior_gamma(k, 0) );
    }
    
    loglik += k_contrib;
  }

  // Document-topic distributions - use pre-computed values
  #ifdef _OPENMP
  #pragma omp parallel for reduction(+:loglik) schedule(static)
  #endif
  for (int d = 0; d < num_doc; ++d) {
    double d_contrib = mylgamma(doc_alpha_sums[d]) - mylgamma(doc_alpha_weighted_sums[d]);
    for (int k = 0; k < num_topics; ++k) {
      d_contrib += mylgamma(n_dk(d,k) + Alpha(d, k)) - mylgamma(Alpha(d, k));
    }
    loglik += d_contrib;
  }

  // Lambda loglik - vectorized computation
  double lambda_contrib = 0.0;
  #ifdef _OPENMP
  #pragma omp parallel for reduction(+:lambda_contrib) collapse(2) schedule(static)
  #endif
  for (int k = 0; k < num_topics; ++k) {
    for (int t = 0; t < num_cov; ++t) {
      lambda_contrib += log_prior_const;
      double lambda_diff = Lambda(k,t) - mu;
      lambda_contrib -= lambda_diff * lambda_diff * inv_2sigma_squared;
    }
  }
  loglik += lambda_contrib;

  return loglik;
}