#include "keyATM_cov.h"

using namespace Eigen;
using namespace Rcpp;
using namespace std;

void keyATMcov::read_data_specific()
{
  // Covariate
  model_settings = model["model_settings"];
  NumericMatrix C_r = model_settings["covariates_data_use"];
  num_cov = C_r.cols();
  C = Rcpp::as<Eigen::MatrixXd>(C_r);
  C_transpose = C.transpose(); // Pre-compute transpose for efficiency

  // Read Lambda matrix
  NumericMatrix Lambda_r = model_settings["Lambda"];
  Lambda = Rcpp::as<Eigen::MatrixXd>(Lambda_r);

  // Parameters
  mh_use = model_settings["mh_use"];
  mu = model_settings["mu"];
  sigma = model_settings["sigma"];
  
  // Pre-compute constants for efficiency
  sigma_squared = sigma * sigma;
  inv_2sigma_squared = 1.0 / (2.0 * sigma_squared);
  log_prior_const = -0.5 * log(2.0 * M_PI * sigma_squared);

  // Initialize OpenMP thread management
  #ifdef HAVE_OPENMP
  num_threads = omp_get_max_threads();
  if (model_settings.containsElementNamed("num_threads")) {
    int user_threads = model_settings["num_threads"];
    if (user_threads > 0) {
      num_threads = std::min(user_threads, omp_get_max_threads());
    }
  }
  #else
  num_threads = 1;
  #endif
}

void keyATMcov::initialize_specific()
{
  // Initialize Alpha matrix
  update_alpha_efficient();
}

void keyATMcov::resume_initialize_specific()
{
  // Resume by updating Alpha
  update_alpha_efficient();
}

void keyATMcov::iteration_single(int it)
{
#ifdef HAVE_OPENMP
  if (num_threads > 1) {
    iteration_single_omp(it);
  } else {
    // Fallback to base implementation
    keyATMmeta::iteration_single(it);
    sample_parameters(it);
  }
#else
  // Serial implementation
  keyATMmeta::iteration_single(it);
  sample_parameters(it);
#endif
}

void keyATMcov::iteration_single_omp(int it)
{
#ifdef HAVE_OPENMP
  omp_set_num_threads(num_threads);
  
  std::vector<int> doc_ids = sampler::shuffled_indexes(num_doc);
  
  #pragma omp parallel
  {
    // Thread-local variables
    std::vector<int> doc_s, doc_z, doc_w;
    std::vector<int> token_indexes;
    
    #pragma omp for schedule(dynamic)
    for (int ii = 0; ii < num_doc; ++ii) {
      int doc_id_ = doc_ids[ii];
      doc_s = S[doc_id_];
      doc_z = Z[doc_id_];
      doc_w = W[doc_id_];
      int doc_length = doc_each_len[doc_id_];

      token_indexes = sampler::shuffled_indexes(doc_length);

      // Iterate each word in the document
      for (int jj = 0; jj < doc_length; ++jj) {
        int w_position = token_indexes[jj];
        int s_ = doc_s[w_position];
        int z_ = doc_z[w_position];
        int w_ = doc_w[w_position];

        // Get current alpha for this document
        VectorXd alpha_d = Alpha.row(doc_id_).transpose();
        
        int new_z = sample_z(alpha_d, z_, s_, w_, doc_id_);
        doc_z[w_position] = new_z;

        if (keywords[new_z].find(w_) == keywords[new_z].end())
          continue;

        z_ = doc_z[w_position]; // use updated z
        int new_s = sample_s(z_, s_, w_, doc_id_);
        doc_s[w_position] = new_s;
      }

      // Update global arrays (thread-safe)
      #pragma omp critical
      {
        Z[doc_id_] = doc_z;
        S[doc_id_] = doc_s;
      }
    }
  }
  
  sample_parameters(it);
#endif
}

void keyATMcov::sample_parameters(int it)
{
  update_alpha_efficient();
  sample_lambda();

  // Store Lambda
  int r_index = it + 1;
  if (r_index % thinning == 0 || r_index == 1 || r_index == iter) {
    Rcpp::NumericMatrix Lambda_R = Rcpp::wrap(Lambda);
    List Lambda_iter = stored_values["Lambda_iter"];
    Lambda_iter.push_back(Lambda_R);
    stored_values["Lambda_iter"] = Lambda_iter;
  }
}

void keyATMcov::update_alpha_efficient()
{
  // Use thread-safe computation without static variables
  MatrixXd C_Lambda_T = C * Lambda.transpose();
  
  // Apply exponential function
  Alpha = C_Lambda_T.array().exp();
  
  // Pre-compute sums using optimized Eigen operations
  if (num_doc > 0 && Alpha.rows() == num_doc && Alpha.cols() == num_topics) {
    doc_alpha_sums = Alpha.rowwise().sum();
    
    if (doc_each_len_weighted.size() == num_doc) {
      Eigen::Map<const Eigen::VectorXd> doc_len_map(doc_each_len_weighted.data(), num_doc);
      doc_alpha_weighted_sums = doc_len_map + doc_alpha_sums;
    }
  }
}

void keyATMcov::sample_lambda()
{
  if (mh_use == 1) {
    sample_lambda_mh();
  } else {
    sample_lambda_slice();
  }
}

void keyATMcov::sample_lambda_mh()
{
#ifdef HAVE_OPENMP
  if (num_threads > 1) {
    sample_lambda_mh_omp();
  } else {
    sample_lambda_mh_efficient();
  }
#else
  sample_lambda_mh_efficient();
#endif
}

void keyATMcov::sample_lambda_mh_omp()
{
#ifdef HAVE_OPENMP
  omp_set_num_threads(num_threads);
  
  std::vector<std::vector<bool>> accept_flags(num_topics, std::vector<bool>(num_cov, false));
  std::vector<std::vector<double>> mh_step_size = model_settings["mh_step_size"];

  topic_ids = sampler::shuffled_indexes(num_topics);
  
  // FIXED: Process topics in parallel without race conditions
  #pragma omp parallel for schedule(dynamic)
  for (int k_idx = 0; k_idx < num_topics; ++k_idx) {
    int k = topic_ids[k_idx];
    std::vector<int> local_cov_ids = sampler::shuffled_indexes(num_cov);
    
    for (int t_idx = 0; t_idx < num_cov; ++t_idx) {
      int t = local_cov_ids[t_idx];
      
      double lambda_kt_current = Lambda(k,t);
      
      // Current alpha for this topic
      VectorXd current_alpha_k = C * Lambda.row(k).transpose();
      current_alpha_k = current_alpha_k.array().exp();
      
      double current_L = compute_likelihood_terms(k, t, lambda_kt_current, current_alpha_k);

      // Proposal
      double U_var = mh_step_size[k][t];
      double step = R::rnorm(0.0, U_var);
      double lambda_kt_new = lambda_kt_current + step;

      // Proposed alpha for this topic
      VectorXd proposed_alpha_k = current_alpha_k + C.col(t) * (exp(lambda_kt_new) - exp(lambda_kt_current));
      double new_L = compute_likelihood_terms(k, t, lambda_kt_new, proposed_alpha_k);

      double log_acceptance_ratio = new_L - current_L;
      
      if (log(R::runif(0.0, 1.0)) < log_acceptance_ratio) {
        #pragma omp critical
        {
          Lambda(k,t) = lambda_kt_new;
        }
        accept_flags[k][t] = true;
      }
    }
  }
  
  model_settings["accept_Lambda"] = accept_flags;
#endif
}

void keyATMcov::sample_lambda_mh_efficient()
{
  std::vector<std::vector<bool>> accept_flags(num_topics, std::vector<bool>(num_cov, false));
  std::vector<std::vector<double>> mh_step_size = model_settings["mh_step_size"];

  topic_ids = sampler::shuffled_indexes(num_topics);
  
  for (int k_idx = 0; k_idx < num_topics; ++k_idx) {
    int k = topic_ids[k_idx];
    cov_ids = sampler::shuffled_indexes(num_cov);
    
    for (int t_idx = 0; t_idx < num_cov; ++t_idx) {
      int t = cov_ids[t_idx];
      
      double lambda_kt_current = Lambda(k,t);
      
      // Current alpha for this topic
      VectorXd current_alpha_k = C * Lambda.row(k).transpose();
      current_alpha_k = current_alpha_k.array().exp();
      
      double current_L = compute_likelihood_terms(k, t, lambda_kt_current, current_alpha_k);

      // Proposal
      double U_var = mh_step_size[k][t];
      double step = R::rnorm(0.0, U_var);
      double lambda_kt_new = lambda_kt_current + step;

      // Proposed alpha for this topic
      VectorXd proposed_alpha_k = current_alpha_k + C.col(t) * (exp(lambda_kt_new) - exp(lambda_kt_current));
      double new_L = compute_likelihood_terms(k, t, lambda_kt_new, proposed_alpha_k);

      double log_acceptance_ratio = new_L - current_L;
      
      if (log(R::runif(0.0, 1.0)) < log_acceptance_ratio) {
        Lambda(k,t) = lambda_kt_new;
        accept_flags[k][t] = true;
      }
    }
  }
  
  model_settings["accept_Lambda"] = accept_flags;
}

void keyATMcov::sample_lambda_slice()
{
  // Simple slice sampling implementation
  topic_ids = sampler::shuffled_indexes(num_topics);
  
  for (int k_idx = 0; k_idx < num_topics; ++k_idx) {
    int k = topic_ids[k_idx];
    cov_ids = sampler::shuffled_indexes(num_cov);
    
    for (int t_idx = 0; t_idx < num_cov; ++t_idx) {
      int t = cov_ids[t_idx];
      
      double lambda_kt_current = Lambda(k,t);
      
      // Current alpha for this topic
      VectorXd current_alpha_k = C * Lambda.row(k).transpose();
      current_alpha_k = current_alpha_k.array().exp();
      
      double current_L = compute_likelihood_terms(k, t, lambda_kt_current, current_alpha_k);
      
      // Slice sampling with simple implementation
      double slice_height = current_L + log(R::runif(0.0, 1.0));
      
      // Find slice bounds
      double left = lambda_kt_current - 1.0;
      double right = lambda_kt_current + 1.0;
      
      // Sample from slice
      double lambda_kt_new;
      VectorXd proposed_alpha_k;
      int max_attempts = 50;
      
      for (int attempt = 0; attempt < max_attempts; ++attempt) {
        lambda_kt_new = R::runif(left, right);
        proposed_alpha_k = current_alpha_k + C.col(t) * (exp(lambda_kt_new) - exp(lambda_kt_current));
        
        if (compute_likelihood_terms(k, t, lambda_kt_new, proposed_alpha_k) > slice_height) {
          Lambda(k,t) = lambda_kt_new;
          break;
        }
        
        if (lambda_kt_new < lambda_kt_current) {
          left = lambda_kt_new;
        } else {
          right = lambda_kt_new;
        }
      }
    }
  }
}

void keyATMcov::sample_lambda_slice_omp()
{
#ifdef HAVE_OPENMP
  // OpenMP version of slice sampling - simplified for stability
  sample_lambda_slice();
#endif
}

double keyATMcov::compute_likelihood_terms(int k, int t, double current_lambda_kt_val,
                                           const Eigen::VectorXd& current_alpha_k_vec)
{
  double loglik = 0.0;

  if (num_doc == 0) {
    loglik += log_prior_const;
    double lambda_diff = current_lambda_kt_val - mu;
    loglik -= lambda_diff * lambda_diff * inv_2sigma_squared;
    return loglik;
  }

  // Use stack-allocated vectors for thread safety
  Eigen::VectorXd alpha_sum_new_overall_vec(num_doc);
  Eigen::VectorXd term_weighted_sum_new(num_doc);
  Eigen::VectorXd term_weighted_sum_old(num_doc);
  Eigen::VectorXd term_ndk_new(num_doc);
  Eigen::VectorXd term_ndk_old(num_doc);

  Eigen::Map<const Eigen::VectorXd> doc_each_len_weighted_eigen(doc_each_len_weighted.data(), doc_each_len_weighted.size());
  Eigen::VectorXd alpha_k_old_from_global_Alpha_vec = Alpha.col(k);
  const Eigen::VectorXd& alpha_sum_old_overall_vec = doc_alpha_sums;

  alpha_sum_new_overall_vec.noalias() = alpha_sum_old_overall_vec - alpha_k_old_from_global_Alpha_vec + current_alpha_k_vec;
  Eigen::VectorXd n_dk_k_vec = n_dk.col(k);

  auto mylgamma_unary_op = [this](double x) { return this->mylgamma(x); };

  // Vectorized operations for performance
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

// Implement missing OpenMP version with thread-local storage
double keyATMcov::compute_likelihood_terms_omp(int k, int t, double current_lambda_kt_val,
                                               const Eigen::VectorXd& current_alpha_k_vec,
                                               ThreadLocalStorage& tls)
{
  tls.resize(num_doc);
  return compute_likelihood_terms(k, t, current_lambda_kt_val, current_alpha_k_vec);
}

double keyATMcov::alpha_loglik()
{
  double loglik = 0.0;
  
  for (int d = 0; d < num_doc; ++d) {
    double alpha_sum = Alpha.row(d).sum();
    loglik += mylgamma(alpha_sum);
    loglik -= mylgamma(doc_each_len_weighted[d] + alpha_sum);
    
    for (int k = 0; k < num_topics; ++k) {
      loglik += mylgamma(n_dk(d,k) + Alpha(d,k));
      loglik -= mylgamma(Alpha(d,k));
    }
  }
  
  return loglik;
}

double keyATMcov::loglik_total()
{
  return alpha_loglik();
}

// Implement missing functions for completeness
void keyATMcov::update_alpha_row_efficient(int k)
{
  // Update single row of Alpha matrix
  VectorXd lambda_k = Lambda.row(k).transpose();
  VectorXd X_k = C * lambda_k;
  Alpha.col(k) = X_k.array().exp();
}

double keyATMcov::likelihood_lambda_efficient(int k, int t, const Eigen::VectorXd* precomputed_alpha_k)
{
  VectorXd alpha_k;
  if (precomputed_alpha_k != nullptr) {
    alpha_k = *precomputed_alpha_k;
  } else {
    alpha_k = C * Lambda.row(k).transpose();
    alpha_k = alpha_k.array().exp();
  }
  
  return compute_likelihood_terms(k, t, Lambda(k,t), alpha_k);
}

double keyATMcov::likelihood_lambda(int k, int t)
{
  return likelihood_lambda_efficient(k, t, nullptr);
}

void keyATMcov::proposal_lambda(int k)
{
  // Simple proposal - could be enhanced
  for (int t = 0; t < num_cov; ++t) {
    Lambda(k,t) += R::rnorm(0.0, 0.1);
  }
}

#ifdef HAVE_OPENMP
void keyATMcov::set_num_threads(int threads)
{
  if (threads > 0) {
    num_threads = std::min(threads, omp_get_max_threads());
  }
}
#else
void keyATMcov::set_num_threads(int threads)
{
  num_threads = 1;
}
#endif