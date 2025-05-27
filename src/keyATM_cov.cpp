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
}


// Helper function to compute log-likelihood terms relevant to Lambda(k,t)
double keyATMcov::compute_likelihood_terms(int k, int t, double current_lambda_kt_val,
                                           const Eigen::VectorXd& current_alpha_k_vec)
{
  double loglik = 0.0;

  for (int d = 0; d < num_doc; ++d) {
    // Alpha(d, k) and doc_alpha_sums[d] are from the global state, reflecting Alpha values
    // *before* the current specific Lambda(k,t) proposal is evaluated.
    double alpha_k_old_from_global_Alpha = Alpha(d, k); 
    double current_eval_alpha_dk = current_alpha_k_vec(d); // Alpha(d,k) if current_lambda_kt_val is used

    double alpha_sum_old_overall = doc_alpha_sums[d]; // Based on global Alpha
    // Calculate new sum_alpha_d if Alpha(d,k) changes from global Alpha(d,k) to current_eval_alpha_dk
    double alpha_sum_new_overall = alpha_sum_old_overall - alpha_k_old_from_global_Alpha + current_eval_alpha_dk;
    
    double weighted_len = doc_each_len_weighted[d];
    
    loglik += mylgamma(alpha_sum_new_overall) - mylgamma(alpha_sum_old_overall);
    loglik -= mylgamma(weighted_len + alpha_sum_new_overall) - mylgamma(weighted_len + alpha_sum_old_overall);
    loglik -= mylgamma(current_eval_alpha_dk) - mylgamma(alpha_k_old_from_global_Alpha);
    loglik += mylgamma(n_dk(d, k) + current_eval_alpha_dk) - mylgamma(n_dk(d, k) + alpha_k_old_from_global_Alpha);
  }

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

  // Create Alpha for this iteration - vectorized computation
  update_alpha_efficient();

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


void keyATMcov::update_alpha_efficient()
{
  // Vectorized computation: Alpha = exp(C * Lambda^T)
  Alpha = (C * Lambda.transpose()).array().exp();
  
  // Pre-compute sums for likelihood computation
  if (num_doc > 0 && Alpha.rows() == num_doc && Alpha.cols() == num_topics) { // Check dimensions
    doc_alpha_sums = Alpha.rowwise().sum(); // Should now work with Eigen::VectorXd
    if (doc_each_len_weighted.size() == num_doc && doc_alpha_sums.size() == num_doc) {
        // Assuming doc_each_len_weighted is std::vector<double>
        Eigen::VectorXd doc_each_len_weighted_eigen = Eigen::Map<const Eigen::VectorXd>(doc_each_len_weighted.data(), doc_each_len_weighted.size());
        doc_alpha_weighted_sums = doc_each_len_weighted_eigen.array() + doc_alpha_sums.array(); // Should now work
    } else {
        // Handle size mismatch, perhaps by re-initializing or erroring
        if (doc_each_len_weighted.size() != num_doc) {
             Rcpp::Rcerr << "Dimension mismatch: doc_each_len_weighted.size() (" << doc_each_len_weighted.size() 
                         << ") != num_doc (" << num_doc << ")" << std::endl;
        }
        if (doc_alpha_sums.size() != num_doc) {
            Rcpp::Rcerr << "Dimension mismatch: doc_alpha_sums.size() (" << doc_alpha_sums.size() 
                        << ") != num_doc (" << num_doc << ")" << std::endl;
        }
        // Fallback or error, ensure vectors are correctly sized if proceeding
        // For safety, one might re-initialize or use the loop if checks fail,
        // or throw an error. If doc_each_len_weighted is std::vector, conversion is needed.
        // For now, assuming it's compatible for direct Eigen operations.
    }
  } else { // Fallback to loop if num_doc is 0 or Alpha dimensions are unexpected
      // If using Eigen::VectorXd, direct assignment from sum is better than loop
      // This manual loop would be inefficient if sizes are large and types are Eigen.
      // However, if the intention is a safe fallback for small/problematic cases:
      if (num_doc > 0) { // Guard against num_doc = 0 for Alpha.row(d)
        for (int d = 0; d < num_doc; ++d) {
          if (d < doc_alpha_sums.size()) doc_alpha_sums(d) = Alpha.row(d).sum();
          if (d < doc_alpha_weighted_sums.size() && d < doc_each_len_weighted.size() && d < doc_alpha_sums.size()) {
             doc_alpha_weighted_sums(d) = doc_each_len_weighted[d] + doc_alpha_sums(d); // Changed (d) to [d] for std::vector
          }
        }
      }
  }
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


double keyATMcov::likelihood_lambda_efficient(int k, int t)
{
  double loglik = 0.0;
  
  // This function computes the full conditional log-likelihood for Lambda(k,t)
  // given the current state of Lambda.row(k) (which includes the current Lambda(k,t)).
  // It uses the global Alpha and doc_alpha_sums as the baseline.

  // Compute new alpha values for topic k based on current Lambda.row(k)
  VectorXd current_lambda_k_row = Lambda.row(k).transpose();
  VectorXd current_alpha_k_for_eval = (C * current_lambda_k_row).array().exp(); // O(num_doc * num_cov)
  
  // Use the helper function by passing the computed current_alpha_k_for_eval
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
  mh_use ? sample_lambda_mh_efficient() : sample_lambda_slice();
}


void keyATMcov::sample_lambda_mh_efficient()
{
  // No static Lambda_proposal, it depends on current Lambda(k,t)
  // static MatrixXd Lambda_proposal = MatrixXd::Zero(num_topics, num_cov);
  std::vector<std::vector<bool>> accept_flags(num_topics, std::vector<bool>(num_cov, false));
  
  topic_ids = sampler::shuffled_indexes(num_topics);
  // cov_ids can be shuffled per topic or globally, original shuffles globally once
  // Shuffling cov_ids per topic k might be slightly better for cache if C.col(t) is accessed.
  // For now, stick to global shuffle of cov_ids as original.
  cov_ids = sampler::shuffled_indexes(num_cov); 
  
  double mh_sigma = 0.4;
  
  for(int kk = 0; kk < num_topics; ++kk) {
    int k = topic_ids[kk];
    
    // Current log(Alpha.col(k)) based on accepted Lambda.row(k) for this topic
    // This is O(num_doc * num_cov) once per topic k
    VectorXd log_alpha_k_current = C * Lambda.row(k).transpose(); 
    VectorXd alpha_k_current_iter_vec = log_alpha_k_current.array().exp();
    
    for(int tt = 0; tt < num_cov; ++tt) {
      int t = cov_ids[tt];
      
      double Lambda_kt_original_val = Lambda(k, t); // Current value from Lambda matrix
      // Proposal is generated based on the current state of Lambda(k,t)
      double Lambda_kt_proposal_val = Lambda_kt_original_val + R::rnorm(0.0, mh_sigma);
      
      // Calculate llk for original Lambda(k,t) using current alpha_k vector for this topic
      // compute_likelihood_terms uses global Alpha and doc_alpha_sums as baseline
      double llk_original = compute_likelihood_terms(k, t, Lambda_kt_original_val, alpha_k_current_iter_vec);
      
      // Efficiently calculate alpha_k vector for the proposal
      double delta_lambda = Lambda_kt_proposal_val - Lambda_kt_original_val;
      // C.col(t) is num_doc x 1. delta_lambda is scalar. Result is num_doc x 1.
      VectorXd alpha_k_proposal_vec = alpha_k_current_iter_vec.array() * (C.col(t) * delta_lambda).array().exp(); // O(num_doc)
      
      // Calculate llk for proposed Lambda(k,t)
      double llk_proposal = compute_likelihood_terms(k, t, Lambda_kt_proposal_val, alpha_k_proposal_vec);
      
      double diffllk = llk_proposal - llk_original;
      // Using r_val to avoid conflict with R::
      double r_val = std::min(0.0, diffllk); // log(min(1, ratio))
      double u = log(R::unif_rand()); // log of uniform_rand
      
      if (u < r_val) {
        // Accept proposal
        Lambda(k, t) = Lambda_kt_proposal_val;
        accept_flags[k][t] = true;
        // Update the base log_alpha_k and alpha_k_current_iter_vec for the next t in THIS topic k
        log_alpha_k_current += C.col(t) * delta_lambda; // O(num_doc)
        alpha_k_current_iter_vec = log_alpha_k_current.array().exp(); // O(num_doc)
        // More direct: alpha_k_current_iter_vec = alpha_k_proposal_vec;
      } else {
        // Reject proposal: Lambda(k,t) remains Lambda_kt_original_val.
        // log_alpha_k_current and alpha_k_current_iter_vec remain unchanged.
        accept_flags[k][t] = false;
      }
    }
    
    // After all covariates 't' for topic 'k' are processed, Lambda.row(k) has its new values.
    // Update the k-th column of global Alpha matrix and related doc_alpha_sums
    // using the final alpha_k_current_iter_vec for this topic.
    // This replaces the old update_alpha_row_efficient(k) call more efficiently.
    for (int d = 0; d < num_doc; ++d) {
      doc_alpha_sums[d] = doc_alpha_sums[d] - Alpha(d, k) + alpha_k_current_iter_vec(d);
      doc_alpha_weighted_sums[d] = doc_each_len_weighted[d] + doc_alpha_sums[d]; // Update weighted sum too
      Alpha(d, k) = alpha_k_current_iter_vec(d);
    }
  }
}


void keyATMcov::sample_lambda_mh()
{
  // Use the efficient version
  sample_lambda_mh_efficient();
}


void keyATMcov::update_alpha_row_efficient(int k)
{
  // Update only row k of Alpha after Lambda(k, :) changes
  // This function is now effectively integrated into sample_lambda_mh_efficient and sample_lambda_slice's logic
  // by updating alpha_k_current_iter_vec and then applying it to global Alpha and sums.
  // It can be kept for other potential uses or if direct full row update is needed elsewhere,
  // but the optimized samplers manage this update more incrementally or at end of topic processing.
  VectorXd lambda_k = Lambda.row(k).transpose();
  VectorXd new_alpha_k_col = (C * lambda_k).array().exp(); // This is Alpha.col(k)
  
  // Update pre-computed sums and Alpha's k-th column
  for (int d = 0; d < num_doc; ++d) {
    doc_alpha_sums[d] = doc_alpha_sums[d] - Alpha(d, k) + new_alpha_k_col(d);
    doc_alpha_weighted_sums[d] = doc_each_len_weighted[d] + doc_alpha_sums[d];
    Alpha(d, k) = new_alpha_k_col(d);
  }
}


void keyATMcov::sample_lambda_slice()
{
  double start_p, end_p; // Renamed to avoid conflict with Eigen::VectorXd::end()

  double previous_p_val = 0.0; // Renamed
  double new_p_val = 0.0; // Renamed

  // double newlikelihood = 0.0; // Not directly used in this form
  double slice_level = 0.0; // Renamed
  // double current_lambda = 0.0; // Will be Lambda(k,t)

  // double store_loglik; // Will be log_f_x0
  // double newlambdallk; // Will be log_f_proposed

  topic_ids = sampler::shuffled_indexes(num_topics);
  cov_ids = sampler::shuffled_indexes(num_cov); // Global shuffle as original
  int k, t;
  const double A = slice_A; // Shrink/expand factor from model_settings

  // newlambdallk = 0.0; // Not needed here

  for (int kk = 0; kk < num_topics; ++kk) {
    k = topic_ids[kk];

    // Current log(Alpha.col(k)) based on accepted Lambda.row(k) values for this topic
    // This is O(num_doc * num_cov) once per topic k
    VectorXd log_alpha_k_topic_base = C * Lambda.row(k).transpose();
    VectorXd alpha_k_topic_base_vec = log_alpha_k_topic_base.array().exp();

    for (int tt = 0; tt < num_cov; ++tt) {
      t = cov_ids[tt];
      
      double original_Lambda_kt = Lambda(k,t); // Current value of Lambda(k,t) for this step
      VectorXd alpha_k_iter_vec = alpha_k_topic_base_vec; // Start with base for this (k,t)
                                                          // This will track Alpha.col(k) if original_Lambda_kt is kept

      // Calculate log f(x_0) + log |dx_0/dp_0| for slice definition
      // Here, x_0 is original_Lambda_kt
      previous_p_val = shrink(original_Lambda_kt, A);
      double log_f_x0 = compute_likelihood_terms(k, t, original_Lambda_kt, alpha_k_iter_vec);
      // Transformation Jacobian term: log( A / (p(1-p)) ) or -log( (1/A) p(1-p) )
      // Original code used: store_loglik - std::log(A * previous_p * (1.0 - previous_p))
      // This suggests store_loglik was log f(x_0) and they subtract log(dp/dx) or add log(dx/dp).
      // If p = shrink(lambda), then dp/dlambda = (1/A) p (1-p).
      // So log(dx/dp) = log( A / (p(1-p)) ) = logA - log(p) - log(1-p).
      // Original code: -log(A) -log(p) -log(1-p). This implies they use f(lambda(p)) * (dlambda/dp).
      // log(f(lambda(p))) + log(dlambda/dp)
      // Let's stick to the original formulation's transformed density for slice level.
      double transformed_log_f_x0 = log_f_x0 - std::log(A * previous_p_val * (1.0 - previous_p_val));
      slice_level = transformed_log_f_x0 + log(R::unif_rand());


      start_p = val_min; // shrinked value from settings
      end_p = val_max;   // shrinked value from settings

      // Stepping-out procedure (simplified here, original doesn't show explicit step-out)
      // The original seems to assume fixed val_min, val_max for p.

      for (int shrink_time = 0; shrink_time < max_shrink_time; ++shrink_time) {
        new_p_val = sampler::slice_uniform(start_p, end_p); 
        double proposed_Lambda_kt = expand(new_p_val, A);
        
        // Efficiently compute alpha_k vector for the proposed_Lambda_kt
        // Delta is from the original_Lambda_kt for this (k,t) sampling step
        double delta_lambda = proposed_Lambda_kt - original_Lambda_kt; 
        VectorXd proposed_alpha_k_vec = alpha_k_topic_base_vec.array() * (C.col(t) * delta_lambda).array().exp(); // O(Ndoc)

        double log_f_proposed = compute_likelihood_terms(k, t, proposed_Lambda_kt, proposed_alpha_k_vec);
        double transformed_log_f_proposed = log_f_proposed - std::log(A * new_p_val * (1.0 - new_p_val));

        if (slice_level < transformed_log_f_proposed) { // Accept proposal
          Lambda(k,t) = proposed_Lambda_kt;
          // Update the topic's base log_alpha and alpha_vector due to accepted change in Lambda(k,t)
          log_alpha_k_topic_base += C.col(t) * delta_lambda; 
          alpha_k_topic_base_vec = log_alpha_k_topic_base.array().exp();
          break; // Exit shrink_time loop
        } else { // Shrink interval
          if (std::abs(end_p - start_p) < 1e-9) {
            Rcpp::Rcerr << "Slice sampler interval shrunk too much for Lambda(" << k << "," << t 
                        << "). Keeping current value: " << original_Lambda_kt << std::endl;
            Lambda(k,t) = original_Lambda_kt; // Keep original value
            break;
          }
          if (previous_p_val < new_p_val) { // Check refers to p_0 vs p_new
             end_p = new_p_val;
          } else if (new_p_val < previous_p_val) {
             start_p = new_p_val;
          } else {
            // This case (new_p_val == previous_p_val after failing slice condition)
            // can happen if interval is tiny or due to precision.
            // Consider it a shrink failure for this step to avoid infinite loop if interval doesn't change.
            Rcpp::Rcerr << "Slice sampler new_p equals previous_p for Lambda(" << k << "," << t 
                        << "). Consider adjusting A_slice or bounds. Keeping current value." << std::endl;
            Lambda(k,t) = original_Lambda_kt;
            break;
          }
        }
      } // End shrink_time loop
      // If loop finished without break (e.g. max_shrink_time reached and no accept):
      // Lambda(k,t) should be original_Lambda_kt. It is, unless current_accepted_Lambda_kt was modified and not reverted.
      // The logic ensures Lambda(k,t) holds the accepted value or original if none accepted or reverted.
      // alpha_k_topic_base_vec should also reflect the final accepted Lambda(k,t).
      // The update to alpha_k_topic_base_vec is done only on acceptance of a *final* value for Lambda(k,t).

    } // End tt loop (covariates)

    // After all t for topic k, Lambda.row(k) is updated. 
    // Update global Alpha.col(k) and sums using the final alpha_k_topic_base_vec for this topic.
    for (int d = 0; d < num_doc; ++d) {
        doc_alpha_sums[d] = doc_alpha_sums[d] - Alpha(d, k) + alpha_k_topic_base_vec(d);
        doc_alpha_weighted_sums[d] = doc_each_len_weighted[d] + doc_alpha_sums[d];
        Alpha(d, k) = alpha_k_topic_base_vec(d);
    }
  } // End kk loop (topics)
}


double keyATMcov::loglik_total()
{
  double loglik = 0.0;
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
    loglik += mylgamma(doc_alpha_sums[d]) - mylgamma(doc_alpha_weighted_sums[d]);
    for (int k = 0; k < num_topics; ++k) {
      loglik += mylgamma(n_dk(d,k) + Alpha(d, k)) - mylgamma(Alpha(d, k));
    }
  }

  // Lambda loglik - vectorized computation
  for (int k = 0; k < num_topics; ++k) {
    for (int t = 0; t < num_cov; ++t) {
      loglik += log_prior_const;
      double lambda_diff = Lambda(k,t) - mu;
      loglik -= lambda_diff * lambda_diff * inv_2sigma_squared;
    }
  }

  return loglik;
}