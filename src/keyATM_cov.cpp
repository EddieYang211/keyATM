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
  // This replaces the element-wise computation in the original code
  Alpha = (C * Lambda.transpose()).array().exp();
  
  // Pre-compute sums for likelihood computation
  for (int d = 0; d < num_doc; ++d) {
    doc_alpha_sums[d] = Alpha.row(d).sum();
    doc_alpha_weighted_sums[d] = doc_each_len_weighted[d] + doc_alpha_sums[d];
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
  
  // Only update the affected column of Alpha
  VectorXd C_col = C.col(t);
  VectorXd lambda_k = Lambda.row(k).transpose();
  
  // Compute exp(C * lambda_k) efficiently
  VectorXd alpha_k_new = (C * lambda_k).array().exp();
  
  // For each document, compute the likelihood contribution
  for (int d = 0; d < num_doc; ++d) {
    double alpha_sum_old = doc_alpha_sums[d];
    double alpha_k_old = Alpha(d, k);
    double alpha_k_new_val = alpha_k_new(d);
    double alpha_sum_new = alpha_sum_old - alpha_k_old + alpha_k_new_val;
    
    // Likelihood terms
    loglik += mylgamma(alpha_sum_new) - mylgamma(doc_each_len_weighted[d] + alpha_sum_new);
    loglik -= mylgamma(alpha_sum_old) - mylgamma(doc_each_len_weighted[d] + alpha_sum_old);
    
    loglik -= mylgamma(alpha_k_new_val) - mylgamma(alpha_k_old);
    loglik += mylgamma(n_dk(d, k) + alpha_k_new_val) - mylgamma(n_dk(d, k) + alpha_k_old);
  }

  // Prior
  loglik += log_prior_const;
  double lambda_diff = Lambda(k, t) - mu;
  loglik -= lambda_diff * lambda_diff * inv_2sigma_squared;

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
  topic_ids = sampler::shuffled_indexes(num_topics);
  cov_ids = sampler::shuffled_indexes(num_cov);
  double Lambda_current = 0.0;
  double llk_current = 0.0;
  double llk_proposal = 0.0;
  double diffllk = 0.0;
  double r = 0.0;
  double u = 0.0;
  double mh_sigma = 0.4;
  int k, t;

  for(int kk = 0; kk < num_topics; ++kk) {
    k = topic_ids[kk];

    for(int tt = 0; tt < num_cov; ++tt) {
      t = cov_ids[tt];

      Lambda_current = Lambda(k, t);

      // Current llk - use efficient computation
      llk_current = likelihood_lambda_efficient(k, t);

      // Proposal
      double proposal_value = Lambda_current + R::rnorm(0.0, mh_sigma);
      Lambda(k, t) = proposal_value;
      
      // Update Alpha for this change only
      update_alpha_row_efficient(k);
      
      llk_proposal = likelihood_lambda_efficient(k, t);

      diffllk = llk_proposal - llk_current;
      r = std::min(0.0, diffllk);
      u = log(unif_rand());

      if (u < r) {
        // accepted - Alpha is already updated
      } else {
        // Put back original values
        Lambda(k, t) = Lambda_current;
        update_alpha_row_efficient(k);
      }
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
  VectorXd lambda_k = Lambda.row(k).transpose();
  VectorXd new_alpha_k = (C * lambda_k).array().exp();
  
  // Update pre-computed sums
  for (int d = 0; d < num_doc; ++d) {
    doc_alpha_sums[d] = doc_alpha_sums[d] - Alpha(d, k) + new_alpha_k(d);
    doc_alpha_weighted_sums[d] = doc_each_len_weighted[d] + doc_alpha_sums[d];
    Alpha(d, k) = new_alpha_k(d);
  }
}


void keyATMcov::sample_lambda_slice()
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

  newlambdallk = 0.0;

  for (int kk = 0; kk < num_topics; ++kk) {
    k = topic_ids[kk];

    for (int tt = 0; tt < num_cov; ++tt) {
      t = cov_ids[tt];
      store_loglik = likelihood_lambda_efficient(k, t);

      start = val_min; // shrinked value
      end = val_max; // shrinked value

      current_lambda = Lambda(k,t);
      previous_p = shrink(current_lambda, A);
      slice_ = store_loglik - std::log(A * previous_p * (1.0 - previous_p))
              + log(unif_rand()); // <-- using R random uniform

      for (int shrink_time = 0; shrink_time < max_shrink_time; ++shrink_time) {
        new_p = sampler::slice_uniform(start, end); // <-- using R function above
        Lambda(k,t) = expand(new_p, A); // expand
        
        // Update Alpha for this change
        update_alpha_row_efficient(k);

        newlambdallk = likelihood_lambda_efficient(k, t);

        newlikelihood = newlambdallk - std::log(A * new_p * (1.0 - new_p));

        if (slice_ < newlikelihood) {
          break;
        } else if (abs(end - start) < 1e-9) {
          Rcerr << "Shrinked too much. Using a current value." << std::endl;
          Lambda(k,t) = current_lambda;
          update_alpha_row_efficient(k);
          break;
        } else if (previous_p < new_p) {
          end = new_p;
        } else if (new_p < previous_p) {
          start = new_p;
        } else {
          Rcpp::stop("Something goes wrong in sample_lambda_slice(). Adjust `A_slice`.");
        }
      } // for loop for shrink time
    } // for loop for num_cov
  } // for loop for num_topics
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
