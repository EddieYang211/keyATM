#include "keyATM_cov.h"

using namespace Eigen;
using namespace Rcpp;
using namespace std;

#define PI_V 3.14159265358979323846 /* pi */

void keyATMcov::read_data_specific() {
  // Covariate
  model_settings = model["model_settings"];
  NumericMatrix C_r = model_settings["covariates_data_use"];
  C = Rcpp::as<Eigen::MatrixXd>(C_r);
  num_cov = C.cols();

  // Slice Sampling
  val_min = model_settings["slice_min"];
  val_min = shrink(val_min, slice_A);

  val_max = model_settings["slice_max"];
  val_max = shrink(val_max, slice_A);

  // Metropolis Hastings
  mh_use = model_settings["mh_use"];
}

void keyATMcov::refresh_alpha_cache() {
  // Full rebuild of (Alpha, alpha_sum) from the current Lambda.
  // Called once at initialization and periodically thereafter to keep
  // floating-point drift from incremental updates bounded.
  Alpha = (C * Lambda.transpose()).array().exp();
  alpha_sum = Alpha.rowwise().sum();
}

void keyATMcov::initialize_specific() {
  // Alpha
  Alpha = MatrixXd::Zero(num_doc, num_topics);
  alpha_sum = VectorXd::Zero(num_doc);
  alpha = VectorXd::Zero(num_topics);

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

  refresh_alpha_cache();
  alpha_refresh_every = 100;
  alpha_refresh_counter = alpha_refresh_every;
}

void keyATMcov::resume_initialize_specific() {
  // Alpha
  Alpha = MatrixXd::Zero(num_doc, num_topics);
  alpha_sum = VectorXd::Zero(num_doc);
  alpha = VectorXd::Zero(num_topics);

  // Lambda
  mu = 0.0;
  sigma = 1.0;
  Lambda = MatrixXd::Zero(num_topics, num_cov);
  List Lambda_iter = stored_values["Lambda_iter"];
  NumericMatrix Lambda_r = Lambda_iter[Lambda_iter.size() - 1];
  Lambda = Rcpp::as<Eigen::MatrixXd>(Lambda_r);

  refresh_alpha_cache();
  alpha_refresh_every = 100;
  alpha_refresh_counter = alpha_refresh_every;
}

void keyATMcov::iteration_single(int it) { // Single iteration
  int doc_id_;
  int doc_length;
  int w_, z_, s_;
  int new_z, new_s;
  int w_position;

  // Periodic full rebuild to control drift in the incremental cache.
  --alpha_refresh_counter;
  if (alpha_refresh_counter <= 0) {
    refresh_alpha_cache();
    alpha_refresh_counter = alpha_refresh_every;
  }

  doc_indexes = sampler::shuffled_indexes(num_doc); // shuffle

  for (int ii = 0; ii < num_doc; ++ii) {
    doc_id_ = doc_indexes[ii];
    doc_s = S[doc_id_], doc_z = Z[doc_id_], doc_w = W[doc_id_];
    doc_length = doc_each_len[doc_id_];

    token_indexes = sampler::shuffled_indexes(doc_length); // shuffle

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

void keyATMcov::sample_parameters(int it) {
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

double keyATMcov::likelihood_lambda_eval(int k, double Lambda_eval,
                                         const VectorXd &cand_col,
                                         const VectorXd &cand_sum) {
  // Evaluate the part of the posterior over Lambda(k, *) that depends on
  // Lambda(k, t), given a candidate column for topic k of Alpha
  // (cand_col(d) = exp(C(d,:) * Lambda(k,:)^T) under the candidate) and the
  // corresponding row-sum vector (cand_sum(d) = sum_k' Alpha(d, k')).
  //
  // The doc-level integrand uses only alpha.sum() and alpha(k) for the
  // current topic k, so other columns of Alpha cancel out exactly.
  double loglik = 0.0;
  for (int d = 0; d < num_doc; ++d) {
    const double s_d = cand_sum(d);
    const double c_d = cand_col(d);
    loglik += mylgamma(s_d);
    loglik -= mylgamma(doc_each_len_weighted[d] + s_d);
    loglik -= mylgamma(c_d);
    loglik += mylgamma(n_dk(d, k) + c_d);
  }

  // Gaussian prior on Lambda(k, t)
  loglik += -0.5 * log(2.0 * PI_V * std::pow(sigma, 2.0));
  loglik -= (std::pow((Lambda_eval - mu), 2.0) / (2.0 * std::pow(sigma, 2.0)));
  return loglik;
}

void keyATMcov::sample_lambda() {
  mh_use ? sample_lambda_mh() : sample_lambda_slice();
}

void keyATMcov::sample_lambda_mh() {
  topic_ids = sampler::shuffled_indexes(num_topics);
  cov_ids = sampler::shuffled_indexes(num_cov);
  const double mh_sigma = 0.4;
  int k, t;

  // Scratch buffers reused across the K * num_cov updates.
  VectorXd col_old(num_doc);
  VectorXd col_new(num_doc);
  VectorXd sum_cand(num_doc);
  VectorXd factor(num_doc);
  VectorXd c_col_t(num_doc);

  for (int kk = 0; kk < num_topics; ++kk) {
    k = topic_ids[kk];

    for (int tt = 0; tt < num_cov; ++tt) {
      t = cov_ids[tt];

      const double Lambda_init = Lambda(k, t);
      col_old = Alpha.col(k);
      c_col_t = C.col(t);

      const double llk_current =
          likelihood_lambda_eval(k, Lambda_init, col_old, alpha_sum);

      // Proposal
      const double Lambda_cand = Lambda_init + R::rnorm(0.0, mh_sigma);
      const double delta = Lambda_cand - Lambda_init;
      factor = (delta * c_col_t.array()).exp();
      col_new = col_old.array() * factor.array();
      sum_cand = alpha_sum - col_old + col_new;

      const double llk_proposal =
          likelihood_lambda_eval(k, Lambda_cand, col_new, sum_cand);

      const double diffllk = llk_proposal - llk_current;
      const double r = std::min(0.0, diffllk);
      const double u = log(unif_rand());

      if (u < r) {
        // accepted: commit
        Lambda(k, t) = Lambda_cand;
        Alpha.col(k) = col_new;
        alpha_sum = sum_cand;
      }
      // rejected: nothing to roll back, no commit happened
    }
  }
}

void keyATMcov::sample_lambda_slice() {
  topic_ids = sampler::shuffled_indexes(num_topics);
  cov_ids = sampler::shuffled_indexes(num_cov);
  const double A = slice_A;
  int k, t;

  // Scratch buffers reused across the K * num_cov updates.
  VectorXd col_old(num_doc);
  VectorXd col_new(num_doc);
  VectorXd sum_cand(num_doc);
  VectorXd factor(num_doc);
  VectorXd c_col_t(num_doc);

  for (int kk = 0; kk < num_topics; ++kk) {
    k = topic_ids[kk];

    for (int tt = 0; tt < num_cov; ++tt) {
      t = cov_ids[tt];

      const double Lambda_init = Lambda(k, t);
      col_old = Alpha.col(k);
      c_col_t = C.col(t);

      const double store_loglik =
          likelihood_lambda_eval(k, Lambda_init, col_old, alpha_sum);

      double start = val_min; // shrinked value
      double end = val_max;   // shrinked value

      const double previous_p = shrink(Lambda_init, A);
      const double slice_ = store_loglik
                            - std::log(A * previous_p * (1.0 - previous_p))
                            + log(unif_rand()); // <-- using R random uniform

      for (int shrink_time = 0; shrink_time < max_shrink_time; ++shrink_time) {
        const double new_p = sampler::slice_uniform(start, end);
        const double Lambda_cand = expand(new_p, A); // expand
        const double delta = Lambda_cand - Lambda_init;

        // Incremental update of column k of Alpha and the row sum.
        //   Alpha_new(d, k) = Alpha_old(d, k) * exp(delta * C(d, t))
        //   alpha_sum_new(d) = alpha_sum(d) - col_old(d) + col_new(d)
        factor = (delta * c_col_t.array()).exp();
        col_new = col_old.array() * factor.array();
        sum_cand = alpha_sum - col_old + col_new;

        const double cand_llk =
            likelihood_lambda_eval(k, Lambda_cand, col_new, sum_cand);
        const double newlikelihood =
            cand_llk - std::log(A * new_p * (1.0 - new_p));

        if (slice_ < newlikelihood) {
          // Accept: commit candidate
          Lambda(k, t) = Lambda_cand;
          Alpha.col(k) = col_new;
          alpha_sum = sum_cand;
          break;
        } else if (std::abs(end - start) < 1e-9) {
          Rcerr << "Shrinked too much. Using a current value." << std::endl;
          // Keep Lambda(k, t), Alpha.col(k), and alpha_sum at their initial values.
          break;
        } else if (previous_p < new_p) {
          end = new_p;
        } else if (new_p < previous_p) {
          start = new_p;
        } else {
          Rcpp::stop("Something goes wrong in sample_lambda_slice(). Adjust "
                     "`A_slice`.");
        }
      } // shrink loop
    } // cov loop
  } // topic loop
}

double keyATMcov::loglik_total() {
  double loglik = 0.0;
  for (int k = 0; k < num_topics; ++k) {
    for (int v = 0; v < num_vocab; ++v) { // word
      loglik += mylgamma(beta + n_s0_kv(k, v)) - mylgamma(beta);
    }

    // word normalization
    loglik += mylgamma(beta * (double)num_vocab) -
              mylgamma(beta * (double)num_vocab + n_s0_k(k));

    if (k < keyword_k) {
      // For keyword topics

      // n_s1_kv
      for (SparseMatrix<double, RowMajor>::InnerIterator it(n_s1_kv, k); it;
           ++it) {
        loglik += mylgamma(beta_s + it.value()) - mylgamma(beta_s);
      }
      loglik += mylgamma(beta_s * (double)keywords_num[k]) -
                mylgamma(beta_s * (double)keywords_num[k] + n_s1_k(k));

      // Normalization
      loglik += mylgamma(prior_gamma(k, 0) + prior_gamma(k, 1)) -
                mylgamma(prior_gamma(k, 0)) - mylgamma(prior_gamma(k, 1));

      // s
      loglik += mylgamma(n_s0_k(k) + prior_gamma(k, 1)) -
                mylgamma(n_s1_k(k) + prior_gamma(k, 0) + n_s0_k(k) +
                         prior_gamma(k, 1)) +
                mylgamma(n_s1_k(k) + prior_gamma(k, 0));
    }
  }

  // z: use cached Alpha and alpha_sum directly (kept in sync by the sampler)
  for (int d = 0; d < num_doc; ++d) {
    const double s_d = alpha_sum(d);
    loglik += mylgamma(s_d) - mylgamma(doc_each_len_weighted[d] + s_d);
    for (int k = 0; k < num_topics; ++k) {
      const double a_dk = Alpha(d, k);
      loglik += mylgamma(n_dk(d, k) + a_dk) - mylgamma(a_dk);
    }
  }

  // Lambda loglik
  const double prior_fixedterm = -0.5 * log(2.0 * PI_V * std::pow(sigma, 2.0));
  for (int k = 0; k < num_topics; ++k) {
    for (int t = 0; t < num_cov; ++t) {
      loglik += prior_fixedterm;
      loglik -=
          (std::pow((Lambda(k, t) - mu), 2.0) / (2.0 * std::pow(sigma, 2.0)));
    }
  }

  return loglik;
}
