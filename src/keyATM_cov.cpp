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

  // Column-compressed nonzero structure of C, used to restrict each Lambda
  // update to the documents it can actually change (see keyATM_cov.h).
  build_cov_sparsity();
}

void keyATMcov::build_cov_sparsity() {
  cov_nz_idx.assign(num_cov, std::vector<int>());
  cov_nz_val.assign(num_cov, std::vector<double>());
  // C is column-major, so scanning d (rows) within a fixed t (column) is a
  // contiguous pass. Document ids are appended in ascending order, which makes
  // the sparse accumulation order match a dense d = 0..num_doc-1 scan.
  for (int t = 0; t < num_cov; ++t) {
    for (int d = 0; d < num_doc; ++d) {
      const double v = C(d, t);
      if (v != 0.0) {
        cov_nz_idx[t].push_back(d);
        cov_nz_val[t].push_back(v);
      }
    }
  }
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

  sampler::shuffle_in_place(doc_indexes, num_doc); // shuffle

  for (int ii = 0; ii < num_doc; ++ii) {
    doc_id_ = doc_indexes[ii];
    doc_s = S[doc_id_], doc_z = Z[doc_id_], doc_w = W[doc_id_];
    doc_length = doc_each_len[doc_id_];

    sampler::shuffle_in_place(token_indexes, doc_length); // shuffle

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

double keyATMcov::lambda_loglik_diff(int k, int t, double delta) {
  // Sum over the documents with C(d, t) != 0 of the change in the per-document
  // log-likelihood integrand when Lambda(k, t) moves by `delta`. For topic k
  // the integrand depends only on alpha_sum(d) and Alpha(d, k); both move by
  // the same multiplicative factor exp(delta * C(d, t)), so all other columns
  // cancel. Documents with C(d, t) == 0 have factor 1 and contribute exactly
  // 0 to this difference, hence are absent from cov_nz_idx[t].
  const std::vector<int> &idx = cov_nz_idx[t];
  const std::vector<double> &val = cov_nz_val[t];
  const int nnz = (int)idx.size();

  double dd = 0.0;
  for (int i = 0; i < nnz; ++i) {
    const int d = idx[i];
    const double c_old = Alpha(d, k);
    const double s_old = alpha_sum(d);
    const double c_new = c_old * std::exp(delta * val[i]);
    const double s_new = s_old - c_old + c_new;
    const double L_d = doc_each_len_weighted[d];
    const double ndk = n_dk(d, k);

    dd += mylgamma(s_new) - mylgamma(s_old);
    dd -= mylgamma(L_d + s_new) - mylgamma(L_d + s_old);
    dd -= mylgamma(c_new) - mylgamma(c_old);
    dd += mylgamma(ndk + c_new) - mylgamma(ndk + c_old);
  }
  return dd;
}

void keyATMcov::commit_lambda(int k, int t, double delta, double Lambda_cand) {
  const std::vector<int> &idx = cov_nz_idx[t];
  const std::vector<double> &val = cov_nz_val[t];
  const int nnz = (int)idx.size();

  for (int i = 0; i < nnz; ++i) {
    const int d = idx[i];
    const double c_old = Alpha(d, k);
    const double c_new = c_old * std::exp(delta * val[i]);
    alpha_sum(d) += c_new - c_old; // += 0 for untouched docs -> exact no-op
    Alpha(d, k) = c_new;
  }
  Lambda(k, t) = Lambda_cand;
}

void keyATMcov::sample_lambda() {
  mh_use ? sample_lambda_mh() : sample_lambda_slice();
}

void keyATMcov::sample_lambda_mh() {
  topic_ids = sampler::shuffled_indexes(num_topics);
  cov_ids = sampler::shuffled_indexes(num_cov);
  const double mh_sigma = 0.4;
  const double inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
  int k, t;

  for (int kk = 0; kk < num_topics; ++kk) {
    k = topic_ids[kk];

    for (int tt = 0; tt < num_cov; ++tt) {
      t = cov_ids[tt];

      const double Lambda_init = Lambda(k, t);

      // Proposal (RNG order matches the original: rnorm then unif_rand).
      const double Lambda_cand = Lambda_init + R::rnorm(0.0, mh_sigma);
      const double delta = Lambda_cand - Lambda_init;

      // log p(cand) - log p(current). The doc-likelihood part is summed over
      // the affected documents only; the Gaussian-prior constant cancels.
      double diffllk = lambda_loglik_diff(k, t, delta);
      diffllk -= (std::pow(Lambda_cand - mu, 2.0) -
                  std::pow(Lambda_init - mu, 2.0)) *
                 inv_2sigma2;

      const double r = std::min(0.0, diffllk);
      const double u = log(unif_rand());

      if (u < r) {
        commit_lambda(k, t, delta, Lambda_cand);
      }
      // rejected: nothing committed, nothing to roll back
    }
  }
}

void keyATMcov::sample_lambda_slice() {
  topic_ids = sampler::shuffled_indexes(num_topics);
  cov_ids = sampler::shuffled_indexes(num_cov);
  const double A = slice_A;
  const double inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
  int k, t;

  for (int kk = 0; kk < num_topics; ++kk) {
    k = topic_ids[kk];

    for (int tt = 0; tt < num_cov; ++tt) {
      t = cov_ids[tt];

      const double Lambda_init = Lambda(k, t);

      double start = val_min; // shrinked value
      double end = val_max;   // shrinked value

      const double previous_p = shrink(Lambda_init, A);

      // The original test is slice_ < newlikelihood with
      //   slice_        = store_loglik - log(A*p*(1-p)) + log(u)
      //   newlikelihood = cand_llk     - log(A*np*(1-np))
      // Subtracting the (state-independent) store_loglik from both sides leaves
      //   base < diff - log(A*np*(1-np)),
      // where base = log(u) - log(A*p*(1-p)) and diff = cand_llk - store_loglik
      // is exactly what lambda_loglik_diff (+ the prior diff) returns. The
      // unif draw stays in the same position in the RNG stream as before.
      const double base =
          log(unif_rand()) - std::log(A * previous_p * (1.0 - previous_p));

      for (int shrink_time = 0; shrink_time < max_shrink_time; ++shrink_time) {
        const double new_p = sampler::slice_uniform(start, end);
        const double Lambda_cand = expand(new_p, A); // expand
        const double delta = Lambda_cand - Lambda_init;

        double diff = lambda_loglik_diff(k, t, delta);
        diff -= (std::pow(Lambda_cand - mu, 2.0) -
                 std::pow(Lambda_init - mu, 2.0)) *
                inv_2sigma2;

        const double lhs = diff - std::log(A * new_p * (1.0 - new_p));

        if (base < lhs) {
          // Accept: write Lambda and refresh the affected Alpha/alpha_sum entries
          commit_lambda(k, t, delta, Lambda_cand);
          break;
        } else if (std::abs(end - start) < 1e-9) {
          Rcerr << "Shrinked too much. Using a current value." << std::endl;
          // Nothing committed: Lambda, Alpha, and alpha_sum keep their values.
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

      // n_s1_kv (dense; zero entries contribute exactly zero so a full
      // column scan is bit-identical to the prior sparse iteration).
      for (int v = 0; v < num_vocab; ++v) {
        const double val = n_s1_kv(k, v);
        if (val != 0.0) {
          loglik += mylgamma(beta_s + val) - mylgamma(beta_s);
        }
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
