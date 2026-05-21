#include "LDA_weightCov.h"

using namespace Eigen;
using namespace Rcpp;
using namespace std;

#define PI_V 3.14159265358979323846 /* pi */

void LDAcov::iteration_single(int it) { // Single iteration
  int doc_id_;
  int doc_length;
  int w_, z_, s_;
  int new_z;
  int w_position;

  s_ = -1;

  // Periodic full rebuild of (Alpha, alpha_sum) to control drift.
  // Between rebuilds, sample_lambda_slice/mh keep both in sync incrementally.
  --alpha_refresh_counter;
  if (alpha_refresh_counter <= 0) {
    refresh_alpha_cache();
    alpha_refresh_counter = alpha_refresh_every;
  }

  doc_indexes = sampler::shuffled_indexes(num_doc); // shuffle

  for (int ii = 0; ii < num_doc; ++ii) {
    doc_id_ = doc_indexes[ii];
    doc_z = Z[doc_id_], doc_w = W[doc_id_];
    doc_length = doc_each_len[doc_id_];

    token_indexes = sampler::shuffled_indexes(doc_length); // shuffle

    // Prepare Alpha for the doc
    alpha = Alpha.row(doc_id_).transpose(); // take out alpha

    // Iterate each word in the document
    for (int jj = 0; jj < doc_length; ++jj) {
      w_position = token_indexes[jj];
      z_ = doc_z[w_position], w_ = doc_w[w_position];

      new_z = sample_z(alpha, z_, s_, w_, doc_id_);
      doc_z[w_position] = new_z;
    }

    Z[doc_id_] = doc_z;
  }
  sample_parameters(it);
}

double LDAcov::loglik_total() {
  double loglik = 0.0;
  for (int k = 0; k < num_topics; ++k) {
    for (int v = 0; v < num_vocab; ++v) { // word
      loglik += mylgamma(beta + n_kv(k, v)) - mylgamma(beta);
    }

    // word normalization
    loglik += mylgamma(beta * (double)num_vocab) -
              mylgamma(beta * (double)num_vocab + n_k(k));
  }

  // z: use the cached Alpha and alpha_sum (kept in sync by the Lambda sampler).
  for (int d = 0; d < num_doc; ++d) {
    const double s_d = alpha_sum(d);
    loglik += mylgamma(s_d) - mylgamma(doc_each_len_weighted[d] + s_d);
    for (int k = 0; k < num_topics; ++k) {
      const double a_dk = Alpha(d, k);
      loglik += mylgamma(n_dk(d, k) + a_dk) - mylgamma(a_dk);
    }
  }

  // Lambda loglik
  double prior_fixedterm = -0.5 * log(2.0 * PI_V * std::pow(sigma, 2.0));
  for (int k = 0; k < num_topics; ++k) {
    for (int t = 0; t < num_cov; ++t) {
      loglik += prior_fixedterm;
      loglik -=
          (std::pow((Lambda(k, t) - mu), 2.0) / (2.0 * std::pow(sigma, 2.0)));
    }
  }

  return loglik;
}
