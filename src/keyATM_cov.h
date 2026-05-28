#ifndef __keyATM_cov__INCLUDED__
#define __keyATM_cov__INCLUDED__
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

#include "keyATM_meta.h"
#include "sampler.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include <unordered_set>

using namespace Eigen;
using namespace Rcpp;
using namespace std;

class keyATMcov : virtual public keyATMmeta {
public:
  //
  // Parameters
  //
  MatrixXd Alpha;       // num_doc x num_topics, exp(C * Lambda^T)
  VectorXd alpha_sum;   // num_doc, row sums of Alpha (kept in sync)
  int num_cov;
  MatrixXd Lambda;
  MatrixXd C;

  // Per-covariate nonzero-document structure (column-compressed view of C).
  // Updating Lambda(k, t) only changes documents with C(d, t) != 0: for every
  // other document the multiplicative factor is exp(delta * 0) = 1, so its
  // Alpha entry, alpha_sum, and all four log-gamma terms are bit-for-bit
  // identical between the current and candidate states and cancel exactly in
  // the slice / Metropolis-Hastings acceptance ratio. Iterating only over the
  // nonzero documents turns each Lambda sweep from O(num_cov * num_doc) into
  // O(nnz(C)) — a large win when the covariate matrix is sparse, which it is
  // whenever factor variables are dummy-coded (the common case, and maximally
  // so under `standardize = "none"`, where the 0/1 dummies keep exact zeros).
  std::vector<std::vector<int>> cov_nz_idx;    // cov_nz_idx[t]: doc ids, ascending
  std::vector<std::vector<double>> cov_nz_val; // matching C(d, t) values

  int mh_use;
  double mu;
  double sigma;

  // During the sampling
  std::vector<int> topic_ids;
  std::vector<int> cov_ids;

  // Slice sampling
  double val_min;
  double val_max;

  // Periodic full rebuild of (Alpha, alpha_sum) to control floating-point drift
  int alpha_refresh_counter;
  int alpha_refresh_every;

  //
  // Functions
  //

  // Constructor
  keyATMcov(List model_) : keyATMmeta(model_) {};

  // Read data
  virtual void read_data_specific() override final;

  // Initialization
  virtual void initialize_specific() override final;

  // Resume
  virtual void resume_initialize_specific() override final;

  // Iteration
  virtual void iteration_single(int it) override;
  virtual void sample_parameters(int it) override final;
  void sample_lambda();
  void sample_lambda_mh();
  void sample_lambda_slice();
  double alpha_loglik();
  virtual double loglik_total() override;

  // Rebuild Alpha = exp(C * Lambda^T) and alpha_sum from scratch
  void refresh_alpha_cache();

  // Build cov_nz_idx / cov_nz_val from C. Call once after C is read.
  void build_cov_sparsity();

  // Change in the doc-likelihood part of the Lambda log-posterior when
  // Lambda(k, t) moves by `delta`, accumulated over the documents with
  // C(d, t) != 0. This equals (candidate - current) log-likelihood in the old
  // dense formulation: zero-covariate documents are skipped because each of
  // their four log-gamma differences is exactly 0 and `acc += 0` is a no-op in
  // IEEE-754, so the result is identical to summing over every document.
  double lambda_loglik_diff(int k, int t, double delta);

  // Commit an accepted Lambda(k, t) move: write Lambda and refresh the cached
  // Alpha column and alpha_sum for the affected documents only. Uses the
  // `alpha_sum += (c_new - c_old)` form so that untouched documents (c_new ==
  // c_old) are exact no-ops, keeping the sparse update equal to a dense one.
  void commit_lambda(int k, int t, double delta, double Lambda_cand);
};

#endif
