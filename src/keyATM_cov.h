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

  // Evaluate log p(Lambda_eval, data | ...) using a candidate column for topic k
  // and its corresponding alpha_sum. Adds Gaussian prior on Lambda_eval.
  double likelihood_lambda_eval(int k, double Lambda_eval,
                                const Eigen::VectorXd &cand_col,
                                const Eigen::VectorXd &cand_sum);
};

#endif
