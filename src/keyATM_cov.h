#ifndef __keyATM_cov__INCLUDED__
#define __keyATM_cov__INCLUDED__
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

#include <Rcpp.h>
#include <RcppEigen.h>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include "sampler.h"
#include "keyATM_meta.h"

using namespace Eigen;
using namespace Rcpp;
using namespace std;

class keyATMcov : virtual public keyATMmeta
{
  public:
    //
    // Parameters
    //
    MatrixXd Alpha;
    int num_cov;
    MatrixXd Lambda;
    MatrixXd C;
    MatrixXd C_transpose; // Pre-computed transpose for efficiency

    int mh_use;
    double mu;
    double sigma;
    
    // Pre-computed constants for efficiency
    double sigma_squared;
    double inv_2sigma_squared;
    double log_prior_const;
    
    // Pre-allocated vectors for likelihood computation
    std::vector<double> doc_alpha_sums;
    std::vector<double> doc_alpha_weighted_sums;
    std::vector<double> temp_alpha_sums;
    std::vector<double> temp_lgamma_cache;

    // Cache for frequently computed lgamma values
    std::unordered_map<int, double> lgamma_cache;

    // During the sampling
    std::vector<int> topic_ids;
    std::vector<int> cov_ids;

    // Slice sampling
    double val_min;
    double val_max;

    //
    // Functions
    //

    // Constructor
    keyATMcov(List model_) :
      keyATMmeta(model_) {};

    // Read data
    virtual void read_data_specific() override final;

    // Initialization
    virtual void initialize_specific() override final;

    // Resume
    virtual void resume_initialize_specific() override final;

    // Iteration
    virtual void iteration_single(int it) override;
    virtual void sample_parameters(int it) override final;
    
    // Optimized core functions
    void update_alpha_vectorized();
    void update_alpha_row_vectorized(int k);
    double likelihood_lambda_vectorized(int k, int t);
    void sample_lambda_optimized();
    void sample_lambda_mh_vectorized();
    void sample_lambda_slice_optimized();
    double loglik_total_optimized();
    
    // Cached lgamma function
    inline double cached_mylgamma(double x);
    
    // Legacy compatibility functions
    void update_alpha_efficient();
    void update_alpha_row_efficient(int k);
    double likelihood_lambda_efficient(int k, int t);
    void sample_lambda_mh_efficient();
    
    // Original functions (now calling optimized versions)
    void sample_lambda();
    void sample_lambda_mh();
    void sample_lambda_slice();
    double alpha_loglik();
    virtual double loglik_total() override;

    double likelihood_lambda(int k, int t);
    void proposal_lambda(int k);
};


#endif