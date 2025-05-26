#ifndef __keyATM_cov_optimized__INCLUDED__
#define __keyATM_cov_optimized__INCLUDED__
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

#include <Rcpp.h>
#include <RcppEigen.h>
#include <unordered_set>
#include <vector>
#include "sampler.h"
#include "keyATM_meta.h"

#ifdef _OPENMP
#include <omp.h>
#endif

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
    
    // Batch computation buffers
    MatrixXd exp_buffer;
    std::vector<double> likelihood_cache;

    // Thread-local storage for OpenMP
    #ifdef _OPENMP
    std::vector<VectorXd> thread_alpha_cache;
    std::vector<VectorXd> thread_prob_cache;
    #endif

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
    
    // Optimized parallel functions
    void process_document_parallel(int ii);
    int sample_z_optimized(VectorXd& alpha, VectorXd& prob_vec, 
                          int z, int s, int w, int doc_id);
    void update_alpha_batch();
    void update_alpha_row_vectorized(int k);
    double likelihood_lambda_vectorized(int k, int t);
    void sample_lambda_parallel();
    void sample_lambda_mh_parallel();
    void sample_lambda_slice_parallel();
    
    // Original functions (kept for compatibility)
    void sample_lambda();
    void sample_lambda_mh();
    void sample_lambda_slice();
    double alpha_loglik();
    virtual double loglik_total() override;

    double likelihood_lambda(int k, int t);
    void proposal_lambda(int k);
    
    // Legacy functions (redirected to optimized versions)
    void update_alpha_efficient() { update_alpha_batch(); }
    void update_alpha_row_efficient(int k) { update_alpha_row_vectorized(k); }
    double likelihood_lambda_efficient(int k, int t) { return likelihood_lambda_vectorized(k, t); }
    void sample_lambda_mh_efficient() { sample_lambda_mh_parallel(); }
};

#endif