#ifndef __keyATM_cov__INCLUDED__
#define __keyATM_cov__INCLUDED__
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

#include <Rcpp.h>
#include <RcppEigen.h>
#include <unordered_set>
#include "sampler.h"
#include "keyATM_meta.h"

// OpenMP support
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
    Eigen::VectorXd doc_alpha_sums;
    Eigen::VectorXd doc_alpha_weighted_sums;

    // During the sampling
      std::vector<int> topic_ids;
      std::vector<int> cov_ids;

      // Slice sampling
      double val_min;
      double val_max;

    // OpenMP optimization settings
    int num_threads;
    bool use_openmp;
    
    // Thread-local storage for vectorized operations
    struct ThreadLocalStorage {
        Eigen::VectorXd alpha_sum_new_overall_vec;
        Eigen::VectorXd term_weighted_sum_new;
        Eigen::VectorXd term_weighted_sum_old;
        Eigen::VectorXd term_ndk_new;
        Eigen::VectorXd term_ndk_old;
        Eigen::VectorXd log_alpha_k_topic_base;
        Eigen::VectorXd alpha_k_topic_base_vec;
        Eigen::VectorXd proposed_alpha_k_vec;
        Eigen::VectorXd X_k_proposal;
        Eigen::VectorXd C_col_t_times_delta;
    };
    
    std::vector<ThreadLocalStorage> thread_storage;

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
    
    // Optimized functions
    void update_alpha_efficient();
    void update_alpha_row_efficient(int k);
    double likelihood_lambda_efficient(int k, int t, const Eigen::VectorXd* precomputed_alpha_k = nullptr);
    void sample_lambda_mh_efficient();
    void sample_lambda_mh_parallel();
    void sample_lambda_slice_parallel();
    
    // Original functions (modified to use efficient versions)
    void sample_lambda();
    void sample_lambda_mh();
    void sample_lambda_slice();
    double alpha_loglik();
    virtual double loglik_total() override;

    double likelihood_lambda(int k, int t);
    void proposal_lambda(int k);
    
    // OpenMP helper functions
    void setup_openmp();
    void init_thread_storage();

  private:
    // Helper function for likelihood computation
    double compute_likelihood_terms(int k, int t, double current_lambda_kt_val,
                                    const Eigen::VectorXd& current_alpha_k_vec);
    
    // Thread-safe likelihood computation
    double compute_likelihood_terms_threadlocal(int k, int t, double current_lambda_kt_val,
                                                const Eigen::VectorXd& current_alpha_k_vec,
                                                ThreadLocalStorage& tls);
};


#endif