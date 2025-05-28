#ifndef __keyATM_cov__INCLUDED__
#define __keyATM_cov__INCLUDED__

// C++ standard compatibility guards
#if __cplusplus < 201703L
#error "This package requires C++17 or later"
#endif

#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

// Include order is important to avoid template conflicts
#include <Rcpp.h>
#include <RcppEigen.h>
#include <unordered_set>
#include "sampler.h"
#include "keyATM_meta.h"

#ifdef HAVE_OPENMP
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
      
    // OpenMP thread management
    int num_threads;
    void set_num_threads(int threads);
    
    // Thread-safe temporary storage for parallel operations
    struct ThreadLocalStorage {
        Eigen::VectorXd alpha_sum_new_overall_vec;
        Eigen::VectorXd term_weighted_sum_new;
        Eigen::VectorXd term_weighted_sum_old;
        Eigen::VectorXd term_ndk_new;
        Eigen::VectorXd term_ndk_old;
        
        void resize(int num_doc) {
            if (alpha_sum_new_overall_vec.size() != num_doc) {
                alpha_sum_new_overall_vec.resize(num_doc);
                term_weighted_sum_new.resize(num_doc);
                term_weighted_sum_old.resize(num_doc);
                term_ndk_new.resize(num_doc);
                term_ndk_old.resize(num_doc);
            }
        }
    };

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
    
    // OpenMP optimized functions  
    void iteration_single_omp(int it);
    void sample_lambda_mh_omp();
    void sample_lambda_slice_omp();
    double compute_likelihood_terms_omp(int k, int t, double current_lambda_kt_val,
                                        const Eigen::VectorXd& current_alpha_k_vec,
                                        ThreadLocalStorage& tls);
    
    // Main functions
    void sample_lambda();
    void sample_lambda_mh();
    void sample_lambda_slice();
    double alpha_loglik();
    virtual double loglik_total() override;

    double likelihood_lambda(int k, int t);
    void proposal_lambda(int k);

  private:
    // Helper function for likelihood computation
    double compute_likelihood_terms(int k, int t, double current_lambda_kt_val,
                                    const Eigen::VectorXd& current_alpha_k_vec);
};


#endif