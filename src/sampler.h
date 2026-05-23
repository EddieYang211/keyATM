#ifndef __sampler__INCLUDED__
#define __sampler__INCLUDED__
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

#include <Rcpp.h>
#include <RcppEigen.h>

using namespace Eigen;

namespace sampler {
// Defines sampler used in keyATM

inline int rand_wrapper(const int n) { return floor(R::unif_rand() * n); }

double slice_uniform(const double lower, const double upper);

std::vector<int> shuffled_indexes(const int m);

// In-place Fisher-Yates shuffle into a pre-allocated buffer. Resizes v to
// length m (preserving capacity for future calls), fills with 0..m-1, then
// permutes. Calls rand_wrapper in the same order as shuffled_indexes() so
// the resulting permutation is bit-identical for a given RNG state.
void shuffle_in_place(std::vector<int> &v, const int m);

int rcat(VectorXd &prob, const int size);
int rcat_without_normalize(VectorXd &prob, const double total, const int size);

int rcat_eqsize(const int size);
int rcat_eqprob(const double prob, const int size);
} // namespace sampler

#endif
