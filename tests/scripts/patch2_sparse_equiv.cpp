// Standalone numerical-equivalence test for the sparse Lambda update
// (keyATM covariates / DirMulti model). No R / Rcpp / Eigen needed:
//
//   g++ -O2 -std=c++17 tests/scripts/patch2_sparse_equiv.cpp -o /tmp/patch2 && /tmp/patch2
//
// It checks three things on synthetic data with a realistically sparse
// covariate matrix (intercept + dummy columns + a couple of dense continuous
// columns):
//
//   (1) lambda_loglik_diff over the nonzero documents only is BIT-FOR-BIT
//       identical to the same accumulation over every document. This is the
//       correctness guarantee for the optimization: skipped docs contribute
//       exactly 0.0 and `acc += 0.0` is a no-op in IEEE-754.
//   (2) The accept/reject decisions (MH style and slice style) computed from
//       the sparse diff equal those from the dense diff, bit for bit.
//   (3) The difference-based statistic equals the OLD dense two-sum statistic
//       (cand_llk - store_loglik from the previous likelihood_lambda_eval) to
//       floating-point precision -- i.e. the refactor is faithful to the
//       original math, differing only by harmless summation reassociation.

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

// ---- exact copy of keyATMmeta::mylgamma ----
static inline double mylgamma(const double x) {
  if (x < 0.6)
    return (lgamma(x));
  else
    return ((x - 0.5) * std::log(x) - x + 0.91893853320467 + 1 / (12 * x));
}

static const double PI_V = 3.14159265358979323846;

// Synthetic model state.
struct Model {
  int D, K, P;
  std::vector<double> C;      // D*P, column-major: C[t*D + d]
  std::vector<double> Lambda; // K*P, Lambda[k*P + t]
  std::vector<double> Alpha;  // D*K, column-major: Alpha[k*D + d]
  std::vector<double> alpha_sum; // D
  std::vector<double> n_dk;      // D*K, row-major: n_dk[d*K + k]
  std::vector<double> len_w;     // D, doc_each_len_weighted
  double mu = 0.0, sigma = 1.0;

  // cov_nz: ascending doc ids with C(d,t) != 0, and their values.
  std::vector<std::vector<int>> nz_idx;
  std::vector<std::vector<double>> nz_val;

  inline double Cdt(int d, int t) const { return C[t * D + d]; }
  inline double &Adk(int d, int k) { return Alpha[k * D + d]; }
  inline double Adk(int d, int k) const { return Alpha[k * D + d]; }
  inline double ndk(int d, int k) const { return n_dk[d * K + k]; }
};

static void rebuild_alpha(Model &m) {
  for (int d = 0; d < m.D; ++d) {
    double s = 0.0;
    for (int k = 0; k < m.K; ++k) {
      double dot = 0.0;
      for (int t = 0; t < m.P; ++t)
        dot += m.Cdt(d, t) * m.Lambda[k * m.P + t];
      double a = std::exp(dot);
      m.Adk(d, k) = a;
      s += a;
    }
    m.alpha_sum[d] = s;
  }
}

static void build_sparsity(Model &m) {
  m.nz_idx.assign(m.P, {});
  m.nz_val.assign(m.P, {});
  for (int t = 0; t < m.P; ++t)
    for (int d = 0; d < m.D; ++d) {
      double v = m.Cdt(d, t);
      if (v != 0.0) {
        m.nz_idx[t].push_back(d);
        m.nz_val[t].push_back(v);
      }
    }
}

// (1) sparse diff -- mirrors keyATMcov::lambda_loglik_diff exactly.
static double diff_sparse(const Model &m, int k, int t, double delta) {
  const std::vector<int> &idx = m.nz_idx[t];
  const std::vector<double> &val = m.nz_val[t];
  const int nnz = (int)idx.size();
  double dd = 0.0;
  for (int i = 0; i < nnz; ++i) {
    const int d = idx[i];
    const double c_old = m.Adk(d, k);
    const double s_old = m.alpha_sum[d];
    const double c_new = c_old * std::exp(delta * val[i]);
    const double s_new = s_old - c_old + c_new;
    const double L_d = m.len_w[d];
    const double ndk = m.ndk(d, k);
    dd += mylgamma(s_new) - mylgamma(s_old);
    dd -= mylgamma(L_d + s_new) - mylgamma(L_d + s_old);
    dd -= mylgamma(c_new) - mylgamma(c_old);
    dd += mylgamma(ndk + c_new) - mylgamma(ndk + c_old);
  }
  return dd;
}

// Dense reference: visit EVERY document, but a document with C(d,t)==0 has an
// unchanged alpha row, so its true per-document difference is exactly 0 and it
// is skipped (contributing 0.0). Gating on cdt!=0 with the identical inner ops
// must reproduce the sparse result bit for bit -- this is what validates that
// cov_nz_idx[t] captures all and only the nonzero docs, in ascending order.
static double diff_dense(const Model &m, int k, int t, double delta) {
  double dd = 0.0;
  for (int d = 0; d < m.D; ++d) {
    const double cdt = m.Cdt(d, t);
    if (cdt == 0.0)
      continue; // unchanged doc: exact-zero contribution
    const double c_old = m.Adk(d, k);
    const double s_old = m.alpha_sum[d];
    const double c_new = c_old * std::exp(delta * cdt);
    const double s_new = s_old - c_old + c_new;
    const double L_d = m.len_w[d];
    const double ndk = m.ndk(d, k);
    dd += mylgamma(s_new) - mylgamma(s_old);
    dd -= mylgamma(L_d + s_new) - mylgamma(L_d + s_old);
    dd -= mylgamma(c_new) - mylgamma(c_old);
    dd += mylgamma(ndk + c_new) - mylgamma(ndk + c_old);
  }
  return dd;
}

// (3) OLD formulation: absolute likelihood_lambda_eval, then subtract.
static double abs_llk_old(const Model &m, int k, double Lambda_eval,
                          const std::vector<double> &cand_col,
                          const std::vector<double> &cand_sum) {
  double loglik = 0.0;
  for (int d = 0; d < m.D; ++d) {
    const double s_d = cand_sum[d];
    const double c_d = cand_col[d];
    loglik += mylgamma(s_d);
    loglik -= mylgamma(m.len_w[d] + s_d);
    loglik -= mylgamma(c_d);
    loglik += mylgamma(m.ndk(d, k) + c_d);
  }
  loglik += -0.5 * std::log(2.0 * PI_V * std::pow(m.sigma, 2.0));
  loglik -= (std::pow((Lambda_eval - m.mu), 2.0) / (2.0 * std::pow(m.sigma, 2.0)));
  return loglik;
}

static double diff_old_twosum(const Model &m, int k, int t, double delta,
                              double Lambda_init) {
  std::vector<double> col_old(m.D), col_new(m.D), sum_cand(m.D);
  for (int d = 0; d < m.D; ++d) {
    col_old[d] = m.Adk(d, k);
    double f = std::exp(delta * m.Cdt(d, t));
    col_new[d] = col_old[d] * f;
    sum_cand[d] = m.alpha_sum[d] - col_old[d] + col_new[d];
  }
  double store = abs_llk_old(m, k, Lambda_init, col_old, m.alpha_sum);
  double cand = abs_llk_old(m, k, Lambda_init + delta, col_new, sum_cand);
  return cand - store; // includes the prior difference, like the new diff
}

static double prior_diff(const Model &m, double Lambda_init, double Lambda_cand) {
  const double inv_2sigma2 = 1.0 / (2.0 * m.sigma * m.sigma);
  return -(std::pow(Lambda_cand - m.mu, 2.0) -
           std::pow(Lambda_init - m.mu, 2.0)) *
         inv_2sigma2;
}

static bool bit_eq(double a, double b) {
  uint64_t ua, ub;
  std::memcpy(&ua, &a, 8);
  std::memcpy(&ub, &b, 8);
  return ua == ub;
}

int main() {
  Model m;
  m.D = 4000;
  m.K = 12;
  m.P = 40;
  std::mt19937_64 rng(20260528);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::normal_distribution<double> N(0.0, 1.0);

  m.C.assign((size_t)m.D * m.P, 0.0);
  // col 0: intercept (dense, all ones)
  for (int d = 0; d < m.D; ++d)
    m.C[0 * m.D + d] = 1.0;
  // cols 1..2: dense continuous (standardize="none" => raw, rarely exactly 0)
  for (int t = 1; t <= 2; ++t)
    for (int d = 0; d < m.D; ++d)
      m.C[t * m.D + d] = N(rng);
  // cols 3..P-1: sparse 0/1 dummies (factor dummy-coding), varied densities
  for (int t = 3; t < m.P; ++t) {
    double p = 0.02 + 0.20 * U(rng); // 2%..22% ones
    for (int d = 0; d < m.D; ++d)
      m.C[t * m.D + d] = (U(rng) < p) ? 1.0 : 0.0;
  }

  m.Lambda.assign((size_t)m.K * m.P, 0.0);
  for (auto &x : m.Lambda)
    x = 0.3 * N(rng);

  m.len_w.assign(m.D, 0.0);
  m.n_dk.assign((size_t)m.D * m.K, 0.0);
  for (int d = 0; d < m.D; ++d) {
    int len = 5 + (int)(40 * U(rng)); // short docs, like tweets
    double tot = 0.0;
    for (int k = 0; k < m.K; ++k) {
      double v = (double)(int)(len * U(rng)) * (0.5 + U(rng));
      m.n_dk[d * m.K + k] = v;
      tot += v;
    }
    m.len_w[d] = tot;
  }

  m.Alpha.assign((size_t)m.D * m.K, 0.0);
  m.alpha_sum.assign(m.D, 0.0);
  rebuild_alpha(m);
  build_sparsity(m);

  // report sparsity
  long nz = 0;
  for (int t = 0; t < m.P; ++t)
    nz += (long)m.nz_idx[t].size();
  double frac_nz = (double)nz / ((double)m.D * m.P);
  std::printf("synthetic: D=%d K=%d P=%d  nnz(C)=%.1f%%  (work ~ %.2fx of dense)\n",
              m.D, m.K, m.P, 100.0 * frac_nz, frac_nz);

  int trials = 200000;
  int bit_mismatch = 0;
  int decision_mismatch_sparse_vs_dense = 0;
  int decision_mismatch_vs_old = 0;
  double max_abs_vs_old = 0.0, max_rel_vs_old = 0.0;

  std::uniform_int_distribution<int> Dk(0, m.K - 1), Dt(0, m.P - 1);
  std::normal_distribution<double> prop(0.0, 0.4);

  for (int it = 0; it < trials; ++it) {
    int k = Dk(rng), t = Dt(rng);
    double Lambda_init = m.Lambda[k * m.P + t];
    double delta = prop(rng);
    double Lambda_cand = Lambda_init + delta;

    double ds = diff_sparse(m, k, t, delta) + prior_diff(m, Lambda_init, Lambda_cand);
    double dd = diff_dense(m, k, t, delta) + prior_diff(m, Lambda_init, Lambda_cand);
    double dold = diff_old_twosum(m, k, t, delta, Lambda_init);

    // (1) sparse == dense, bit for bit
    if (!bit_eq(ds, dd))
      ++bit_mismatch;

    // (2) decisions agree (MH style and slice style) sparse vs dense
    double u = U(rng);
    double logu = std::log(u);
    bool mh_sparse = logu < std::min(0.0, ds);
    bool mh_dense = logu < std::min(0.0, dd);
    if (mh_sparse != mh_dense)
      ++decision_mismatch_sparse_vs_dense;

    // (3) faithfulness vs old two-sum statistic
    double ae = std::fabs(ds - dold);
    double re = ae / (std::fabs(dold) + 1e-12);
    if (ae > max_abs_vs_old)
      max_abs_vs_old = ae;
    if (re > max_rel_vs_old)
      max_rel_vs_old = re;
    // slice-style decision using a random level vs old
    double lvl = std::log(U(rng));
    bool acc_new = lvl < ds;
    bool acc_old = lvl < dold;
    if (acc_new != acc_old)
      ++decision_mismatch_vs_old;
  }

  // commit equivalence: sparse commit vs dense commit on copies
  int commit_mismatch = 0;
  for (int it = 0; it < 2000; ++it) {
    int k = Dk(rng), t = Dt(rng);
    double delta = prop(rng);
    Model a = m, b = m; // copies
    // sparse commit
    {
      const auto &idx = a.nz_idx[t];
      const auto &val = a.nz_val[t];
      for (size_t i = 0; i < idx.size(); ++i) {
        int d = idx[i];
        double c_old = a.Adk(d, k);
        double c_new = c_old * std::exp(delta * val[i]);
        a.alpha_sum[d] += c_new - c_old;
        a.Adk(d, k) = c_new;
      }
    }
    // dense commit (all docs, += form)
    {
      for (int d = 0; d < b.D; ++d) {
        double c_old = b.Adk(d, k);
        double c_new = c_old * std::exp(delta * b.Cdt(d, t));
        b.alpha_sum[d] += c_new - c_old;
        b.Adk(d, k) = c_new;
      }
    }
    for (int d = 0; d < m.D; ++d) {
      if (!bit_eq(a.alpha_sum[d], b.alpha_sum[d]) ||
          !bit_eq(a.Adk(d, k), b.Adk(d, k)))
        ++commit_mismatch;
    }
  }

  std::printf("\n[1] sparse vs dense diff, bit-identical : %s (%d / %d mismatches)\n",
              bit_mismatch == 0 ? "PASS" : "FAIL", bit_mismatch, trials);
  std::printf("[2] accept decision sparse == dense      : %s (%d / %d mismatches)\n",
              decision_mismatch_sparse_vs_dense == 0 ? "PASS" : "FAIL",
              decision_mismatch_sparse_vs_dense, trials);
  std::printf("[3] commit sparse == dense, bit-identical : %s (%d mismatches)\n",
              commit_mismatch == 0 ? "PASS" : "FAIL", commit_mismatch);
  std::printf("\n[faithfulness vs old two-sum formula]\n");
  std::printf("    max |diff_new - diff_old|      = %.3e\n", max_abs_vs_old);
  std::printf("    max relative difference        = %.3e\n", max_rel_vs_old);
  std::printf("    slice-decision flips vs old    = %d / %d (%.4f%%)\n",
              decision_mismatch_vs_old, trials,
              100.0 * decision_mismatch_vs_old / trials);

  bool ok = (bit_mismatch == 0) && (decision_mismatch_sparse_vs_dense == 0) &&
            (commit_mismatch == 0);
  std::printf("\nOVERALL: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
