<a href = "https://keyatm.github.io/keyATM/"><img src="docs/reference/figures/keyATM_logoFull.svg" alt="keyATM: Keyword Assisted Topic Models" width="290"/></a>

<!-- badges: start -->

[![CRAN
status](https://www.r-pkg.org/badges/version/keyATM)](https://CRAN.R-project.org/package=keyATM)
[![metacran
downloads](https://cranlogs.r-pkg.org/badges/grand-total/keyATM)](https://cran.r-project.org/package=keyATM)
[![Lifecycle:
stable](https://lifecycle.r-lib.org/articles/figures/lifecycle-stable.svg)](https://lifecycle.r-lib.org/articles/stages.html#stable)
[![R build status](https://github.com/keyATM/keyATM/actions/workflows/R-CMD-check.yml/badge.svg)](https://github.com/keyATM/keyATM/actions)
<!-- badges: end -->

# About
An R package for Keyword Assisted Topic Models, created by [Shusei Eshima](https://shusei-e.github.io), [Tomoya Sasaki](https://polisci.mit.edu/people/tomoya-sasaki), and [Kosuke Imai](https://imai.fas.harvard.edu/).

# Website
Please visit [our website](https://keyatm.github.io/keyATM/) for a complete reference.

# keyATM: Keyword Assisted Topic Models

<!-- badges: start -->
[![CRAN status](https://www.r-pkg.org/badges/version/keyATM)](https://CRAN.R-project.org/package=keyATM)
[![keyATM status badge](https://keyatm.r-universe.dev/badges/keyATM)](https://keyatm.r-universe.dev)
[![R-CMD-check](https://github.com/keyATM/keyATM/workflows/R-CMD-check/badge.svg)](https://github.com/keyATM/keyATM/actions)
[![Codecov test coverage](https://codecov.io/gh/keyATM/keyATM/branch/main/graph/badge.svg)](https://app.codecov.io/gh/keyATM/keyATM?branch=main)
[![CRAN downloads](https://cranlogs.r-pkg.org/badges/grand-total/keyATM)](https://CRAN.R-project.org/package=keyATM)
<!-- badges: end -->

`keyATM` performs keyword assisted topic modeling. 

## How to cite
Eshima, Shusei, Kosuke Imai, and Tomoya Sasaki. 2024. "Keyword Assisted Topic Models." *American Journal of Political Science* 68(1): 363-378. [[Paper](https://doi.org/10.1111/ajps.12779)] [[arXiv](https://arxiv.org/abs/2004.05964)]

## Performance Optimization with OpenMP

🚀 **New Feature**: The keyATM package now includes OpenMP parallelization for significant performance improvements!

### Key Benefits
- **2-6x speedup** for large datasets (1000+ documents, 10+ topics)
- **Automatic thread management** with optimal defaults
- **Thread-safe implementation** maintaining statistical correctness
- **Cross-platform support** (Windows, macOS, Linux)

### Usage
```r
# OpenMP is enabled automatically when available
model <- keyATM(
  docs = documents,
  keywords = keyword_list, 
  model = "covariates",
  covariates_data = covariates,
  options = list(
    iterations = 1000,
    num_threads = 4  # Optional: specify thread count
  )
)
```

See `OPENMP_OPTIMIZATION.md` for detailed performance benchmarks and configuration options.

## Installation

