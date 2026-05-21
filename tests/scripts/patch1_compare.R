## Re-run the same cov-DirMulti configs after the Patch 1 refactor and
## diff against the baseline snapshot.

suppressPackageStartupMessages({
  library(keyATM)
})

out_dir <- "tests/scripts/patch1_data"
baseline_path <- file.path(out_dir, "baseline.rds")
if (!file.exists(baseline_path)) {
  stop(sprintf("Baseline not found at %s. Run patch1_baseline.R first.", baseline_path))
}
baseline <- readRDS(baseline_path)

data(keyATM_data_bills)
bills_dfm      <- keyATM_data_bills$doc_dfm
bills_keywords <- keyATM_data_bills$keywords
bills_cov      <- keyATM_data_bills$cov
keyATM_docs    <- keyATM_read(bills_dfm)

run_cfg <- function(label, options, model_settings) {
  set.seed(options$seed)
  t0 <- proc.time()
  out <- keyATM(
    docs              = keyATM_docs,
    no_keyword_topics = 3,
    keywords          = bills_keywords,
    model             = "covariates",
    model_settings    = model_settings,
    options           = options,
    keep              = c("Z", "S")
  )
  elapsed <- (proc.time() - t0)[["elapsed"]]
  list(out = out, elapsed = elapsed, label = label)
}

cfgs <- list(
  short = list(
    options = list(
      seed = 250, store_theta = TRUE, iterations = 20, store_pi = 1,
      thinning = 5, verbose = FALSE
    ),
    model_settings = list(
      covariates_data    = bills_cov,
      standardize        = "all",
      covariates_formula = ~.,
      covariates_model   = "DirMulti"
    )
  ),
  long = list(
    options = list(
      seed = 250, store_theta = TRUE, iterations = 300, store_pi = 1,
      thinning = 10, llk_per = 50, verbose = FALSE
    ),
    model_settings = list(
      covariates_data    = bills_cov,
      standardize        = "all",
      covariates_formula = ~.,
      covariates_model   = "DirMulti"
    )
  )
)

max_abs_diff <- function(a, b) {
  a <- as.numeric(a); b <- as.numeric(b)
  if (length(a) != length(b)) return(NA_real_)
  max(abs(a - b), na.rm = TRUE)
}

cat("\n=== Patch 1 vs baseline ===\n")
cat(sprintf("baseline pkg=%s | now=%s\n",
            baseline$packageVersion, packageVersion("keyATM")))

for (nm in names(cfgs)) {
  cat(sprintf("\n[%s]\n", nm))
  res <- run_cfg(nm, cfgs[[nm]]$options, cfgs[[nm]]$model_settings)
  cat(sprintf("  elapsed: baseline=%.2fs  patch1=%.2fs  (speedup=%.2fx)\n",
              baseline[[nm]]$elapsed, res$elapsed,
              baseline[[nm]]$elapsed / res$elapsed))

  cat(sprintf("  max|Δtheta|       = %.3e  (max baseline=%.3f)\n",
              max_abs_diff(res$out$theta, baseline[[nm]]$theta),
              max(abs(baseline[[nm]]$theta))))
  cat(sprintf("  max|Δphi|         = %.3e\n",
              max_abs_diff(res$out$phi, baseline[[nm]]$phi)))
  cat(sprintf("  max|Δpi|          = %.3e\n",
              max_abs_diff(res$out$pi$Proportion, baseline[[nm]]$pi$Proportion)))
  # Compare last Lambda
  last_b <- baseline[[nm]]$Lambda_iter[[length(baseline[[nm]]$Lambda_iter)]]
  last_n <- res$out$values_iter$Lambda_iter[[length(res$out$values_iter$Lambda_iter)]]
  cat(sprintf("  max|ΔLambda_last| = %.3e\n", max_abs_diff(last_n, last_b)))
  # Compare logLik / perplexity trace
  if (!is.null(res$out$model_fit) && !is.null(baseline[[nm]]$model_fit)) {
    cat(sprintf("  max|ΔlogLik|      = %.3e\n",
                max_abs_diff(res$out$model_fit$`Log Likelihood`,
                             baseline[[nm]]$model_fit$`Log Likelihood`)))
  }
  # Spot-check the top-words table
  tw_b <- baseline[[nm]]$top_words3
  tw_n <- top_words(res$out, n = 3)
  agree <- mean(as.matrix(tw_b) == as.matrix(tw_n), na.rm = TRUE)
  cat(sprintf("  top_words3 agreement = %.1f%%\n", 100 * agree))
}
