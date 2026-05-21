## Snapshot the cov-DirMulti output on the bundled bills dataset
## using the currently installed keyATM. Outputs are stashed for the
## post-patch comparison run.

suppressPackageStartupMessages({
  library(keyATM)
})

out_dir <- "tests/scripts/patch1_data"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

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

results <- lapply(names(cfgs), function(nm) {
  cat(sprintf("[baseline] running cfg=%s ...\n", nm))
  res <- run_cfg(nm, cfgs[[nm]]$options, cfgs[[nm]]$model_settings)
  cat(sprintf("[baseline]   elapsed: %.2fs\n", res$elapsed))
  res
})
names(results) <- names(cfgs)

snap <- lapply(results, function(r) {
  list(
    elapsed     = r$elapsed,
    theta       = r$out$theta,
    phi         = r$out$phi,
    pi          = r$out$pi,
    model_fit   = r$out$model_fit,
    Lambda_iter = r$out$values_iter$Lambda_iter,
    top_words3  = top_words(r$out, n = 3)
  )
})

snap$packageVersion <- as.character(packageVersion("keyATM"))
snap$Rversion       <- R.version.string

saveRDS(snap, file = file.path(out_dir, "baseline.rds"))
cat(sprintf("[baseline] saved -> %s\n", file.path(out_dir, "baseline.rds")))
