## Synthetic tweet-like benchmark to expose the Lambda block at scale.
## Writes the timing line for the current installed keyATM to
## tests/scripts/patch1_data/bench_<tag>.rds where <tag> is the first arg
## (e.g. "old" or "new").

suppressPackageStartupMessages({
  library(keyATM)
  library(quanteda)
})

args <- commandArgs(trailingOnly = TRUE)
tag <- if (length(args) >= 1) args[1] else "current"

out_dir <- "tests/scripts/patch1_data"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

set.seed(42)
N <- 6000L      # docs
V <- 4000L      # vocab
doc_len_mean <- 18L  # tweet-ish
K_no_kw <- 20L  # no-keyword topics
n_kw_topics <- 5L  # keyword topics
M_cov <- 10L    # numeric covariates (after intercept => 11 columns)

# Synthetic vocab
vocab <- sprintf("w%04d", seq_len(V))

# Synthetic docs: each doc is a vector of word tokens
sim_doc <- function() {
  L <- max(2L, rpois(1, doc_len_mean))
  sample(vocab, L, replace = TRUE)
}
docs_raw <- replicate(N, sim_doc(), simplify = FALSE)

# Convert to dfm
toks <- quanteda::as.tokens(docs_raw)
dfm  <- quanteda::dfm(toks)
keyATM_docs <- keyATM_read(dfm)

# Keywords: pick a few words per topic
all_words <- colnames(dfm)
kw_list <- setNames(
  lapply(seq_len(n_kw_topics), function(k) sample(all_words, 5)),
  paste0("kw_topic_", seq_len(n_kw_topics))
)

# Covariates: a numeric data.frame, M_cov columns
cov_df <- as.data.frame(matrix(rnorm(N * M_cov), nrow = N))
colnames(cov_df) <- paste0("x", seq_len(M_cov))

iters <- 50L  # short fit; we want per-iter timing, not posterior quality

cat(sprintf("[%s] N=%d V=%d K=%d M=%d iters=%d\n",
            tag, N, V, K_no_kw + n_kw_topics, M_cov + 1L, iters))

t0 <- proc.time()
out <- keyATM(
  docs              = keyATM_docs,
  no_keyword_topics = K_no_kw,
  keywords          = kw_list,
  model             = "covariates",
  model_settings    = list(
    covariates_data    = cov_df,
    standardize        = "all",
    covariates_formula = ~.,
    covariates_model   = "DirMulti"
  ),
  options = list(
    seed = 250, iterations = iters, thinning = 25, llk_per = 100,
    store_theta = FALSE, store_pi = FALSE, verbose = FALSE
  )
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("[%s] elapsed: %.2fs  (%.0f ms/iter approx for fitting)\n",
            tag, elapsed, 1000 * elapsed / iters))

saveRDS(list(
  tag = tag,
  N = N, V = V, K = K_no_kw + n_kw_topics, M = M_cov + 1L,
  iters = iters,
  elapsed = elapsed,
  model_fit = out$model_fit,
  pkg_version = as.character(packageVersion("keyATM"))
), file = file.path(out_dir, sprintf("bench_%s.rds", tag)))
cat(sprintf("[%s] saved -> %s\n", tag, file.path(out_dir, sprintf("bench_%s.rds", tag))))
