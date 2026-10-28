# Main hourly GAMM (M4). Univariate smooths + fire/monitor random intercepts.
# No two-dimensional ti() terms in this model.
#
# From the repository root:
#   Rscript code/gamm/step05_fit_gamm_main.R
#
# Reads (first match wins):
#   data/gamm/csv/step05_gamm_model_data.csv.gz
#   data/gamm/csv/step05_gamm_model_data.csv
#
# Optional argument: path to the .csv / .csv.gz, or to a folder that contains
# csv/step05_gamm_model_data.csv(.gz).
#
# write.csv() of the gzipped table is the GitHub-sized file (~6 MB).
# Do not commit the uncompressed ~62 MB CSV.
#
# Optional decompress (not required; this script reads the .gz):
#   python code/gamm/check_gamm_n.py

args <- commandArgs(trailingOnly = TRUE)

script_dir <- tryCatch(
  {
    ca <- commandArgs(trailingOnly = FALSE)
    f <- ca[grepl("^--file=", ca)]
    if (length(f)) {
      dirname(normalizePath(sub("^--file=", "", f[1])))
    } else {
      getwd()
    }
  },
  error = function(e) getwd()
)
pkg_root <- normalizePath(file.path(script_dir, "..", ".."))

find_gamm_csv <- function(root_or_file) {
  candidates <- c()
  if (length(root_or_file) && nzchar(root_or_file[1])) {
    p <- root_or_file[1]
    if (file.exists(p) && !dir.exists(p)) {
      return(p)
    }
    candidates <- c(
      candidates,
      file.path(p, "csv", "step05_gamm_model_data.csv.gz"),
      file.path(p, "csv", "step05_gamm_model_data.csv"),
      file.path(p, "step05_gamm_model_data.csv.gz"),
      file.path(p, "step05_gamm_model_data.csv")
    )
  }
  candidates <- c(
    candidates,
    file.path(pkg_root, "data", "gamm", "csv", "step05_gamm_model_data.csv.gz"),
    file.path(pkg_root, "data", "gamm", "csv", "step05_gamm_model_data.csv")
  )
  hit <- candidates[file.exists(candidates)]
  if (length(hit)) hit[[1]] else NA_character_
}

dat_path <- find_gamm_csv(if (length(args) >= 1) args[1] else "")
if (is.na(dat_path)) {
  stop(
    "Missing step05_gamm_model_data.csv.gz. ",
    "Expected data/gamm/csv/step05_gamm_model_data.csv.gz. ",
    "See code/gamm/README.md."
  )
}

csv_dir <- dirname(dat_path)
png_dir <- file.path(dirname(csv_dir), "png")
if (!dir.exists(file.path(pkg_root, "results", "figures"))) {
  dir.create(file.path(pkg_root, "results", "figures"), recursive = TRUE, showWarnings = FALSE)
}
dir.create(png_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(csv_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(mgcv)
})

read_gamm <- function(path) {
  if (grepl("\\.gz$", path, ignore.case = TRUE)) {
    read.csv(gzfile(path), stringsAsFactors = FALSE)
  } else {
    read.csv(path, stringsAsFactors = FALSE)
  }
}

df <- read_gamm(dat_path)
df$phase <- factor(df$phase, levels = c("Before", "During", "After"))
df$downwind_f <- factor(df$downwind, levels = c("0", "1"), labels = c("not_downwind", "downwind"))
if (is.numeric(df$downwind) || is.integer(df$downwind)) {
  df$downwind_f <- factor(df$downwind, levels = c(0, 1), labels = c("not_downwind", "downwind"))
}
df$FireID <- as.factor(df$FireID)
df$MonitorID <- as.factor(df$MonitorID)

n <- nrow(df)
n_fires <- nlevels(df$FireID)
n_mon <- nlevels(df$MonitorID)
cat("Model data:", n, "rows;", n_fires, "fires;", n_mon, "monitors\n")
cat("Read from:", dat_path, "\n")
if (n != 379433L || n_fires != 761L || n_mon != 21L) {
  warning(
    "Expected 379,433 hours / 761 fires / 21 monitors; got ",
    n, " / ", n_fires, " / ", n_mon
  )
}

form_main <- PM25 ~ phase * downwind_f +
  s(distance_km, k = 10) +
  s(wind_speed, k = 8) +
  s(temp, k = 8) +
  s(humidity, k = 8) +
  s(hour, bs = "cc", k = 24) +
  s(day_of_year, bs = "cc", k = 12) +
  s(FireID, bs = "re") +
  s(MonitorID, bs = "re")

t0 <- Sys.time()
model_main <- bam(
  form_main,
  data = df,
  method = "fREML",
  discrete = TRUE,
  nthreads = max(1L, parallel::detectCores() - 1L)
)
cat("bam fit elapsed:", round(difftime(Sys.time(), t0, units = "mins"), 2), "min\n")

sink(file.path(csv_dir, "step05_gamm_main_summary.txt"))
cat("=== Main model: bam (fREML) ===\n")
print(form_main)
cat("\n")
print(summary(model_main))
sink()

saveRDS(model_main, file.path(csv_dir, "step05_gamm_main_model.rds"))

param_tab <- as.data.frame(summary(model_main)$p.table)
param_tab$term <- rownames(param_tab)
rownames(param_tab) <- NULL
write.csv(param_tab, file.path(csv_dir, "step05_gamm_parametric_coefficients.csv"), row.names = FALSE)

sm <- summary(model_main)$s.table
if (!is.null(sm)) {
  sm_tab <- as.data.frame(sm)
  sm_tab$smooth <- rownames(sm_tab)
  rownames(sm_tab) <- NULL
  write.csv(sm_tab, file.path(csv_dir, "step05_gamm_smooth_terms.csv"), row.names = FALSE)
}

fit_stats <- data.frame(
  metric = c("n", "n_fires", "n_monitors", "r_sq", "dev_expl", "aic"),
  value = c(
    n,
    n_fires,
    n_mon,
    summary(model_main)$r.sq,
    summary(model_main)$dev.expl,
    AIC(model_main)
  )
)
write.csv(fit_stats, file.path(csv_dir, "step05_gamm_fit_statistics.csv"), row.names = FALSE)

fig_png <- file.path(pkg_root, "results", "figures", "Figure_S4_gamm_smooth_terms.png")
png(fig_png, width = 1400, height = 1000, res = 120)
plot(model_main, pages = 1, shade = TRUE)
dev.off()
png(file.path(png_dir, "step05_gamm_smooth_terms_panels.png"), width = 1400, height = 1000, res = 120)
plot(model_main, pages = 1, shade = TRUE)
dev.off()

cat("Done M4 bam fit. Figure S4 ->", fig_png, "\n")
