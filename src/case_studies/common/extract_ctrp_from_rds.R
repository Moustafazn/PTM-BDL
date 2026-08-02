#!/usr/bin/env Rscript
# ══════════════════════════════════════════════════════════════════════════════
# Extract CTRPv2 flat files from PharmacoSet RDS (ORCESTRA/PharmacoGx format)
#
# INPUT:  data/raw/ctrp/PSet_CTRPv2.rds  (downloaded from orcestra.ca)
# OUTPUT: data/raw/ctrp/v20.data.curves_post_qc.txt
#         data/raw/ctrp/v20.meta.per_cell_line.txt
#         data/raw/ctrp/v20.meta.per_compound.txt
#         data/raw/ctrp/v20.meta.per_experiment.txt
#
# USAGE:  Rscript src/case_studies/common/extract_ctrp_from_rds.R
#
# REQUIRES: PharmacoGx (installed from Bioconductor)
#   if (!require("BiocManager")) install.packages("BiocManager")
#   BiocManager::install("PharmacoGx")
# ══════════════════════════════════════════════════════════════════════════════

cat("╔══════════════════════════════════════════════════════════════╗\n")
cat("║  Extract CTRPv2 data from PharmacoSet RDS                   ║\n")
cat("╚══════════════════════════════════════════════════════════════╝\n")

# ── Install PharmacoGx if needed ────────────────────────────────────────────
if (!requireNamespace("PharmacoGx", quietly = TRUE)) {
  cat("Installing PharmacoGx from Bioconductor...\n")
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
  }
  BiocManager::install("PharmacoGx", ask = FALSE, update = FALSE)
}

library(PharmacoGx)

# ── Paths ───────────────────────────────────────────────────────────────────
rds_path <- "data/raw/ctrp/PSet_CTRPv2.rds"
out_dir <- "data/raw/ctrp"

if (!file.exists(rds_path)) {
  stop(paste("RDS file not found:", rds_path,
             "\nDownload from https://orcestra.ca/pset/canonical → CTRPv2_2015"))
}

# ── Load PharmacoSet ────────────────────────────────────────────────────────
cat(sprintf("Loading PharmacoSet from %s (~40MB)...\n", rds_path))
pset <- readRDS(rds_path)
cat(sprintf("  ✓ Loaded: %s\n", psetName(pset)))

# ── Extract sensitivity info (cell_line + compound + AUC + EC50) ────────────
cat("Extracting sensitivity data...\n")
sens_info <- sensitivityInfo(pset)
sens_prof <- sensitivityProfiles(pset)

# Merge info + profiles
curves <- merge(sens_info, sens_prof, by = "row.names", all = TRUE)
names(curves)[1] <- "experiment_id"

cat(sprintf("  ✓ Sensitivity data: %d experiments\n", nrow(curves)))
cat(sprintf("    Columns: %s\n", paste(head(names(curves), 15), collapse = ", ")))

# ── Extract cell line metadata ──────────────────────────────────────────────
cat("Extracting cell line metadata...\n")
cell_info <- cellInfo(pset)
cell_info$ccl_name <- rownames(cell_info)

cat(sprintf("  ✓ Cell lines: %d\n", nrow(cell_info)))
cat(sprintf("    Columns: %s\n", paste(head(names(cell_info), 10), collapse = ", ")))

# ── Extract drug/compound metadata ──────────────────────────────────────────
cat("Extracting compound metadata...\n")
drug_info <- drugInfo(pset)
drug_info$cpd_name <- rownames(drug_info)

cat(sprintf("  ✓ Compounds: %d\n", nrow(drug_info)))
cat(sprintf("    Columns: %s\n", paste(head(names(drug_info), 10), collapse = ", ")))

# ── Save as tab-separated flat files ────────────────────────────────────────
cat("\nSaving flat files...\n")

curves_path <- file.path(out_dir, "v20.data.curves_post_qc.txt")
write.table(curves, curves_path, sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("  ✓ %s (%d rows)\n", curves_path, nrow(curves)))

cells_path <- file.path(out_dir, "v20.meta.per_cell_line.txt")
write.table(cell_info, cells_path, sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("  ✓ %s (%d rows)\n", cells_path, nrow(cell_info)))

compounds_path <- file.path(out_dir, "v20.meta.per_compound.txt")
write.table(drug_info, compounds_path, sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("  ✓ %s (%d rows)\n", compounds_path, nrow(drug_info)))

# ── Also save a pre-merged summary CSV for direct Python consumption ────────
cat("\nCreating merged summary CSV for Python...\n")

# Build a clean merged table: cell_line, drug, AUC, EC50
summary_cols_curves <- intersect(
  c("experiment_id", "cellid", "drugid",
    "auc_recomputed", "ic50_recomputed",
    "aac_recomputed", "auc_published",
    "ic50_published"),
  names(curves)
)
summary_df <- curves[, summary_cols_curves, drop = FALSE]

# Rename for compatibility
if ("cellid" %in% names(summary_df)) names(summary_df)[names(summary_df) == "cellid"] <- "cell_line_name"
if ("drugid" %in% names(summary_df)) names(summary_df)[names(summary_df) == "drugid"] <- "drug_name"

summary_path <- file.path(out_dir, "ctrp_sensitivity_summary.csv")
write.csv(summary_df, summary_path, row.names = FALSE)
cat(sprintf("  ✓ %s (%d rows)\n", summary_path, nrow(summary_df)))

# ── Print drug overlap with our case studies ────────────────────────────────
cat("\n═══ Drug overlap with PTM-BDL case studies ═══\n")
our_drugs <- c("erlotinib", "gefitinib", "lapatinib", "afatinib",
               "vorinostat", "romidepsin",
               "imatinib", "dasatinib", "paclitaxel", "cytarabine", "methotrexate")

available_drugs <- tolower(unique(as.character(summary_df$drug_name)))
for (d in our_drugs) {
  matches <- grep(d, available_drugs, value = TRUE, ignore.case = TRUE)
  if (length(matches) > 0) {
    n <- sum(tolower(summary_df$drug_name) %in% matches)
    cat(sprintf("  ✓ %s: %d records (as '%s')\n", d, n, paste(matches, collapse = "', '")))
  } else {
    cat(sprintf("  ✗ %s: NOT FOUND\n", d))
  }
}

cat("\n✓ CTRPv2 extraction complete!\n")
cat(sprintf("  Files in: %s/\n", out_dir))
