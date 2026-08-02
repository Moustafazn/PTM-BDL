#!/usr/bin/env Rscript
# ══════════════════════════════════════════════════════════════════════════════
# Extract CTRPv2 data from PharmacoSet RDS — SIMPLE version (no PharmacoGx)
#
# This script reads the S4 object slots directly using base R, without
# requiring the PharmacoGx Bioconductor package.
#
# INPUT:  data/raw/ctrp/PSet_CTRPv2.rds
# OUTPUT: data/raw/ctrp/ctrp_sensitivity_summary.csv
#         data/raw/ctrp/v20.data.curves_post_qc.txt
#         data/raw/ctrp/v20.meta.per_cell_line.txt
#         data/raw/ctrp/v20.meta.per_compound.txt
#
# USAGE:  Rscript src/case_studies/common/extract_ctrp_simple.R
# ══════════════════════════════════════════════════════════════════════════════

cat("╔══════════════════════════════════════════════════════════════╗\n")
cat("║  Extract CTRPv2 data (simple — no PharmacoGx required)      ║\n")
cat("╚══════════════════════════════════════════════════════════════╝\n")

rds_path <- "data/raw/ctrp/PSet_CTRPv2.rds"
out_dir <- "data/raw/ctrp"

if (!file.exists(rds_path)) {
  stop(paste("RDS file not found:", rds_path))
}

cat(sprintf("Loading %s (~40 MB)...\n", rds_path))
pset <- readRDS(rds_path)
cat(sprintf("  Class: %s\n", class(pset)))

# ── Explore the object structure ────────────────────────────────────────────
slot_names <- tryCatch(slotNames(pset), error = function(e) NULL)
if (!is.null(slot_names)) {
  cat(sprintf("  Slots: %s\n", paste(slot_names, collapse = ", ")))
}

# ── Try to extract sensitivity data ─────────────────────────────────────────
# PharmacoSet stores sensitivity in @sensitivity (a list with $info, $profiles, $raw)
sens <- NULL
cell_info <- NULL
drug_info <- NULL

# Method 1: Direct S4 slot access
tryCatch({
  sens_slot <- slot(pset, "sensitivity")
  if (is.list(sens_slot)) {
    cat(sprintf("  Sensitivity components: %s\n", paste(names(sens_slot), collapse = ", ")))
    
    sens_info <- sens_slot$info
    sens_prof <- sens_slot$profiles
    
    if (!is.null(sens_info)) {
      cat(sprintf("  Sensitivity info: %d rows x %d cols\n", nrow(sens_info), ncol(sens_info)))
      cat(sprintf("    Columns: %s\n", paste(head(colnames(sens_info), 15), collapse = ", ")))
    }
    if (!is.null(sens_prof)) {
      cat(sprintf("  Sensitivity profiles: %d rows x %d cols\n", nrow(sens_prof), ncol(sens_prof)))
      cat(sprintf("    Columns: %s\n", paste(head(colnames(sens_prof), 15), collapse = ", ")))
    }
    
    # Merge info + profiles
    if (!is.null(sens_info) && !is.null(sens_prof)) {
      sens <- merge(sens_info, sens_prof, by = "row.names", all = TRUE)
      names(sens)[1] <- "experiment_id"
      cat(sprintf("  ✓ Merged sensitivity: %d rows\n", nrow(sens)))
    } else if (!is.null(sens_info)) {
      sens <- sens_info
      sens$experiment_id <- rownames(sens_info)
    }
  }
}, error = function(e) {
  cat(sprintf("  Could not access @sensitivity: %s\n", e$message))
})

# Method 2: Try @cell and @drug slots for metadata
tryCatch({
  cell_info <- slot(pset, "cell")
  if (is.data.frame(cell_info)) {
    cell_info$ccl_name <- rownames(cell_info)
    cat(sprintf("  ✓ Cell info: %d cell lines\n", nrow(cell_info)))
    cat(sprintf("    Columns: %s\n", paste(head(colnames(cell_info), 10), collapse = ", ")))
  }
}, error = function(e) {
  cat(sprintf("  No @cell slot: %s\n", e$message))
})

tryCatch({
  drug_info <- slot(pset, "drug")
  if (is.data.frame(drug_info)) {
    drug_info$cpd_name <- rownames(drug_info)
    cat(sprintf("  ✓ Drug info: %d compounds\n", nrow(drug_info)))
    cat(sprintf("    Columns: %s\n", paste(head(colnames(drug_info), 10), collapse = ", ")))
  }
}, error = function(e) {
  cat(sprintf("  No @drug slot: %s\n", e$message))
})

# Method 3: If above failed, try alternative slot names
if (is.null(sens)) {
  for (sn in c("treatmentResponse", "sensitivity", "dose.response")) {
    tryCatch({
      s <- slot(pset, sn)
      if (is.list(s) && !is.null(s$profiles)) {
        cat(sprintf("  Found data in @%s\n", sn))
        sens_info <- s$info
        sens_prof <- s$profiles
        sens <- merge(sens_info, sens_prof, by = "row.names", all = TRUE)
        names(sens)[1] <- "experiment_id"
        cat(sprintf("  ✓ Merged: %d rows from @%s\n", nrow(sens), sn))
        break
      }
    }, error = function(e) NULL)
  }
}

if (is.null(cell_info)) {
  for (sn in c("sample", "cell", "cellInfo")) {
    tryCatch({
      ci <- slot(pset, sn)
      if (is.data.frame(ci)) {
        cell_info <- ci
        cell_info$ccl_name <- rownames(ci)
        cat(sprintf("  ✓ Cell info from @%s: %d rows\n", sn, nrow(ci)))
        break
      }
    }, error = function(e) NULL)
  }
}

if (is.null(drug_info)) {
  for (sn in c("treatment", "drug", "drugInfo")) {
    tryCatch({
      di <- slot(pset, sn)
      if (is.data.frame(di)) {
        drug_info <- di
        drug_info$cpd_name <- rownames(di)
        cat(sprintf("  ✓ Drug info from @%s: %d rows\n", sn, nrow(di)))
        break
      }
    }, error = function(e) NULL)
  }
}

# ── Save outputs ────────────────────────────────────────────────────────────
if (is.null(sens)) {
  cat("\n✗ Could not extract sensitivity data. Object structure:\n")
  str(pset, max.level = 2)
  stop("Failed to extract sensitivity data from PharmacoSet RDS")
}

cat("\nSaving extracted data...\n")

# Save curves (main sensitivity data)
curves_path <- file.path(out_dir, "v20.data.curves_post_qc.txt")
write.table(sens, curves_path, sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("  ✓ %s (%d rows)\n", curves_path, nrow(sens)))

# Save cell line metadata
if (!is.null(cell_info)) {
  cells_path <- file.path(out_dir, "v20.meta.per_cell_line.txt")
  write.table(cell_info, cells_path, sep = "\t", row.names = FALSE, quote = FALSE)
  cat(sprintf("  ✓ %s (%d rows)\n", cells_path, nrow(cell_info)))
}

# Save drug metadata
if (!is.null(drug_info)) {
  compounds_path <- file.path(out_dir, "v20.meta.per_compound.txt")
  write.table(drug_info, compounds_path, sep = "\t", row.names = FALSE, quote = FALSE)
  cat(sprintf("  ✓ %s (%d rows)\n", compounds_path, nrow(drug_info)))
}

# Create pre-merged summary CSV for direct Python consumption
cat("\nCreating summary CSV for Python...\n")

# Find the cell and drug ID columns in sensitivity data
cellid_col <- intersect(c("cellid", "sampleid", "cell_line"), colnames(sens))
drugid_col <- intersect(c("drugid", "treatmentid", "compound"), colnames(sens))
auc_cols <- intersect(c("auc_recomputed", "auc_published", "AUC"), colnames(sens))
ic50_cols <- intersect(c("ic50_recomputed", "ic50_published", "IC50"), colnames(sens))

keep_cols <- c("experiment_id", cellid_col, drugid_col, auc_cols, ic50_cols)
keep_cols <- intersect(keep_cols, colnames(sens))
summary_df <- sens[, keep_cols, drop = FALSE]

# Standardize column names
if (length(cellid_col) > 0) colnames(summary_df)[colnames(summary_df) == cellid_col[1]] <- "cell_line_name"
if (length(drugid_col) > 0) colnames(summary_df)[colnames(summary_df) == drugid_col[1]] <- "drug_name"

summary_path <- file.path(out_dir, "ctrp_sensitivity_summary.csv")
write.csv(summary_df, summary_path, row.names = FALSE)
cat(sprintf("  ✓ %s (%d rows, columns: %s)\n", summary_path, nrow(summary_df),
            paste(colnames(summary_df), collapse = ", ")))

# ── Drug overlap check ──────────────────────────────────────────────────────
cat("\n═══ Drug overlap with PTM-BDL case studies ═══\n")
our_drugs <- c("erlotinib", "gefitinib", "lapatinib", "afatinib",
               "vorinostat", "romidepsin",
               "imatinib", "dasatinib", "paclitaxel", "cytarabine", "methotrexate")

if ("drug_name" %in% colnames(summary_df)) {
  available <- tolower(unique(as.character(summary_df$drug_name)))
  for (d in our_drugs) {
    matches <- grep(d, available, value = TRUE, ignore.case = TRUE)
    if (length(matches) > 0) {
      n <- sum(tolower(summary_df$drug_name) %in% matches)
      cat(sprintf("  ✓ %s: %d records\n", d, n))
    } else {
      cat(sprintf("  ✗ %s: NOT FOUND\n", d))
    }
  }
}

cat("\n✓ CTRPv2 extraction complete!\n")
