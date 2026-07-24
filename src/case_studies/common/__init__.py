"""
Common data pipeline scripts shared across all case studies.

Steps 01-04 are config-driven and work for ANY case study via the
run(case_study) entry point.

Usage from thin wrappers (recommended):
    from src.case_studies.common.step01_download_gdsc import run
    run(case_study="hela_hdac")

Usage from CLI (backward-compatible):
    PYTHONPATH=. python src/case_studies/common/step01_download_gdsc.py --case-study hela_hdac

Each script reads drug IDs, gene symbols, PDB structures, and UniProt
accessions from the case study's config.yaml. Step 05 (DrugPTM extraction)
and step 06 (harmonization) are case-study-specific and live in each
case study's own data_pipeline/ directory.
"""
