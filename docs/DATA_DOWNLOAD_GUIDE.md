# Data Download Guide

**All raw data must be downloaded manually before running the pipeline.**
Steps 01–05 will check for these files and print instructions if anything is missing.

---

## Quick Checklist

| #  | Source  | Files Needed                         | Download URL                                                                                                                 | Place In         |
|----|---------|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------|------------------|
| 1  | GDSC2   | IC50 dose-response (.xlsx)           | [Sanger Cell Model Passports](https://cellmodelpassports.sanger.ac.uk/downloads) → "Drug Sensitivity Data" → "GDSC2 dataset" | `data/raw/gdsc/` |
| 2  | GDSC2   | Model annotation (.csv)              | Same page → "Model Annotation" → "list of all annotated models"                                                              | `data/raw/gdsc/` |
| 3  | DepMap  | OmicsSomaticMutations.csv (~300 MB)  | [DepMap Portal](https://depmap.org/portal/data_page/?tab=allData) → search "OmicsSomaticMutations"                           | `data/raw/ccle/` |
| 4  | DepMap  | Model.csv                            | Same page → search "Model"                                                                                                   | `data/raw/ccle/` |
| 5  | DepMap  | ccle_somatic_mutations.csv (~595 MB) | Same page → search "OmicsSomaticMutations" (earlier format)                                                                  | `data/raw/ccle/` |
| 6  | DepMap  | ccle_model_info.csv                  | Same page → search "Model" (earlier format)                                                                                  | `data/raw/ccle/` |
| 7  | UniProt | EGFR reference sequence              | `curl -o data/raw/ccle/egfr_P00533.fasta https://rest.uniprot.org/uniprotkb/P00533.fasta`                                    | `data/raw/ccle/` |
| 8  | UniProt | HER2 reference sequence              | `curl -o data/raw/ccle/erbb2_P04626.fasta https://rest.uniprot.org/uniprotkb/P04626.fasta`                                   | `data/raw/ccle/` |
| 9  | PDB     | Crystal structures                   | See Step 03 below                                                                                                            | `data/raw/pdb/`  |
| 10 | UniProt | PTM annotations (JSON)               | `curl -o data/raw/ptm/uniprot_P00533.json https://rest.uniprot.org/uniprotkb/P00533.json`                                    | `data/raw/ptm/`  |

---

## Detailed Instructions by Step

### Step 01 — GDSC Drug Response Data

**Source**: Sanger Cell Model Passports (formerly cancerrxgene.org)

1. Go to: https://cellmodelpassports.sanger.ac.uk/downloads
2. Download **GDSC2 IC50 Data** (~21 MB .xlsx) → place in `data/raw/gdsc/`
3. Download **Model List** (~934 KB .csv) → place in `data/raw/gdsc/`

The script auto-detects filenames containing "gdsc2" or "model_list".

### Step 02 — Mutation Data

**Source**: DepMap (Broad Institute)

1. Go to: https://depmap.org/portal/data_page/?tab=allData
2. Download **OmicsSomaticMutations.csv** (~300 MB) → `data/raw/ccle/`
3. Download **Model.csv** → `data/raw/ccle/`
4. Download EGFR FASTA:
   ```bash
   curl -o data/raw/ccle/egfr_P00533.fasta https://rest.uniprot.org/uniprotkb/P00533.fasta
   curl -o data/raw/ccle/erbb2_P04626.fasta https://rest.uniprot.org/uniprotkb/P04626.fasta
   ```

### Step 03 — PDB Crystal Structures

**Source**: RCSB Protein Data Bank

Download all structures automatically:

```bash
cd data/raw/pdb/
for PDB in 2GS6 2JIT 4HJO 5EDP 3IKA 6LUD 2ITY 4ZAU 4G5J 3PP0 3RCD; do
  curl -o ${PDB}.pdb https://files.rcsb.org/download/${PDB}.pdb
done
```

Or let step03 download them — it handles this automatically if `requests` is installed.

### Step 04 — PTM Site Annotations

**Source**: UniProt

```bash
mkdir -p data/raw/ptm
curl -o data/raw/ptm/uniprot_P00533.json "https://rest.uniprot.org/uniprotkb/P00533.json"
curl -o data/raw/ptm/uniprot_P04626.json "https://rest.uniprot.org/uniprotkb/P04626.json"
```

PTM site definitions are also hardcoded in `config/config.yaml` as a fallback.

### Step 05 — Drug-PTM Phosphoproteomic Data

**Source**: Multiple publications (8 studies)

| Study           | Data File                            | Download From                            |
|-----------------|--------------------------------------|------------------------------------------|
| DrugPTM-Bench   | CSV files from repo                  | https://github.com/Xie-lab/DrugPTM-Bench |
| Tozuka 2024     | mmc2.xlsx                            | Journal supplement (PMID 38646155)       |
| Hsu 2025        | 44320_2025_141_MOESM3_ESM.xlsx       | Journal supplement (PMID 41023502)       |
| PNAS 2025       | pnas.2522090123.sd02.xlsx, sd04.xlsx | PNAS supplement                          |
| FEBS 2025       | mol270091-sup-0006-tables5.xlsx      | Journal supplement                       |
| Cancer Res 2021 | table_s2_phosphosites.xlsx           | Journal supplement                       |
| MCP 2025        | table_s8_phospho_glyco_summary.xlsx  | Journal supplement                       |
| Ruprecht 2017   | (included in DrugPTM-Bench)          | —                                        |

Place all files in `data/raw/drugptm/` following the directory structure expected by step05 (the script prints exact
paths if files are missing).

### Step 05 — DrugPTM-Bench Data for Non-TKI Case Studies

**Source**: DrugPTM-Bench (Badkul et al., 2024) — https://github.com/Xie-lab/DrugPTM-Bench

For the HeLa/HDAC and K562/CML case studies, download the cell-line-specific CSV files:

| Case Study | File | Size | Place In |
|------------|------|------|----------|
| HeLa/HDAC | `PTM_CellLine_HeLa.csv` | ~508 MB | `data/raw/drugptm/30394195/` |
| K562/CML | `PTM_CellLine_K562.csv` | ~991 MB | `data/raw/drugptm/30394195/` |

These files contain dose-response PTM data:
- **HeLa**: 980,608 rows (921K phospho + 59K acetylation), 6 drugs, 15 dose points
- **K562**: 1,608,421 rows (all phosphorylation), 5 drugs, 14 dose points

### PDB Structures for Non-TKI Case Studies

**HeLa/HDAC structures:**
```bash
cd data/raw/pdb/
for PDB in 4LXZ 3MAX 5EDU 4BKX; do
  curl -o ${PDB}.pdb https://files.rcsb.org/download/${PDB}.pdb
done
```

**K562/CML structures (ABL1):**
```bash
cd data/raw/pdb/
for PDB in 1IEP 2GQG 2HYY; do
  curl -o ${PDB}.pdb https://files.rcsb.org/download/${PDB}.pdb
done
```

---

### CTRPv2 — Cross-Dataset Generalization Testing (Reviewer Q4)

**Source**: ORCESTRA PharmacoGx — https://orcestra.ca/pset/canonical

CTRPv2 (Cancer Therapeutics Response Portal v2) provides an independent drug sensitivity
dataset for cross-dataset generalization testing (train on GDSC2, test on CTRPv2).

**Download and extraction (2 steps):**

1. **Download the PharmacoSet RDS file:**
   - Go to: https://orcestra.ca/pset/canonical
   - Find **CTRPv2** in the list → click **CTRPv2_2015** → Download
   - Save as: `data/raw/ctrp/PSet_CTRPv2.rds` (~40 MB)

2. **Extract flat files using R** (one-time, requires R + PharmacoGx):
   ```bash
   # Install R if needed: brew install r  (macOS)
   Rscript src/case_studies/common/extract_ctrp_from_rds.R
   ```
   This auto-installs PharmacoGx from Bioconductor and produces:
   - `data/raw/ctrp/ctrp_sensitivity_summary.csv` — pre-merged summary for Python
   - `data/raw/ctrp/v20.data.curves_post_qc.txt` — raw dose-response curves
   - `data/raw/ctrp/v20.meta.per_cell_line.txt` — cell line metadata
   - `data/raw/ctrp/v20.meta.per_compound.txt` — compound metadata + SMILES

3. **Process for our case studies:**
   ```bash
   python -m src.case_studies.common.download_ctrp
   ```
   This filters to overlapping drugs and saves to `data/processed/ctrp/`.

**Overlapping drugs with our case studies:**

| Drug | CS1 (EGFR/ERBB2) | CS2 (HDAC) | CS3 (BCR-ABL) |
|------|:-:|:-:|:-:|
| Erlotinib | ✓ | | |
| Gefitinib | ✓ | | |
| Lapatinib | ✓ | | |
| Afatinib | ✓ | | |
| Vorinostat | | ✓ | |
| Dasatinib | | | ✓ |
| Imatinib | | | ✓ |
| Paclitaxel | | | ✓ |
| Cytarabine | | | ✓ |
| Methotrexate | | | ✓ |

**References:**
- Basu et al., Cell 2013 (PMID 23993102) — original CTRPv1
- Rees et al., Nat Chem Biol 2016 (PMID 26656090) — CTRPv2
- Seashore-Ludlow et al., Cancer Discovery 2015 (PMID 26482930) — connectivity analysis

---

## Directory Structure After Download

```
data/
├── raw/
│   ├── gdsc/
│   │   ├── gdsc2_dose_response.xlsx     # or any file containing "gdsc2"
│   │   └── model_list.csv               # or any file containing "model_list"
│   ├── ccle/
│   │   ├── OmicsSomaticMutations.csv
│   │   ├── Model.csv
│   │   ├── egfr_P00533.fasta
│   │   └── erbb2_P04626.fasta
│   ├── pdb/
│   │   ├── 2GS6.pdb
│   │   ├── 2JIT.pdb
│   │   ├── ... (11 PDB files total)
│   │   └── 3PP0.pdb
│   ├── ptm/
│   │   ├── uniprot_P00533.json
│   │   └── uniprot_P04626.json
│   ├── ctrp/                            # CTRPv2 cross-dataset (Reviewer Q4)
│   │   ├── PSet_CTRPv2.rds              # PharmacoSet RDS (~40 MB)
│   │   ├── ctrp_sensitivity_summary.csv # Extracted by R script
│   │   ├── v20.data.curves_post_qc.txt
│   │   ├── v20.meta.per_cell_line.txt
│   │   └── v20.meta.per_compound.txt
│   └── drugptm/
│       └── (study-specific subdirectories)
```

---

## Notes

- **Steps 01–05 will NOT crash** if data is missing — they print clear instructions and either skip gracefully or create
  development placeholders.
- **Steps 07–09 (feature extraction)** also have fallbacks: if ESM-2, GearNet, or ChemBERTa can't load, they create
  placeholder embeddings for development.
- The full pipeline requires ~1 GB of raw data downloads.
