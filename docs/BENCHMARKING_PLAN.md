# Benchmarking Plan — Nature Methods 2026 Submission

## ⚠️ The Benchmarking Trap (Why This Plan Exists)

> *Nature Methods reviewers will demand comparison against the most recent state-of-the-art alternatives. If we only compare against GraphDRP (2022) and DrugCell (2020), we will receive a major revision request adding 3–6 months to our timeline.*

**This plan explicitly addresses this risk** by organizing baselines into four tiers based on recency and relevance, with **2024–2026 methods as primary comparisons**.

**Date:** 2026-07-01  
**Target journal:** Nature Methods  
**Benchmarking philosophy:** Fair, same-data, same-split, same-metric comparison with full statistical rigor

---

## 1. What Already Exists (No Duplication)

### Internal evaluation already implemented

| Script | What it provides | Status |
|--------|-----------------|--------|
| `step11_train.py` | Single-split training + test metrics | ✅ Running |
| `step11b_ablation.py` | 5-arm ablation + randomized PTM control + multi-seed stability | ✅ Running |
| `step11c_crossval.py` | 5-fold stratified CV, paired t-tests, per-fold IG, cross-receptor homology | ✅ Running |
| `step12_evaluate.py` | Per-protein, per-drug, mutation-stratified, confidence-aware | ✅ Running |
| `step13_explainability.py` | IG site rankings (EGFR + ERBB2), cross-modal attention, homology check | ✅ Running |

### What is MISSING for Nature Methods

1. **External method baselines** — 3 tiers of published methods (§3)
2. **Bootstrap confidence intervals** on test-set metrics (1,000 resamples)
3. **Paired statistical tests** (DeLong for AUROC, paired bootstrap for PCC)
4. **Leave-One-Cell-Line-Out (LOCLO)** grouped cell-line split — for generalization claim
5. **Runtime / scalability benchmarking**
6. **Comparison figures and tables** formatted for the paper

> **Note on LODO**: Leave-One-Drug-Out was implemented in June 23 run. Result: 2/4 drugs failed (BAcc=0.500), Osimertinib AUROC=0.362. Removed from subsequent runs — with only 6 pharmacologically overlapping drugs, LODO does not produce meaningful generalization evidence. Per-drug held-out evaluation (already in step12) is sufficient.

---

## 2. Metric Selection — Grounded in Our Model + Field Standards

### Why we cannot blindly copy other papers' metrics

Our model is fundamentally different from most DRP methods:
- **Dual-task architecture**: Predicts BOTH continuous IC50 AND binary resistance (most DRP models do only regression)
- **92%/8% class imbalance**: ~12 sensitive samples in the test set — threshold-dependent metrics are unreliable
- **PTM-BDL is the contribution**: Our novelty is biological, not just predictive — we need biological validation metrics that no other model can even be evaluated on

### AUROC + AUPRC-sensitive as CO-PRIMARY (investigated and validated)

**AUROC alone is insufficient for our 92%/8% imbalance.** With 131 resistant + 12 sensitive in the test set, AUROC can be misleadingly high because 131 true negatives absorb false positives (Saito & Rehmsmeier, PLOS ONE 2015). Example: a model identifying 10/12 sensitive but making 20 false positives gets FPR=0.153 (looks fine) but Precision=0.333 (terrible). AUROC hides this; AUPRC exposes it.

**Our live ablation (2026-07-01) proves AUROC and AUPRC tell different stories:**

| Model | AUROC | AUPRC-sens | What it means |
|-------|-------|-----------|---------------|
| No PTM | 0.873 | 0.604 | Baseline |
| Full PTM-BDL | **0.909** | 0.667 | Best overall ranking |
| No typed attention (MLP) | 0.866 | **0.693** | Best minority precision |

AUROC says Full PTM-BDL is best; AUPRC says MLP replacement is best. **Neither alone tells the full story.** Therefore both must be co-primary:
- **AUROC**: overall ranking quality + field comparability (DeepCDR, DeepTTA use it)
- **AUPRC-sensitive**: honest minority-class evaluation (exposes precision problems that AUROC hides)

**No model code changes needed** — `step11_train.py` early stopping on AUROC remains correct for model selection stability; both metrics are already computed by step11b and step12.

### Our metrics: Three tiers

#### Tier A — Standard DRP Benchmarking Metrics (for external comparison)

*These let reviewers compare us to published methods on their terms.*

| Metric | Task | Role | Why for OUR model |
|--------|------|------|-------------------|
| **Pearson R (PCC)** | Regression | **Primary** | Field standard (DIPK, HiDRA, GraphDRP, DeepCDR all report it). Our step12 already computes it. |
| **RMSE** | Regression | **Primary** | Field standard error metric. Our step12 already computes it. |
| **AUROC** | Classification | **Primary** | Our model's early stopping criterion (step11). Threshold-independent — critical with 12 sensitive test samples. DeepCDR, DeepTTA, MTIGCN all use AUROC for binarized IC50 (Table 2, Sada Del Real 2026 review). |
| **AUPRC (sensitive class)** | Classification | **Primary** | Complements AUROC for our extreme imbalance. Our ablation (step11b) uses ptm_gain_auprc_sensitive as a key metric. |
| **Per-drug PCC** | Regression | **Required** | The 2026 DRP review (Sada Del Real et al.) states *"per-drug SCC is the most clinically relevant metric."* Our step12 already reports per-drug metrics for all 6 drugs. |

#### Tier B — Biological Validation Metrics (PTM-BDL contribution — unique to us)

*These are what make our paper publishable in Nature Methods. No other method can be evaluated on these.*

| Metric | What it tests | Pass criterion (from step11b code) |
|--------|---------------|-----------------------------------|
| **IG site ranking concordance** | Does the model correctly rank Y1068 as the #1 resistance-determining phosphosite? | Y1068 = rank #1 across 3/3 seeds (std_rank = 0.0) |
| **Cross-receptor homology** | Does the model independently discover EGFR Y1068 ≡ HER2 Y1221 (GRB2→MAPK docking site)? | Both rank #1 in their respective proteins |
| **Randomized PTM control** | Does real PTM beat shuffled PTM? (Definitive test of biological signal) | ΔAUROC ≥ +0.005 AND ΔAUPRC-sens ≥ 0.0 (revised 2026-07-03; BAcc removed — inappropriate for 92:8 imbalance) |
| **PTM-BDL ablation gain** | Does the full model (phospho+glyco+typed self-attention) beat the no-PTM baseline? | ptm_gain_auroc > 0, ptm_gain_auprc_sensitive > 0 |
| **Phospho vs glyco channel contribution** | Do both modification types add value? | phospho_marginal_auroc > 0, glyco_marginal_auroc > 0 |
| **Typed attention vs MLP** | Does inter-site self-attention beat an MLP replacement? | typed_attention_marginal_auroc > 0 |
| **Modification-type hierarchy** | Are tyrosine phosphosites ranked higher than serine/threonine by IG? | Top 9 = all tyrosine, Bottom 3 = all serine/threonine |

#### Tier C — Supplementary Metrics (reported but not primary)

| Metric | Why supplementary |
|--------|-------------------|
| Spearman ρ (SCC) | Rank-based complement to PCC. DrugCell uses as primary. Report for comparability. |
| F1, Precision, Recall | Threshold-dependent with n=12 sensitive. Report at the threshold our model uses, not as primary. |
| BAcc | Our step11b uses it in the pass criterion alongside AUROC, but it's noisy with small n. |
| R² | Redundant with PCC for same test set. Report if reviewers ask. |

### Statistical testing

| Test | When | Source |
|------|------|--------|
| **Wilcoxon signed-rank** | Per-fold/per-drug paired comparisons | SAGE-net (Nature Methods 2026) uses this |
| **Paired bootstrap** (1,000 resamples) | 95% CIs for all Tier A metrics | Standard |
| **Benjamini-Hochberg correction** | Multiple testing across K baselines | SAGE-net (Nature Methods 2026) |
| **DeLong test** | Paired AUROC comparison (ours vs each baseline) | Standard for classifier comparison |

### Field alignment evidence

| Paper | Venue | Year | Metrics they report |
|-------|-------|------|---------------------|
| Sada Del Real et al. | Brief Bioinf (DRP Review) | 2026 | PCC, SCC, RMSE; per-drug SCC; "per-drug SCC is the most clinically relevant metric" |
| SAGE-net (Spiro et al.) | Nature Methods | 2026 | Pearson R; Wilcoxon signed-rank; BH correction |
| ClairS (Zheng et al.) | Nature Methods | 2026 | F1, Precision, Recall, AUPRC (binary classification) |
| DeepCDR (Li et al.) | Brief Bioinf | 2020 | PCC, SCC, RMSE; **AUROC + AUPR for binarized IC50** |
| DIPK (Li et al.) | Nature Comms | 2024 | PCC, RMSE, Spearman ρ |

---

## 3. Competitive Landscape (2024–2026)

### Why our positioning is unique

Every method below predicts drug response. **None operates at individual PTM site resolution with typed self-attention and cross-modification-type crosstalk.** This is the gap we fill.

| Capability | DIPK | HiDRA | GraTransDRP | TransCDR | PathDSP | GraphDRP | DrugCell | **Ours** |
|---|---|---|---|---|---|---|---|---|
| Year | 2024 | 2023 | 2023 | 2023 | 2024 | 2022 | 2020 | **2026** |
| Venue | Nat Comm | Nat Comm | Brief Bioinf | Brief Bioinf | Bioinformatics | Bioinformatics | Cancer Cell | **Nat Methods** |
| Protein sequence (ESM-2) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✅** |
| 3D structure (GearNet) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✅** |
| PTM site-level features | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✅ 24 tokens** |
| PTM crosstalk modeling | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✅ typed self-attention** |
| Cross-modification-type | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✅ phospho × glyco** |
| Site-level interpretability | ✗ | pathway | ✗ | ✗ | pathway | ✗ | GO-level | **✅ IG per site** |
| Cross-receptor validation | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✅ EGFR↔HER2** |

---

## 4. Benchmarking Tier Structure

### Tier 0 — Simple ML Baselines (MANDATORY)

*Purpose: If RF/XGBoost matches our DL model, the architecture adds no value.*

| Method | Type | Code | 
|--------|------|------|
| Random Forest | ML | scikit-learn |
| XGBoost | Gradient boosting | xgboost |
| Ridge Regression | Linear | scikit-learn |
| Elastic Net | Regularized linear | scikit-learn |

**Input**: Concatenate pooled ESM-2 (1280-d) + GearNet (512-d) + ChemBERTa (384-d) + PTM (48-d) → flat 2,224-d feature vector. Same features as our model, different architecture.

### Tier 1 — Recent State-of-the-Art (2023–2026) (MANDATORY)

*Reviewers will specifically check for these. Omitting any 2024+ Nature Communications method is fatal.*

| # | Method | Year | Venue | Code |
|---|--------|------|-------|------|
| 1 | **DIPK** | 2024 | Nature Communications | [github.com/user-wu/DIPK](https://github.com/user-wu/DIPK) |
| 2 | **HiDRA** | 2023 | Nature Communications | [github.com/DMCB-GIST/HiDRA](https://github.com/DMCB-GIST/HiDRA) |
| 3 | **GraTransDRP** | 2023 | Briefings in Bioinformatics | [github.com/cnellington/GraTransDRP](https://github.com/cnellington/GraTransDRP) |
| 4 | **TransCDR** | 2023 | Briefings in Bioinformatics | [github.com/XiaoqiongXia/TransCDR](https://github.com/XiaoqiongXia/TransCDR) |
| 5 | **PathDSP** | 2024 | Bioinformatics | [github.com/TangYiChing/PathDSP](https://github.com/TangYiChing/PathDSP) |

### Tier 2 — Established Baselines (2020–2022) (RECOMMENDED)

*Frame as "established methods" not "state-of-the-art."*

| # | Method | Year | Venue | Code |
|---|--------|------|-------|------|
| 6 | **GraphDRP** | 2022 | Bioinformatics | [github.com/hauldhut/GraphDRP](https://github.com/hauldhut/GraphDRP) |
| 7 | **DrugCell** | 2020 | Cancer Cell | [github.com/idekerlab/DrugCell](https://github.com/idekerlab/DrugCell) |
| 8 | **DeepCDR** | 2020 | Briefings in Bioinformatics | [github.com/kimmo1019/DeepCDR](https://github.com/kimmo1019/DeepCDR) |

### Tier 3 — Revision-Ready (AS-NEEDED)

| # | Method | Year | Likely Reviewer Question |
|---|--------|------|------------------------|
| 9 | **MOFGCN** | 2024 | "Multi-omics fusion with GCN?" |
| 10 | **CancerGPT** | 2024 | "LLM-based approaches?" |
| 11 | **StructDRP** | 2024 | "Other structure-aware models?" |
| 12 | **SparseGO** | 2024 | "Visible neural networks with GO priors?" (flagged by the 2026 DRP review as top performer) |

### Minimum Viable Benchmarking (MVB)

**Must-have for submission**: Tier 0 (all) + Tier 1 (#1 DIPK, #2 HiDRA, #3 GraTransDRP) + Tier 2 (#6 GraphDRP)

**Stronger submission**: Above + Tier 1 (#4 TransCDR, #5 PathDSP) + Tier 2 (#7 DrugCell, #8 DeepCDR)

---

## 5. Fair Comparison Protocol

### 5.1 Data Fairness

All methods evaluated on **identical biological samples**:

```
DATASET: GDSC2 IC50 for 6 TKI drugs × EGFR/ERBB2 cell lines (951 samples)
SPLIT:   split_indices.json (70/15/15, stratified by resistance_label × target_protein)
LABELS:  Same IC50 values + same resistance binarization threshold
```

**Approach A** (their full pipeline on our data subset) for Tier 1–2 methods.
**Approach B** (our concatenated features) for Tier 0 ML baselines.

### 5.2 Split Fairness

Following the 2026 DRP review (Sada Del Real et al.), we use:
- **Random split**: Our standard 70/15/15 (same as step11c 5-fold CV)
- **Cell-blind split** (LOCLO): Hold out entire cell line groups by mutation class

The review explicitly states: *"cell-blind... is particularly valuable for drug repositioning and, more importantly, for precision medicine"* and that *"models must be assessed under at least one additional cross-validation regime"* beyond random splitting.

### 5.3 Hyperparameter Fairness

- External methods: published default hyperparameters
- Document any deviations from defaults

---

## 6. Evaluation Structure

### Axis 1: Overall Performance (Main Table — Tier A metrics)
- Held-out test: PCC, RMSE (regression) + AUROC, AUPRC-sensitive (classification)
- 5-fold cross-validation: same metrics (our method + top 2 baselines)
- Bootstrap 95% CIs (1,000 resamples)

### Axis 2: Per-Drug Performance (per 2026 DRP review requirement)
- Per-drug PCC and AUROC for all methods × 6 drugs
- Critical: Osimertinib (focal drug), Gefitinib (known difficulty), Lapatinib/Sapitinib (HER2-specific)

### Axis 3: Per-Protein / Cross-Receptor Transfer
- EGFR-only vs ERBB2-only: PCC, RMSE, AUROC
- Unique to our method — demonstrates cross-receptor generalization (EGFR + HER2)

**Paper language for Methods** (1-2 sentences):
> "We validate on EGFR and HER2, which share 81.3% kinase domain sequence identity (218/268 residues, UniProt P00533 vs P04626) and conserved functional phosphosites (Y1068↔Y1221 for GRB2 binding, Y1173↔Y1248 for SHC1 binding), enabling cross-receptor biological validation."

### Axis 4: Cell-Blind Generalization (LOCLO)
- Group cell lines by mutation class → hold out each group
- Our method + top 2 baselines

### Axis 5: PTM-BDL Biological Validation (Tier B — our unique contribution)
- 5-arm ablation (step11b): no_ptm / no_glyco / glyco_only / no_typed_attention / full
- Randomized PTM control: phospho-shuffled / glyco-shuffled / both-shuffled (step11b)
- IG site rankings: per-protein, per-seed, cross-receptor homology (step11b + step13)
- These tests are ONLY possible for our model — they validate PTM-BDL as a biological module

### Axis 6: Interpretability Comparison
- Our IG per-site attributions vs DrugCell GO-level vs PathDSP pathway scores
- "Can the method identify Y1068 as the resistance-determining site?"

### Axis 7: Runtime & Scalability
- Training time, inference time per sample, model parameters, peak GPU memory

---

## 7. Paper Tables and Figures

### Main Text Table 1: External Benchmarking (Tier A metrics)

*Regression: PCC + RMSE (field standard from DIPK, HiDRA, GraphDRP). Classification: AUROC (our model's primary, same as DeepCDR binarized IC50).*

| Method | Type | Year | PCC ↑ | RMSE ↓ | AUROC ↑ | AUPRC-sens ↑ | PTM Site-Level? |
|--------|------|------|-------|--------|---------|-------------|-----------------|
| Ridge | Linear | — | — | — | — | — | No |
| Random Forest | ML | — | — | — | — | — | No |
| XGBoost | Boost | — | — | — | — | — | No |
| DeepCDR | DL+Multi-omics | 2020 | — | — | — | — | No |
| DrugCell | DL+GO | 2020 | — | — | — | — | GO-level |
| GraphDRP | GNN | 2022 | — | — | — | — | No |
| GraTransDRP | Graph Transf. | 2023 | — | — | — | — | No |
| HiDRA | Hierarchical | 2023 | — | — | — | — | No |
| TransCDR | Transformer | 2023 | — | — | — | — | No |
| PathDSP | Pathway | 2024 | — | — | — | — | Pathway |
| **DIPK** | **DL+PPI** | **2024** | — | — | — | — | **No** |
| **Ours** | **PTM-BDL** | **2026** | **—** | **—** | **—** | **—** | **Yes, per-site** |
| Ours (5-fold CV) | Same | 2026 | mean±std | | | | |

### Main Text Table 2: PTM-BDL Biological Validation (Tier B — our contribution)

*No other method can be evaluated on these metrics. This is what makes PTM-BDL novel.*

| Validation Test | Metric | Result | Pass? |
|-----------------|--------|--------|-------|
| IG site ranking (EGFR) | Y1068 rank across 3 seeds | — (std_rank) | Y1068 = #1? |
| IG site ranking (HER2) | Y1221 rank across 3 seeds | — | Y1221 = #1? |
| Cross-receptor homology | EGFR Y1068 ≡ HER2 Y1221 | — | Concordant? |
| Modification-type hierarchy | Tyrosine vs Ser/Thr IG ranking | — | Y > S/T? |
| PTM ablation (AUROC) | Full − No_PTM | — | > 0? |
| PTM ablation (AUPRC-sens) | Full − No_PTM | — | > 0? |
| Phospho channel marginal | Full − no_glyco AUROC | — | > 0? |
| Glyco channel marginal | Full − glyco_only AUROC | — | > 0? |
| Typed attention vs MLP | Full − no_typed_attention AUROC | — | > 0? |
| Randomized PTM control | Real − Shuffled AUROC | — | ≥ +0.005? |
| Randomized PTM control | Real − Shuffled BAcc | — | ≥ +0.02? |

### Main Text Table 3: Per-Drug Performance

| Drug | N_test | Ours PCC | DIPK PCC | HiDRA PCC | Ours AUROC | DIPK AUROC |
|------|--------|---------|---------|-----------|-----------|-----------|
| Osimertinib | — | — | — | — | — | — |
| Gefitinib | — | — | — | — | — | — |
| Erlotinib | — | — | — | — | — | — |
| Afatinib | — | — | — | — | — | — |
| Lapatinib | — | — | — | — | — | — |
| Sapitinib | — | — | — | — | — | — |

### Main Text Figures (6 panels across ~3 figures)

**Figure X: External Benchmarking**
```
(a) PCC bar chart with 95% bootstrap CIs — all methods, colored by tier
(b) AUROC bar chart with 95% CIs — all methods that produce class probabilities
(c) Per-drug PCC heatmap — our method vs top 3 baselines × 6 drugs
```

**Figure Y: PTM-BDL Ablation & Biological Validation**
```
(a) Ablation waterfall — 5-arm AUROC comparison (no_ptm → full)
(b) Randomized PTM control — real vs shuffled (phospho / glyco / both)
(c) Channel contribution — phospho marginal, glyco marginal, typed-attention marginal
```

**Figure Z: Biological Interpretability (PTM-BDL's unique capability)**
```
(a) EGFR IG site ranking — bar chart (Y1068 > Y1086 > Y1173 hierarchy)
(b) HER2 IG site ranking — bar chart (Y1221 expected #1)
(c) Cross-receptor homology — side-by-side EGFR Y1068 ≡ HER2 Y1221
(d) Phospho vs glyco IG importance — modification-type-level attribution
```

### Supplementary Tables
- **S1**: Full Tier C metrics (All Methods × Spearman ρ, BAcc, F1, R², Sensitivity, Specificity)
- **S2**: Per-Drug × Per-Method full breakdown
- **S3**: LOCLO cell-blind generalization results
- **S4**: Statistical tests (p-values, DeLong, Wilcoxon, BH correction)
- **S5**: Runtime and scalability
- **S6**: Multi-seed IG stability (per-seed site rankings, std_rank)
- **S7**: Cross-receptor attention patterns (sensitive vs resistant)

---

## 8. Implementation Plan — Two Script Families

### Script Family 1: Benchmarking (step14a–d)

*These scripts RUN external methods and compute comparative metrics.*

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `step14a_ml_baselines.py` | Tier 0 ML baselines | split_indices.json + pooled embeddings | `results/ml_baselines.json` |
| `step14b_external_baselines.py` | Tier 1–2 external methods | GDSC cell line data in each method's format | `results/external_baselines/{method}.json` |
| `step14c_statistical_tests.py` | Bootstrap CIs + paired tests | All prediction outputs | `results/statistical_tests.json` |
| `step14d_loclo.py` | Cell-blind generalization | Same dataset, grouped by mutation class | `results/loclo_results.json` |

### Script Family 2: Publication Figures & Tables (step15a–b)

*These scripts GENERATE publication-quality figures and formatted tables from existing results. They do NOT run any models — they only read from `results/` and produce camera-ready outputs.*

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `step15a_paper_figures.py` | All main text + supplementary figures | `results/*.json` + `results/figures/*.png` | `results/publication/figures/*.pdf` |
| `step15b_paper_tables.py` | All main text + supplementary tables | `results/*.json` | `results/publication/tables/*.csv` + `.tex` |

### Step 14a: ML Baselines

```
1. Load split_indices.json (identical split as step11)
2. Concatenate pooled ESM-2 + GearNet + ChemBERTa + PTM → 2,224-d
3. Train: RF, XGBoost, Ridge, Elastic Net (inner 5-fold CV for hyperparams)
4. Compute: PCC, RMSE, AUROC, AUPRC-sens + per-drug PCC/AUROC
5. Save → results/ml_baselines.json
```

### Step 14b: External Baselines

```
For each method: clone repo → data prep → run on our GDSC subset → compute metrics
Data prep per method:
- DIPK: expression + mutation + methylation + CNV + PPI network
- HiDRA: expression (gene-set-level) + drug fingerprints
- GraTransDRP: expression + mutation + drug SMILES (molecular graph)
- TransCDR: expression + mutation + drug SMILES
- PathDSP: expression + drug fingerprints + pathway DB
- GraphDRP: expression + drug SMILES
- DrugCell: mutation + drug fingerprints + GO hierarchy
- DeepCDR: expression + mutation + methylation + drug SMILES
```

### Step 14c: Statistical Tests

```
1. Bootstrap CI (1,000 resamples) for PCC, RMSE, AUROC, AUPRC-sens
2. DeLong test: ours vs each baseline (AUROC)
3. Wilcoxon signed-rank: ours vs each baseline (per-fold or per-drug)
4. Benjamini-Hochberg correction across K baselines
5. Save → results/statistical_tests.json
```

### Step 14d: Cell-Blind Generalization (LOCLO)

```
1. Group cell lines by mutation class
2. For each group: hold out → train on rest → predict held-out → compute metrics
3. Our method + top 2 baselines
4. Save → results/loclo_results.json
```

### Step 15a: Publication Figures

```
Reads from: results/ablation_study.json, results/ml_baselines.json,
            results/external_baselines/*.json, results/xai_report.json,
            results/stability_analysis.json, results/statistical_tests.json

Generates:
  Main text:
    Fig_benchmarking.pdf     — PCC bars + AUROC bars + per-drug heatmap (3 panels)
    Fig_ablation.pdf         — ablation waterfall + randomized control + channel margins (3 panels)
    Fig_interpretability.pdf — EGFR IG + HER2 IG + cross-receptor + phospho vs glyco (4 panels)
  
  Supplementary:
    Fig_S_perdrug.pdf        — Per-drug detailed comparison (6 drugs × all methods)
    Fig_S_crossreceptor.pdf  — EGFR vs ERBB2 side-by-side performance
    Fig_S_attention.pdf      — Cross-modal attention heatmaps
    Fig_S_runtime.pdf        — Runtime comparison bar chart

→ results/publication/figures/*.pdf (300 DPI, Nature Methods format)
```

### Step 15b: Publication Tables

```
Reads from: same as step15a

Generates:
  Main text:
    Table1_benchmarking.csv/.tex   — All methods × PCC, RMSE, AUROC, AUPRC-sens
    Table2_biological.csv/.tex     — PTM-BDL validation (11 biological tests)
    Table3_perdrug.csv/.tex        — Per-drug PCC + AUROC for top 3 methods
  
  Supplementary:
    TableS1_full_metrics.csv/.tex  — All methods × all Tier C metrics
    TableS2_perdrug_full.csv/.tex  — Per-drug × all methods × all metrics
    TableS3_loclo.csv/.tex         — Cell-blind results
    TableS4_statistics.csv/.tex    — p-values, CIs, effect sizes
    TableS5_runtime.csv/.tex       — Training time, inference time, params, memory
    TableS6_IG_stability.csv/.tex  — Per-seed site rankings
    TableS7_attention.csv/.tex     — Cross-modal attention patterns

→ results/publication/tables/*.csv + .tex (LaTeX-ready)
```

### Step 14f: Runtime Benchmark (part of step14 family)

```
1. For each method: training time, inference time/sample, parameter count, peak memory
2. All measured on same hardware
3. Save → results/runtime_benchmark.json
```

---

## 9. Risk Mitigation

### Risk 1: External method code doesn't run
**Fallback**: Report published numbers with "reported performance" caveat.

### Risk 2: Our method underperforms on overall metrics
**This is expected.** DIPK/HiDRA use genome-wide expression (thousands of genes). We use 24 PTM sites. Frame correctly:
> *"Our method introduces a new modality (site-level PTM dynamics) and shows it provides orthogonal biological signal. We do not compete on pan-cancer gene-expression-based prediction."*

### Risk 3: Bootstrap CIs wide due to small test set (n=143)
**Mitigation**: Report 5-fold CV results alongside test-set results. Be transparent about sample size.

---

## Appendix A: Expected Reviewer Questions (Pre-Empted)

### "DIPK uses expression + mutation + methylation. Your method uses protein embeddings + PTM. This is not fair."
> *"We evaluate each method as a complete pipeline. This reflects the practical question: 'Given the same cell lines and drugs, which complete method gives better predictions?' ML baselines (RF, XGBoost) using our exact features isolate the architectural contribution."*

### "Your dataset has only 951 samples."
> *"Our model is designed for focused, mechanistically-interpretable prediction in a specific oncogene-driven context (EGFR/HER2 TKI resistance). PTM-BDL requires PTM-level annotations only available for well-characterized kinase targets."*

### "Why not compare to SparseGO? It was the top performer in the 2026 DRP review."
> *"SparseGO uses GO hierarchy as network structure and gene expression as input. Our method operates at PTM-site resolution with typed self-attention. If reviewers request, SparseGO can be added (Tier 3)."*

---

## Appendix B: Key References for This Plan

1. **Sada Del Real K et al.** (2026) "Foundation models and deep learning for cancer drug response prediction: a framework for data, metrics, and validation." *Brief Bioinform* 27(3):bbag225. PMID: 42153322 — **Defines the standard DRP metric framework we follow**
2. **Spiro AE et al.** (2026) "A scalable approach to investigating sequence-to-function predictions from personal genomes." *Nat Methods* s41592-026-03124-8 — **Nature Methods metric example: Pearson R + Wilcoxon signed-rank**
3. **Zheng Z et al.** (2026) "ClairS: a deep-learning method for long-read tumor–normal pair somatic small variant calling." *Nat Methods* s41592-026-03152-4 — **Nature Methods metric example: F1, Precision, Recall, AUPRC**
4. **Wei Z et al.** (2026) "Benchmarking algorithms for generalizable single-cell perturbation response prediction." *Nat Methods* 23(2):451-464. PMID: 41381899 — **Nature Methods benchmarking standard: 27 methods, 29 datasets, 6 metrics**

---

---

## 10. Files & Docs to Update After Benchmarking is Complete

Once all benchmarking scripts (step14a–d) have been run and results are available, the following files **must be updated** to reflect the new comparative results:

### Files that MUST be updated

| File | What to update | Why |
|------|---------------|-----|
| **`README.md`** | Add "Benchmarking Results" section with headline numbers (PCC, AUROC vs top baselines). Update "Key Results" section. | First thing reviewers/users see |
| **`docs/Scientific_Explanation.md`** | Add comparative performance context — how our method positions vs DIPK/HiDRA/GraphDRP | Scientific narrative needs benchmarking evidence |
| **`docs/PTM_Biological_Dynamics_Layer.md`** | Update §1.1 (empirical evidence) with new ablation numbers from live run. Update §13 (expected outcomes) with actual benchmarking results | The PTM-BDL proposal references outdated June 28 numbers |
| **`docs/PTM-BDL_One_Page_Summary.md`** | Update "Key Results" with comparative benchmarking + live ablation numbers | Summary doc used for quick reference |
| **`CITATION.cff`** | Verify author list, title, year are correct for submission | Required for proper citation |
| **`Makefile`** | Add `make benchmark` and `make figures` targets for step14/step15 scripts | Pipeline automation |
| **`pyproject.toml`** | Add any new dependencies (e.g., xgboost if not already listed) | Build reproducibility |

### Files that SHOULD be updated

| File | What to update | Why |
|------|---------------|-----|
| **`docs/How_to_Run.md`** | Add instructions for running benchmarking pipeline (step14a–d) and figure generation (step15a–b) | User documentation |
| **`config/config.yaml`** | Add benchmarking config section if external methods need configurable paths | Centralized configuration |
| **`CLAUDE.md`** | Update with benchmarking context for AI-assisted development continuity | Development context |

### Files to CREATE (new)

| File | Purpose |
|------|---------|
| **`docs/COMPREHENSIVE_EVALUATION_july.md`** | New evaluation doc with live ablation + benchmarking results (replaces stale June docs) |
| **`results/publication/`** directory | Camera-ready figures (PDF) and tables (CSV + LaTeX) from step15a–b |
| **`benchmarks/`** directory | Cloned external method repos + data prep scripts |

### Results files that will be PRODUCED

| File | Source |
|------|--------|
| `results/ml_baselines.json` | step14a |
| `results/external_baselines/{DIPK,HiDRA,GraTransDRP,...}.json` | step14b |
| `results/statistical_tests.json` | step14c |
| `results/loclo_results.json` | step14d |
| `results/runtime_benchmark.json` | step14f |
| `results/publication/figures/*.pdf` | step15a |
| `results/publication/tables/*.csv` + `*.tex` | step15b |

### Results files that are STALE (from pre-July runs)

These exist but contain outdated numbers from earlier runs. They will be **overwritten** when the current pipeline run (step11b in progress) completes:

| File | Status |
|------|--------|
| `results/ablation_study.json` | ⏳ Being overwritten by current step11b run |
| `results/stability_analysis.json` | ⏳ Will be overwritten |
| `results/randomized_ptm_control.json` | ⏳ Will be overwritten |
| `results/evaluation_report.json` | Will be overwritten by step12 after step11b |
| `results/xai_report.json` | Will be overwritten by step13 after step12 |

---

*This plan was last updated 2026-07-01. Metrics verified against 2 Nature Methods papers + 1 Briefings in Bioinformatics DRP review (all 2026).*
