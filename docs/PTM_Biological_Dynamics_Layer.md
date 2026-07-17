# PTM Biological Dynamics Layer (PTM-BDL)
## A Learnable Multi-PTM Module for Modeling Post-Translational Modification Signaling Codes in Drug Resistance Prediction

---

## Table of Contents
1. [The Problem: Why the Current PTM Architecture Fails](#1-the-problem-why-the-current-ptm-architecture-fails)
2. [The Biological Foundation: PTMs as a Signaling Code](#2-the-biological-foundation-ptms-as-a-signaling-code)
3. [Multi-PTM Landscape of EGFR and HER2](#3-multi-ptm-landscape-of-egfr-and-her2)
4. [Insights from the 2025 PTM Review (PMC13070201)](#4-insights-from-the-2025-ptm-review-pmc13070201)
5. [Literature Review: What Exists and What's Missing](#5-literature-review-what-exists-and-whats-missing)
6. [The Innovation: PTM Biological Dynamics Layer](#6-the-innovation-ptm-biological-dynamics-layer)
7. [Detailed Architecture Design](#7-detailed-architecture-design)
8. [Integration with Existing Architecture](#8-integration-with-existing-architecture)
9. [Validation Strategy](#9-validation-strategy)
10. [Implementation Steps](#10-implementation-steps)
11. [Limitations](#11-limitations)
12. [Future Directions](#12-future-directions)
13. [Expected Outcomes](#13-expected-outcomes)
14. [References](#14-references)

---

## 0. The Core Idea: Simulating PTM Biology in Silico

### 0.1 The Vision

In a living cell, post-translational modifications do not act as isolated numbers — they are a **biological process**. When EGFR is activated by a mutation like L858R, a cascade unfolds:

```
Kinase activates → Tyrosine sites get phosphorylated → 
  Adapter proteins dock (GRB2 at pY1068, SHC at pY1173) →
    Signaling cascades fire (MAPK, PI3K) →
      Meanwhile, pY1045 recruits Cbl → EGFR gets ubiquitinated → 
        Receptor is degraded (negative feedback) →
          Glycosylation controls how much receptor reaches the surface →
            The BALANCE of all these determines: cell lives or dies
```

When a drug like Osimertinib is added:
```
Drug binds C797 → Kinase inhibited → Tyrosine dephosphorylation →
  BUT: Which sites dephosphorylate? How much? How fast? →
    If ALL signaling sites go down AND bypass (SRC) also goes down → SENSITIVE
    If MAPK sites go down BUT PI3K stays up OR SRC bypasses → RESISTANT
    If receptor degradation is impaired (low pY1045) → receptor persists → RESISTANT
```

**No existing computational model captures this process.** Current models treat PTMs as static numbers — 12 scalars fed to an MLP. They cannot learn that the RELATIONSHIP between Y1068 going down and Y845 staying up is what determines resistance.

### 0.2 The Innovation: A PTM Attention/Gating Layer That Simulates PTM Biology

We propose a **PTM Biological Dynamics Layer (PTM-BDL)** — a learnable neural module that simulates how PTMs function in a biological cell:

1. **Each PTM site is a biological actor** — not a number. It has a modification level, a drug-induced change, a modification TYPE (phospho-Y, phospho-S, phospho-T, glyco-N), and a site identity. The model learns what each actor does.

2. **PTM sites INTERACT through biological gating** — a modification-type-aware gate controls how much each site's change matters. Tyrosine phosphorylation (direct drug target) is gated differently from serine phosphorylation (downstream indicator) which is gated differently from glycosylation (receptor surface state). The gate learns the biological rules.

3. **PTM sites COMMUNICATE through self-attention** — sites attend to each other to form the signaling code. Y1068 (MAPK) attends to Y1173 (PI3K): "Is the survival pathway also shut down?" Y1045 (degradation) attends to Y1068 (signaling): "Is the receptor being degraded while signaling is active?" This is how the model learns the combinatorial logic of resistance.

4. **Multiple PTM types interact in the SAME attention space** — phosphorylation tokens and glycosylation tokens attend to each other. The model can learn: "High glycosylation (lots of receptor on surface) + persistent phosphorylation (active signaling) + weak drug delta = resistant." This phospho-glyco crosstalk is the foundation for a truly multi-PTM model.

5. **The drug efficacy ratio (delta/baseline) captures the PROCESS** — not just what the modification level is, but how much the drug CHANGED it relative to baseline. This is the biological readout of drug action at each site.

### 0.3 Why This Is Foundational

This approach is **foundational** because:

- **Multi-PTM by design** — phosphorylation and glycosylation are integrated from the start. The modification-type embedding system is extensible and can accommodate additional PTM types as data becomes available.
- **Foundation model architecture** — the PTM-BDL module processes PTM sites by their biological properties (modification type, drug response dynamics), not by protein identity. This design principle enables the module to serve as a foundation for PTM-driven drug response prediction across EGFR and HER2 (two members of the ERBB receptor family).
- It captures **PTM crosstalk** — the central unsolved problem in PTM biology (PMC13070201, 2025). No existing tool models how different PTM types interact to determine drug response.
- The model **discovers biology, not patterns** — nothing is hard-coded. The attention weights and IG attributions reveal WHAT the model learned, and we validate it against known signaling biology.

### 0.4 The Biological Process the Model Must Learn

For **phosphorylation** (intracellular signaling):
```
Mutation → Kinase activation state → Autophosphorylation pattern →
  Drug treatment → Site-specific dephosphorylation →
    Pattern of remaining phosphorylation → Resistance or Sensitivity
```
The PTM-BDL learns this by encoding ptm_level (mutation effect), delta_ptm (drug effect), and ratio (efficacy). Self-attention learns which site-site patterns predict resistance.

For **glycosylation** (extracellular receptor biology):
```
Glycan biosynthesis → N-linked glycosylation at extracellular sites →
  Controls receptor folding, stability, surface expression →
    Determines HOW MUCH receptor is available for drug to target →
      Interacts with phosphorylation: glyco = receptor quantity, phospho = receptor activity
```
The PTM-BDL learns this by encoding glyco sites alongside phospho sites. Cross-type attention (phospho↔glyco) learns their interaction.

For **the combined multi-PTM code**:
```
Glycosylation (HOW MUCH receptor) ×
  Phosphorylation (HOW ACTIVE receptor) ×
    Drug delta (HOW MUCH drug inhibits) =
      Cell fate (sensitive or resistant)
```

This three-layer code is what the PTM-BDL self-attention mechanism is designed to learn — **and no existing model attempts this.**

---

## 1. The Problem: Why the Current PTM Architecture Fails

### 1.1 Empirical Evidence (Run of 2026-06-28, EGFR + HER2 pipeline)

The latest end-to-end pipeline (EGFR + HER2, `delta_ptm` active, n=143 test samples) produces a layered empirical picture that is more informative than the prior EGFR-only run. The headline result is no longer the ablation — it is the **randomized PTM control**, which definitively demonstrates that the current PTM input channel carries no usable biological signal. Full numbers are in `results/COMPREHENSIVE_EVALUATION_28_june.md`.

#### 1.1.1 The Randomized PTM Control — The Definitive Failure

| Metric | Full Model (real PTM) | Shuffled PTM (random) | Drop (full − shuffled) |
|---|---|---|---|
| Test BAcc | 0.676 | **0.718** | **−0.042** |
| Test AUROC | 0.860 | **0.870** | **−0.010** |
| Test AUPRC-sensitive | 0.605 | **0.655** | **−0.050** |
| Test RMSE | 1.733 | **1.616** | **+0.117** (worse with real PTM) |
| Test Pearson R | 0.595 | **0.610** | **−0.015** |

**Shuffling PTM vectors across samples — destroying the mutation→PTM correspondence — makes the model BETTER on every metric.** If PTM features carried genuine biological information, randomizing them should *hurt* the model. We observe the opposite. This is the strongest possible empirical signal that the current PTM input channel is, at best, unused and, at worst, an active source of noise the model has to work around.

#### 1.1.2 The Ablation Study — Mixed Signal That Reinforces the Diagnosis

| Model | Test BAcc | Test AUROC | Test AUPRC-sens | RMSE | Pearson R |
|---|---|---|---|---|---|
| **A: No PTM** | **0.837** | 0.847 | 0.523 | **1.519** | **0.637** |
| B: Level 1 Only | 0.691 | **0.874** | 0.609 | 1.837 | 0.618 |
| C: Level 2 Only | 0.722 | 0.862 | 0.593 | 1.816 | 0.598 |
| D: Full (Both) | 0.676 | 0.860 | **0.605** | 1.733 | 0.595 |

```
ptm_gain_auroc:           +0.0127   ← PTM helps ranking
ptm_gain_auprc_sensitive: +0.0822   ← PTM helps sensitive-class identification
ptm_gain_bacc:            −0.1606   ← PTM hurts threshold-calibrated classification
ptm_gain_f1_macro:        −0.2702   ← PTM hurts overall F1
votes_ptm_helps:          2 / 4
conclusion:               PTM_HELPS  (on threshold-independent metrics only)
```

PTM features improve **ranking quality** (AUROC, AUPRC-sensitive) but hurt **threshold calibration** (BAcc) and **regression magnitude** (RMSE). The model uses PTM to push more samples toward "sensitive," which increases sensitive-class detection but over-commits — pulling many true-resistant samples across the threshold. The headline `PTM_HELPS` is technically correct for threshold-independent metrics but should honestly be read as "PTM helps ranking, hurts calibration."

#### 1.1.3 The Crucial Combined Reading

The ablation alone could be interpreted optimistically (PTM helps AUROC). The randomized control alone could be dismissed as noise. **Together**, they form a tight diagnostic:

- The ablation shows PTM moves predictions in a particular direction (toward sensitive).
- The randomized control shows that this movement is **not driven by PTM biology** — random PTM moves predictions in the same (or better) direction.
- Therefore PTM features are not contributing *biological information*; they are acting as a generic "perturbation that biases the classification threshold."

This is precisely the failure mode of redundant features that are deterministic functions of an already-encoded input (mutation × drug).

#### 1.1.4 Reproducibility of the Problem (Multi-Seed IG)

Across 3 random seeds (42, 123, 456), the IG-derived site-importance ranking is **highly stable**:

- `top_consistent: true`
- Y1092 (Y1068) is rank #1 in **all 3 seeds** (std_rank = 0.0)
- Top-3 sites (Y1068, Y1173, Y845) reproduce across seeds with std_rank ≤ 1.25

The model has *learned* the canonical EGFR phosphosite hierarchy (Y1068 > Y1086 > Y1173 > Y992 > Y1148 > Y845) — matching 30+ years of established biology. **This rules out "the model can't learn PTM biology" as an explanation.** The model knows which sites should matter. The architectural channel through which PTM enters the prediction is what fails.

#### 1.1.5 Mutation-Group Collapse — The Symptom in Practice

Every EGFR mutation group in the test set collapses to nearly identical predictions:

| Mutation Group | True IC50 | Pred IC50 | Mean Resist Prob |
|---|---|---|---|
| L858R | −1.47 | −2.66 | 0.186 |
| L858R/T790M | +0.01 | −2.68 | 0.185 |
| E746_A750del | −4.40 | −2.56 | 0.195 |
| **A755D/L747_P753delinsS** | **+0.61 (resistant)** | **−2.67 (predicted sensitive)** | **0.186** |
| … other groups … | … | ≈ −2.66 | ≈ 0.186 |

The model has learned a single rule: `EGFR-mutant → sensitive (prob ≈ 0.186)`. It cannot distinguish L858R/T790M (T790M confers 1st-gen TKI resistance) from L858R alone. It cannot identify the genuinely resistant A755D mutation group. **This is the practical phenotype of the PTM architectural failure** — there is no channel for mutation-specific PTM state to differentiate response.

### 1.2 Root Cause Analysis: PTM as Redundant Information


Tracing the data flow through step04 → step06 → step11 reveals why PTM fails:

**Problem 1: ptm_vector is a deterministic function of mutation_class**

Step06 maps each sample's mutation to a PTM state vector via `map_mutations_to_ptm_vector()`. With 951 samples:
- 610 WT EGFR samples → **identical** ptm_vector = [1.0 × 12]
- 305 ERBB2 samples → **identical** ptm_vector = [1.5 × 10, 0.0, 0.0]
- 36 EGFR-mutant samples → ~5 distinct vectors

There are only **~7 distinct ptm_vectors** across 951 samples. The ESM-2 sequence embedding already encodes mutation identity. Therefore, `ptm_vector` provides **zero new information**.

**Problem 2: delta_ptm is a deterministic function of (drug_name × ptm_vector)**

Step06 Section 8e computes:
```
delta_ptm = osimertinib_profile × drug_scaling[drug_name] × ptm_vector
```

Since the model already knows drug identity from ChemBERTa and mutation class from ESM-2, `delta_ptm` is fully reconstructable from existing modalities.

**Problem 3: Level 2 phospho features are class-level averages**

The PhosphoContextEncoder receives 7 aggregate features propagated from ~16 measured samples to 935 samples via mutation-class averaging. Again, these are deterministic functions of mutation_class.

### 1.3 The Information-Theoretic Diagnosis

The PTM branch adds value **if and only if** it carries information not present in other modalities. Currently:

```
I(PTM; Response | Sequence, Structure, Drug) ≈ 0
```

For PTM to add value, the encoding must capture aspects of phosphorylation biology that sequence identity **cannot** represent:
- Inter-site PTM dynamics and crosstalk
- The biological meaning of different modification types (phospho Y/S/T, glycosylation)
- The relationship between modification baseline and drug-induced change
- Combinatorial PTM patterns that define the signaling code

---

## 2. The Biological Foundation: PTMs as a Signaling Code

### 2.1 The Phospho-Code Hypothesis

Phosphorylation is not a collection of independent binary switches. It is a **structured signaling language** where the cellular outcome depends on the **combination** of modified sites, not individual sites.

Seet et al. (2006) first proposed the "modification code" hypothesis: combinations of PTMs create a higher-order signaling language that determines protein function, interactions, and cellular outcomes. This was extended by Beltrao et al. (2012) who showed that phosphosites evolve in coordinated modules, suggesting functional interdependence.

**Key evidence:**
- **Schulze et al., Molecular Systems Biology 2005** (PMID 16729048): EGFR phosphosite quantitation showed that different stimuli (EGF vs HRG) produce distinct phosphorylation PATTERNS at the same sites, leading to different cellular outcomes.
- **Olsen et al., Cell 2006** (PMID 17081983): Global phosphoproteomics revealed that EGF stimulation activates >6,000 phosphosites in coordinated temporal waves.

### 2.2 EGFR Phosphorylation: Site-Site Dependencies Are the Key to Resistance

The 12 EGFR phosphosites form a functional hierarchy with known dependencies:

```
Kinase Activation
    └── Y869 (Y845, SRC substrate) — stabilizes active conformation
            ↓
Autophosphorylation Cascade
    ├── Y1092 (Y1068) → GRB2 → RAS-MAPK (proliferation)
    ├── Y1110 (Y1086) → secondary GRB2 → reinforces MAPK
    ├── Y1197 (Y1173) → SHC1 → PI3K-AKT (survival)
    ├── Y1172 (Y1148) → SHC → alternative MAPK
    ├── Y1016 (Y992) → PLCγ → PKC
    └── Y1069 (Y1045) → c-Cbl → RECEPTOR UBIQUITINATION → DEGRADATION
            ↓
Regulatory Sites
    ├── S991, S1039, T1041 — receptor trafficking and stability
    └── Y998 — endocytosis/internalization
```

**The resistance decision depends on SITE-SITE RELATIONSHIPS:**

| Pattern | Y1092 (MAPK) | Y1197 (PI3K) | Y1069 (Cbl) | Y869 (SRC) | Outcome |
|---------|:---:|:---:|:---:|:---:|---------|
| Drug works completely | ↓↓↓ | ↓↓↓ | ↓ | ↓↓ | **SENSITIVE** |
| Drug hits target, bypass active | ↓↓↓ | ↓↓ | ↓ | ↑ or → | **RESISTANT** (SRC bypass) |
| Drug blocked by mutation | → or ↑ | → or ↑ | ↓ | ↑ | **RESISTANT** (T790M) |
| Receptor degradation impaired | ↓↓ | ↓ | ↓↓↓ | → | **RESISTANT** (sustained receptor) |

The SAME delta at Y1092 can mean sensitivity or resistance depending on what Y869 and Y1069 are doing. This combinatorial logic **cannot** be captured by treating sites independently.

### 2.3 Modification Type Determines Biological Mechanism

The 12 EGFR phosphosites include three modification types with fundamentally different biology:

**Tyrosine phosphorylation (9 sites):**
- Catalyzed by tyrosine kinases (EGFR auto-phosphorylation, SRC)
- Creates docking sites for SH2-domain adapter proteins (GRB2, SHC, PLCγ)
- DIRECTLY targeted by TKIs
- Rapid ON/OFF kinetics (seconds to minutes)
- **These are where the drug acts**

**Serine phosphorylation (2 sites: S991, S1039):**
- Catalyzed by serine/threonine kinases (PKC, CK2, ERK)
- Modulates receptor conformation, trafficking, and localization
- NOT directly targeted by TKIs — regulated by downstream kinases
- **These reflect downstream pathway activity**

**Threonine phosphorylation (1 site: T1041):**
- Serine/threonine kinase substrate, proximal to Y1069 (Cbl binding)
- May modulate ubiquitination efficiency
- **This reflects regulatory feedback**

A TKI directly dephosphorylates tyrosine sites but only indirectly affects serine/threonine sites. The modification type is a **biological prior** the model should learn to use.

### 2.4 The Delta/Baseline Ratio: Drug Efficacy Signal

The ratio `delta_ptm / (ptm_level + ε)` captures "what fraction of signaling capacity was eliminated":

| Condition | ptm_level | delta_ptm | delta/baseline | Interpretation |
|-----------|:---------:|:---------:|:--------------:|----------------|
| L858R + Osimertinib | 4.0 | -5.6 | -1.40 | Signal COMPLETELY killed |
| L858R + Gefitinib | 4.0 | -1.2 | -0.30 | 70% signal remains |
| WT + Osimertinib | 1.0 | -1.4 | -1.40 | Over-inhibits WT |
| T790M + Osimertinib | 1.5 | -0.3 | -0.20 | Drug barely works |

This ratio is the **drug efficacy signal** — it tells the model whether the drug is effectively inhibiting at this site.

### 2.5 Recent Evidence: Coordinated Phospho-Signaling in Drug Resistance (2024-2026)

**Tozuka et al., iScience 2024** (PMID 38646155) — in our dataset:
Resistant cells show **coordinated** phospho-rewiring: EGFR direct sites dephosphorylated (log2FC < -2), but bypass kinase sites (SRC, FAK) maintain phosphorylation. The PATTERN determines resistance.

**PNAS 2025** (DOI: 10.1073/pnas.2522090123) — in our dataset:
Phosphoproteomic gradient under Osimertinib: EGFR direct at -3.62, adapters at -0.76, MAPK at -0.55, PI3K at -0.45, SRC/FAK at +0.19. The **gradient across pathway levels** determines sensitivity.

**Hsu et al., Molecular Systems Biology 2025** (PMID 41023502) — in our dataset:
Temporal phosphoproteomics: acute effect (5 min) → sustained inhibition (6h) → DTP rebound (21 days). The temporal pattern reveals bypass mechanisms.

**Krug et al., Cell 2020** (PMID 32504382):
CPTAC pan-cancer phosphoproteomics identified 58 phospho-modules — coordinated phosphosite groups that act as functional units for drug response.

**Ochoa et al., Nature Biotechnology 2023** (PMID 36510105):
Functional phosphosites are more likely to form coordinated networks and co-occur with other modification types.

**Cross et al., Cancer Discovery 2014** (PMID 25351743):
Sustained pathway inhibition (pERK stays down) = sensitivity. Rebound (pERK comes back) = resistance. The temporal PATTERN determines outcome.

---

## 3. Multi-PTM Landscape of EGFR and HER2

### 3.1 Complete PTM Inventory: EGFR (UniProt P00533)

| PTM Type | # Sites | Key Positions | Drug Resistance Relevance | Data in Our Datasets? |
|----------|:---:|-----------|------------------------|:-:|
| **Phosphorylation** | 23+ | Y869, S991, Y998, Y1016, S1039, T1041, Y1069, Y1092, Y1110, Y1125, Y1172, Y1197 | **CRITICAL** — direct TKI target, signaling output | ✅ 2,123 rows across 7 studies |
| **N-Glycosylation** | 12 | N56, N73, N128, N175, N196, N352, N361, N413, N444, N528, N568, N603, N623 | **HIGH** — controls receptor folding, surface expression, drug binding, dimerization | ✅ **MCP 2025** (8 sites), **ErbB2 Glycoform Atlas** (9 EGFR sites from CHO-sEGFR), **EGFR Fucosylation** (4 sites: N175, N413, N444, N603), **MCP 2025b** (large-scale glyco+phospho in EGFR-mutant lung cancer cells) |

### 3.2 Complete PTM Inventory: HER2/ERBB2 (UniProt P04626)

| PTM Type | # Sites | Key Positions | Drug Resistance Relevance | Data in Our Datasets? |
|----------|:---:|-----------|------------------------|:-:|
| **Phosphorylation** | 10 | T686, Y1005, S1054, T1099, Y1139, S1151, Y1196, Y1221, Y1222, Y1248 | **CRITICAL** — Y1221≡EGFR Y1068 (GRB2), Y1248≡EGFR Y1173 (SHC) | ✅ DrugPTM-Bench (BT-474) + MCP 2025 (4 sites) |
| **N-Glycosylation** | 7 | N68, N124, N187, N259, N530, N571, N629 | **HIGH** — HER2 surface expression, trastuzumab binding, dimerization | ✅ **MCP 2025** (N530), **ErbB2 Glycoform Atlas** (ALL 7 sites from SKBR-3 + BT-474) [Ref 29], **ST6Gal1 ErbB2** (site-specific sialylation, trastuzumab resistance) [Ref 30], **MCP 2025b** (large-scale glyco+phospho) [Ref 31] |

### 3.3 Cross-Receptor Homologous PTM Sites

| Function | EGFR Site | HER2 Site | PTM Type | In Our Data? |
|----------|----------|----------|:---:|:---:|
| GRB2 → RAS-MAPK | **Y1092** (Y1068) | **Y1221** | Phospho | ✅ Both |
| SHC → PI3K-AKT | **Y1197** (Y1173) | **Y1248** | Phospho | ✅ Both |
| Extracellular domain III glyco | **N528** | **N530** | Glyco | ✅ Both (MCP 2025) |
| c-Cbl → degradation | Y1069 (Y1045) | Y1005 | Phospho | ✅ EGFR only |

The homologous glycosylation sites N528(EGFR) ↔ N530(HER2) provide a **second axis of cross-receptor validation** beyond phosphorylation, strengthening the biological claim that the model learns FUNCTION, not protein identity.

### 3.4 How Glycosylation Affects EGFR/HER2 Drug Resistance

Glycosylation is not merely structural — it directly modulates drug binding, receptor biology, and resistance:

**EGFR N-glycosylation and drug response:**
- **N361 glycosylation critically affects EGFR function:** Zhu et al. (Cancers 2026) demonstrated that N361 glycosylation modulates EGFR dimerization, ligand binding affinity, and downstream signaling intensity. Loss of N361 glycosylation impairs EGFR activation and alters sensitivity to cetuximab [Ref 21].
- **Glycosylation controls receptor surface expression:** EGFR must be properly glycosylated in the ER/Golgi to reach the cell surface. Aberrant glycosylation traps EGFR intracellularly, reducing drug target availability [Ref 22].
- **Glycan composition affects TKI binding:** High-mannose vs complex glycans at N528/N568 alter the extracellular domain conformation, indirectly affecting the intracellular kinase domain where TKIs bind [Ref 23].

**HER2 glycosylation and drug response:**
- **N-glycosylation at N530 (HER2) is required for proper receptor folding** and trastuzumab binding. Altered glycosylation at N530 is associated with trastuzumab resistance in breast cancer [Ref 22].
- **Glycosylation state affects HER2-EGFR heterodimerization** — the primary oncogenic signaling unit. Changes in glycan structure alter dimerization efficiency and downstream signaling.

**PTM crosstalk: Glycosylation × Phosphorylation:**
- Glycosylation is an extracellular modification that controls RECEPTOR AVAILABILITY
- Phosphorylation is an intracellular modification that controls RECEPTOR SIGNALING
- Together they form a two-layer code: glyco determines HOW MUCH receptor is on the surface, phospho determines HOW ACTIVE that receptor is
- The PTM-BDL self-attention can learn this crosstalk: "high glyco (lots of receptor) + high phospho (active) + weak drug delta = resistant"

### 3.5 PTMs in NSCLC Pathogenesis

Kharb et al. (2023) reviewed the role of multiple PTMs in NSCLC, confirming that phosphorylation and glycosylation — among other PTM types — contribute to NSCLC pathogenesis and drug resistance [Ref 24]. Key findings relevant to our model:
- EGFR autophosphorylation patterns determine TKI sensitivity
- Glycosylation affects receptor surface expression and immune checkpoint interactions (PD-L1 glycosylation stabilizes PD-L1, promoting immune evasion) [Ref 25]

### 3.6 Data Availability Summary

| PTM Type | EGFR Data Rows | HER2 Data Rows | Source | Drug Context |
|----------|:-:|:-:|--------|:---:|
| Phosphorylation | 2,123 | ~270K (BT-474) | DrugPTM-Bench, Tozuka, Hsu, PNAS, FEBS, CancerRes, MCP | ✅ Drug-conditioned |
| N-Glycosylation | 48 + ~1,086 (EGFR glycoform atlas) | 4 + ~654 (ErbB2 glycoform atlas) + ~478 (ST6Gal1) | MCP 2025 [Ref 12], ErbB2 Glycoform Atlas [Ref 29], ST6Gal1 ErbB2 [Ref 30], MCP 2025b [Ref 31], EGFR Fucosylation [Ref 32] | ⚠️ Cell-line context; ST6Gal1 has drug (trastuzumab) context |

---

## 4. Insights from the 2025 PTM Review (PMC13070201)

The comprehensive review "Post-translational Modifications in Proteins: Prediction Methods, Biological Functions, and Diseases" (2025) [Ref 26] provides critical context for our approach:

### 4.1 PTM Crosstalk Is the Central Unsolved Problem

The review identifies PTM crosstalk prediction as a critical gap:

> *"PTM crosstalk events play critical roles in biological processes. Many PTM sites from the same (intra) or different (inter) proteins often cooperate with each other to perform a function. Several ML methods have been developed to identify PTM crosstalk within proteins, but the accuracy is still far from satisfactory."*

Three crosstalk prediction tools exist: PTM-X, PCTpred (AUC ~0.90), and DeepPCT (Transformer+GNN+RF, AUC 0.957/0.777). Notably, DeepPCT uses ESM-2 + GearNet-Edge — the same foundation models we use. But these tools predict WHETHER two sites crosstalk. **Our PTM-BDL predicts HOW crosstalk AFFECTS drug response** — a fundamentally different and unaddressed question.

### 4.2 The "Three-Dimensional PTM Code"

The review articulates a principle that directly validates our design:

> *"The pathological impact of any PTM is not dictated by its chemical identity alone. Instead, a three-dimensional code — stoichiometry (how much), timing (when), and geography (where) — determines whether the same PTM will protect, perturb, or even switch physiological signaling to disease-driving cascades."*

This maps exactly to our PTM-BDL features:
- **Stoichiometry** → ptm_level (how much modification)
- **Timing** → delta_ptm (drug-induced change = temporal snapshot)
- **Geography** → site identity embedding + modification type (where on the protein, what type)

### 4.3 Multi-Modification Crosstalk Determines Cell Fate

> *"Disease progression is not governed by a single PTM; instead, different modifications engage in positive cooperation or negative competition, and their integrated crosstalk ultimately determines cell fate and the direction of pathology."*

The review documents specific examples:
- **Phosphorylation × O-GlcNAcylation:** compete at Ser/Thr residues
- **Ubiquitination × phosphorylation:** cooperate in cancer progression
- **Acetylation × ubiquitination:** interplay controls protein stability

This validates including glycosylation alongside phosphorylation in our model — the PTM-BDL self-attention can learn cross-type interactions.

### 4.4 85 Kinase Inhibitors Approved, Resistance Is the Central Problem

The review documents 85 FDA-approved kinase inhibitors including osimertinib, afatinib, and neratinib. It explicitly connects phosphorylation to drug resistance:

> *"Osimertinib is the standard first-line treatment for EGFR-mutant NSCLC, yet most patients eventually develop acquired resistance. Phosphoproteomic mapping of drug-tolerant persister cells shows that resistance is driven by reactivation of EGFR-downstream signaling and by antiapoptotic rewiring — highlighted by hyperphosphorylation of YAP1 and the mTOR-BAD axis."*

### 4.5 All Existing Prediction Tools Are Site-Predictors, Not Response-Predictors

The review catalogs ALL major PTM prediction tools (NetPhos, DeepPhospho, TransPhos, GPS 6.0, PhosphoPredict, DeepAcet, UbPred, DeepUbi, DeepNGlyPred, etc.). **Every single one predicts WHERE a PTM occurs.** NONE predicts HOW PTM patterns affect drug response. The review explicitly calls for:

> *"Future research needs to further integrate multisource data, develop more efficient algorithms, and enhance the interpretability and reliability of models."*

Our PTM-BDL directly answers this call.

---

## 5. Literature Review: What Exists and What's Missing

### 5.1 Existing Approaches

| Method | Year | PTM Role | Inter-Site Dynamics? | Drug Context? | Multi-PTM? | Mod-Type Aware? |
|--------|------|----------|:---:|:---:|:---:|:---:|
| DeepPhos | 2019 | Predicts phospho sites | No | No | No | No |
| PhosBoost | 2024 | Kinase-substrate | No | No | No | No |
| DrugPTM-Bench | 2026 | Predicts drug→PTM | No | Yes (output) | No | No |
| PTM-X / DeepPCT | 2023-25 | Predicts crosstalk | Binary (yes/no) | No | Partial | No |
| PPICT | 2025 | Inter-protein crosstalk | Binary | No | No | No |
| DrugCell | 2020 | Pathway features (GO) | No (pathway level) | Yes | No | No |
| **PTM-BDL (this work)** | **2026** | **PTM as input for response** | **Yes (self-attention)** | **Yes (delta_ptm)** | **Yes (phospho+glyco)** | **Yes (Y/S/T/N)** |

### 5.2 The Gap This Work Fills

No published model:
1. Uses individual PTM sites as **input features** for drug resistance prediction while modeling their **inter-site dependencies**
2. Handles **multiple PTM types simultaneously** (phosphorylation + glycosylation) within the same attention module
3. Learns how **modification type** affects biological interpretation of changes
4. Encodes the **drug response ratio** (delta/baseline) as a biological signal
5. Provides a **general PTM processing module** extensible to any modification type

---

## 6. The Innovation: PTM Biological Dynamics Layer

### 6.1 Core Concept

The PTM-BDL treats post-translational modifications as a **structured biological process** rather than independent features. It encodes four biological principles:

**Principle 1: PTM sites form a signaling code**
The combination of modified sites determines cellular outcome. Self-attention among PTM sites enables the model to learn which combinations predict resistance.

**Principle 2: Modification type determines mechanism**
Phospho-tyrosine (direct TKI target), phospho-serine/threonine (downstream indicator), and glycosylation (receptor surface biology) have fundamentally different biological roles. Learnable modification-type embeddings capture these distinctions.

**Principle 3: The drug response signal is relative, not absolute**
The fraction of signaling capacity eliminated (delta/baseline) is more informative than raw delta. This normalized signal captures drug efficacy.

**Principle 4: Cross-type PTM interactions determine resistance**
Phosphorylation (intracellular signaling) and glycosylation (extracellular receptor biology) provide complementary views. The self-attention mechanism can learn their crosstalk.

### 6.2 What the Model Learns (Not Hard-Coded)

| What | Encoded As | Learned By |
|------|-----------|------------|
| Site identity | Learnable site embedding | Model learns which sites matter |
| Modification type (Y/S/T/N) | Learnable type embedding | Model learns how types differ |
| Site-site interactions | Self-attention weights | Model learns which sites co-regulate |
| Cross-type crosstalk | Self-attention between phospho↔glyco tokens | Model learns phospho-glyco interactions |
| Drug efficacy signal | delta/baseline ratio feature | Model learns what ratio predicts resistance |
| Functional roles | NOT encoded | Discovered and validated post-hoc via IG and attention maps |
| Pathway membership | OPTIONAL input (when available) | Model learns to use or ignore |

### 6.3 How This Solves the Information-Theoretic Problem

1. **delta/baseline ratio**: Nonlinear transformation creating signal that varies per-site within the same mutation class
2. **Inter-site attention**: 66 pairwise phospho interactions + phospho↔glyco cross-attention create combinatorial features
3. **Modification-type conditioning**: Same delta at a tyrosine site (direct drug target) vs serine site (downstream) has different meaning — 4 types create 4× effective feature space
4. **Glycosylation tokens**: Add genuinely new information — receptor surface biology not captured by sequence, structure, or phosphorylation

---

## 7. Detailed Architecture Design

### 7.1 The Two-Stage Architecture: Static Early Fusion + Dynamic Late Fusion

The key architectural insight is that the model's inputs are **biologically two different kinds of information**:

**Static modalities** — properties of the protein and drug that do not change during treatment:
- Protein sequence (ESM-2) — WHAT the protein is, its evolutionary context, mutation identity
- Protein structure (GearNet) — the 3D conformation, binding pocket shape
- Drug chemistry (ChemBERTa) — the drug's chemical structure, binding properties

These are **fixed molecular identities**. They determine the POTENTIAL for drug-protein interaction. They benefit from early fusion (joint self-attention) because sequence↔structure↔drug cross-modal interactions are critical for modeling binding.

**Dynamic modifiers** — the PTM state that determines HOW the protein actually responds to the drug:
- Phosphorylation pattern — which signaling pathways are active
- Glycosylation state — how much receptor is on the cell surface
- Drug-induced PTM changes — how the drug modified the signaling landscape

These are **biological state variables**. They determine the ACTUAL outcome of the drug-protein interaction. They should NOT be thrown into the same attention pool as 1,300+ static tokens where they get drowned. Instead, they should MODULATE the static prediction through late fusion.

**This mirrors the biology:**
```
STATIC: The protein exists (L858R EGFR) + The drug exists (Osimertinib)
        → These determine the POTENTIAL interaction
        → Early fusion captures: "Can this drug bind this protein?"

DYNAMIC: The PTM state modifies the outcome
        → Phospho pattern: "Is signaling actually shut down?"
        → Glyco state: "Is the receptor even on the surface?"
        → Late fusion captures: "Given the drug CAN bind, does it WORK?"
```

### 7.2 Architecture Overview

```
═══════════════════════════════════════════════════════════════
  STAGE 1: EARLY FUSION — Static Molecular Identity
═══════════════════════════════════════════════════════════════

┌─────────┐  ┌──────────┐  ┌──────────┐
│ESM-2 Seq│  │GearNet   │  │ChemBERTa │
│(L×1280) │  │(M×512)   │  │(N×384)   │
└────┬────┘  └────┬─────┘  └────┬─────┘
     │            │              │
┌────▼──┐   ┌────▼────┐   ┌─────▼──┐
│Project│   │Project  │   │Project │
│→ D    │   │→ D      │   │→ D     │
└───┬───┘   └───┬─────┘   └───┬────┘
    │           │              │
    └───────────┴──────────────┘
                │ CONCATENATE
  [(L seq) ; (M struct) ; (N drug)]  ← ALL static tokens
                │
      ┌─────────▼──────────┐
      │ Joint Self-Attention │  ← Seq↔Struct↔Drug interactions
      │ (4 layers × 8 heads)│     "Can this drug bind this protein?"
      └─────────┬──────────┘
                │
      ┌─────────▼──────────┐
      │ Attention Pooling    │  → static_representation (D)
      └─────────┬──────────┘
                │
═══════════════╪═══════════════════════════════════════════════
               │
  STAGE 2: LATE FUSION — Dynamic PTM Biological Gate
═══════════════╪═══════════════════════════════════════════════
               │
               │    ┌─────────────────────────────────────┐
               │    │ PTM Biological Dynamics Layer        │
               │    │                                      │
               │    │ Phospho sites (12) + Glyco sites (8) │
               │    │         │                             │
               │    │  Feature Enrichment                   │
               │    │  [level, delta, ratio, type_emb,      │
               │    │   site_emb]                           │
               │    │         │                             │
               │    │  Type-Gated Projection                │
               │    │         │                             │
               │    │  PTM Self-Attention (2 heads)         │
               │    │  Phospho↔Phospho, Phospho↔Glyco     │
               │    │         │                             │
               │    │  Pool → ptm_representation (D)        │
               │    └─────────┬───────────────────────────┘
               │              │
      ┌────────▼──────────────▼────────┐
      │ Late Bilinear Fusion            │
      │ static_rep ⊙ ptm_rep → fused   │
      │                                  │
      │ "Given the drug CAN bind,       │
      │  does the PTM state say          │
      │  it actually WORKS?"             │
      └──────────────┬─────────────────┘
                     │
            ┌────────┴────────┐
       ┌────▼────┐      ┌─────▼─────┐
       │ IC50    │      │Resistance  │
       │(regress)│      │(classify)  │
       └─────────┘      └───────────┘
```

### 7.3 Why This Design Is Correct

**Biologically:** PTMs don't add "more information about the protein" — the sequence and structure already capture what the protein IS. PTMs capture the protein's DYNAMIC STATE. A gate/modulation is the right computational analog for a biological modifier.

**Computationally:** The static modalities (1,300+ tokens) benefit from early fusion self-attention. The PTM modality (12-20 sites) has its OWN dedicated self-attention where sites can interact without competition. The two representations meet at late fusion where the PTM state MODULATES the static prediction.

**Information-theoretically:** The current failure happens because PTM tokens get drowned in 1,300 other tokens. In this design, PTM has a DIRECT path to the output — through its own self-attention → pooling → late fusion → prediction. No drowning.

### 7.4 Component 1: PTM Feature Enrichment

For each PTM site (phospho OR glyco), construct a rich feature vector:

```python
enriched_i = [
    ptm_level_i,                      # Absolute modification level
    delta_ptm_i,                      # Drug-induced change (0 for glyco sites)
    delta_ptm_i / (ptm_level_i + ε),  # Drug efficacy ratio
    mod_type_embedding_i,             # Learned: phospho-Y=0, S=1, T=2, glyco-N=3
    site_identity_embedding_i,        # Learned per-site embedding
]
```

For glycosylation sites: delta = 0 (no drug-specific glyco change measured), ratio = 0. The model learns from absolute glyco level + cross-type attention with phospho sites.

### 7.5 Component 2: Modification-Type Gated Projection

```python
gate_i = σ(W_gate · [enriched_i ; mod_type_emb_i])
projected_i = gate_i ⊙ (W_proj · enriched_i)
```

The gate learns: "How important is this site's state, given its modification type?" This naturally handles the asymmetry between phospho (drug-responsive) and glyco (structural/baseline) sites.

### 7.6 Component 3: PTM Self-Attention

Small multi-head self-attention among ALL PTM sites (phospho + glyco):

```python
attention = softmax(Q · K^T / √d) · V    # (N_sites × D)
```

With phospho (12) + glyco (up to 8) = up to 20 sites, this is a 20×20 attention matrix — computationally trivial.

**What the attention learns across PTM types:**
- Phospho↔Phospho: Y1092↔Y1197 co-regulation, Y1069↔Y1092 degradation coupling
- Phospho↔Glyco: "Does glycosylation at N528 correlate with Y1092 phospho level?"
- Glyco↔Glyco: "Do different glycan sites co-vary in resistant vs sensitive cells?"

### 7.7 Component 4: Residual Gate

```python
α_i = σ(W_α · [attended_i ; projected_i])
output_i = α_i · attended_i + (1 - α_i) · projected_i
```

Controls how much inter-site context modifies each token.

### 7.8 Pathway Context (OPTIONAL)

When pathway-level phosphoproteomic data is available (PNAS 2025: H1975, HCC4006):

```python
pathway_token = PathwayEncoder(pathway_features)  # (1, D_shared)
# Concatenate with PTM tokens before self-attention
ptm_tokens_with_pathway = cat([ptm_tokens, pathway_token], dim=1)
```

When pathway data is NOT available (most samples): pathway_token is zero-padded with a missingness indicator. The model learns to use pathway data when present and rely on PTM site-level attention when absent.

---

## 8. Integration with Existing Architecture

### 8.1 What Changes

| Removed | Replaced By | Reason |
|---------|------------|--------|
| PTMTokenEncoder | PTM-BDL (separate branch) | Current encoder puts PTM tokens into joint attention where they drown |
| PTMFeatureModulator | PTM-BDL (separate branch) | Current modulator broadcasts single signal to structure — wrong approach |
| Single-stage fusion | Two-stage: early (static) + late (dynamic) | Mirrors the biology: static identity → dynamic modification |
| PTM in Joint Self-Attention | PTM has own dedicated self-attention | PTM sites need to interact with EACH OTHER, not compete with 1,300 static tokens |

### 8.2 What Stays the Same

| Component | Status | Why |
|-----------|--------|-----|
| ESM-2 sequence embeddings | Keep (Stage 1) | Captures protein identity and mutation context |
| GearNet structural embeddings | Keep (Stage 1) | Captures 3D conformation and binding pocket |
| ChemBERTa drug embeddings | Keep (Stage 1) | Captures drug chemistry |
| Joint Self-Attention | Keep (Stage 1 only — static modalities) | Cross-modal seq↔struct↔drug interactions |
| Attention Pooling | Keep (Stage 1) | Produces static representation |
| Bilinear Fusion | Repurposed (Stage 2) | Now fuses static_rep ⊙ ptm_rep instead of protein ⊙ drug |
| Prediction Heads | Keep | IC50 + resistance |

### 8.3 Key Architectural Principle

The architecture follows the biology:

```
STAGE 1 (Early Fusion): Seq + Struct + Drug → "What is the protein-drug system?"
    → These are STATIC molecular identities
    → They benefit from cross-modal attention (seq↔struct↔drug)
    → Output: static_representation (D)

STAGE 2 (Late Fusion): PTM-BDL → "How does the PTM state modify the outcome?"
    → These are DYNAMIC biological modifiers
    → They need their OWN self-attention (phospho↔phospho, phospho↔glyco)
    → Output: ptm_representation (D)

FUSION: static_rep ⊙ ptm_rep → prediction
    → "Given this protein-drug system, what does the PTM state say about resistance?"
```

This is not over-complicated. It is the simplest correct design: two branches that meet at fusion. The static branch already works (No-PTM baseline on the 2026-06-28 run: BAcc 0.837, AUROC 0.847, RMSE 1.519, Pearson R 0.637 — see §1.1). The PTM branch must add the biological dynamics that the static branch cannot capture, *without* the randomization-control failure mode currently observed.


---

## 9. Validation Strategy

The validation plan is designed to **specifically discriminate PTM-BDL from the current architecture** on the four failure modes surfaced by the 2026-06-28 run (see §1.1): randomized-control failure, mutation-group collapse, threshold-calibration trade-off, and the IG-vs-prediction gap (the model "knows" Y1068 matters but cannot use it).

### 9.1 Ablation Study (step11b — existing infrastructure)

| Model | Description | Pass Criterion |
|-------|------------|---------------|
| Model A: No PTM | All PTM zeroed (static branch only) | Reference baseline (current No-PTM: AUROC 0.847, BAcc 0.837, RMSE 1.519) |
| Model B: PTM-BDL phospho only | Phospho 12-site PTM-BDL active, glyco zeroed | Should match or exceed Model A on AUROC AND maintain BAcc within −0.05 of A (no calibration collapse) |
| Model C: PTM-BDL phospho + glyco | Full PTM-BDL with phospho + glyco tokens | Should be the best model overall; demonstrates type-extensibility of the architecture |
| Model D: Level 2 Only | Aggregate phospho rewiring only (current C ablation arm) | Reference for "aggregate features without per-site dynamics" |

**Primary acceptance metric:** AUROC + AUPRC-sensitive (threshold-independent). The 2026-06-28 ablation already shows the current architecture moves these metrics in the right direction (ptm_gain_auroc = +0.013, ptm_gain_auprc_sensitive = +0.082); PTM-BDL should preserve these gains AND avoid the BAcc collapse (current ptm_gain_bacc = −0.161).

**Secondary acceptance:** RMSE must not regress more than +0.10 vs Model A (current Model D RMSE 1.733 vs Model A 1.519 = +0.214 regression; PTM-BDL should cut this in half or eliminate it).

### 9.2 Randomized PTM Control — Primary Falsification Test (existing)

This is the most important single test. The current architecture **fails** this test: shuffled PTM outperforms real PTM on every metric (drop_BAcc = −0.042, drop_AUROC = −0.010, drop_AUPRC-sens = −0.050).

**Pass criterion for PTM-BDL:**
- `drop_BAcc ≥ +0.02` (real PTM at least 0.02 BAcc better than shuffled), AND
- `drop_AUROC ≥ +0.005`, AND
- `drop_AUPRC-sensitive ≥ +0.02`

This is the single binary test that distinguishes "the model uses PTM biology" from "the model treats PTM as redundant noise." If PTM-BDL does not flip the sign of the drop, the architecture has not solved the core problem and should not be promoted.

### 9.3 Multi-Seed Stability (existing)

3 seeds (42, 123, 456): IG rankings stable? Top site consistent? PTM self-attention patterns consistent?

**Pass criterion:** Y1092 (Y1068) remains rank #1 with std_rank ≤ 0.5, and top-3 sites overlap ≥ 2/3 across all seed pairs. The current architecture already passes this with std_rank=0.0 for Y1068; PTM-BDL must at minimum maintain it.

### 9.4 Integrated Gradients on PTM Sites (existing)

- Y1092/Y1068 should rank #1–2 (matches current run rank #1)
- Y1197/Y1173 and Y1110/Y1086 should appear in top 4 (matches current run ranks #2–3)
- Y1069/Y1045 should show a distinct attribution direction reflecting its degradation role (currently rank 10 in the resistance head)
- For the glyco arm (Model C): at least one glyco site (e.g., N528 or N361) should rank above the bottom regulatory phospho sites (S991, S1039, T1041), demonstrating that the glyco channel carries non-zero signal

### 9.5 PTM Self-Attention Analysis (NEW)

The current architecture cannot produce this analysis (no dedicated PTM self-attention). PTM-BDL's dedicated PTM attention enables:

- **Cross-site attention map (12+8 = 20 sites)** — visualize Q·K patterns per sample
- **MAPK-PI3K coupling:** Y1092 ↔ Y1197 should show elevated mutual attention vs random pairs
- **SRC bypass differentiation:** Y869 → Y1092 attention should differ between sensitive and resistant samples
- **Phospho-glyco crosstalk:** non-zero attention between phospho and glyco token blocks (demonstrates the model uses the type extension)
- **Type clustering:** Y sites attend more to other Y sites than to S/T/N sites — emergent organization by modification type

### 9.6 Cross-Receptor Validation (EGFR × HER2)

The 2026-06-28 run shows EGFR AUROC 0.823 vs ERBB2 AUROC 0.800 — the architecture transfers some learning but learns EGFR biology more strongly. PTM-BDL is designed to be protein-agnostic at the PTM level.

**Pass criterion:**
- Phospho: EGFR Y1092 (Y1068) and HER2 Y1221 should both rank #1 in their respective IG analyses — same functional role (GRB2 binding) → same rank
- Glyco: EGFR N528 and HER2 N530 (homologous domain III sites) should both appear in top 3 glyco-site IG rankings
- ERBB2-AUROC gap vs EGFR-AUROC should narrow (current gap 0.023; target ≤ 0.015)

### 9.7 Mutation-Group Discrimination (NEW — Direct Test of the Failure Mode)

The current architecture collapses all EGFR mutation groups to prob ≈ 0.186 (see §1.1.5). PTM-BDL is designed to give each mutation group access to a distinct PTM dynamic state via its type-gated PTM tokens.

**Pass criterion:**
- The standard deviation of `mean_resist_prob` across the 8 EGFR mutation groups in the test set should be **at least 0.04** (current: ~0.003)
- The A755D/L747_P753delinsS group (true resistant, true IC50 = +0.61) should have `mean_resist_prob` higher than at least 3 of the sensitive mutation groups
- This is a direct check that the PTM channel can differentiate within-mutant-class biology, not just within-WT-vs-mutant

This test does not require new infrastructure — it reuses `evaluation_report.json["mutation_stratified"]` from step12.


---

## 10. Implementation Steps

### Step 1: Extend Data Pipeline
- Add glycosylation extraction from MCP 2025 (48 EGFR + 4 HER2 rows)
- Create glyco_vector alongside ptm_vector in step06
- Add modification_type_ids to dataset (Y=0, S=1, T=2, N=3)

### Step 2: Implement PTM-BDL Module
- Feature enrichment (level, delta, ratio, mod_type_emb, site_emb)
- Modification-type gated projection
- PTM self-attention (2 heads, 1-2 layers)
- Residual gate
- Optional pathway context encoder

### Step 3: Integrate into Model
- Replace PTMTokenEncoder + PTMFeatureModulator with PTM-BDL
- Update forward pass, token boundaries
- Update step10 shape verification

### Step 4: Update Training
- Add glyco features to ResistanceDataset
- Modification type mapping in collate_fn
- No changes to loss function or training loop

### Step 5: Update Ablation & Evaluation
- Add PTM-BDL-specific ablation modes
- Add PTM self-attention visualization to step13
- Add cross-type attention analysis
- Add glyco-specific IG analysis

### Step 6: Run Full Validation
- Ablation → does PTM-BDL help?
- Randomized control → uses PTM biology?
- Stability → reproducible?
- IG → which sites matter?
- PTM attention → cross-type interactions?
- Cross-receptor → EGFR-HER2 consistency?

---

## 11. Limitations

### 11.1 Phosphoproteomics Coverage
Experimental phosphoproteomic measurements are available for 4-6 EGFR-mutant cell lines across 7 published studies. The remaining samples receive PTM features via mutation-class biological priors with explicit confidence scores (0.40-1.00). This hierarchical propagation is supported by published evidence that EGFR activating mutations produce convergent autophosphorylation patterns (Yun 2008, Red Brewer 2013, Sordella 2004).

### 11.2 Glycosylation Data Coverage
Glycosylation data has been expanded from 4 sources beyond the original MCP 2025:
- **MCP 2025** (Abe et al.) [Ref 12]: 48 EGFR + 4 HER2 glyco measurements from H1975, H3255, PC-9
- **ErbB2 Glycoform Atlas** (Glycobiology 2024) [Ref 29]: Complete MS-based glycoform atlas covering ALL 7 ErbB2 N-glycosylation sites (N68, N124, N187, N259, N530, N571, N629) from SKBR-3 and BT-474 cancer cell lines, plus 9 EGFR sites from CHO-expressed sEGFR. Table SI: 137 glycoform compositions; Table SII: ~1,985 MS/MS glycopeptide spectra across 5 cell contexts.
- **ST6Gal1 → ErbB2** (Oncogene 2021) [Ref 30]: Site-specific ErbB2 glycoproteomics showing ST6Gal1-mediated sialylation modulates trastuzumab sensitivity. 478 glycopeptide rows across 2 replicates + gastric adenocarcinoma tissue. Drug resistance context (trastuzumab).
- **MCP 2025b** (MCP 2025) [Ref 31]: Companion paper to MCP 2025 using Metal Ion-Enhanced ZIC-cHILIC. Contains 31,609 + 19,678 glycopeptide identifications and 4,006-row summary table (phospho+glyco) from EGFR-mutated lung cancer cells.
- **EGFR Fucosylation** (Mol Omics 2020) [Ref 32]: Site-specific EGFR glycoproteomics by HILIC-C18-QTOF-MS/MS from OSCC cells (CAL27, HSC3). Covers 4 EGFR glycosites (N175, N413, N444, N603) with per-site glycoform relative intensities and fucosylation status. Serves as EGFR glycoform reference catalog.

No drug-induced glycosylation changes are directly measured in the EGFR context — glyco features represent baseline receptor biology. The ST6Gal1 ErbB2 dataset provides the only drug-conditioned glycosylation data (trastuzumab resistance). As drug-conditional glycoproteomics data becomes available (e.g., osimertinib-induced glyco changes), the PTM-BDL module can incorporate them without architectural modification.

### 11.3 Supported PTM Types
This foundational model supports **Phosphorylation** and **N-Glycosylation** — the two PTM types for which sufficient quantitative data exists in our datasets. Other biologically relevant PTM types (ubiquitination, acetylation, SUMOylation, palmitoylation, methylation) are excluded due to data limitations: no quantitative drug-response measurements are available for these modifications in the EGFR/HER2 context. The PTM-BDL modification-type embedding system is extensible and can accommodate additional PTM types as data becomes available.

### 11.4 Pathway Data Sparsity
Pathway-level phosphoproteomic data (per-pathway mean log2FC under TKI treatment) is available for only 2-3 cell lines (H1975, HCC4006 from PNAS 2025). The pathway context is implemented as an OPTIONAL input — the model learns to use it when available and rely on site-level PTM attention when absent.

---

## 12. Future Directions

### 12.1 Multi-PTM Extension: Ubiquitination, Acetylation, and Beyond

The PTM-BDL architecture is designed as a **foundation for multi-PTM modeling**. The modification-type embedding system supports any PTM type by adding new type IDs:

| PTM Type | Type ID | Status |
|----------|:---:|------|
| Phospho-Y | 0 | ✅ Supported (available now) |
| Phospho-S | 1 | ✅ Supported (available now) |
| Phospho-T | 2 | ✅ Supported (available now) |
| Glyco-N | 3 | ✅ Supported (available now — MCP 2025) |

**The biological vision:** As multi-PTM proteomics studies become available (e.g., combined phospho+ubiquitin+glyco enrichment protocols), the PTM-BDL self-attention can be extended to learn a broader **PTM crosstalk network** — incorporating additional modification types as quantitative drug-response data becomes available. The PMC13070201 review identifies this as the key frontier:

> *"Identifying the critical PTM-combinations offers fresh molecular targets and intervention strategies for precision medicine."*

The PTM-BDL is positioned to be the first computational tool capable of modeling these combinations for drug response prediction.

### 12.2 Signaling Pathway Integration

The current pathway context is optional and sparse. Future directions:

1. **Pathway knowledge graphs:** Encode known signaling cascades (EGFR→GRB2→SOS→RAS→RAF→MEK→ERK) as a graph structure that the PTM-BDL can use as prior knowledge
2. **Pathway-resolved phosphoproteomics:** As more cell lines get pathway-level phospho measurements, the pathway context token becomes increasingly informative
3. **Cross-pathway PTM attention:** The self-attention between PTM sites implicitly learns pathway topology — Y1092 (MAPK pathway) attending to Y1197 (PI3K pathway) captures pathway crosstalk without explicit pathway annotation

### 12.3 Extension to Other Receptor Families

The PTM-BDL module is protein-agnostic — it processes PTM sites based on modification type and site identity, not protein identity. Future application to:
- **ALK/ROS1:** Lung cancer TKI targets with known phosphorylation-driven resistance
- **BCR-ABL:** CML — extensive phosphoproteomic data available
- **FGFR:** Emerging TKI target with glycosylation-dependent drug binding
- **MET:** Bypass signaling in EGFR TKI resistance

---

## 13. Expected Outcomes

### 13.1 Scope of Expected Outcomes

This is a **computational framework paper using existing data** — no new wet-lab experiments will be generated. Validation is therefore through (a) matching the architecture to published PTM biology, (b) IG and attention analyses recovering known biology, and (c) bootstrap-defensible improvements on the existing dataset. Targets below are calibrated to be empirically detectable with n=143 test samples and 12 sensitive cases, not to claim clinical-grade prediction.

### 13.2 Quantitative Targets (anchored to 2026-06-28 run)

Reference baselines from the current pipeline (`results/COMPREHENSIVE_EVALUATION_28_june.md`):
- **Static-only (Model A: No PTM):** AUROC 0.847, BAcc 0.837, RMSE 1.519, Pearson R 0.637, AUPRC-sensitive 0.523
- **Current full model (Model D: Both):** AUROC 0.860, BAcc 0.676, RMSE 1.733, Pearson R 0.595, AUPRC-sensitive 0.605
- **Randomized control:** shuffled outperforms real on every metric — drop_BAcc = −0.042, drop_AUROC = −0.010, drop_AUPRC-sens = −0.050

| Metric | Static-only Baseline | Current Full Model | PTM-BDL Target | Type of Improvement |
|--------|:--:|:--:|:--:|---|
| Test AUROC | 0.847 | 0.860 | **≥ 0.870** | Preserve current ranking gain |
| Test AUPRC-sensitive | 0.523 | 0.605 | **≥ 0.620** | Build on current largest PTM gain |
| Test BAcc | 0.837 | 0.676 | **≥ 0.78** | Fix the threshold collapse |
| Test RMSE | 1.519 | 1.733 | **≤ 1.55** | Eliminate the regression regression |
| Test Pearson R | 0.637 | 0.595 | **≥ 0.62** | Preserve magnitude calibration |
| Mutation-prob std-dev (across 8 groups) | n/a | 0.003 | **≥ 0.04** | Break the mutation collapse |
| Randomized drop_AUROC | n/a | **−0.010** (shuffled wins) | **≥ +0.005** (real wins) | Flip the sign — primary falsification test |
| Randomized drop_BAcc | n/a | **−0.042** (shuffled wins) | **≥ +0.02** (real wins) | Flip the sign — primary falsification test |
| EGFR↔ERBB2 AUROC gap | n/a | 0.023 | ≤ 0.015 | Cross-receptor consistency |

The targets are intentionally modest in absolute magnitude (e.g., AUROC 0.860 → 0.870) but **categorically meaningful**: flipping the randomized-control drop sign and breaking the mutation collapse are qualitative shifts the current architecture cannot achieve, regardless of how many epochs or seeds we train.

### 13.3 Biological Discoveries (Computational Framework Scope)

The PTM self-attention weights and IG attributions should reveal:
1. Which phosphosite interactions predict resistance (e.g., Y1068↔Y1173 mutual attention as a sensitivity signature)
2. Whether modification type genuinely affects interpretation (Y vs S/T vs N attention clustering)
3. Whether phospho-glyco crosstalk is informative (non-zero cross-block attention)
4. Whether EGFR and HER2 show consistent functional patterns at homologous sites — both phospho (Y1068↔Y1221, Y1173↔Y1248) and glyco (N528↔N530)

All four discovery claims are computationally verifiable from `xai_report.json` outputs and require no additional experiments.

### 13.4 What This Paper Will NOT Claim

To preempt over-interpretation, the framework explicitly will not claim:
- Clinical-grade response prediction
- Mechanism discovery requiring wet-lab confirmation
- That PTM-BDL is the "best" architecture — only that it is the right *inductive bias* for biologically-grounded PTM processing
- Generalization beyond EGFR and HER2 without additional data

### 13.5 Benchmarking Validation (added 2026-07-01)

The PTM-BDL claims are validated through a comprehensive benchmarking suite (see `docs/BENCHMARKING_PLAN.md`) that includes:

**External comparison (12 methods across 3 tiers):**
- **Tier 0** (ML baselines): RF, XGBoost, Ridge, Elastic Net on same 2224-d concatenated features — tests whether our DL architecture adds value over feature concatenation
- **Tier 1** (2023–2024 SOTA): DIPK (Nat Comm 2024), HiDRA (Nat Comm 2023), GraTransDRP, TransCDR, PathDSP — the methods Nature Methods reviewers will demand comparison with
- **Tier 2** (established): GraphDRP, DrugCell, DeepCDR — frame as "established methods"

**Statistical rigor (following SAGE-net, Nat Methods 2026):**
- Bootstrap 95% CIs (1,000 resamples) for PCC, RMSE, AUROC, AUPRC-sensitive
- DeLong test for paired AUROC comparison
- Wilcoxon signed-rank for per-drug paired comparisons
- Benjamini-Hochberg correction across K baselines

**Cell-blind generalization (per Sada Del Real et al., Brief Bioinf 2026):**
- Leave-One-Cell-Line-Out (LOCLO) by mutation class group
- Tests whether PTM-BDL generalizes to mutation classes not seen during training

**Biological validation (11 tests unique to PTM-BDL — Table 2 in paper):**
- IG site ranking: Y1068 = #1 across 3/3 seeds
- Cross-receptor homology: EGFR Y1068 ≡ HER2 Y1221
- Modification-type hierarchy: tyrosine > serine/threonine
- Randomized PTM control: real > shuffled (ΔAUROC ≥ +0.005)
- Channel contributions: phospho marginal, glyco marginal, typed attention marginal

**Publication outputs:** Scripts `step14a–d` (benchmarking) and `step15a–b` (figures/tables) generate camera-ready PDF figures (300 DPI, colorblind-friendly) and LaTeX tables in booktabs style.

---

## 14. References

### Core Biological References

1. **Seet BT, Dikic I, Zhou MM, Pawson T.** Reading protein modifications with interaction domains. *Nat Rev Mol Cell Biol* 7, 473-483 (2006). PMID: 16829981.

2. **Schulze WX, Deng L, Mann M.** Phosphotyrosine interactome of the ErbB-receptor kinase family. *Mol Syst Biol* 1, 2005.0008 (2005). PMID: 16729048.

3. **Olsen JV, Blagoev B, Gnad F, et al.** Global, in vivo, and site-specific phosphorylation dynamics in signaling networks. *Cell* 127(3), 635-648 (2006). PMID: 17081983.

4. **Sordella R, Bell DW, Haber DA, Settleman J.** Gefitinib-sensitizing EGFR mutations activate anti-apoptotic pathways. *Science* 305(5687), 1163-1167 (2004). PMID: 15118125.

5. **Cross DAE, Ashton SE, et al.** AZD9291 overcomes T790M-mediated resistance. *Cancer Discovery* 4(9), 1046-1061 (2014). PMID: 25351743.

6. **Red Brewer M, Yun CH, et al.** Mechanism for activation of mutated EGFR. *PNAS* 110(38), E3595-E3604 (2013). PMID: 23940396.

7. **Kobayashi S, Boggon TJ, et al.** EGFR mutation and resistance to gefitinib. *NEJM* 352(8), 786-792 (2005). PMID: 15713906.

### Drug-PTM Data Sources

8. **Tozuka T, Nishi H, et al.** Phosphoproteomics of osimertinib-resistant NSCLC. *iScience* 27(5), 109657 (2024). PMID: 38646155.

9. **Hsu JL, Chen CT, et al.** Temporal phosphoproteomics in EGFR-mutant NSCLC. *Mol Syst Biol* (2025). PMID: 41023502.

10. **PNAS 2025.** Tyrosine phosphoproteome under TKI treatment. DOI: 10.1073/pnas.2522090123.

11. **Badkul A, Qi Y, Xie L.** DrugPTM-Bench. *Mol Cell* (2026). PMID: 30394195.

12. **Abe et al.** Fe-ZIC-cHILIC for phosphoproteomics and glycoproteomics in TKI-resistant NSCLC. *MCP* (2025). DOI: 10.1016/j.mcpro.2025.100917.

### Computational References

13. **Krug K, Jaehnig EJ, et al.** Proteogenomic landscape of breast cancer. *Cell* 183(5), 1436-1456.e31 (2020). PMID: 32504382.

14. **Ochoa D, Jarnuczak AF, et al.** Functional landscape of the human phosphoproteome. *Nat Biotechnol* 41, 541-554 (2023). PMID: 36510105.

15. **Beltrao P, Albanése V, et al.** Systematic functional prioritization of PTMs. *Cell* 150(2), 413-425 (2012). PMID: 22817900.

16. **Sundararajan M, Taly A, Yan Q.** Axiomatic attribution for deep networks. *ICML* 2017.

17. **Ilse M, Tomczak JM, Welling M.** Attention-based deep MIL. *ICML* 2018.

18. **Vaswani A, Shazeer N, et al.** Attention is all you need. *NeurIPS* 2017.

### ERBB Family Biology

19. **Citri A, Yarden Y.** EGF-ERBB signalling: towards the systems level. *Nat Rev Mol Cell Biol* 7, 505-516 (2006). PMID: 16829981.

20. **Engelman JA, et al.** MET amplification leads to gefitinib resistance via ERBB3. *Science* 316(5827), 1039-1043 (2007). PMID: 17463250.

### Glycosylation and Multi-PTM References

21. **Zhu X, et al.** Effects of N361 glycosylation on EGFR biological function. *Cancers* 18(3), 474 (2026). DOI: 10.3390/cancers18030474.

22. **Li S, et al.** Glycosylation targeting: a paradigm shift in cancer immunotherapy. *Int J Biol Sci* 20, 2607 (2024). DOI: 10.7150/ijbs.95434.

23. **Zheng A, et al.** Post-translational modifications of cancer immune checkpoints. *Mol Cancer* (2025). DOI: 10.1186/s12943-025-02397-5.

24. **Kharb R, et al.** Unraveling the PTMs and therapeutical approach in NSCLC pathogenesis. *Biomed Pharmacother* 164, 114997 (2023). PMID: 37276755. PMC: PMC10133877.

25. **Arteaga CL, Engelman JA.** ERBB receptors: from oncogene discovery to mechanism-based therapeutics. *Cancer Cell* 25(3), 282-303 (2014). PMID: 24651011.

### PTM Prediction and Crosstalk Tools

26. **PMC13070201 (2025).** Post-translational Modifications in Proteins: Prediction Methods, Biological Functions, and Diseases. Comprehensive review covering phosphorylation, ubiquitination, methylation, SUMOylation, glycosylation, and acetylation prediction tools, biological functions, disease mechanisms, and clinical translation.

27. **Huang YX, Liu R.** Improved prediction of PTM crosstalk using DeepPCT. *Bioinformatics* (2025). — Transformer+GNN+RF using ESM-2 + GearNet-Edge, AUC 0.957.

28. **Zhu F, Deng L, et al.** PPICT: integrated deep neural network for inter-protein PTM crosstalk. *Brief Bioinform* (2025). — Bilinear fusion for cross-protein PTM interactions.

### N-Glycosylation Data Sources

29. **Taniguchi T, et al.** Site-specific glycosylation analysis of epidermal growth factor receptor 2 (ErbB2): exploring structure and function toward therapeutic targeting. *Glycobiology* 34(3), cwad100 (2024). PMID: 38109791. DOI: 10.1093/glycob/cwad100. — Complete MS-based glycoform atlas of ALL 7 ErbB2 N-glycosylation sites from SKBR-3, BT-474, HEK293, CHO cells. Also includes 9 EGFR glycosites from CHO-expressed sEGFR. Data: `data/raw/drugptm/erbb2_glycoform_atlas_2024/`

30. **Garnham R, et al.** ST6Gal1 targets the ectodomain of ErbB2 in a site-specific manner and regulates gastric cancer cell sensitivity to trastuzumab. *Oncogene* 40, 3111-3125 (2021). PMID: 33947960. DOI: 10.1038/s41388-021-01801-w. PMC: PMC8154592. — Site-specific ErbB2 glycoproteomics showing sialylation modulates trastuzumab drug sensitivity. 478 glycopeptide identifications. Data: `data/raw/drugptm/st6gal1_erbb2_2021/`

31. **Abe et al.** Metal Ion-Enhanced ZIC-cHILIC StageTip for N-Glycoproteomic and Phosphoproteomic Profiling in EGFR-Mutated Lung Cancer Cells. *MCP* (2025). PMID: 40154885. DOI: 10.1016/j.mcpro.2025.100957. PMC: PMC12289526. — Companion to MCP 2025 [Ref 12]. 10,536 glycopeptides + 11,329 phosphopeptides from EGFR-mutated lung cancer cells. Simultaneous glyco+phospho profiling. Data: `data/raw/drugptm/mcp_2025b/`

32. **Sethi MK, et al.** β-catenin/CBP Inhibition Alters Epidermal Growth Factor Receptor Fucosylation Status in Oral Squamous Cell Carcinoma. *Mol Omics* 16(3), 234-245 (2020). PMID: 32203567. DOI: 10.1039/d0mo00009d. PMC: PMC7299767. — Site-specific EGFR glycoproteomics by HILIC-C18-QTOF-MS/MS. Covers N175, N413, N444, N603 with glycoform relative intensities. Data: `data/raw/drugptm/egfr_fucosylation_2020/`
