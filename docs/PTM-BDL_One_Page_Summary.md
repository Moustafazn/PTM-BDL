# PTM Biological Dynamics Layer (PTM-BDL)
## A Learnable Neural Module That Simulates PTM Signaling Biology for Drug Resistance Prediction

---

### The Idea

In a living cell, post-translational modifications (PTMs) are not static numbers — they are a **dynamic biological process**. When EGFR is activated by a mutation like L858R, a phosphorylation cascade fires: Y1068 recruits GRB2→MAPK, Y1173 recruits SHC→PI3K, Y1045 recruits Cbl→degradation. When osimertinib is added, different sites dephosphorylate at different rates. The **pattern** of remaining phosphorylation — not any single site — determines whether the cell lives or dies. Meanwhile, N-glycosylation on the extracellular domain controls how much receptor reaches the surface in the first place.

**No existing computational model captures this process.** Current models treat PTMs as flat numbers fed to an MLP. They cannot learn that the *relationship* between Y1068 going down and Y845 (SRC bypass) staying up is what determines resistance. Furthermore, the quantitative PTM data needed to model this process is scattered across 7 independent phosphoproteomic studies and 5 glycoproteomic sources — each using different quantification methods (TMT, SILAC, LFQ, DIA-MS). We harmonize all of these into a unified per-cell-line, per-drug PTM state vector across two proteins and two cancer types — a foundational data integration that enables the entire approach.

### The Innovation: PTM-BDL

We propose a **learnable neural module** that simulates how PTMs function in biological cells:

1. **Each PTM site is a biological actor** — encoded as `[level, delta, ratio]` where `ratio = delta/(level+ε)` captures drug efficacy (§7.4)
2. **Modification type determines biological mechanism** — a type-gated projection learns that phospho-Y (direct TKI target) matters differently than phospho-S (downstream) or glyco-N (receptor surface) (§7.5)
3. **Sites communicate through self-attention** — Y1068 (MAPK) attends to Y1173 (PI3K): "Is the survival pathway also shut down?" This is how the model learns the combinatorial signaling code (§7.6)
4. **Phospho and glyco tokens interact in the same attention space** — the model can learn: "High glyco (receptor on surface) + persistent phospho (active signaling) = resistant" (§7.6)
5. **A residual gate controls attention influence** — sites that don't benefit from inter-site context keep their independent representation (§7.7)

### Why This Is Novel (Gap in the Literature)

| What Exists | What's Missing | What PTM-BDL Does |
|---|---|---|
| DeepPhos, PhosBoost predict WHERE PTMs occur | No model predicts HOW PTM patterns affect drug response | PTM sites as input features WITH inter-site dependencies |
| PTM-X, DeepPCT predict WHETHER two sites crosstalk | No model predicts HOW crosstalk DETERMINES resistance | Self-attention learns which site combinations predict outcome |
| DrugPTM-Bench predicts drug→PTM effects | No model uses multiple PTM TYPES simultaneously | Phospho + glyco in same attention space with type-aware gating |
| DrugCell uses pathway-level features | No model operates at individual PTM site resolution | 24 per-site tokens with biological embeddings |

**The 2025 PTM Review (PMC13070201) explicitly calls for this**: *"PTM crosstalk prediction is a critical gap... Future research needs to integrate multisource data and enhance interpretability."* PTM-BDL directly answers this call.

### Architecture (Two-Stage Fusion)

```
STAGE 1 — STATIC: What is the protein-drug system?
  ESM-2 (sequence) + GearNet (structure) + ChemBERTa (drug)
  → Joint self-attention → S_rep (drug-aware protein representation)

STAGE 2 — DYNAMIC: Does the PTM state say the drug works?
  12 phospho + 12 glyco tokens → [level, delta, ratio]
  → Type-gated projection (Y/S/T/N aware)
  → Typed self-attention (phospho↔phospho, phospho↔glyco)
  → Residual gate
  → P_rep (PTM biological state)

FUSION: S_rep ⊙ P_rep → IC50 + P(resistance)
  "Given the drug CAN bind, does the PTM signaling code say it WORKS?"
```

### Cross-Cancer Validation (EGFR × HER2)

The module is **protein-agnostic** — it processes PTM sites by biological properties, not protein identity. We validate on:
- **EGFR** (NSCLC, 12 phospho + 12 glyco sites) — 4 TKI drugs
- **HER2/ERBB2** (breast cancer, 10 phospho + 7 glyco sites) — 6 TKI drugs (4 shared)
- **Cross-receptor homology**: EGFR Y1068 ≡ HER2 Y1221 (both GRB2→MAPK). If the model has learned **function** not **protein identity**, these homologous sites should rank #1 in both proteins independently.

### Falsification Test

**Randomized PTM control**: shuffle PTM features across samples, destroying the mutation→PTM correspondence, and retrain. If the model is using PTM biology, shuffled PTM must perform **worse**. If shuffled PTM performs the same or better, the PTM channel is redundant. This is a binary pass/fail test — no ambiguity.

### Why Nature Methods

1. **Methodological contribution**: First neural module that encodes PTM biology (type-aware gating + inter-site attention + cross-type crosstalk) as an inductive bias for drug response prediction
2. **Generalizable architecture**: The modification-type embedding system is extensible to ubiquitination, acetylation, SUMOylation as data becomes available
3. **Interpretable by design**: IG attributions bucketed by modification type + cross-type attention heatmaps + sensitive-vs-resistant attention pattern comparison — the model's discoveries are directly readable by biologists
4. **Cross-receptor validation**: Same module, two proteins, two cancers, shared drugs — demonstrates the architecture learns receptor-family biology, not dataset-specific patterns
5. **Reproducible falsification**: The randomized PTM control is a rigorous negative control that no existing PTM-prediction paper includes
6. **Comprehensive benchmarking**: Compared against 12 methods across 3 tiers (ML baselines, recent SOTA, established methods) with full statistical rigor — Bootstrap 95% CIs, DeLong paired AUROC tests, Wilcoxon signed-rank, BH correction — following the benchmarking standards of SAGE-net (Nat Methods 2026) and the 2026 DRP review (Sada Del Real et al., Brief Bioinf)

### Benchmarking Position

Our model introduces a **unique capability** that no other DRP method provides — per-PTM-site resolution with typed self-attention and cross-modification-type crosstalk. This is validated through:
- **Tier 0** (ML baselines): RF, XGBoost, Ridge, Elastic Net on same 2224-d features
- **Tier 1** (2023–2024 SOTA): DIPK, HiDRA, GraTransDRP, TransCDR, PathDSP
- **Tier 2** (established): GraphDRP, DrugCell, DeepCDR
- **Cell-blind generalization**: Leave-One-Cell-Line-Out (LOCLO) by mutation class — per Sada Del Real et al. recommendation
- **11 biological validation tests** that ONLY our model can be evaluated on (Table 2 in paper)

### Key References

- Seet et al., NRMCB 2006 — "modification code" hypothesis
- PMC13070201 (2025) — PTM crosstalk as central unsolved problem
- Schulze et al., MSB 2005 — EGFR phosphosite quantitation
- Krug et al., Cell 2020 — CPTAC phospho-modules
- Sundararajan et al., ICML 2017 — Integrated Gradients
