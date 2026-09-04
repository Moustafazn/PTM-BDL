"""
PTM-BDL Evaluation Package — Evaluation, baselines, statistical tests, and generalization.

Components:
    evaluator   — Full evaluation (collect_predictions, compute_full_metrics)
    baselines   — ML baseline tool (RF, XGBoost, Ridge, ElasticNet)
    statistical — Bootstrap CIs, DeLong, Wilcoxon, BH correction, ECE, IG stability
    loclo       — Leave-One-Class-Line-Out generalization tool
    cold_split  — Cold-drug (LODO) and cold-cell evaluation
"""
