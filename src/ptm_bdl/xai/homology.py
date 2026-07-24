"""
Cross-protein homology validation for PTM-BDL models.

Validates that the model learns biological FUNCTION (e.g., GRB2 docking)
rather than protein identity, by checking if homologous sites across
different proteins receive concordant importance rankings.

The module is protein-agnostic — homologous slot pairs are provided by the caller
(from the case study's biology.py).
"""

from __future__ import annotations


def check_homology(
        ig_results: dict,
        homology_pairs: list[dict],
) -> dict:
    """
    Check if homologous sites across proteins have concordant IG rankings.

    Args:
        ig_results: Output from compute_ig_batch() — dict keyed by protein_id,
                    each containing channel_name → attribution array.
        homology_pairs: List of dicts, each defining a homology check:
            {
                "name": str — human-readable name (e.g., "GRB2_docking"),
                "protein_a": int — protein ID for first protein,
                "protein_b": int — protein ID for second protein,
                "channel": str — channel key (e.g., "ptm_vector"),
                "slot_a": int — expected top slot index in protein A,
                "slot_b": int — expected top slot index in protein B,
            }

    Returns:
        Dict with per-pair results:
        {
            "pair_name": {
                "protein_a_top_slot": int,
                "protein_b_top_slot": int,
                "expected_a": int,
                "expected_b": int,
                "concordant": bool — both match expected slots,
                "a_matches": bool,
                "b_matches": bool,
            },
            ...
            "all_concordant": bool,
            "n_concordant": int,
            "n_total": int,
        }
    """
    results = {}
    n_concordant = 0

    for pair in homology_pairs:
        name = pair["name"]
        pid_a = pair["protein_a"]
        pid_b = pair["protein_b"]
        channel = pair["channel"]
        expected_a = pair["slot_a"]
        expected_b = pair["slot_b"]

        # Get attribution arrays for each protein
        attrs_a = ig_results.get(pid_a, {}).get(channel)
        attrs_b = ig_results.get(pid_b, {}).get(channel)

        if attrs_a is None or attrs_b is None:
            results[name] = {
                "status": "missing_data",
                "concordant": False,
            }
            continue

        import numpy as np

        # Handle real-slot masking if provided
        real_mask_a = pair.get("real_mask_a")
        real_mask_b = pair.get("real_mask_b")

        if real_mask_a is not None:
            masked_a = attrs_a.copy()
            masked_a[~np.array(real_mask_a)] = -np.inf
            top_a = int(np.argmax(masked_a))
        else:
            top_a = int(np.argmax(attrs_a))

        if real_mask_b is not None:
            masked_b = attrs_b.copy()
            masked_b[~np.array(real_mask_b)] = -np.inf
            top_b = int(np.argmax(masked_b))
        else:
            top_b = int(np.argmax(attrs_b))

        a_matches = (top_a == expected_a)
        b_matches = (top_b == expected_b)
        concordant = a_matches and b_matches

        if concordant:
            n_concordant += 1

        results[name] = {
            "protein_a_top_slot": top_a,
            "protein_b_top_slot": top_b,
            "expected_a": expected_a,
            "expected_b": expected_b,
            "a_matches": a_matches,
            "b_matches": b_matches,
            "concordant": concordant,
        }

    results["all_concordant"] = (n_concordant == len(homology_pairs))
    results["n_concordant"] = n_concordant
    results["n_total"] = len(homology_pairs)
    return results
