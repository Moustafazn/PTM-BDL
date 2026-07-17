#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 03 — Download ERBB Family 3D Crystal Structures (PDB)                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    Download experimentally determined 3D crystal structures of the EGFR      ║
║    and HER2 (ERBB2) kinase domains from the Protein Data Bank (PDB).        ║
║    These structures capture how mutations physically alter drug binding.      ║
║                                                                              ║
║  ERBB FAMILY EXPANSION (v2):                                                 ║
║    Now includes HER2/ERBB2 structures alongside EGFR:                        ║
║    • 3PP0 — HER2 kinase domain (apo, active, 2.25Å)                        ║
║    • 3RCD — HER2 kinase + Lapatinib (DFG-out inactive, 2.40Å)              ║
║    Key finding (Section 7a HER2_EXPANSION_PLAN.md):                          ║
║    • Neratinib NOT in GDSC2 → replaced with Sapitinib                       ║
║    • GDSC2 tissue = "Breast Carcinoma" (not "BRCA")                        ║
║                                                                              ║
║  WHY PDB STRUCTURES?                                                         ║
║    • Sequence alone can't capture 3D spatial relationships                   ║
║    • Drug binding depends on the SHAPE of the ATP-binding pocket             ║
║    • Different mutations cause different conformational changes              ║
║    • T790M introduces a bulkier residue → steric clash with 1st-gen TKIs    ║
║    • C797S removes the cysteine → Osimertinib can't form covalent bond      ║
║    • GearNet converts these 3D coordinates into graph embeddings            ║
║                                                                              ║
║  STRUCTURE SELECTION:                                                        ║
║    We select structures spanning key biological states:                       ║
 ║    ┌─────────┬─────────────────────┬──────────────────────────────────┐     ║
 ║    │ PDB ID  │ EGFR State          │ Drug Complex                     │     ║
 ║    ├─────────┼─────────────────────┼──────────────────────────────────┤     ║
 ║    │ 2GS6    │ Wild-type (apo)     │ None (baseline)                  │     ║
 ║    │ 2ITY    │ Wild-type           │ Gefitinib (1st-gen)              │     ║
 ║    │ 4ZAU    │ Wild-type           │ Osimertinib (3rd-gen)            │     ║
 ║    │ 5EDP    │ L858R/T790M (apo)   │ None (double mutant)             │     ║
 ║    │ 4G5J    │ Wild-type           │ Afatinib (2nd-gen)               │     ║
 ║    │ 3IKA    │ T790M               │ WZ4002 (3rd-gen covalent)        │     ║
 ║    │ 6LUD    │ L858R/T790M/C797S   │ Osimertinib (full resistance)    │     ║
 ║    └─────────┴─────────────────────┴──────────────────────────────────┘     ║
║                                                                              ║
║  OUTPUT:                                                                     ║
║    data/raw/pdb/*.pdb              — raw PDB coordinate files                ║
║    data/processed/pdb/structure_catalog.csv — enriched structural metadata   ║
║                                                                              ║
║  WHAT THE ENRICHED CATALOG PROVIDES (for Steps 06, 08, 10):                  ║
║    • best_chain      — which chain to use (Step 08 GearNet needs this)       ║
║    • resolution      — crystallographic quality (Å)                          ║
║    • num_residues     — residue count and range                              ║
║    • res_790/797/858 — amino acid at key mutation sites (validation)         ║
║    • ligand_ids      — drug molecules in the structure                       ║
║    • ptm_sites_resolved — which phospho sites are structurally resolved      ║
║      (Step 10 needs this for site-specific vs. global PTM modulation)        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import json
import yaml
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

RAW_DIR = PROJECT_ROOT / cfg["paths"]["raw_data"] / "pdb"           # input: downloaded PDB files
OUT_DIR = PROJECT_ROOT / cfg["paths"]["processed_data"] / "pdb"     # output: structure catalog
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Download PDB Structures
# ══════════════════════════════════════════════════════════════════════════════

MANUAL_DOWNLOAD_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  MANUAL DOWNLOAD REQUIRED — PDB Crystal Structures                          ║
║                                                                              ║
║  Download each PDB file from RCSB and save to: {raw_dir}/                   ║
║                                                                              ║
║  For each PDB ID below, visit:                                              ║
║    https://www.rcsb.org/structure/<PDB_ID>                                  ║
║  Click "Download Files" → "PDB Format"                                      ║
║  Save as: <PDB_ID>.pdb  in the folder above.                               ║
║                                                                              ║
║  Required structures:                                                        ║
{structure_list}║                                                                              ║
║  Or download all at once via command line:                                   ║
║    cd {raw_dir}                                                              ║
║    for id in {pdb_ids}; do                                                  ║
║      curl -o $id.pdb https://files.rcsb.org/download/$id.pdb               ║
║    done                                                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def check_pdb_structure(pdb_id: str, description: str = "") -> Path | None:
    """Check if a PDB file exists in the raw directory."""
    pdb_id = pdb_id.upper()
    dest_path = RAW_DIR / f"{pdb_id}.pdb"
    
    if dest_path.exists():
        print(f"  ✓ {pdb_id}: Found — {description}")
        return dest_path
    else:
        print(f"  ✗ {pdb_id}: MISSING — {description}")
        print(f"         → Download: https://files.rcsb.org/download/{pdb_id}.pdb")
        return None


def download_all_structures():
    """
    Download all EGFR structures defined in config.yaml.
    
    WHY THESE SPECIFIC STRUCTURES:
    ──────────────────────────────
    We carefully selected structures to cover the biological state space:
    
    1. MUTATION COVERAGE:
       - Wild-type (baseline for comparison)
       - Single mutants (L858R, T790M, C797S each individually)
       - Double mutants (L858R+T790M = the classic Osimertinib-sensitive combo)
       - Triple mutants (if available, L858R+T790M+C797S = full resistance)
    
    2. DRUG COVERAGE:
       - Apo structures (no drug) — shows natural conformation
       - Drug-bound structures — shows how each TKI generation binds
       - Comparing apo vs. bound reveals drug-induced conformational changes
    
    3. RESOLUTION:
       - We prefer structures with resolution < 3.0 Å for accuracy
       - Higher resolution = more precise atomic coordinates
    
    For GearNet processing (Step 08), each PDB will be converted to:
       - A graph where nodes = amino acid residues (Cα atoms)
       - Edges = spatial proximity (residues within ~10Å of each other)
       - Node features = amino acid type, secondary structure, B-factor
       - This captures the 3D geometry of the kinase pocket
    """
    print("\n" + "="*70)
    print("STEP 3.1: Downloading EGFR Crystal Structures from PDB")
    print("="*70)
    
    structures = cfg["pdb"]["structures"]
    downloaded = []
    
    for struct in structures:
        pdb_id = struct["id"]
        desc = struct["description"]
        mutations = struct.get("mutations", [])
        drug = struct.get("drug", "none")
        
        path = check_pdb_structure(pdb_id, desc)
        
        if path:
            # Determine target_protein: HER2 structures have target_protein in config
            target_protein = struct.get("target_protein", "EGFR")
            downloaded.append({
                "pdb_id": pdb_id,
                "description": desc,
                "target_protein": target_protein,
                "mutations": "; ".join(mutations) if mutations else "wild_type",
                "drug": drug or "apo",
                "mapping_role": struct.get("mapping_role", ""),
                "file_path": str(path)
            })
    
    # Save structure catalog
    catalog = pd.DataFrame(downloaded)
    catalog_path = OUT_DIR / "structure_catalog.csv"
    catalog.to_csv(catalog_path, index=False)
    print(f"\n  ✓ Structure catalog saved: {catalog_path}")
    
    return catalog


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Validate, Characterize & Save Structural Metadata
# ══════════════════════════════════════════════════════════════════════════════

# 3-letter → 1-letter amino acid mapping
AA3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# The 12 phosphorylation sites tracked in the project (from Step 04).
# Step 10 needs to know which of these are resolved in each crystal structure
# to decide site-specific vs. global PTM conditioning.
PTM_PHOSPHO_SITES = [845, 991, 992, 998, 1039, 1041, 1045, 1068, 1086, 1101, 1148, 1173]


def _get_resolution(pdb_path: Path) -> float | None:
    """Extract crystallographic resolution from PDB REMARK 2 records."""
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("REMARK   2 RESOLUTION."):
                parts = line.split()
                for i, token in enumerate(parts):
                    if token == "RESOLUTION.":
                        try:
                            return float(parts[i + 1])
                        except (IndexError, ValueError):
                            pass
    return None


# ── Expected residues at key UniProt positions (for validation) ───────────
# These are what we expect to see at each key position based on mutation status.
# Wild-type EGFR: T790, C797, L858
# Mutants: T790→M (T790M), C797→S (C797S), L858→R (L858R)
EXPECTED_WT_RESIDUES = {790: "T", 797: "C", 858: "L"}
EXPECTED_MUTANT_RESIDUES = {790: "M", 797: "S", 858: "R"}


def validate_and_characterize(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Validate downloaded PDB files and extract structural metadata that
    downstream steps need.
    
    For each structure, we extract and SAVE:
    1. File integrity (can it be parsed?)
    2. Resolution — crystallographic quality
    3. best_chain — chain with most residues (Step 08 GearNet needs this
       instead of blindly defaulting to chain "A")
    4. num_residues, residue_start, residue_end — structural coverage
    5. Ligand IDs — drug molecules present in the structure
    6. res_790, res_797, res_858 — amino acid identity at key mutation
       positions (validates that mutations match expectations)
    7. ptm_sites_resolved — which of the 12 phosphorylation sites from
       Step 04 are structurally resolved (Step 10 needs this to decide
       whether to inject PTM levels at specific residue nodes or use
       global conditioning; the C-terminal tail where most phospho sites
       reside is typically disordered in kinase domain crystals)
    
    Returns the catalog DataFrame enriched with all extracted columns.
    """
    print("\n" + "="*70)
    print("STEP 3.2: Validating & Characterizing PDB Structures")
    print("="*70)
    
    try:
        from Bio.PDB import PDBParser
    except ImportError:
        print("  ⚠ BioPython PDB module not available. Skipping validation.")
        print("    Install with: pip install biopython")
        return catalog
    
    parser = PDBParser(QUIET=True)
    
    # New columns we will populate
    catalog["resolution"] = None
    catalog["best_chain"] = "A"
    catalog["num_residues"] = None
    catalog["residue_start"] = None
    catalog["residue_end"] = None
    catalog["ligand_ids"] = ""
    catalog["res_790"] = ""
    catalog["res_797"] = ""
    catalog["res_858"] = ""
    catalog["ptm_sites_resolved"] = ""      # comma-separated resolved site positions
    catalog["ptm_sites_resolved_count"] = 0
    
    # Detailed analysis dict for JSON output
    detailed_analysis = {}
    
    for idx, row in catalog.iterrows():
        pdb_id = row["pdb_id"]
        pdb_file = Path(row["file_path"])
        
        if not pdb_file.exists():
            print(f"\n  ⚠ {pdb_id}: file not found — skipping")
            continue
        
        print(f"\n  Validating {pdb_id}...")
        
        try:
            structure = parser.get_structure(pdb_id, pdb_file)
            model = structure[0]
            
            # ── Resolution ────────────────────────────────────────────────
            resolution = _get_resolution(pdb_file)
            catalog.at[idx, "resolution"] = resolution
            if resolution:
                print(f"    Resolution: {resolution:.2f} Å")
            
            # ── Find best chain (most standard residues) ──────────────────
            chains = list(model.get_chains())
            best_chain_id = "A"
            best_count = -1
            chain_info = {}
            
            for chain in chains:
                std_residues = [r for r in chain.get_residues() if r.id[0] == " "]
                het_residues = [r for r in chain.get_residues()
                               if r.id[0] != " " and r.id[0] != "W"]
                
                n_std = len(std_residues)
                chain_info[chain.id] = {
                    "num_residues": n_std,
                    "first_residue": std_residues[0].id[1] if std_residues else None,
                    "last_residue": std_residues[-1].id[1] if std_residues else None,
                    "ligands": sorted(set(r.resname for r in het_residues)),
                }
                
                if n_std > best_count:
                    best_count = n_std
                    best_chain_id = chain.id
            
            catalog.at[idx, "best_chain"] = best_chain_id
            print(f"    Chains: {[c.id for c in chains]} → best: {best_chain_id}")
            
            # ── Residue info for best chain ───────────────────────────────
            best_chain = model[best_chain_id]
            std_residues = [r for r in best_chain.get_residues() if r.id[0] == " "]
            
            if std_residues:
                catalog.at[idx, "num_residues"] = len(std_residues)
                catalog.at[idx, "residue_start"] = std_residues[0].id[1]
                catalog.at[idx, "residue_end"] = std_residues[-1].id[1]
                print(f"    Chain {best_chain_id}: {len(std_residues)} residues "
                      f"({std_residues[0].id[1]}–{std_residues[-1].id[1]})")
            
            # ── Ligands (non-water HETATM) ────────────────────────────────
            het_residues = [r for r in best_chain.get_residues()
                          if r.id[0] != " " and r.id[0] != "W"]
            if het_residues:
                het_names = sorted(set(r.resname for r in het_residues))
                catalog.at[idx, "ligand_ids"] = "|".join(het_names)
                print(f"    Ligands: {het_names}")
            
            # ── Key mutation residues (790, 797, 858) ─────────────────────
            # Check what amino acid is at each key position in the PDB.
            # Expected for WT: T790, C797, L858
            # Expected for mutants: M790 (T790M), S797 (C797S), R858 (L858R)
            # If the residue doesn't match → the PDB may use non-standard
            # numbering (e.g. 2GS6 has a -24 offset). We flag this as a
            # warning but still record what's actually at that position.
            expected_mutations = row.get("mutations", "wild_type")
            key_positions = {"790": 790, "797": 797, "858": 858}
            for col_suffix, pos in key_positions.items():
                try:
                    residue = best_chain[(" ", pos, " ")]
                    aa_1 = AA3_TO_1.get(residue.resname, "?")
                    catalog.at[idx, f"res_{col_suffix}"] = aa_1
                    
                    # Validate against expected
                    wt_aa = EXPECTED_WT_RESIDUES[pos]
                    mut_aa = EXPECTED_MUTANT_RESIDUES[pos]
                    if aa_1 in (wt_aa, mut_aa):
                        print(f"    Residue {pos}: {aa_1} ({residue.resname}) ✓")
                    else:
                        print(f"    Residue {pos}: {aa_1} ({residue.resname}) "
                              f"⚠ unexpected (expected {wt_aa} or {mut_aa} "
                              f"— PDB may use non-standard numbering)")
                except KeyError:
                    catalog.at[idx, f"res_{col_suffix}"] = "—"
                    print(f"    Residue {pos}: NOT RESOLVED ✗")
            
            # ── PTM phosphorylation site coverage (for Step 10) ───────────
            # Check which of the 12 phospho sites from Step 04 are resolved.
            # Uses PDB residue numbering directly (works for most structures;
            # structures with non-standard numbering like 2GS6 may report
            # fewer resolved sites, which is safe — Step 10 will use global
            # PTM conditioning for those).
            resolved_sites = []
            for pos in PTM_PHOSPHO_SITES:
                try:
                    residue = best_chain[(" ", pos, " ")]
                    resolved_sites.append(str(pos))
                except KeyError:
                    pass
            
            catalog.at[idx, "ptm_sites_resolved"] = ",".join(resolved_sites)
            catalog.at[idx, "ptm_sites_resolved_count"] = len(resolved_sites)
            
            n_total = len(PTM_PHOSPHO_SITES)
            print(f"    PTM sites resolved: {len(resolved_sites)}/{n_total}")
            if resolved_sites:
                print(f"      Resolved: {', '.join(resolved_sites)}")
            missing_sites = [str(p) for p in PTM_PHOSPHO_SITES
                            if str(p) not in resolved_sites]
            if missing_sites:
                print(f"      Missing:  {', '.join(missing_sites)}")
            
            # ── Save detailed analysis for JSON ──────────────────────────
            detailed_analysis[pdb_id] = {
                "resolution": resolution,
                "best_chain": best_chain_id,
                "chain_info": chain_info,
                "num_residues": len(std_residues) if std_residues else 0,
                "residue_start": std_residues[0].id[1] if std_residues else None,
                "residue_end": std_residues[-1].id[1] if std_residues else None,
                "ligand_ids": sorted(set(r.resname for r in het_residues)) if het_residues else [],
                "key_residues": {
                    str(pos): catalog.at[idx, f"res_{col_suffix}"]
                    for col_suffix, pos in key_positions.items()
                },
                "ptm_sites_resolved": resolved_sites,
                "ptm_sites_resolved_count": len(resolved_sites),
                "ptm_sites_total": n_total,
            }
                    
        except Exception as e:
            print(f"    ✗ Validation error: {e}")
    
    # ── Save enriched catalog ─────────────────────────────────────────────
    catalog_path = OUT_DIR / "structure_catalog.csv"
    catalog.to_csv(catalog_path, index=False)
    print(f"\n  ✓ Enriched structure catalog saved: {catalog_path}")
    print(f"    Columns: {list(catalog.columns)}")
    
    # ── Save detailed JSON analysis ───────────────────────────────────────
    json_path = OUT_DIR / "structure_analysis.json"
    with open(json_path, "w") as f:
        json.dump(detailed_analysis, f, indent=2, default=str)
    print(f"  ✓ Detailed analysis saved: {json_path}")
    
    return catalog


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  STEP 03: Download ERBB Family 3D Crystal Structures       ║")
    print("║  (EGFR + HER2/ERBB2)                                      ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Source: RCSB Protein Data Bank (https://www.rcsb.org/)    ║")
    print("║  Output: PDB files for EGFR + HER2 kinase domain variants  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    # Step 1: Check which PDB files are present (saves basic catalog)
    catalog = download_all_structures()
    
    # Step 2: Validate & characterize — extracts structural metadata from
    # the actual PDB files and enriches the catalog with columns needed by:
    #   Step 08: best_chain (which chain GearNet should parse)
    #   Step 10: ptm_sites_resolved (which phospho sites are in the structure)
    catalog = validate_and_characterize(catalog)
    
    print("\n✓ Step 03 complete!")
    print("  Enriched catalog ready for:")
    print("    • Step 06 (harmonization — mutation validation via res_790/797/858)")
    print("    • Step 08 (GearNet — best_chain, residue range)")
    print("    • Step 10 (PTM integration — ptm_sites_resolved)")
