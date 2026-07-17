#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 05b — Search for N-Glycosylation Papers (EGFR & ERBB2/HER2)         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    Search Europe PMC for recent papers with quantitative N-glycosylation     ║
║    data for EGFR and ERBB2 proteins, to expand the glycosylation data       ║
║    layer in the PTM-BDL module beyond the current MCP 2025 source           ║
║    (48 EGFR + 4 HER2 glyco measurements).                                   ║
║                                                                              ║
║  SEARCH CRITERIA:                                                            ║
║    1. Quantitative glycoproteomics (site-specific, mass spectrometry)       ║
║    2. EGFR or HER2/ERBB2 specific                                          ║
║    3. Drug resistance context preferred (TKI, antibody)                     ║
║    4. Recent publications (2020-2026)                                       ║
║    5. Downloadable supplementary data                                       ║
║                                                                              ║
║  CURRENT GLYCO DATA INVENTORY:                                              ║
║    EGFR: 48 rows from MCP 2025 (8 sites: N128,N175,N352,N361,N413,         ║
║          N528,N568,N603) across H1975, H3255, PC-9                          ║
║    HER2: 4 rows from MCP 2025 (N530 only, 4 glycan compositions)           ║
║                                                                              ║
║  KNOWN EGFR N-GLYCOSYLATION SITES (UniProt P00533):                        ║
║    N56, N73, N128, N175, N196, N352, N361, N413, N444, N528, N568,         ║
║    N603, N623 (12-13 sites in extracellular domain)                         ║
║                                                                              ║
║  KNOWN HER2 N-GLYCOSYLATION SITES (UniProt P04626):                        ║
║    N68, N124, N187, N259, N530, N571, N629 (7 sites)                       ║
║                                                                              ║
║  OUTPUT: Prints ranked paper candidates for manual review                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import subprocess
import time
import urllib.parse
from datetime import datetime


def epmc_search(query: str, label: str, page_size: int = 5) -> list[dict]:
    """
    Search Europe PMC REST API using curl (avoids Python SSL issues).
    Returns list of result dicts.
    """
    params = {
        "query": query,
        "format": "json",
        "resultType": "core",
        "pageSize": str(page_size),
        "sort": "CITED desc",
    }
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{urllib.parse.urlencode(params)}"

    try:
        result = subprocess.run(
            ["curl", "-s", "-m", "20",
             "-H", "User-Agent: PTM-Glyco-Search/1.0",
             url],
            capture_output=True, text=True, timeout=25
        )
        if result.returncode != 0:
            print(f"\n  [{label}] CURL ERROR: {result.stderr[:200]}")
            return []

        data = json.loads(result.stdout)
        results = data.get("resultList", {}).get("result", [])
        return results

    except Exception as e:
        print(f"\n  [{label}] ERROR: {e}")
        return []


def print_paper(r: dict, rank: int, label: str):
    """Pretty-print a single paper result."""
    title = r.get("title", "N/A")
    year = r.get("pubYear", "?")
    journal = r.get("journalTitle", "?")
    pmid = r.get("pmid", "")
    doi = r.get("doi", "")
    pmc = r.get("pmcid", "")
    cited = r.get("citedByCount", 0)
    abstract = r.get("abstractText", "") or ""

    print(f"\n  {'='*72}")
    print(f"  [{label}] Paper #{rank}")
    print(f"  {'='*72}")
    print(f"  Title:    {title[:120]}")
    print(f"  Year:     {year} | Journal: {journal}")
    print(f"  PMID:     {pmid} | DOI: {doi}")
    print(f"  PMC:      {pmc} | Cited by: {cited}")
    if abstract:
        # Show first 400 chars, looking for glyco-relevant keywords
        print(f"  Abstract: {abstract[:400]}...")
        # Highlight glyco keywords
        glyco_kw = ["glycosyl", "glycan", "glycoprot", "N-linked", "N-glyc",
                     "lectin", "PNGase", "mannose", "fucose", "sialic",
                     "GlcNAc", "galactose"]
        found = [kw for kw in glyco_kw if kw.lower() in abstract.lower()]
        if found:
            print(f"  🔑 Glyco keywords: {', '.join(found)}")

        drug_kw = ["TKI", "osimertinib", "gefitinib", "erlotinib", "afatinib",
                    "lapatinib", "trastuzumab", "cetuximab", "resistance",
                    "drug", "inhibitor", "treatment"]
        found_drug = [kw for kw in drug_kw if kw.lower() in abstract.lower()]
        if found_drug:
            print(f"  💊 Drug keywords: {', '.join(found_drug)}")

        quant_kw = ["mass spec", "proteomics", "quantitative", "LC-MS",
                     "TMT", "SILAC", "DIA", "LFQ", "glycoproteom",
                     "site-specific", "intact glycopeptide"]
        found_quant = [kw for kw in quant_kw if kw.lower() in abstract.lower()]
        if found_quant:
            print(f"  📊 Quantitative keywords: {', '.join(found_quant)}")


def search_specific_papers():
    """Search for specific known papers by DOI/PMCID."""
    print("\n" + "▓"*74)
    print("  SECTION 0: Verifying Known/Referenced Papers")
    print("▓"*74)

    known_papers = [
        ("DOI:10.1016/j.mcpro.2025.100917",
         "MCP 2025 (Abe) — Fe-ZIC-cHILIC phospho+glyco (ALREADY IN DATASET)"),
        ("DOI:10.3390/cancers18030474",
         "Zhu 2026 — N361 glycosylation effects on EGFR function"),
        ("DOI:10.1093/glycob/cwad100",
         "Glycobiology 2024 — EGFR/ERBB glycosylation review"),
        ("DOI:10.1002/mas.21882",
         "Mass Spec Rev 2023 — Glycoproteomics methods review"),
        ("DOI:10.1016/j.jbc.2022.101950",
         "JBC 2022 — EGFR glycoform analysis"),
        ("DOI:10.1093/glycob/cwaf066",
         "Glycobiology 2025 — HER2 glycosylation"),
        ("DOI:10.1016/j.trecan.2022.02.008",
         "Trends Cancer 2022 — Glycosylation in cancer drug resistance"),
        ("DOI:10.1093/jb/mvad065",
         "J Biochem 2023 — EGFR N-glycosylation functional analysis"),
    ]

    for query, label in known_papers:
        results = epmc_search(query, label, page_size=1)
        if results:
            r = results[0]
            print(f"\n  ✅ {label}")
            print(f"     {r.get('title', '')[:100]}")
            print(f"     Year: {r.get('pubYear','')} | PMID: {r.get('pmid','')} | "
                  f"Cited: {r.get('citedByCount', 0)}")
        else:
            print(f"\n  ❌ {label} — NOT FOUND in Europe PMC")
        time.sleep(0.5)


def search_egfr_glyco():
    """Search for EGFR N-glycosylation papers."""
    print("\n\n" + "▓"*74)
    print("  SECTION 1: EGFR N-Glycosylation Papers (Quantitative Data)")
    print("▓"*74)

    queries = [
        # Query 1: Site-specific EGFR glycoproteomics
        (
            '(TITLE_ABS:"EGFR" OR TITLE_ABS:"epidermal growth factor receptor") '
            'AND (TITLE_ABS:"N-glycosyl*" OR TITLE_ABS:"glycoproteom*" OR TITLE_ABS:"glycopeptide") '
            'AND (TITLE_ABS:"mass spectrometry" OR TITLE_ABS:"proteomics" OR TITLE_ABS:"LC-MS") '
            'AND (PUB_YEAR:[2020 TO 2026])',
            "EGFR-Q1: Site-specific glycoproteomics (MS-based)"
        ),
        # Query 2: EGFR glycosylation + drug resistance
        (
            '(TITLE_ABS:"EGFR" OR TITLE_ABS:"epidermal growth factor receptor") '
            'AND (TITLE_ABS:"glycosyl*" OR TITLE_ABS:"glycan") '
            'AND (TITLE_ABS:"resistance" OR TITLE_ABS:"TKI" OR TITLE_ABS:"inhibitor") '
            'AND (PUB_YEAR:[2020 TO 2026])',
            "EGFR-Q2: Glycosylation + drug resistance"
        ),
        # Query 3: EGFR glycoform analysis
        (
            '(TITLE_ABS:"EGFR") '
            'AND (TITLE_ABS:"glycoform" OR TITLE_ABS:"intact glycopeptide" '
            'OR TITLE_ABS:"N-linked glycan") '
            'AND (PUB_YEAR:[2018 TO 2026])',
            "EGFR-Q3: Glycoform / intact glycopeptide analysis"
        ),
        # Query 4: EGFR glycosylation + NSCLC / lung cancer
        (
            '(TITLE_ABS:"EGFR") '
            'AND (TITLE_ABS:"glycosyl*" OR TITLE_ABS:"glycan") '
            'AND (TITLE_ABS:"lung cancer" OR TITLE_ABS:"NSCLC") '
            'AND (PUB_YEAR:[2020 TO 2026])',
            "EGFR-Q4: Glycosylation + lung cancer"
        ),
        # Query 5: Broader — EGFR N-glycosylation function
        (
            '(TITLE:"EGFR" OR TITLE:"EGF receptor") '
            'AND (TITLE:"glycosyl*" OR TITLE:"N-glycan" OR TITLE:"N-linked") '
            'AND (PUB_YEAR:[2020 TO 2026])',
            "EGFR-Q5: Title-specific glycosylation function"
        ),
        # Query 6: EGFR glycosylation sites quantitative
        (
            '(TITLE_ABS:"EGFR") '
            'AND (TITLE_ABS:"N-glycosylation site" OR TITLE_ABS:"glycosylation site") '
            'AND (TITLE_ABS:"quantitative" OR TITLE_ABS:"mass spec") '
            'AND (PUB_YEAR:[2018 TO 2026])',
            "EGFR-Q6: Specific glycosylation site quantification"
        ),
        # Query 7: Cetuximab glycosylation (cetuximab binds EGFR domain III
        # where N528 glycosylation site is)
        (
            '(TITLE_ABS:"EGFR") '
            'AND (TITLE_ABS:"cetuximab" OR TITLE_ABS:"panitumumab") '
            'AND (TITLE_ABS:"glycosyl*") '
            'AND (PUB_YEAR:[2020 TO 2026])',
            "EGFR-Q7: Cetuximab + EGFR glycosylation"
        ),
    ]

    all_results = {}
    for query, label in queries:
        print(f"\n  Searching: {label}...")
        results = epmc_search(query, label, page_size=5)
        print(f"  → Found {len(results)} results")

        for i, r in enumerate(results, 1):
            pmid = r.get("pmid", r.get("id", ""))
            if pmid and pmid not in all_results:
                all_results[pmid] = (r, label, i)
                print_paper(r, i, label)

        time.sleep(0.5)

    return all_results


def search_her2_glyco():
    """Search for HER2/ERBB2 N-glycosylation papers."""
    print("\n\n" + "▓"*74)
    print("  SECTION 2: HER2/ERBB2 N-Glycosylation Papers (Quantitative Data)")
    print("▓"*74)

    queries = [
        # Query 1: HER2 glycoproteomics (MS-based)
        (
            '(TITLE_ABS:"HER2" OR TITLE_ABS:"ERBB2" OR TITLE_ABS:"ErbB-2") '
            'AND (TITLE_ABS:"N-glycosyl*" OR TITLE_ABS:"glycoproteom*" OR TITLE_ABS:"glycopeptide") '
            'AND (TITLE_ABS:"mass spectrometry" OR TITLE_ABS:"proteomics" OR TITLE_ABS:"LC-MS") '
            'AND (PUB_YEAR:[2020 TO 2026])',
            "HER2-Q1: Site-specific glycoproteomics (MS-based)"
        ),
        # Query 2: HER2 glycosylation + drug resistance
        (
            '(TITLE_ABS:"HER2" OR TITLE_ABS:"ERBB2") '
            'AND (TITLE_ABS:"glycosyl*" OR TITLE_ABS:"glycan") '
            'AND (TITLE_ABS:"resistance" OR TITLE_ABS:"trastuzumab" OR TITLE_ABS:"lapatinib") '
            'AND (PUB_YEAR:[2020 TO 2026])',
            "HER2-Q2: Glycosylation + drug resistance"
        ),
        # Query 3: HER2 glycoform / intact glycopeptide
        (
            '(TITLE_ABS:"HER2" OR TITLE_ABS:"ERBB2") '
            'AND (TITLE_ABS:"glycoform" OR TITLE_ABS:"intact glycopeptide" '
            'OR TITLE_ABS:"N-linked glycan") '
            'AND (PUB_YEAR:[2018 TO 2026])',
            "HER2-Q3: Glycoform / intact glycopeptide analysis"
        ),
        # Query 4: HER2 glycosylation + breast cancer
        (
            '(TITLE_ABS:"HER2" OR TITLE_ABS:"ERBB2") '
            'AND (TITLE_ABS:"glycosyl*") '
            'AND (TITLE_ABS:"breast cancer") '
            'AND (PUB_YEAR:[2020 TO 2026])',
            "HER2-Q4: Glycosylation + breast cancer"
        ),
        # Query 5: Trastuzumab glycosylation binding
        (
            '(TITLE_ABS:"trastuzumab") '
            'AND (TITLE_ABS:"glycosyl*" OR TITLE_ABS:"glycan") '
            'AND (TITLE_ABS:"HER2" OR TITLE_ABS:"ERBB2") '
            'AND (PUB_YEAR:[2020 TO 2026])',
            "HER2-Q5: Trastuzumab binding + glycosylation"
        ),
        # Query 6: HER2 N-glycosylation site-specific
        (
            '(TITLE_ABS:"HER2" OR TITLE_ABS:"ERBB2") '
            'AND (TITLE_ABS:"N-glycosylation site" OR TITLE_ABS:"glycosylation site") '
            'AND (PUB_YEAR:[2018 TO 2026])',
            "HER2-Q6: N-glycosylation site-specific"
        ),
    ]

    all_results = {}
    for query, label in queries:
        print(f"\n  Searching: {label}...")
        results = epmc_search(query, label, page_size=5)
        print(f"  → Found {len(results)} results")

        for i, r in enumerate(results, 1):
            pmid = r.get("pmid", r.get("id", ""))
            if pmid and pmid not in all_results:
                all_results[pmid] = (r, label, i)
                print_paper(r, i, label)

        time.sleep(0.5)

    return all_results


def search_combined_glyco():
    """Search for combined phospho+glyco or multi-PTM papers."""
    print("\n\n" + "▓"*74)
    print("  SECTION 3: Combined Phospho+Glyco / Multi-PTM Papers")
    print("▓"*74)

    queries = [
        # Query 1: Combined phospho + glyco enrichment
        (
            '(TITLE_ABS:"phospho*" AND TITLE_ABS:"glyco*") '
            'AND (TITLE_ABS:"EGFR" OR TITLE_ABS:"HER2" OR TITLE_ABS:"ERBB" '
            'OR TITLE_ABS:"lung cancer" OR TITLE_ABS:"breast cancer") '
            'AND (TITLE_ABS:"mass spec*" OR TITLE_ABS:"proteom*") '
            'AND (PUB_YEAR:[2020 TO 2026])',
            "MULTI-Q1: Phospho+glyco combined enrichment"
        ),
        # Query 2: PTM crosstalk glycosylation phosphorylation
        (
            '(TITLE_ABS:"glycosyl*") AND (TITLE_ABS:"phosphoryl*") '
            'AND (TITLE_ABS:"crosstalk" OR TITLE_ABS:"interplay" OR TITLE_ABS:"cross-talk") '
            'AND (TITLE_ABS:"receptor" OR TITLE_ABS:"RTK" OR TITLE_ABS:"kinase") '
            'AND (PUB_YEAR:[2020 TO 2026])',
            "MULTI-Q2: Phospho-glyco crosstalk in RTKs"
        ),
        # Query 3: Cancer glycoproteomics cell lines
        (
            '(TITLE_ABS:"glycoproteom*") '
            'AND (TITLE_ABS:"cell line" OR TITLE_ABS:"cell lines") '
            'AND (TITLE_ABS:"lung" OR TITLE_ABS:"NSCLC" OR TITLE_ABS:"breast") '
            'AND (TITLE_ABS:"quantitative" OR TITLE_ABS:"site-specific") '
            'AND (PUB_YEAR:[2022 TO 2026])',
            "MULTI-Q3: Cancer cell line glycoproteomics"
        ),
        # Query 4: N-glycan profiling drug response
        (
            '(TITLE_ABS:"glycan profiling" OR TITLE_ABS:"glycan analysis" '
            'OR TITLE_ABS:"glycomic") '
            'AND (TITLE_ABS:"drug" OR TITLE_ABS:"treatment" OR TITLE_ABS:"resistance") '
            'AND (TITLE_ABS:"cancer") '
            'AND (PUB_YEAR:[2022 TO 2026])',
            "MULTI-Q4: Glycan profiling + drug response in cancer"
        ),
    ]

    all_results = {}
    for query, label in queries:
        print(f"\n  Searching: {label}...")
        results = epmc_search(query, label, page_size=5)
        print(f"  → Found {len(results)} results")

        for i, r in enumerate(results, 1):
            pmid = r.get("pmid", r.get("id", ""))
            if pmid and pmid not in all_results:
                all_results[pmid] = (r, label, i)
                print_paper(r, i, label)

        time.sleep(0.5)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  STEP 05b: Search N-Glycosylation Papers (EGFR & ERBB2/HER2)  ║")
    print("║  Purpose: Expand glyco data for PTM-BDL module                 ║")
    print(f"║  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}                                        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    print("\n  Current glyco data inventory:")
    print("    EGFR:  48 rows from MCP 2025 (8 sites)")
    print("    HER2:  4 rows from MCP 2025 (N530 only)")
    print("    TOTAL: 52 glyco measurements — NEED MORE DATA")
    print("\n  Search criteria:")
    print("    ✓ Quantitative glycoproteomics (site-specific MS)")
    print("    ✓ EGFR or HER2/ERBB2 specific")
    print("    ✓ Drug resistance context preferred")
    print("    ✓ Recent (2020-2026)")
    print("    ✓ Downloadable supplementary data")

    # Section 0: Verify known/referenced papers
    search_specific_papers()

    # Section 1: EGFR glyco papers
    egfr_results = search_egfr_glyco()

    # Section 2: HER2 glyco papers
    her2_results = search_her2_glyco()

    # Section 3: Combined multi-PTM papers
    multi_results = search_combined_glyco()

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n\n" + "▓"*74)
    print("  SUMMARY: Unique Papers Found")
    print("▓"*74)

    all_unique = {}
    for results, section in [(egfr_results, "EGFR"),
                              (her2_results, "HER2"),
                              (multi_results, "MULTI")]:
        for pmid, (r, label, rank) in results.items():
            if pmid not in all_unique:
                all_unique[pmid] = (r, section, label)

    print(f"\n  Total unique papers found: {len(all_unique)}")
    print(f"    EGFR-specific:  {len(egfr_results)}")
    print(f"    HER2-specific:  {len(her2_results)}")
    print(f"    Multi-PTM:      {len(multi_results)}")

    print("\n  ─── All Unique Papers (sorted by year, descending) ───")
    sorted_papers = sorted(all_unique.items(),
                           key=lambda x: x[1][0].get("pubYear", "0"),
                           reverse=True)
    for i, (pmid, (r, section, label)) in enumerate(sorted_papers, 1):
        title = r.get("title", "N/A")[:100]
        year = r.get("pubYear", "?")
        journal = r.get("journalTitle", "?")
        doi = r.get("doi", "")
        cited = r.get("citedByCount", 0)
        print(f"\n  {i:2d}. [{section}] {title}")
        print(f"      Year: {year} | Journal: {journal} | "
              f"PMID: {pmid} | Cited: {cited}")
        print(f"      DOI: {doi}")
        print(f"      Found via: {label}")

    print("\n\n" + "═"*74)
    print("  NEXT STEPS:")
    print("  1. Review papers above for quantitative glycosite data")
    print("  2. Check supplementary tables for downloadable glycopeptide data")
    print("  3. Select top 5 per protein (EGFR & ERBB2)")
    print("  4. Download and integrate into step05 pipeline")
    print("═"*74)
