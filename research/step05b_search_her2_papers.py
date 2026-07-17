#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 05b — Search for HER2 Resistance Phosphoproteomics Papers            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    One-time search script to find published HER2/ERBB2 phosphoproteomic     ║
║    datasets for the ERBB family expansion.                                   ║
║    Sources: Europe PMC, PRIDE/ProteomeXchange, Google Scholar                ║
║                                                                              ║
║  OUTPUT:                                                                     ║
║    research/her2_phospho_papers.json  — Europe PMC papers                   ║
║    research/her2_proteomexchange.json — PRIDE deposited datasets            ║
║    research/her2_scholar_papers.json  — Google Scholar results              ║
║                                                                              ║
║  WORKFLOW:                                                                   ║
║    1. Run this script: python scripts/step05b_search_her2_papers.py         ║
║    2. Review output JSON files                                               ║
║    3. Manually download supplementary data from selected papers             ║
║    4. Place downloaded files in data/raw/drugptm/her2_*/                    ║
║    5. Add processing functions to step05_download_drugptm.py                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import ssl
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from datetime import datetime

import certifi

# Build a default SSL context using certifi's CA bundle
# (macOS Python often cannot locate the system root certificates)
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PubMed / Europe PMC Search
# ══════════════════════════════════════════════════════════════════════════════

EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

PUBMED_QUERIES = [
    # ── PRIORITY 1: RECENT (2023–2026) — phosphoproteomics data papers ────
    {
        "query": '(HER2 OR ERBB2) AND phosphoproteom* AND "breast cancer" AND (FIRST_PDATE:[2023 TO 2026])',
        "description": "2023-2026: HER2 phosphoproteomics in breast cancer",
    },
    {
        "query": 'ERBB2 AND phosphopeptide* AND (resistance OR resistant) AND (FIRST_PDATE:[2023 TO 2026])',
        "description": "2023-2026: ERBB2 phosphopeptide + resistance",
    },
    {
        "query": '(HER2 OR ERBB2) AND (TMT OR SILAC OR "label-free" OR DIA) AND phospho* AND (FIRST_PDATE:[2023 TO 2026])',
        "description": "2023-2026: HER2 quantitative MS phosphoproteomics",
    },
    {
        "query": '(lapatinib OR trastuzumab OR neratinib OR tucatinib OR "T-DXd") AND phosphoproteom* AND (FIRST_PDATE:[2023 TO 2026])',
        "description": "2023-2026: HER2 drugs + phosphoproteomics",
    },
    {
        "query": '"breast cancer" AND phosphoproteom* AND (resistance OR resistant) AND (FIRST_PDATE:[2024 TO 2026])',
        "description": "2024-2026: Breast cancer phosphoproteomics + resistance",
    },
    {
        "query": '(HER2 OR ERBB2) AND "mass spectrometry" AND phospho* AND (resistance OR drug) AND (FIRST_PDATE:[2023 TO 2026])',
        "description": "2023-2026: HER2 MS-based phospho + drug/resistance",
    },
    {
        "query": 'phosphoproteom* AND "drug resistance" AND "breast cancer" AND (FIRST_PDATE:[2024 TO 2026])',
        "description": "2024-2026: Drug resistance phosphoproteomics in breast cancer",
    },
    {
        "query": '(HER2 OR ERBB2) AND phospho* AND (PRIDE OR ProteomeXchange OR "data deposited") AND (FIRST_PDATE:[2023 TO 2026])',
        "description": "2023-2026: HER2 phospho with deposited data",
    },
    {
        "query": '(HER2 OR ERBB2) AND ("drug-tolerant persister" OR DTP) AND phospho* AND (FIRST_PDATE:[2023 TO 2026])',
        "description": "2023-2026: HER2 drug-tolerant persister phospho",
    },
    {
        "query": '"breast cancer" AND phosphoproteom* AND (lapatinib OR trastuzumab OR neratinib OR tucatinib) AND (FIRST_PDATE:[2023 TO 2026])',
        "description": "2023-2026: Breast cancer phosphoproteomics + HER2 drugs",
    },

    # ── PRIORITY 2: SLIGHTLY OLDER but with real phospho data (2020–2022) ──
    {
        "query": '(HER2 OR ERBB2) AND phosphoproteom* AND (resistance OR resistant) AND (FIRST_PDATE:[2020 TO 2022])',
        "description": "2020-2022: HER2 phosphoproteomics + resistance",
    },
    {
        "query": '(HER2 OR ERBB2) AND phosphoproteom* AND (lapatinib OR trastuzumab OR neratinib) AND (FIRST_PDATE:[2020 TO 2022])',
        "description": "2020-2022: HER2 phosphoproteomics + drugs",
    },
    {
        "query": '(TMT OR SILAC) AND (HER2 OR ERBB2) AND phosphopeptide* AND "breast cancer" AND (FIRST_PDATE:[2020 TO 2022])',
        "description": "2020-2022: Quantitative phosphopeptide data in HER2+ breast",
    },
    {
        "query": '"breast cancer" AND phosphoproteom* AND "drug resistance" AND "mass spectrometry" AND (FIRST_PDATE:[2020 TO 2022])',
        "description": "2020-2022: Breast cancer phosphoproteomics + drug resistance MS data",
    },
]


def search_europe_pmc(query: str, max_results: int = 25) -> list[dict]:
    """
    Search Europe PMC (covers PubMed + PMC + preprints) via REST API.
    Returns list of paper metadata dicts.
    """
    params = {
        "query": query,
        "format": "json",
        "pageSize": str(max_results),
        "resultType": "core",  # includes abstract
        "sort": "CITED desc",  # sort by citation count
    }

    url = f"{EUROPE_PMC_API}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PTM-Resistance-Project/1.0"})
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"    ⚠ API error: {e}")
        return []

    results = []
    result_list = data.get("resultList", {}).get("result", [])

    for paper in result_list:
        entry = {
            "pmid": paper.get("pmid", ""),
            "pmcid": paper.get("pmcid", ""),
            "doi": paper.get("doi", ""),
            "title": paper.get("title", ""),
            "authorString": paper.get("authorString", ""),
            "journalTitle": paper.get("journalTitle", ""),
            "pubYear": paper.get("pubYear", ""),
            "abstractText": paper.get("abstractText", "")[:500] if paper.get("abstractText") else "",
            "citedByCount": paper.get("citedByCount", 0),
            "isOpenAccess": paper.get("isOpenAccess", "N"),
            "source": paper.get("source", ""),
        }

        # Check for data availability keywords in abstract
        abstract_lower = (entry["abstractText"] or "").lower()
        entry["has_data_keywords"] = any(kw in abstract_lower for kw in [
            "proteomexchange", "pride", "massive", "supplementary",
            "dataset", "deposited", "data availability", "raw data",
            "phosphosite", "phosphopeptide",
        ])

        # Flag papers likely to have downloadable phospho data
        entry["likely_has_phospho_data"] = any(kw in abstract_lower for kw in [
            "phosphoproteom", "phosphopeptide", "phos-tag",
            "tmt", "silac", "label-free", "dia-ms",
            "phosphosite", "kinase substrate",
        ])

        results.append(entry)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ProteomeXchange / PRIDE Search
# ══════════════════════════════════════════════════════════════════════════════

PRIDE_API = "https://www.ebi.ac.uk/pride/ws/archive/v2/search/projects"

PRIDE_QUERIES = [
    "HER2 phospho",
    "ERBB2 phospho",
    "HER2 resistance",
    "breast cancer phosphoproteomics",
    "lapatinib phospho",
    "trastuzumab phospho",
    "breast cancer kinase",
    "HER2 proteomics",
]


def search_pride(query: str, max_results: int = 15) -> list[dict]:
    """
    Search PRIDE/ProteomeXchange for deposited proteomics datasets.
    """
    params = {
        "keyword": query,
        "pageSize": str(max_results),
        "page": "0",
        "sortDirection": "DESC",
        "sortConditions": "submissionDate",
    }

    url = f"{PRIDE_API}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "PTM-Resistance-Project/1.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"    ⚠ PRIDE API error: {e}")
        return []

    results = []

    # Handle both list and dict responses
    projects = data if isinstance(data, list) else data.get("_embedded", {}).get("compactprojects", [])

    for proj in projects:
        entry = {
            "accession": proj.get("accession", ""),
            "title": proj.get("title", ""),
            "projectDescription": (proj.get("projectDescription", "") or "")[:500],
            "submissionDate": proj.get("submissionDate", ""),
            "publicationDate": proj.get("publicationDate", ""),
            "organisms": proj.get("organisms", []),
            "instrumentNames": proj.get("instrumentNames", []),
            "projectTags": proj.get("projectTags", []),
            "numFiles": proj.get("numAssays", 0),
        }

        # Check if it's phosphoproteomics
        desc_lower = (entry["projectDescription"] + " " + entry["title"]).lower()
        entry["is_phospho"] = any(kw in desc_lower for kw in [
            "phospho", "ptm", "kinase", "phos-tag",
        ])
        entry["is_her2"] = any(kw in desc_lower for kw in [
            "her2", "erbb2", "erb-b2", "breast cancer",
        ])

        results.append(entry)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Google Scholar Search
# ══════════════════════════════════════════════════════════════════════════════

SCHOLAR_QUERIES = [
    # Targeted: parental vs resistant phosphoproteomics in HER2+ cells
    'BT-474 lapatinib resistant phosphoproteomics parental ERBB2',
    'SKBR3 lapatinib resistant phosphoproteomics HER2 breast cancer',
    'trastuzumab resistant BT-474 phosphoproteomics phosphopeptide fold change',
    'HER2 breast cancer lapatinib resistance phosphoproteomics log2 fold change',
    'ERBB2 amplified breast cancer phosphoproteomics resistance quantitative mass spectrometry',
    'HER2 positive breast cancer phosphoproteomics parental versus resistant cell line',
    'lapatinib neratinib phosphoproteomics HER2 breast cancer TMT SILAC 2024 2025',
    'trastuzumab resistance phosphoproteomics BT474 SKBR3 AU565 HCC1954',
    'HER2 acquired resistance phosphoproteome breast cancer drug treatment',
    'ERBB2 phosphosite quantification resistant sensitive breast cancer cells',
]


def search_google_scholar(query: str, max_results: int = 10) -> list[dict]:
    """
    Search Google Scholar using the scholarly library.
    Returns list of paper metadata dicts.
    """
    try:
        from scholarly import scholarly
    except ImportError:
        print("    ⚠ scholarly library not available — skipping Google Scholar")
        return []

    results = []
    try:
        search_results = scholarly.search_pubs(query)
        for i, paper in enumerate(search_results):
            if i >= max_results:
                break

            bib = paper.get("bib", {})
            entry = {
                "title": bib.get("title", ""),
                "author": bib.get("author", ""),
                "pub_year": bib.get("pub_year", ""),
                "venue": bib.get("venue", "") or bib.get("journal", ""),
                "abstract": (bib.get("abstract", "") or "")[:500],
                "num_citations": paper.get("num_citations", 0),
                "url_scholarbib": paper.get("url_scholarbib", ""),
                "pub_url": paper.get("pub_url", "") or paper.get("eprint_url", ""),
                "source": "google_scholar",
            }

            # Check for data availability keywords in abstract
            abstract_lower = (entry["abstract"] or "").lower()
            title_lower = (entry["title"] or "").lower()
            combined = abstract_lower + " " + title_lower

            entry["likely_has_phospho_data"] = any(kw in combined for kw in [
                "phosphoproteom", "phosphopeptide", "phos-tag",
                "tmt", "silac", "label-free", "dia-ms",
                "phosphosite", "kinase substrate", "kinome",
            ])

            entry["has_data_keywords"] = any(kw in combined for kw in [
                "proteomexchange", "pride", "supplementary",
                "dataset", "deposited", "data availability",
                "phosphosite", "phosphopeptide",
            ])

            results.append(entry)

            # Be polite to Google — short delay between result fetches
            time.sleep(0.3)

    except Exception as e:
        print(f"    ⚠ Google Scholar error: {e}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main Search Execution
# ══════════════════════════════════════════════════════════════════════════════

def run_search():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  STEP 05b: Search for HER2 Phosphoproteomics Papers        ║")
    print("║  Sources: Europe PMC · PRIDE · Google Scholar               ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ══════════════════════════════════════════════════════════════════
    # PART 1: PubMed / Europe PMC Search
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  {'='*55}")
    print(f"  PART 1: PubMed / Europe PMC Search")
    print(f"  {'='*55}")

    all_papers = {}  # Deduplicate by PMID or DOI
    query_results = []

    for i, qinfo in enumerate(PUBMED_QUERIES):
        query = qinfo["query"]
        desc = qinfo["description"]
        print(f"\n  Query {i+1}/{len(PUBMED_QUERIES)}: {desc}")
        print(f"    Search: {query[:80]}...")

        papers = search_europe_pmc(query, max_results=20)
        print(f"    Found: {len(papers)} results")

        # Deduplicate
        new_count = 0
        for p in papers:
            key = p.get("pmid") or p.get("doi") or p.get("title", "")[:50]
            if key and key not in all_papers:
                all_papers[key] = p
                p["found_by_query"] = desc
                new_count += 1

        query_results.append({
            "query": query,
            "description": desc,
            "n_results": len(papers),
            "n_new": new_count,
        })

        # Show top 3 results
        for p in papers[:3]:
            phospho_flag = "📊" if p.get("likely_has_phospho_data") else "  "
            data_flag = "💾" if p.get("has_data_keywords") else "  "
            print(f"    {phospho_flag}{data_flag} [{p.get('pubYear','')}] "
                  f"{p.get('title','')[:70]}...")
            if p.get("pmid"):
                print(f"         PMID: {p['pmid']} | {p.get('journalTitle','')}")

        time.sleep(0.5)  # Rate limiting

    # ── Rank papers by relevance ────────────────────────────────────
    ranked_papers = sorted(
        all_papers.values(),
        key=lambda p: (
            p.get("likely_has_phospho_data", False),
            p.get("has_data_keywords", False),
            p.get("citedByCount", 0),
        ),
        reverse=True,
    )

    print(f"\n\n  {'='*55}")
    print(f"  TOP PAPERS — Europe PMC (ranked by phospho data likelihood)")
    print(f"  {'='*55}")
    print(f"  📊 = likely has phospho data | 💾 = mentions data deposit")

    for i, p in enumerate(ranked_papers[:20]):
        phospho_flag = "📊" if p.get("likely_has_phospho_data") else "  "
        data_flag = "💾" if p.get("has_data_keywords") else "  "
        oa_flag = "🔓" if p.get("isOpenAccess") == "Y" else "  "

        print(f"\n  {i+1:2d}. {phospho_flag}{data_flag}{oa_flag} {p.get('title','')[:80]}")
        print(f"      {p.get('authorString','')[:60]}")
        print(f"      {p.get('journalTitle','')} ({p.get('pubYear','')}) "
              f"| Cited: {p.get('citedByCount',0)}")
        if p.get("pmid"):
            print(f"      PMID: {p['pmid']}", end="")
        if p.get("doi"):
            print(f" | DOI: {p['doi']}", end="")
        print()
        if p.get("abstractText"):
            print(f"      Abstract: {p['abstractText'][:150]}...")

    # ══════════════════════════════════════════════════════════════════
    # PART 2: PRIDE / ProteomeXchange Search
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n  {'='*55}")
    print(f"  PART 2: PRIDE / ProteomeXchange Datasets")
    print(f"  {'='*55}")

    all_datasets = {}

    for query in PRIDE_QUERIES:
        print(f"\n  PRIDE query: '{query}'")
        datasets = search_pride(query, max_results=10)
        print(f"    Found: {len(datasets)} datasets")

        for d in datasets:
            key = d.get("accession", "")
            if key and key not in all_datasets:
                all_datasets[key] = d
                phospho_flag = "📊" if d.get("is_phospho") else "  "
                her2_flag = "🎯" if d.get("is_her2") else "  "
                print(f"    {phospho_flag}{her2_flag} {key}: {d.get('title','')[:60]}...")

        time.sleep(0.5)

    # Filter for relevant datasets
    relevant_datasets = [
        d for d in all_datasets.values()
        if d.get("is_phospho") or d.get("is_her2")
    ]

    if relevant_datasets:
        print(f"\n  RELEVANT PRIDE DATASETS ({len(relevant_datasets)} found):")
        for d in relevant_datasets:
            print(f"    {d['accession']}: {d.get('title','')[:70]}")
            print(f"      Date: {d.get('publicationDate','')}")

    # ══════════════════════════════════════════════════════════════════
    # PART 3: Google Scholar Search
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n  {'='*55}")
    print(f"  PART 3: Google Scholar Search")
    print(f"  {'='*55}")

    all_scholar = {}  # Deduplicate by title

    for i, query in enumerate(SCHOLAR_QUERIES):
        print(f"\n  Scholar query {i+1}/{len(SCHOLAR_QUERIES)}: {query[:60]}...")

        papers = search_google_scholar(query, max_results=8)
        print(f"    Found: {len(papers)} results")

        for p in papers:
            # Deduplicate by normalized title
            key = p.get("title", "").lower().strip()[:60]
            if key and key not in all_scholar:
                all_scholar[key] = p
                phospho_flag = "📊" if p.get("likely_has_phospho_data") else "  "
                data_flag = "💾" if p.get("has_data_keywords") else "  "
                year = p.get("pub_year", "")
                cites = p.get("num_citations", 0)
                print(f"    {phospho_flag}{data_flag} [{year}] (cited:{cites}) "
                      f"{p.get('title','')[:60]}...")

        # Longer delay between Scholar queries to avoid rate-limiting
        time.sleep(3)

    # Rank Scholar papers
    ranked_scholar = sorted(
        all_scholar.values(),
        key=lambda p: (
            p.get("likely_has_phospho_data", False),
            p.get("has_data_keywords", False),
            p.get("num_citations", 0),
        ),
        reverse=True,
    )

    print(f"\n\n  {'='*55}")
    print(f"  TOP PAPERS — Google Scholar (ranked)")
    print(f"  {'='*55}")

    for i, p in enumerate(ranked_scholar[:15]):
        phospho_flag = "📊" if p.get("likely_has_phospho_data") else "  "
        data_flag = "💾" if p.get("has_data_keywords") else "  "
        print(f"\n  {i+1:2d}. {phospho_flag}{data_flag} {p.get('title','')[:80]}")
        author = p.get("author", "")
        if isinstance(author, list):
            author = ", ".join(author[:3])
        print(f"      {str(author)[:60]}")
        print(f"      {p.get('venue','')} ({p.get('pub_year','')}) "
              f"| Cited: {p.get('num_citations',0)}")
        if p.get("pub_url"):
            print(f"      URL: {p['pub_url']}")
        if p.get("abstract"):
            print(f"      Abstract: {p['abstract'][:150]}...")

    # ══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════════

    # Save PubMed results
    pubmed_output = {
        "search_date": timestamp,
        "n_queries": len(PUBMED_QUERIES),
        "n_unique_papers": len(all_papers),
        "n_with_phospho_data": sum(1 for p in all_papers.values()
                                    if p.get("likely_has_phospho_data")),
        "n_with_data_deposit": sum(1 for p in all_papers.values()
                                    if p.get("has_data_keywords")),
        "query_summary": query_results,
        "papers": ranked_papers,
    }

    pubmed_path = RESEARCH_DIR / "her2_phospho_papers.json"
    with open(pubmed_path, "w") as f:
        json.dump(pubmed_output, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ PubMed results saved: {pubmed_path}")
    print(f"    Total unique papers: {len(all_papers)}")
    print(f"    With phospho data: {pubmed_output['n_with_phospho_data']}")
    print(f"    With data deposit: {pubmed_output['n_with_data_deposit']}")

    # Save PRIDE results
    pride_output = {
        "search_date": timestamp,
        "n_queries": len(PRIDE_QUERIES),
        "n_unique_datasets": len(all_datasets),
        "n_relevant": len(relevant_datasets),
        "datasets": list(all_datasets.values()),
    }

    pride_path = RESEARCH_DIR / "her2_proteomexchange.json"
    with open(pride_path, "w") as f:
        json.dump(pride_output, f, indent=2, ensure_ascii=False)
    print(f"  ✓ PRIDE results saved: {pride_path}")
    print(f"    Total datasets: {len(all_datasets)}")
    print(f"    Relevant (phospho + HER2): {len(relevant_datasets)}")

    # Save Google Scholar results
    scholar_output = {
        "search_date": timestamp,
        "n_queries": len(SCHOLAR_QUERIES),
        "n_unique_papers": len(all_scholar),
        "n_with_phospho_data": sum(1 for p in all_scholar.values()
                                    if p.get("likely_has_phospho_data")),
        "n_with_data_deposit": sum(1 for p in all_scholar.values()
                                    if p.get("has_data_keywords")),
        "papers": ranked_scholar,
    }

    scholar_path = RESEARCH_DIR / "her2_scholar_papers.json"
    with open(scholar_path, "w") as f:
        json.dump(scholar_output, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Google Scholar results saved: {scholar_path}")
    print(f"    Total unique papers: {len(all_scholar)}")
    print(f"    With phospho data: {scholar_output['n_with_phospho_data']}")

    # ══════════════════════════════════════════════════════════════════
    # COMBINED SUMMARY
    # ══════════════════════════════════════════════════════════════════
    total_unique = len(all_papers) + len(all_scholar)
    total_phospho = (pubmed_output['n_with_phospho_data'] +
                     scholar_output['n_with_phospho_data'])

    print(f"\n\n  {'='*55}")
    print(f"  COMBINED SUMMARY")
    print(f"  {'='*55}")
    print(f"  Europe PMC papers:  {len(all_papers):3d}  (phospho: {pubmed_output['n_with_phospho_data']})")
    print(f"  Google Scholar:     {len(all_scholar):3d}  (phospho: {scholar_output['n_with_phospho_data']})")
    print(f"  PRIDE datasets:     {len(all_datasets):3d}  (relevant: {len(relevant_datasets)})")
    print(f"  ────────────────────────────────────")
    print(f"  Total papers:       {total_unique:3d}  (with phospho data: {total_phospho})")

    # ══════════════════════════════════════════════════════════════════
    # ACTION ITEMS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n  {'='*55}")
    print(f"  NEXT STEPS")
    print(f"  {'='*55}")
    print(f"""
  1. Review the three output JSON files:
     → research/her2_phospho_papers.json   (Europe PMC)
     → research/her2_scholar_papers.json   (Google Scholar)
     → research/her2_proteomexchange.json  (PRIDE datasets)

  2. Look for papers with 📊 (phospho data) and 💾 (deposited data)
     → Priority: supplementary tables with per-site phosphorylation
       fold-changes in HER2+ cell lines

  3. For each selected paper:
     → Download supplementary data files
     → Place in data/raw/drugptm/her2_<author>_<year>/
     → Add a process_<paper>() function to step05_download_drugptm.py

  4. Target data to find:
     → BT-474 + Lapatinib: parental vs resistant phosphoproteomics
     → SKBR3 + Lapatinib/Neratinib: dose-response phospho
     → Any HER2+ cell line + TKI resistance phosphoproteomics
     → Patient-derived HER2+ tumor phosphoproteomics

  5. Known papers to prioritize:
     → Stuhlmiller et al., Cell Rep 2015: kinome reprogramming, Lapatinib
     → Rexer et al., Cancer Res 2011: Lapatinib-resistant BT-474
     → Chandarlapaty et al., Cancer Cell 2012: PI3K in Trastuzumab resistance
""")

    print(f"✓ Step 05b complete! Review the JSON files and download selected data.")


if __name__ == "__main__":
    run_search()
