"""External data downloader for the SupplyMind corpus (public sources, no paid auth).

This is the single tracked script that rebuilds the ``external_data/`` corpus from a
fresh clone. The downloaded data files themselves stay gitignored (large, regeneratable);
only this fetcher + ``PROVENANCE.md`` are tracked. See ``external_data/PROVENANCE.md`` for
per-dataset source URLs, checksums, and consumers.

Sources (all currently live / verified 2026-07-02):
  - SEC EDGAR 10-K filings for 20 supply-chain Fortune 500 companies (data.sec.gov)
  - FRED daily Brent crude (DCOILBRENTEU) + monthly truck-transport PPI (PCU484121484121)
  - Policy papers: FRBNY GSCPI staff report, BIS bottlenecks bulletin, FRBSF supply/demand
    inflation decomposition (all real, on-topic supply-chain economics PDFs)
  - Wikipedia supply-chain crisis / chokepoint articles (via wikipedia-api)
  - World Bank macro indicators (GDP, trade, inflation, container throughput) for CN/DE/IN/JP
  - NOAA NHC active-storm feed (live JSON)

SEC's fair-access policy requires a contact e-mail in the User-Agent. Supply it via the
``SEC_CONTACT_EMAIL`` environment variable (no personal address is baked into the code).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUT = Path(__file__).parent
OUT.mkdir(exist_ok=True)

# NOAA (and some CDN-fronted sites) reject non-browser User-Agents; SEC requires a contact
# e-mail. We keep two header profiles and pick per-source.
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SupplyMind/1.0"


def contact_email() -> str:
    """Return the SEC contact e-mail from the environment, failing loud if unset.

    SEC EDGAR's fair-access policy (https://www.sec.gov/os/webmaster-faq#developers)
    requires a declared contact address in the User-Agent header. We never hardcode a
    personal address; the operator supplies one via SEC_CONTACT_EMAIL.
    """
    email = os.environ.get("SEC_CONTACT_EMAIL", "").strip()
    if not email:
        raise RuntimeError(
            "SEC_CONTACT_EMAIL is not set. SEC EDGAR requires a contact e-mail in the "
            "User-Agent header (fair-access policy). Set it before running, e.g.\n"
            "  PowerShell:  $env:SEC_CONTACT_EMAIL = 'you@example.com'\n"
            "  bash:        export SEC_CONTACT_EMAIL='you@example.com'\n"
            "Add SEC_CONTACT_EMAIL to your .env / .env.example."
        )
    return email


def contact_headers() -> dict:
    return {"User-Agent": f"Sleep-Token-SupplyMind/1.0 ({contact_email()})"}


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, fname: str, hdr: dict, mode: str = "wb") -> bool:
    """Download ``url`` to ``OUT/fname``. Skips if a non-trivial file already exists.

    Returns True on success. Failures are logged loudly (never silently swallowed) and
    return False so callers can surface a partial-corpus warning.
    """
    p = OUT / fname
    if p.exists() and p.stat().st_size > 1000:
        log.info(f"  exists: {fname} ({p.stat().st_size:,} bytes)")
        return True
    try:
        r = requests.get(url, headers=hdr, timeout=60, stream=True)
        if r.status_code != 200:
            log.warning(f"  {fname}: HTTP {r.status_code} from {url}")
            return False
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, mode) as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        log.info(f"  saved: {fname} ({p.stat().st_size:,} bytes)")
        return True
    except Exception as e:  # noqa: BLE001 — logged loudly, not swallowed
        log.warning(f"  {fname}: FETCH FAILED — {e}")
        return False


# ============================================================
# 1. SEC EDGAR — recent 10-K filings for supply-chain Fortune 500
# ============================================================

def sec_edgar(hdr: dict) -> int:
    """Pull the most recent 10-K for 20 major supply-chain companies."""
    log.info("[1] SEC EDGAR 10-K filings...")
    out = OUT / "sec_10k"
    out.mkdir(exist_ok=True)

    companies = {
        "0000320193": "AAPL", "0000789019": "MSFT", "0000051143": "IBM",
        "0000018230": "CAT", "0000040533": "GE", "0001318605": "TSLA",
        "0000936468": "LMT", "0000063908": "MCD", "0000034088": "XOM",
        "0000037996": "F", "0001652044": "GOOG", "0000093410": "CVX",
        "0000078003": "PFE", "0000200406": "JNJ", "0000080424": "PG",
        "0000732717": "T", "0000731766": "UNH", "0000093751": "TGT",
        "0000104169": "WMT", "0000050863": "INTC",
    }

    count = 0
    for cik, ticker in companies.items():
        sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            r = requests.get(sub_url, headers=hdr, timeout=30)
            if r.status_code != 200:
                log.warning(f"  {ticker}: submissions HTTP {r.status_code}")
                continue
            j = r.json()
            forms = j.get("filings", {}).get("recent", {})
            forms_arr = forms.get("form", [])
            accns = forms.get("accessionNumber", [])
            primary = forms.get("primaryDocument", [])
            for i, f in enumerate(forms_arr):
                if f == "10-K":
                    accn = accns[i].replace("-", "")
                    doc = primary[i]
                    doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn}/{doc}"
                    if fetch(doc_url, f"sec_10k/{ticker}_10k.html", hdr):
                        count += 1
                    break
            time.sleep(0.5)  # SEC rate limit (10 req/s max; be polite)
        except Exception as e:  # noqa: BLE001
            log.warning(f"  {ticker}: {e}")

    log.info(f"  SEC: {count}/20 10-Ks present")
    return count


# ============================================================
# 2. FRED time series (daily Brent + monthly truck-transport PPI)
# ============================================================

def fred_series(hdr: dict) -> int:
    """FRED CSV endpoint (keyless). DCOILBRENTEU is the real daily Brent series used by
    the v5 platinum counterfactual (method C); PCU484121484121 is the monthly
    truck-transportation PPI.

    NOTE: the old GLBLALSCMINDX 'supply chain pressure' pull was removed — that series id
    does not exist on FRED (returned HTTP 404). The NY Fed GSCPI is not on FRED; its paper
    is fetched under policy_papers instead.
    """
    log.info("[2] FRED time series (keyless CSV endpoint)...")
    urls = [
        ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU",
         "fred_brent_daily.csv"),          # real daily Brent crude, USD/bbl
        ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCU484121484121",
         "fred_truck_transport.csv"),       # monthly truck-transportation PPI (index)
    ]
    count = sum(1 for u, n in urls if fetch(u, n, hdr))
    log.info(f"  FRED: {count}/{len(urls)} series")
    return count


# ============================================================
# 3. Policy papers (real, on-topic supply-chain economics PDFs)
# ============================================================

def policy_papers(hdr: dict) -> int:
    """Three real supply-chain policy papers. Each URL was verified to return the named
    paper (page-1 title checked), replacing three previously-mislabeled files (an HTML
    error page and two off-topic PDFs on unrelated economics topics).
    """
    log.info("[3] Policy papers (real supply-chain economics PDFs)...")
    papers = [
        # FRBNY Staff Report No. 1017 — "The GSCPI: A New Barometer of Global Supply
        # Chain Pressures" (Benigno, di Giovanni, Groen, Noble; May 2022)
        ("https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr1017.pdf",
         "policy_papers/frbny_gscpi_sr1017.pdf"),
        # BIS Bulletin No 48 — "Bottlenecks: causes and macroeconomic implications"
        # (Rees & Rungcharoenkitkul; Nov 2021)
        ("https://www.bis.org/publ/bisbull48.pdf",
         "policy_papers/bis_bottlenecks_bull48.pdf"),
        # FRBSF Working Paper 2022-18 — "Decomposing Supply and Demand Driven Inflation"
        # (Adam Hale Shapiro)
        ("https://www.frbsf.org/economic-research/wp-content/uploads/sites/4/wp2022-18.pdf",
         "policy_papers/frbsf_supply_demand_inflation_wp2022-18.pdf"),
    ]
    count = sum(1 for u, n in papers if fetch(u, n, hdr))
    log.info(f"  policy papers: {count}/{len(papers)}")
    return count


# ============================================================
# 4. Wikipedia supply-chain crisis / chokepoint articles
# ============================================================

# Canonical article titles (redirects resolved 2026-07-02 via the MediaWiki API). The six
# titles that previously 404'd are mapped to their current canonical pages:
#   COVID-19_supply_chain_crisis            -> 2021–2023 global supply chain crisis
#   Global_supply_chain_issues_(2020–present) -> 2021–2023 global supply chain crisis (dup)
#   2024_Baltimore_bridge_collapse          -> Francis Scott Key Bridge collapse
#   2022_Russian_invasion_of_Ukraine_economic_impact -> Economic impact of the Russian invasion of Ukraine
#   Panama_Canal_drought_2023               -> Panama Canal (drought / draft-restriction chokepoint)
#   North_Field_(Qatar)                     -> South Pars/North Dome Gas-Condensate field
WIKIPEDIA_TITLES = [
    "2011 Tōhoku earthquake and tsunami",
    "2021 Suez Canal obstruction",
    "Ever Given",
    "2020–2023 global chip shortage",
    "2021–2023 global supply chain crisis",       # COVID / global supply-chain crisis
    "Red Sea crisis",
    "Francis Scott Key Bridge collapse",           # 2024 Baltimore bridge collapse
    "Bullwhip effect",
    "Supply chain attack",
    "Just-in-time manufacturing",
    "TSMC",
    "Samsung Electronics",
    "Foxconn",
    "Semiconductor industry",
    "CHIPS and Science Act",
    "Economic impact of the Russian invasion of Ukraine",
    "Port of Los Angeles",
    "Port of Singapore",
    "Panama Canal",                                # 2023-24 drought / draft restrictions
    "South Pars/North Dome Gas-Condensate field",  # North Field (Qatar)
    "Strait of Hormuz",
    "Strait of Malacca",
    "Bab-el-Mandeb",
    "Suez Canal",
    "Baltic Dry Index",
    "Container ship",
    "Supply chain management",
    "Enterprise resource planning",
    "Logistics",
    "Warehouse",
    "Inventory",
]


def wikipedia_crises() -> int:
    """Fetch crisis / chokepoint articles into wikipedia_crises/ (consumed by the v3 RAG
    beast + BEIR eval). Uses wikipedia-api, which resolves redirects to canonical titles.
    """
    log.info("[4] Wikipedia supply-chain crisis articles...")
    try:
        import wikipediaapi
    except Exception:  # noqa: BLE001
        log.error("  wikipedia-api not installed. `pip install wikipedia-api`")
        return 0

    out_dir = OUT / "wikipedia_crises"
    out_dir.mkdir(exist_ok=True)
    wiki = wikipediaapi.Wikipedia(
        user_agent=f"Sleep-Token-SupplyMind ({contact_email()})",
        language="en",
    )

    count = 0
    for title in WIKIPEDIA_TITLES:
        try:
            page = wiki.page(title)
            if not page.exists():
                log.warning(f"  MISSING (no such article): {title}")
                continue
            # Filename follows the requested title with the existing underscore convention
            # (NOT page.title, which may be a redirect target with spaces and would create
            # duplicates alongside the already-cached underscore files).
            dest = out_dir / f"{title.replace(' ', '_').replace('/', '_')}.txt"
            if dest.exists() and dest.stat().st_size > 1000:
                log.info(f"  exists: {dest.name} ({dest.stat().st_size:,} bytes)")
                count += 1
                continue
            dest.write_text(page.text, encoding="utf-8")
            resolved = "" if page.title == title else f"  (redirect -> {page.title!r})"
            log.info(f"  saved: {dest.name} ({dest.stat().st_size:,} bytes){resolved}")
            count += 1
        except Exception as e:  # noqa: BLE001
            log.warning(f"  {title}: FETCH FAILED — {e}")
    log.info(f"  wikipedia: {count}/{len(WIKIPEDIA_TITLES)} articles")
    return count


# ============================================================
# 5. World Bank macro indicators (keyless API v2)
# ============================================================

WB_INDICATORS = {
    "GDP_USD": "NY.GDP.MKTP.CD",
    "GDP_growth": "NY.GDP.MKTP.KD.ZG",
    "Inflation_CPI": "FP.CPI.TOTL.ZG",
    "Exports_pct_GDP": "NE.EXP.GNFS.ZS",
    "Imports_pct_GDP": "NE.IMP.GNFS.ZS",
    "Container_throughput": "IS.SHP.GOOD.TU",
}
WB_COUNTRIES = "CN;DE;IN;JP"  # China, Germany, India, Japan


def worldbank_macro(hdr: dict) -> int:
    """World Bank API v2 (keyless). One JSON per indicator across CN/DE/IN/JP, full history.
    Consumed by versions/v3_arcadia/40_granite/r5_rag_beast.py (world_bank corpus chunks).
    """
    log.info("[5] World Bank macro indicators (keyless API v2)...")
    out_dir = OUT / "world_bank_macro"
    out_dir.mkdir(exist_ok=True)
    count = 0
    for label, indicator in WB_INDICATORS.items():
        url = (f"https://api.worldbank.org/v2/country/{WB_COUNTRIES}/indicator/{indicator}"
               f"?format=json&per_page=500")
        if fetch(url, f"world_bank_macro/wb_{label}.json", hdr):
            count += 1
        time.sleep(0.3)
    log.info(f"  World Bank macro: {count}/{len(WB_INDICATORS)}")
    return count


# ============================================================
# 6. NOAA NHC active storms (live JSON)
# ============================================================

def noaa_storms() -> int:
    """NOAA National Hurricane Center active-storm feed. Requires a browser-style
    User-Agent (the SEC contact UA gets connection-reset by NOAA's CDN).
    """
    log.info("[6] NOAA NHC active storms (live JSON)...")
    hdr = {"User-Agent": BROWSER_UA, "Accept": "application/json"}
    ok = fetch("https://www.nhc.noaa.gov/CurrentStorms.json", "noaa/current_storms.json", hdr)
    log.info(f"  NOAA: {1 if ok else 0}/1")
    return 1 if ok else 0


def main() -> None:
    hdr = contact_headers()  # fails loud here if SEC_CONTACT_EMAIL is unset
    results = {
        "sec_edgar": sec_edgar(hdr),
        "fred_series": fred_series(hdr),
        "policy_papers": policy_papers(hdr),
        "wikipedia_crises": wikipedia_crises(),
        "world_bank_macro": worldbank_macro(hdr),
        "noaa_storms": noaa_storms(),
    }
    (OUT / "fetch_results.json").write_text(json.dumps(results, indent=2))
    log.info(f"\nExternal data fetch complete: {results}")


if __name__ == "__main__":
    main()
