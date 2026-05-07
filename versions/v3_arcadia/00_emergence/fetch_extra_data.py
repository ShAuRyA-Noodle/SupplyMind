"""Fetch extra free data: UN COMTRADE + IMF IFS + Wikipedia supply-chain-crisis articles."""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "external_data"
OUT.mkdir(parents=True, exist_ok=True)

HDR = {"User-Agent": "Sleep-Token-SupplyMind (paneermomos10@gmail.com)"}

results: dict = {}


def fetch(url: str, dest: Path, mode: str = "wb") -> int:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 1000:
            return dest.stat().st_size
        r = requests.get(url, headers=HDR, timeout=60, stream=True)
        if r.status_code != 200:
            return 0
        with open(dest, mode) as f:
            for ch in r.iter_content(8192):
                f.write(ch)
        return dest.stat().st_size
    except Exception as e:
        print(f"    fail {dest.name}: {str(e)[:120]}")
        return 0


# ============================================================
# 1) UN COMTRADE (no-auth public preview endpoint)
# ============================================================
def fetch_comtrade():
    print("[1] UN COMTRADE trade flows...")
    out_dir = OUT / "un_comtrade"
    out_dir.mkdir(exist_ok=True)
    count = 0
    # Top reporters × HS=ALL × year 2023 preview (no auth)
    for rep in ["842", "276", "156", "392", "356"]:  # USA, DEU, CHN, JPN, IND
        url = f"https://comtradeapi.un.org/public/v1/preview/C/A/HS/all/2023/00/{rep}"
        dest = out_dir / f"comtrade_{rep}_2023.json"
        sz = fetch(url, dest)
        if sz > 1000:
            count += 1
            print(f"    saved {dest.name} ({sz/1e6:.2f} MB)")
        time.sleep(0.5)
    return {"downloaded": count, "dir": str(out_dir)}


# ============================================================
# 2) IMF IFS (public JSON Data Services)
# ============================================================
def fetch_imf_ifs():
    print("[2] IMF IFS macro indicators...")
    out_dir = OUT / "imf_ifs"
    out_dir.mkdir(exist_ok=True)
    count = 0
    # Free "DataServices" endpoints - CPI, GDP, Trade balance for key supply chain countries
    indicators = [
        ("PCPI_IX", "CPI_Index"),
        ("NGDP_R_SA_XDC", "GDP_RealSeasonallyAdjusted"),
        ("TXG_FOB_USD", "Exports_USD"),
        ("TMG_CIF_USD", "Imports_USD"),
    ]
    countries = ["US", "CN", "DE", "JP", "IN"]
    for ind_code, label in indicators:
        for c in countries:
            url = f"https://www.imf.org/external/datamapper/api/v1/{ind_code}/{c}"
            dest = out_dir / f"imf_{ind_code}_{c}.json"
            sz = fetch(url, dest)
            if sz > 100:
                count += 1
            time.sleep(0.3)
    print(f"    total IMF files: {count}")
    return {"downloaded": count, "dir": str(out_dir)}


# ============================================================
# 3) Wikipedia supply-chain-crisis articles
# ============================================================
def fetch_wikipedia():
    print("[3] Wikipedia supply-chain crisis articles...")
    out_dir = OUT / "wikipedia_crises"
    out_dir.mkdir(exist_ok=True)
    try:
        import wikipediaapi
    except Exception:
        import subprocess
        subprocess.run(["pip", "install", "-q", "wikipedia-api"], check=False)
        import wikipediaapi

    wiki = wikipediaapi.Wikipedia(
        user_agent="Sleep-Token-SupplyMind (paneermomos10@gmail.com)",
        language="en",
    )

    titles = [
        "2011_Tōhoku_earthquake_and_tsunami",
        "2021_Suez_Canal_obstruction",
        "Ever_Given",
        "2020–2023_global_chip_shortage",
        "COVID-19_supply_chain_crisis",
        "Red_Sea_crisis",
        "2024_Baltimore_bridge_collapse",
        "Global_supply_chain_issues_(2020–present)",
        "Bullwhip_effect",
        "Supply_chain_attack",
        "Just-in-time_manufacturing",
        "TSMC",
        "Samsung_Electronics",
        "Foxconn",
        "Semiconductor_industry",
        "CHIPS_and_Science_Act",
        "2022_Russian_invasion_of_Ukraine_economic_impact",
        "Port_of_Los_Angeles",
        "Port_of_Singapore",
        "Panama_Canal_drought_2023",
        "North_Field_(Qatar)",
        "Strait_of_Hormuz",
        "Strait_of_Malacca",
        "Bab-el-Mandeb",
        "Suez_Canal",
        "Baltic_Dry_Index",
        "Container_ship",
        "Supply_chain_management",
        "Enterprise_resource_planning",
        "Logistics",
        "Warehouse",
        "Inventory",
    ]

    count = 0
    for t in titles:
        p = wiki.page(t)
        if p.exists():
            dest = out_dir / f"{t.replace('/', '_')}.txt"
            dest.write_text(p.text, encoding="utf-8")
            count += 1
    print(f"    wikipedia: {count}/{len(titles)} articles")
    return {"downloaded": count, "of_attempted": len(titles), "dir": str(out_dir)}


def main():
    results["un_comtrade"] = fetch_comtrade()
    results["imf_ifs"] = fetch_imf_ifs()
    results["wikipedia_crises"] = fetch_wikipedia()
    (OUT / "extra_data_results.json").write_text(json.dumps(results, indent=2))
    print("\nAll extra data fetches complete:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
