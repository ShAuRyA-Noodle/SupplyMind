"""causal_methods.py — REAL 4-method causal counterfactual on historical analogs.

R6 rebuild (REBUILD_BACKLOG.md). The deleted version drew ``rng.normal`` samples
and labelled them "ARIMA-BSTS" / "do-calculus". This module implements the four
methods *for real* on documented historical supply-chain shocks. Default analog:
the 2011 Tōhoku earthquake (auto / semiconductor supply chain).

Each method estimates the counterfactual USD impact of the shock (actual world
minus the no-shock counterfactual), using a DIFFERENT data source and estimand.
They therefore measure DIFFERENT SCOPES and are NOT expected to agree tightly —
the honest signal is the *spread* and *which scope each method captures*:

  A. Paired-bootstrap Monte-Carlo   — the engine's real MonteCarloEngine
     (server/engine/monte_carlo.py). N seeded rollouts of the disrupted world;
     bootstrap CI on the mean simulated customer-revenue-at-risk. A genuinely
     PAIRED secondary compares no-mitigation vs a real safety-stock action
     (golden-path "value of acting"). SCOPE: modelled auto supply-graph subset.
  B. Synthetic control (Abadie)      — real World-Bank macro series. Treated unit
     = Japan (fetched live from the keyless World-Bank API; donors China/Germany/
     India from external_data/world_bank_macro/*.json on disk). Convex donor
     weights fit on the pre-2011 GDP-growth path; 2011 gap × Japan GDP = macro
     impact. SCOPE: whole-economy.
  C. ARIMA(p,1,0)                    — real FRED daily Brent (DCOILBRENTEU,
     external_data/fred_brent_daily.csv). AR(p) (AIC-selected) fit on the
     pre-event window, forecast the counterfactual price path, compare to actual;
     per-bbl gap × Japan crude-import volume. SCOPE: oil-price channel only
     (heavily confounded by the concurrent Libya / Arab-Spring rally — surfaced).
  D. Do-calculus / SCM intervention  — the engine's real supply-chain DAG. do(all
     Japan tier-1 suppliers offline) via graph node intervention, then the
     engine's own path-based get_total_revenue_at_risk(). SCOPE: modelled auto
     supply-graph subset (structural / deterministic).

statsmodels is not installed in the .venv, so method C uses a clean AR(p) fit by
OLS on the differenced series (numpy/scipy) — an honest ARIMA(p,1,0). No random
numbers stand in for a model anywhere; the only RNG use is a genuine bootstrap /
Monte-Carlo resampler with a fixed seed.

Reproduce:
  .venv/Scripts/python.exe -m supplymind.phoenix.counterfactual_v2.causal_methods \
      --analog tohoku_2011 --receipt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
WB_MACRO_DIR = REPO_ROOT / "external_data" / "world_bank_macro"
FRED_BRENT_CSV = REPO_ROOT / "external_data" / "fred_brent_daily.csv"
GRAPH_DIR = REPO_ROOT / "server" / "data" / "graphs"
TREATED_CACHE = Path(__file__).resolve().parent / "_wb_treated_cache.json"


# =====================================================================
# Documented real outcomes (for the bracket check — CITED, never used as a
# method input; these only tell us whether the estimates land near reality).
# =====================================================================

DOCUMENTED_OUTCOMES: dict[str, dict] = {
    "tohoku_2011": {
        "headline_low_usd": 210_000_000_000,
        "headline_high_usd": 235_000_000_000,
        "citations": [
            "World Bank (2011-03-21): total economic damage of the Great East "
            "Japan Earthquake estimated at up to US$235B — the costliest natural "
            "disaster on record.",
            "Japan Cabinet Office (2011-06): direct capital-stock damage ¥16.9 "
            "trillion ≈ US$210B.",
            "Toyota FY2011 (2011-06 earnings): quake cut operating income by "
            "≈¥110B ≈ US$1.2B and ≈800,000 units of lost production.",
            "Renesas Naka fab (auto MCUs) offline ~3 months — the single largest "
            "automotive semiconductor disruption of 2011.",
        ],
        "firm_level_usd": {"toyota_operating_income_hit": 1_200_000_000},
        "scope_note": "Headline $210–235B is WHOLE-ECONOMY damage; the auto "
                      "industry is a subset, and any single-firm / supply-graph "
                      "estimate should be well below the headline.",
    },
    "suez_2021": {
        "headline_low_usd": 6 * 9_600_000_000,
        "headline_high_usd": 54_000_000_000,
        "citations": [
            "Lloyd's List (2021-03-29): Ever Given blockage held up ≈US$9.6B/day "
            "of trade; canal blocked 6 days (23–29 Mar 2021).",
            "Allianz (2021): estimated the blockage cost global trade $6–10B/week.",
        ],
        "scope_note": "Global-trade flow disruption, not a single-country loss.",
    },
    "hormuz_2019": {
        "headline_low_usd": None,
        "headline_high_usd": None,
        "citations": [
            "Strait of Hormuz carries ≈21M bbl/day (~20% of global oil, EIA). "
            "Jun–Jul 2019 tanker attacks spiked Brent and insurance war-risk "
            "premia; no single consensus USD total exists (chokepoint-risk event).",
        ],
        "scope_note": "Oil-price / chokepoint-risk channel; no consensus headline.",
    },
}


# =====================================================================
# Analog configuration — real dates, real nodes, real series windows.
# =====================================================================

ANALOGS: dict[str, dict] = {
    "tohoku_2011": {
        "event_date": "2011-03-11",
        "description": "Great East Japan (Tōhoku) earthquake + tsunami; "
                       "auto / semiconductor supply chain.",
        # Method A + D: the engine's hard (automotive) graph. Japan tier-1
        # suppliers that actually failed (Renesas MCUs, Denso, Murata).
        "graph_file": "hard_graph.json",
        "do_nodes": ["SUP_RENESAS", "SUP_DENSO", "SUP_MURATA"],
        "mc_severity": 0.85,
        "mc_duration_days": 90.0,          # Renesas Naka fab down ~3 months
        "mitigation_warehouse": "WH_JAPAN",
        "mitigation_extra_stock_days": 90,
        # Method B: synthetic control.
        "sc_treated": "Japan",
        "sc_treated_iso3": "JPN",
        "sc_donors": ["China", "Germany", "India"],
        "sc_event_year": 2011,
        "sc_pre_years": list(range(2000, 2011)),
        # Method C: FRED Brent oil-price channel.
        "fred_pre_days": 60,
        "fred_post_days": 20,
        "oil_import_bbl_per_day": 3_500_000,  # Japan crude imports ≈3.5M bbl/d (EIA 2011)
        "oil_confounder": "Concurrent Libyan civil war / Arab-Spring rally drove "
                          "Brent from ~$95 (Feb) to ~$115 (early Mar) 2011; the "
                          "Tōhoku oil signal is confounded by it — surfaced, not hidden.",
    },
    "suez_2021": {
        "event_date": "2021-03-23",
        "description": "Ever Given grounding, Suez Canal (23–29 Mar 2021).",
        "graph_file": "hard_graph.json",
        "do_nodes": ["PORT_HAMBURG"],       # Europe-facing port (Suez-Europe trade)
        "mc_severity": 0.8,
        "mc_duration_days": 6.0,
        "mitigation_warehouse": "WH_GERMANY",
        "mitigation_extra_stock_days": 20,
        "sc_treated": "Germany",
        "sc_treated_iso3": "DEU",
        "sc_donors": ["China", "India", "Japan"],
        "sc_event_year": 2021,
        "sc_pre_years": list(range(2010, 2021)),
        "fred_pre_days": 60,
        "fred_post_days": 15,
        "oil_import_bbl_per_day": 1_800_000,   # Germany crude imports ≈1.8M bbl/d
        "oil_confounder": "Brent in Mar 2021 was recovering from the COVID demand "
                          "collapse; a Suez oil signal is small and confounded.",
    },
    "hormuz_2019": {
        "event_date": "2019-06-13",
        "description": "Gulf of Oman tanker attacks near the Strait of Hormuz "
                       "(2019-06-13).",
        "graph_file": "hard_graph.json",
        "do_nodes": ["PORT_MUMBAI"],        # India imports Gulf crude via Hormuz
        "mc_severity": 0.6,
        "mc_duration_days": 14.0,
        "mitigation_warehouse": "WH_INDIA",
        "mitigation_extra_stock_days": 20,
        "sc_treated": "India",
        "sc_treated_iso3": "IND",
        "sc_donors": ["China", "Germany", "Japan"],
        "sc_event_year": 2019,
        "sc_pre_years": list(range(2008, 2019)),
        "fred_pre_days": 45,
        "fred_post_days": 15,
        "oil_import_bbl_per_day": 4_500_000,   # India crude imports ≈4.5M bbl/d
        "oil_confounder": "US-China trade-war demand fears were the dominant Brent "
                          "driver in mid-2019; the Hormuz signal is confounded.",
    },
}


# =====================================================================
# Result container
# =====================================================================

@dataclass
class MethodEstimate:
    method: str                       # a|b|c|d
    label: str
    estimate_usd: float | None        # None when blocked
    ci_low_usd: float | None
    ci_high_usd: float | None
    unit: str
    scope: str                        # "supply_graph" | "macro_economy" | "oil_channel"
    status: str                       # "ok" | "degraded" | "blocked"
    data_source: str
    inputs: dict = field(default_factory=dict)
    notes: str = ""


# =====================================================================
# Method A — Paired-bootstrap Monte-Carlo (engine's real MonteCarloEngine)
# =====================================================================

def method_a_paired_bootstrap_mc(
    cfg: dict, n_seeds: int = 200, n_boot: int = 2000, seed: int = 20110311,
) -> MethodEstimate:
    """Run the engine's real MonteCarloEngine over N seeded rollouts of the
    disrupted world and bootstrap a CI on the mean simulated customer
    revenue-at-risk. Also runs a genuinely PAIRED (same-seed) no-mitigation vs
    safety-stock comparison → the golden-path "value of acting" number.

    Estimand: E[customer revenue-at-risk | do(Japan suppliers disrupted)].
    Scope: modelled auto supply-graph subset (a fraction of the whole economy).
    """
    try:
        from server.engine.graph import SupplyChainGraph
        from server.engine.monte_carlo import MonteCarloEngine
        from supplymind.contracts import DisruptionSignal, SupplyMindAction
    except Exception as e:  # noqa: BLE001 — surfaced, not swallowed
        return MethodEstimate(
            method="a", label="paired_bootstrap_mc",
            estimate_usd=None, ci_low_usd=None, ci_high_usd=None,
            unit="USD", scope="supply_graph", status="blocked",
            data_source="server/engine/monte_carlo.py",
            notes=f"engine import failed: {type(e).__name__}: {e}",
        )

    graph_path = GRAPH_DIR / cfg["graph_file"]
    base = SupplyChainGraph()
    base.load_from_json(str(graph_path))

    do_nodes = [n for n in cfg["do_nodes"] if n in base.G]
    if not do_nodes:
        return MethodEstimate(
            method="a", label="paired_bootstrap_mc",
            estimate_usd=None, ci_low_usd=None, ci_high_usd=None,
            unit="USD", scope="supply_graph", status="blocked",
            data_source=str(graph_path),
            notes=f"none of do_nodes {cfg['do_nodes']} present in {cfg['graph_file']}",
        )

    signal = DisruptionSignal(
        signal_id="ANALOG", disruption_type="earthquake",
        severity=cfg["mc_severity"], confidence=1.0,
        affected_region=cfg.get("sc_treated", ""), affected_node_ids=do_nodes,
        time_to_impact_hours=0.0, estimated_duration_days=cfg["mc_duration_days"],
        description=cfg["description"], lifecycle_phase="active",
    )

    rng = np.random.default_rng(seed)
    loss_no_action: list[float] = []
    loss_mitigated: list[float] = []

    mit_wh = cfg.get("mitigation_warehouse")
    mit_days = cfg.get("mitigation_extra_stock_days", 90)

    for _ in range(n_seeds):
        s = int(rng.integers(0, 2**31 - 1))
        # No-mitigation world
        g_na = base.deep_copy()
        r_na = MonteCarloEngine(seed=s).run_simulation(g_na, [signal], n_simulations=1)
        loss_no_action.append(float(r_na["p50_loss"]))
        # Mitigated world (same seed → paired): real safety-stock action
        g_mit = base.deep_copy()
        if mit_wh and mit_wh in g_mit.G:
            g_mit.apply_action(SupplyMindAction(
                action_type="increase_safety_stock",
                target_node_id=mit_wh, additional_stock_days=int(mit_days),
            ))
        r_mit = MonteCarloEngine(seed=s).run_simulation(g_mit, [signal], n_simulations=1)
        loss_mitigated.append(float(r_mit["p50_loss"]))

    na = np.array(loss_no_action)
    mit = np.array(loss_mitigated)
    paired_diff = na - mit  # value of acting (>=0 if mitigation helps)

    # Bootstrap the mean event impact (no-action loss)
    boot_event = np.empty(n_boot)
    boot_action = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        boot_event[i] = na[idx].mean()
        boot_action[i] = paired_diff[idx].mean()

    point = float(na.mean())
    lo = float(np.percentile(boot_event, 2.5))
    hi = float(np.percentile(boot_event, 97.5))

    return MethodEstimate(
        method="a", label="paired_bootstrap_mc",
        estimate_usd=point, ci_low_usd=lo, ci_high_usd=hi,
        unit="USD (annual customer revenue-at-risk)", scope="supply_graph",
        status="ok", data_source=f"engine MonteCarloEngine on {cfg['graph_file']}",
        inputs={
            "graph_file": cfg["graph_file"],
            "do_nodes": do_nodes,
            "mc_severity": cfg["mc_severity"],
            "mc_duration_days": cfg["mc_duration_days"],
            "n_seeds": n_seeds, "n_boot": n_boot,
            "mean_loss_usd": point,
            "median_loss_usd": float(np.median(na)),
            "std_loss_usd": float(na.std()),
            "frac_seeds_nonzero": float((na > 0).mean()),
        },
        notes=(
            "N seeded engine rollouts (n_sim=1 each → raw per-seed loss draw), "
            "bootstrap CI on the mean. PAIRED same-seed no-mitigation vs "
            f"safety-stock(+{mit_days}d @ {mit_wh}) gives value_of_acting "
            f"mean=${paired_diff.mean():,.0f} "
            f"[bootstrap 95% ${np.percentile(boot_action,2.5):,.0f}, "
            f"${np.percentile(boot_action,97.5):,.0f}]. Estimand: expected "
            "customer revenue-at-risk under the modelled disruption — a supply-"
            "graph subset, not whole-economy damage."
        ),
    )


# =====================================================================
# Method B — Synthetic control (Abadie): real World-Bank macro series
# =====================================================================

def _wb_load_ondisk(indicator_file: str) -> dict[str, dict[int, float]]:
    """Parse a World-Bank API JSON dump on disk. The file is a 2-element list
    ``[metadata, observations]`` (R17's list-vs-dict quirk); observations[i]
    is a dict with country/date/value. Returns {country: {year: value}}."""
    path = WB_MACRO_DIR / indicator_file
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not (isinstance(raw, list) and len(raw) >= 2 and isinstance(raw[1], list)):
        raise ValueError(f"unexpected WB JSON shape in {path.name}: "
                         f"expected [meta, [obs...]] got {type(raw)}")
    out: dict[str, dict[int, float]] = {}
    for o in raw[1]:
        try:
            country = o["country"]["value"]
            year = int(o["date"])
            val = o["value"]
        except (KeyError, TypeError, ValueError):
            continue
        if val is None:
            continue
        out.setdefault(country, {})[year] = float(val)
    return out


def _wb_fetch_treated(iso3: str, indicator: str, y0: int, y1: int) -> dict[int, float]:
    """Fetch a single country's real series from the keyless World-Bank API.
    Falls back to the on-disk cache (real, previously-fetched) when the network
    is unavailable. Raises when neither is available — never fabricates."""
    url = (f"https://api.worldbank.org/v2/country/{iso3}/indicator/{indicator}"
           f"?format=json&per_page=400&date={y0}:{y1}")
    cache_key = f"{iso3}:{indicator}:{y0}:{y1}"
    try:
        with urllib.request.urlopen(url, timeout=25) as resp:
            data = json.load(resp)
        obs = data[1] if isinstance(data, list) and len(data) > 1 else []
        series = {int(o["date"]): float(o["value"])
                  for o in obs if o.get("value") is not None}
        if series:
            _cache_write(cache_key, url, series)
            logger.info("[method_b] fetched %s %s live: %d points", iso3, indicator, len(series))
            return series
        raise ValueError("WB API returned no non-null observations")
    except Exception as e:  # noqa: BLE001 — try cache, else raise
        cached = _cache_read(cache_key)
        if cached is not None:
            logger.warning("[method_b] WB API unreachable (%s); using on-disk cache "
                           "for %s %s", type(e).__name__, iso3, indicator)
            return cached
        raise RuntimeError(
            f"treated series {iso3}/{indicator} unavailable: live fetch failed "
            f"({type(e).__name__}: {e}) and no cache at {TREATED_CACHE}"
        ) from e


def _cache_read(key: str) -> dict[int, float] | None:
    if not TREATED_CACHE.exists():
        return None
    try:
        blob = json.loads(TREATED_CACHE.read_text(encoding="utf-8"))
        entry = blob.get(key)
        if entry:
            return {int(k): float(v) for k, v in entry["series"].items()}
    except Exception:  # noqa: BLE001
        return None
    return None


def _cache_write(key: str, url: str, series: dict[int, float]) -> None:
    try:
        blob = {}
        if TREATED_CACHE.exists():
            blob = json.loads(TREATED_CACHE.read_text(encoding="utf-8"))
        blob[key] = {
            "url": url,
            "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "series": {str(k): v for k, v in series.items()},
        }
        TREATED_CACHE.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.warning("[method_b] could not write treated cache: %s", e)


def _fit_synthetic_weights(donor_pre: np.ndarray, treated_pre: np.ndarray) -> np.ndarray:
    """Abadie convex weights: min ||donor_pre @ w - treated_pre||^2 s.t. w>=0,
    sum(w)=1. SLSQP on the simplex."""
    from scipy.optimize import minimize
    n = donor_pre.shape[1]
    w0 = np.full(n, 1.0 / n)

    def obj(w):
        r = donor_pre @ w - treated_pre
        return float(r @ r)

    cons = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    bounds = [(0.0, 1.0)] * n
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, None)
    return w / w.sum() if w.sum() > 0 else w0


def method_b_synthetic_control(cfg: dict) -> MethodEstimate:
    """Abadie synthetic control on real World-Bank GDP-growth series.
    Treated = the analog's country (live WB fetch); donors from on-disk files.
    Estimand: whole-economy GDP-growth gap in the event year × treated GDP.
    Scope: whole-economy (this is the method that should approach the headline).
    """
    treated_name = cfg["sc_treated"]
    iso3 = cfg["sc_treated_iso3"]
    donors = cfg["sc_donors"]
    ev = cfg["sc_event_year"]
    pre_years = cfg["sc_pre_years"]

    try:
        donor_growth = _wb_load_ondisk("wb_GDP_growth.json")
        donor_gdp = _wb_load_ondisk("wb_GDP_USD.json")
    except Exception as e:  # noqa: BLE001
        return MethodEstimate(
            method="b", label="synthetic_control_abadie",
            estimate_usd=None, ci_low_usd=None, ci_high_usd=None,
            unit="USD", scope="macro_economy", status="blocked",
            data_source="external_data/world_bank_macro/",
            notes=f"donor WB files unreadable: {type(e).__name__}: {e}",
        )

    # Treated series (live fetch + cache fallback). GDP growth + GDP level.
    try:
        treated_growth = _wb_fetch_treated(
            iso3, "NY.GDP.MKTP.KD.ZG", min(pre_years) - 1, ev + 1)
        treated_gdp = _wb_fetch_treated(
            iso3, "NY.GDP.MKTP.CD", min(pre_years) - 1, ev + 1)
    except Exception as e:  # noqa: BLE001
        return MethodEstimate(
            method="b", label="synthetic_control_abadie",
            estimate_usd=None, ci_low_usd=None, ci_high_usd=None,
            unit="USD", scope="macro_economy", status="blocked",
            data_source="World Bank API (keyless) + on-disk cache",
            notes=(f"treated unit '{treated_name}' unavailable — the on-disk WB "
                   f"dumps are truncated (page 1/2) and omit {treated_name}; live "
                   f"fetch also failed: {type(e).__name__}: {e}. No fabrication."),
        )

    # Common pre-period years present for treated + all donors, with an event value.
    usable_pre = [y for y in pre_years
                  if y in treated_growth and all(y in donor_growth.get(d, {}) for d in donors)]
    donors_ok = [d for d in donors if ev in donor_growth.get(d, {})]
    if len(usable_pre) < 3 or ev not in treated_growth or len(donors_ok) < 2:
        return MethodEstimate(
            method="b", label="synthetic_control_abadie",
            estimate_usd=None, ci_low_usd=None, ci_high_usd=None,
            unit="USD", scope="macro_economy", status="blocked",
            data_source="World Bank",
            notes=(f"insufficient overlap: {len(usable_pre)} pre-years, "
                   f"{len(donors_ok)} donors with event-year value."),
        )
    donors = donors_ok

    donor_pre = np.array([[donor_growth[d][y] for d in donors] for y in usable_pre])
    treated_pre = np.array([treated_growth[y] for y in usable_pre])
    w = _fit_synthetic_weights(donor_pre, treated_pre)

    pre_fit = donor_pre @ w
    pre_rmse = float(np.sqrt(np.mean((pre_fit - treated_pre) ** 2)))

    synth_ev = float(sum(w[i] * donor_growth[d][ev] for i, d in enumerate(donors)))
    actual_ev = float(treated_growth[ev])
    gap_pp = actual_ev - synth_ev  # negative = treated grew less than synthetic

    # Convert growth-gap (percentage points) to USD via prior-year nominal GDP.
    base_gdp = treated_gdp.get(ev - 1) or treated_gdp.get(ev)
    if base_gdp is None:
        return MethodEstimate(
            method="b", label="synthetic_control_abadie",
            estimate_usd=None, ci_low_usd=None, ci_high_usd=None,
            unit="USD", scope="macro_economy", status="blocked",
            data_source="World Bank",
            notes=f"no nominal GDP for {treated_name} near {ev}.",
        )
    impact_usd = abs(gap_pp) / 100.0 * base_gdp  # magnitude of lost output

    # CI via leave-one-pre-year-out refit (placebo-style weight sensitivity).
    boot_gaps = []
    for skip in range(len(usable_pre)):
        keep = [i for i in range(len(usable_pre)) if i != skip]
        w_k = _fit_synthetic_weights(donor_pre[keep], treated_pre[keep])
        synth_k = sum(w_k[i] * donor_growth[d][ev] for i, d in enumerate(donors))
        boot_gaps.append((actual_ev - synth_k) / 100.0 * base_gdp)
    boot_gaps = np.abs(np.array(boot_gaps))
    lo = float(boot_gaps.min())
    hi = float(boot_gaps.max())

    return MethodEstimate(
        method="b", label="synthetic_control_abadie",
        estimate_usd=impact_usd, ci_low_usd=min(lo, impact_usd),
        ci_high_usd=max(hi, impact_usd),
        unit="USD (lost annual output, |growth gap| × GDP)", scope="macro_economy",
        status="ok", data_source="World Bank GDP-growth (treated live; donors on-disk)",
        inputs={
            "treated": treated_name, "donors": donors,
            "donor_weights": {d: round(float(w[i]), 4) for i, d in enumerate(donors)},
            "pre_years": usable_pre, "pre_fit_rmse_pp": round(pre_rmse, 4),
            "actual_growth_pct": round(actual_ev, 3),
            "synthetic_growth_pct": round(synth_ev, 3),
            "gap_pp": round(gap_pp, 3),
            "base_gdp_usd": base_gdp, "event_year": ev,
        },
        notes=(
            f"Abadie convex synthetic control. Pre-fit RMSE {pre_rmse:.2f}pp over "
            f"{len(usable_pre)} years. {treated_name} {ev} grew {actual_ev:.2f}% vs "
            f"synthetic {synth_ev:.2f}% → gap {gap_pp:.2f}pp → |gap|×GDP{ev-1}. "
            "CI = leave-one-pre-year-out weight sensitivity. Whole-economy scope — "
            "captures far more than the auto supply chain, so expected to sit near "
            "the documented headline, above the supply-graph methods."
        ),
    )


# =====================================================================
# Method C — ARIMA(p,1,0) on real FRED daily Brent
# =====================================================================

def _load_fred_brent() -> tuple[list[str], np.ndarray]:
    """Parse real FRED Brent CSV (observation_date,DCOILBRENTEU); '.' = missing."""
    dates: list[str] = []
    prices: list[float] = []
    with open(FRED_BRENT_CSV, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                v = float(row[-1].strip())
            except ValueError:
                continue  # FRED missing-value marker '.'
            if v > 0:
                dates.append(row[0].strip())
                prices.append(v)
    return dates, np.array(prices)


def _fit_ar_ols(diff: np.ndarray, p: int) -> tuple[np.ndarray, float, np.ndarray]:
    """OLS fit of AR(p) with intercept on a differenced series. Returns
    (coeffs [c, phi1..phip], sigma2, residuals)."""
    n = len(diff)
    y = diff[p:]
    X = np.column_stack([np.ones(n - p)] + [diff[p - k - 1: n - k - 1] for k in range(p)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = float(resid @ resid / max(1, len(resid) - X.shape[1]))
    return beta, sigma2, resid


def method_c_arima_fred(cfg: dict, seed: int = 7, n_boot: int = 1000) -> MethodEstimate:
    """ARIMA(p,1,0) counterfactual on real FRED daily Brent. Fit AR(p) (AIC over
    p=1..5) on the pre-event window, forecast the no-shock counterfactual price
    path, compare to the actual post-event path. Per-bbl gap × treated crude
    imports × window = oil-price-channel USD impact.

    Scope: oil-price channel only, and heavily confounded (surfaced) — this is
    the method most likely to disagree with the others, by design/honesty.
    """
    if not FRED_BRENT_CSV.exists():
        return MethodEstimate(
            method="c", label="arima_p10_fred_brent",
            estimate_usd=None, ci_low_usd=None, ci_high_usd=None,
            unit="USD", scope="oil_channel", status="blocked",
            data_source=str(FRED_BRENT_CSV),
            notes=f"FRED Brent CSV not present at {FRED_BRENT_CSV}",
        )

    try:
        dates, prices = _load_fred_brent()
    except Exception as e:  # noqa: BLE001
        return MethodEstimate(
            method="c", label="arima_p10_fred_brent",
            estimate_usd=None, ci_low_usd=None, ci_high_usd=None,
            unit="USD", scope="oil_channel", status="blocked",
            data_source=str(FRED_BRENT_CSV),
            notes=f"FRED CSV read error: {type(e).__name__}: {e}",
        )

    ev_date = cfg["event_date"]
    pre_win = cfg["fred_pre_days"]
    post_win = cfg["fred_post_days"]
    ev_idx = next((i for i, d in enumerate(dates) if d >= ev_date), None)
    if ev_idx is None or ev_idx < pre_win + 5 or ev_idx + post_win > len(prices):
        return MethodEstimate(
            method="c", label="arima_p10_fred_brent",
            estimate_usd=None, ci_low_usd=None, ci_high_usd=None,
            unit="USD", scope="oil_channel", status="blocked",
            data_source=str(FRED_BRENT_CSV),
            notes=f"event date {ev_date} outside usable FRED range.",
        )

    pre = prices[ev_idx - pre_win: ev_idx]
    post = prices[ev_idx: ev_idx + post_win]
    diff = np.diff(pre)

    # AIC-select AR order p.
    best = None
    for p in range(1, 6):
        if len(diff) <= p + 5:
            continue
        beta, sigma2, resid = _fit_ar_ols(diff, p)
        k = p + 2  # intercept + p coeffs + variance
        n_eff = len(resid)
        aic = n_eff * np.log(max(sigma2, 1e-12)) + 2 * k
        if best is None or aic < best[0]:
            best = (aic, p, beta, sigma2, resid)
    if best is None:
        return MethodEstimate(
            method="c", label="arima_p10_fred_brent",
            estimate_usd=None, ci_low_usd=None, ci_high_usd=None,
            unit="USD", scope="oil_channel", status="blocked",
            data_source=str(FRED_BRENT_CSV), notes="AR fit failed (window too short).",
        )
    _, p, beta, sigma2, resid = best

    def forecast_path(shocks: np.ndarray | None) -> np.ndarray:
        """Iteratively forecast the differenced series then integrate to levels."""
        hist = list(diff[-p:])
        level = float(pre[-1])
        out = []
        for t in range(post_win):
            d_hat = beta[0] + sum(beta[1 + k] * hist[-1 - k] for k in range(p))
            if shocks is not None:
                d_hat += shocks[t]
            hist.append(d_hat)
            level += d_hat
            out.append(level)
        return np.array(out)

    cf = forecast_path(None)
    effect_per_bbl = float((post - cf).mean())
    bbl_per_day = cfg["oil_import_bbl_per_day"]
    point = abs(effect_per_bbl) * bbl_per_day * post_win

    # Residual-bootstrap CI on the per-bbl effect → USD.
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        shocks = rng.choice(resid, size=post_win, replace=True)
        cf_b = forecast_path(shocks)
        eff_b = float((post - cf_b).mean())
        boot[i] = abs(eff_b) * bbl_per_day * post_win
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))

    return MethodEstimate(
        method="c", label="arima_p10_fred_brent",
        estimate_usd=point, ci_low_usd=lo, ci_high_usd=hi,
        unit="USD (oil-import cost, |per-bbl gap| × imports × window)",
        scope="oil_channel", status="ok",
        data_source="FRED DCOILBRENTEU daily (external_data/fred_brent_daily.csv)",
        inputs={
            "event_date": ev_date, "ar_order_p": p,
            "pre_window_days": pre_win, "post_window_days": post_win,
            "pre_window_dates": [dates[ev_idx - pre_win], dates[ev_idx - 1]],
            "actual_post_avg_usd_bbl": round(float(post.mean()), 3),
            "counterfactual_avg_usd_bbl": round(float(cf.mean()), 3),
            "effect_per_bbl_usd": round(effect_per_bbl, 4),
            "signed_direction": "actual_below_counterfactual" if effect_per_bbl < 0
                                else "actual_above_counterfactual",
            "oil_import_bbl_per_day": bbl_per_day,
            "n_prices_loaded": int(len(prices)),
        },
        notes=(
            f"AR({p}) (AIC-selected) on the differenced pre-event Brent window, "
            f"iteratively forecast + integrated to a counterfactual price path; "
            f"residual-bootstrap CI (n={n_boot}). Effect {effect_per_bbl:+.2f} $/bbl. "
            "OIL-CHANNEL scope only — a narrow slice, and confounded: "
            + cfg.get("oil_confounder", "")
        ),
    )


# =====================================================================
# Method D — do-calculus / SCM intervention on the engine's real DAG
# =====================================================================

def method_d_scm_do_intervention(cfg: dict) -> MethodEstimate:
    """Structural do() intervention on the engine's real supply-chain graph:
    set the analog's Japan tier-1 suppliers OFFLINE (the do(X) operator), then
    let the engine's own path-based get_total_revenue_at_risk() propagate the
    structural consequence (customers with no operational supply path).

    Estimand: customer revenue with no operational supply path under do(disrupt).
    Scope: modelled auto supply-graph subset (deterministic / structural).
    """
    try:
        from server.engine.graph import SupplyChainGraph
    except Exception as e:  # noqa: BLE001
        return MethodEstimate(
            method="d", label="scm_do_intervention",
            estimate_usd=None, ci_low_usd=None, ci_high_usd=None,
            unit="USD", scope="supply_graph", status="blocked",
            data_source="server/engine/graph.py",
            notes=f"engine import failed: {type(e).__name__}: {e}",
        )

    graph_path = GRAPH_DIR / cfg["graph_file"]

    def do_offline(nodes: Sequence[str]) -> float:
        g = SupplyChainGraph()
        g.load_from_json(str(graph_path))
        base = g.get_total_revenue_at_risk()
        present = [n for n in nodes if n in g.G]
        for nid in present:
            g.G.nodes[nid]["is_operational"] = False
            g._ever_offline.add(nid)
        return g.get_total_revenue_at_risk() - base

    do_nodes = cfg["do_nodes"]
    point = do_offline(do_nodes)

    # Sensitivity band across plausible intervention sets → honest CI.
    # (Structural model is discrete: redundancy makes it near all-or-nothing.)
    scenarios: dict[str, float] = {"do(all_target_suppliers)": point}
    # leave-one-out (partial disruption — redundancy may keep customers served)
    for drop in do_nodes:
        subset = [n for n in do_nodes if n != drop]
        if subset:
            scenarios[f"do(minus:{drop})"] = do_offline(subset)
    # escalation: also take the downstream factory offline
    try:
        g0 = SupplyChainGraph(); g0.load_from_json(str(graph_path))
        facs = [n for n in g0.G if str(n).startswith("FAC_")
                and any(g0.G.has_edge(pred, n) for pred in ["WH_JAPAN", "WH_GERMANY",
                        "WH_INDIA"] if pred in g0.G)]
    except Exception:  # noqa: BLE001
        facs = []
    if facs:
        scenarios["do(+downstream_factory)"] = do_offline(list(do_nodes) + facs[:1])

    vals = np.array([v for v in scenarios.values()])
    lo, hi = float(vals.min()), float(vals.max())

    return MethodEstimate(
        method="d", label="scm_do_intervention",
        estimate_usd=float(point), ci_low_usd=lo, ci_high_usd=hi,
        unit="USD (annual customer revenue with no operational supply path)",
        scope="supply_graph", status="ok",
        data_source=f"engine SupplyChainGraph structural propagation on {cfg['graph_file']}",
        inputs={
            "graph_file": cfg["graph_file"],
            "do_nodes": do_nodes,
            "intervention": "set do_nodes is_operational=False, then "
                            "get_total_revenue_at_risk() (path-based)",
            "sensitivity_scenarios_usd": {k: round(v, 0) for k, v in scenarios.items()},
        },
        notes=(
            "do(X): the engine's real graph.get_total_revenue_at_risk() counts "
            "customers whose every supply path passes through a disrupted node. "
            "CI = sensitivity across intervention subsets (leave-one-supplier-out "
            "and +downstream-factory). Discrete/structural: redundancy makes the "
            "effect near all-or-nothing. Supply-graph subset scope (a few "
            "automakers), so well below the whole-economy headline."
        ),
    )


# =====================================================================
# Honest pooling + bracket check
# =====================================================================

def pool_estimates(
    estimates: Sequence[MethodEstimate],
    documented: dict | None = None,
) -> dict:
    """Pool the OK estimates HONESTLY: report each, the median/mean, the min/max,
    the order-of-magnitude spread, a scope breakdown, and whether the estimates
    bracket the documented real outcome. Never hides disagreement."""
    ok = [e for e in estimates if e.status in ("ok", "degraded")
          and e.estimate_usd is not None]
    points = [float(e.estimate_usd) for e in ok]
    result: dict[str, Any] = {
        "n_methods_ok": len(ok),
        "n_methods_total": len(estimates),
        "per_method_usd": {e.label: round(float(e.estimate_usd), 0) for e in ok},
        "per_method_scope": {e.label: e.scope for e in ok},
        "per_method_status": {e.label: e.status for e in estimates},
    }
    if not points:
        result["pooled"] = None
        result["note"] = "No method produced an estimate (all blocked)."
        return result

    arr = np.array(points)
    pos = arr[arr > 0]
    spread_ratio = float(pos.max() / pos.min()) if len(pos) and pos.min() > 0 else None
    result["pooled_median_usd"] = float(np.median(arr))
    result["pooled_mean_usd"] = float(arr.mean())
    result["min_usd"] = float(arr.min())
    result["max_usd"] = float(arr.max())
    result["spread_max_over_min"] = (round(spread_ratio, 1) if spread_ratio else None)
    result["orders_of_magnitude_spread"] = (
        round(float(np.log10(spread_ratio)), 2) if spread_ratio else None)

    # Scope grouping — the honest explanation for disagreement.
    by_scope: dict[str, list[float]] = {}
    for e in ok:
        by_scope.setdefault(e.scope, []).append(float(e.estimate_usd))
    result["by_scope_usd"] = {
        s: {"n": len(v), "median": float(np.median(v)),
            "range": [float(min(v)), float(max(v))]}
        for s, v in by_scope.items()
    }

    agree = spread_ratio is not None and spread_ratio < 3.0
    result["methods_agree_within_3x"] = bool(agree)

    if documented and documented.get("headline_low_usd"):
        lo_d = documented["headline_low_usd"]
        hi_d = documented["headline_high_usd"]
        # Which methods' CIs overlap the documented headline band?
        bracketing = []
        for e in ok:
            e_lo = e.ci_low_usd if e.ci_low_usd is not None else e.estimate_usd
            e_hi = e.ci_high_usd if e.ci_high_usd is not None else e.estimate_usd
            if e_hi >= lo_d and e_lo <= hi_d:
                bracketing.append(e.label)
        result["documented_headline_usd"] = [lo_d, hi_d]
        result["documented_bracketed_by"] = bracketing
        result["pooled_range_brackets_headline"] = bool(
            arr.max() >= lo_d and arr.min() <= hi_d)
        result["documented_citations"] = documented.get("citations", [])
        result["documented_scope_note"] = documented.get("scope_note", "")

    result["honest_reading"] = _honest_reading(ok, spread_ratio, documented)
    return result


def _honest_reading(ok, spread_ratio, documented) -> str:
    if not ok:
        return "All methods blocked; no estimate."
    parts = []
    if spread_ratio and spread_ratio >= 3.0:
        parts.append(
            f"The {len(ok)} methods DISAGREE by ~{spread_ratio:.0f}× — expected, "
            "because they measure different SCOPES: the macro synthetic control "
            "captures whole-economy output loss while the supply-graph MC/SCM "
            "methods capture only the modelled auto-supplier subset, and the "
            "ARIMA method captures only the (confounded) oil-price channel.")
    else:
        parts.append(f"The {len(ok)} methods agree within {spread_ratio:.1f}×.")
    if documented and documented.get("headline_low_usd"):
        parts.append(
            "Against the documented whole-economy headline, the macro method is "
            "the scope-appropriate comparison; the supply-graph methods are a "
            "subset and correctly land below it. Do not average across scopes as "
            "if they estimated the same quantity.")
    return " ".join(parts)


# =====================================================================
# Orchestration + receipt
# =====================================================================

def run_all_methods(
    analog: str = "tohoku_2011",
    n_seeds_mc: int = 200,
) -> dict:
    """Run all four real methods on the named analog and pool honestly."""
    if analog not in ANALOGS:
        raise ValueError(f"unknown analog {analog!r}; choices: {list(ANALOGS)}")
    cfg = ANALOGS[analog]
    documented = DOCUMENTED_OUTCOMES.get(analog)

    a = method_a_paired_bootstrap_mc(cfg, n_seeds=n_seeds_mc)
    b = method_b_synthetic_control(cfg)
    c = method_c_arima_fred(cfg)
    d = method_d_scm_do_intervention(cfg)
    estimates = [a, b, c, d]
    pooled = pool_estimates(estimates, documented)

    return {
        "analog": analog,
        "event_date": cfg["event_date"],
        "description": cfg["description"],
        "methods": {e.method: asdict(e) for e in estimates},
        "pooled": pooled,
        "documented_outcome": documented,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_receipt(result: dict, path: Path | None = None) -> Path:
    """Write the receipt with sha256 self-hash + reproduce command."""
    path = path or (REPO_ROOT / "FINAL_SUBMIT" / "receipts"
                    / "counterfactual_4method_REAL.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    analog = result["analog"]
    receipt = {
        "receipt": "counterfactual_4method_REAL",
        "capability": "R6 — real 4-method causal counterfactual",
        "analog": analog,
        "git_sha": _git_sha(),
        "python": "c:/Users/Dell/Desktop/Sleep-Token/.venv/Scripts/python.exe",
        "reproduce": (
            "c:/Users/Dell/Desktop/Sleep-Token/.venv/Scripts/python.exe -m "
            f"supplymind.phoenix.counterfactual_v2.causal_methods --analog {analog} "
            "--receipt"),
        "data_sources": {
            "method_a_and_d_engine": "server/engine/monte_carlo.py + "
                                     "server/engine/graph.py (server/data/graphs/)",
            "method_b_worldbank": "World Bank API (keyless, treated) + "
                                  "external_data/world_bank_macro/*.json (donors)",
            "method_c_fred": "external_data/fred_brent_daily.csv (DCOILBRENTEU)",
        },
        "result": result,
    }
    # sha256 over the SUBSTANTIVE, deterministic content only (drop the wall-clock
    # timestamp) so a reviewer re-running on the same code + data reproduces the
    # exact same hash — all RNG in every method is seeded. The volatile
    # generated_utc is kept in the file for provenance but excluded from the hash.
    hashable = {k: v for k, v in receipt.items()}
    res_copy = {k: v for k, v in result.items() if k != "generated_utc"}
    hashable["result"] = res_copy
    blob = json.dumps(hashable, indent=2, sort_keys=True, default=str)
    sha = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    receipt["sha256"] = sha
    receipt["sha256_scope"] = ("sha256 of the receipt with result.generated_utc "
                               "removed; deterministic across re-runs (seeded RNG).")
    # A tighter stamp: hash of just the per-method (estimate, ci_low, ci_high).
    values = {m: [receipt["result"]["methods"][m]["estimate_usd"],
                  receipt["result"]["methods"][m]["ci_low_usd"],
                  receipt["result"]["methods"][m]["ci_high_usd"]]
              for m in ("a", "b", "c", "d")}
    receipt["values_sha256"] = hashlib.sha256(
        json.dumps(values, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    return path


def _print_summary(result: dict) -> None:
    print(f"\n{'='*70}\nR6 4-METHOD CAUSAL COUNTERFACTUAL — {result['analog']} "
          f"({result['event_date']})\n{'='*70}")
    for m in ("a", "b", "c", "d"):
        e = result["methods"][m]
        if e["estimate_usd"] is None:
            print(f"  [{m}] {e['label']:26s} [{e['status'].upper()}] {e['notes'][:90]}")
        else:
            print(f"  [{m}] {e['label']:26s} ${e['estimate_usd']:>16,.0f}  "
                  f"CI95 [${e['ci_low_usd']:,.0f}, ${e['ci_high_usd']:,.0f}]  "
                  f"scope={e['scope']}")
    p = result["pooled"]
    print(f"\n  POOLED  median ${p.get('pooled_median_usd', 0):,.0f}   "
          f"range [${p.get('min_usd', 0):,.0f}, ${p.get('max_usd', 0):,.0f}]   "
          f"spread {p.get('spread_max_over_min')}×")
    if "documented_headline_usd" in p:
        d = p["documented_headline_usd"]
        print(f"  DOCUMENTED headline ${d[0]:,.0f}–${d[1]:,.0f}  "
              f"bracketed_by={p.get('documented_bracketed_by')}")
    print(f"\n  {p.get('honest_reading', '')}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="R6 real 4-method causal counterfactual")
    ap.add_argument("--analog", default="tohoku_2011", choices=list(ANALOGS))
    ap.add_argument("--n-seeds", type=int, default=200, help="Method A MC seeds")
    ap.add_argument("--receipt", action="store_true", help="write the receipt")
    args = ap.parse_args()
    # Windows consoles default to cp1252; the summary contains typographic chars
    # (em-dash, ×, ō). Force UTF-8 so printing never crashes. The receipt JSON is
    # already portable (json.dumps ensure_ascii escapes non-ASCII).
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    result = run_all_methods(args.analog, n_seeds_mc=args.n_seeds)
    _print_summary(result)
    if args.receipt:
        path = write_receipt(result)
        print(f"  receipt written: {path}")
        print(f"  sha256: {json.loads(path.read_text(encoding='utf-8'))['sha256']}")


if __name__ == "__main__":
    main()
