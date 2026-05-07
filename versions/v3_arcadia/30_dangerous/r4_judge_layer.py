"""R4 Dangerous — 3-judge LLM layer for supply-chain risk analysis.

Judges (all SOTA, all local via Ollama):
  - deepseek-r1-local (F16 7B, reasoning specialist)
  - qwen25-14b-local (Q4_K_M 14B, generalist)
  - mistral-nemo-local (Q4_K_M 12B, 128K long-context)

Per scenario: parallel inference -> structured JSON -> consensus scoring.

Outputs:
  versions/v3_arcadia/results/R4_DANGEROUS.json
  versions/v3_arcadia/plots/dangerous/r4_agreement.png
"""
from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CRISES = ROOT / "external_data" / "wikipedia_crises"
RESULTS = ROOT / "v3_arcadia" / "results"
PLOTS = ROOT / "v3_arcadia" / "plots" / "dangerous"
PLOTS.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
JUDGES = ["deepseek-r1-local", "qwen25-14b-local", "mistral-nemo-local"]
RISK_ORDINAL = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

SYSTEM_PROMPT = """You are a supply-chain risk analyst. Given a factual context, you produce a structured
JSON assessment. Be calibrated: CRITICAL only when demonstrable global disruption; HIGH for regional/sectoral
disruption; MEDIUM for localized with spillover; LOW when recoverable in <30 days without industry impact.
Return ONLY valid JSON, no prose outside the JSON object."""

USER_TEMPLATE = """CONTEXT (from historical crisis documentation):
---
{context}
---

Assess supply-chain risk. Return JSON with exactly these keys:
{{
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "confidence": float between 0 and 1,
  "primary_vulnerabilities": [list of 3 chokepoints or concentrated nodes],
  "mitigations": [list of 3 concrete actions],
  "reasoning_one_line": "why this risk level"
}}"""


def load_scenarios(n: int = 10) -> list[dict]:
    files = sorted(CRISES.glob("*.txt"))
    out = []
    for f in files[:n]:
        txt = f.read_text(encoding="utf-8", errors="ignore")
        # Truncate to first 2500 chars (fits in 8K context easily + leaves room for output)
        out.append({"name": f.stem, "context": txt[:2500]})
    return out


def call_ollama(model: str, system: str, user: str, timeout: int = 360) -> dict:
    t0 = time.time()
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "format": "json",
            "stream": False,
            "keep_alive": "30m",  # keep loaded across sequential scenarios
            "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 800},
        }, timeout=timeout)
        r.raise_for_status()
        content = r.json()["message"]["content"]
        dt = time.time() - t0
        parsed = parse_json_loose(content)
        return {"raw": content, "parsed": parsed, "latency_s": dt, "ok": parsed is not None}
    except Exception as e:
        return {"raw": None, "parsed": None, "latency_s": time.time() - t0,
                "ok": False, "error": str(e)[:200]}


def parse_json_loose(text: str) -> dict | None:
    if not text: return None
    # Strip <think>...</think> blocks DeepSeek-R1 emits even in json mode
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find outermost {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def judge_scenario(scenario: dict, parallel: bool = True) -> dict:
    user = USER_TEMPLATE.format(context=scenario["context"])
    results = {}
    if parallel:
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(call_ollama, j, SYSTEM_PROMPT, user): j for j in JUDGES}
            for f in as_completed(futs):
                j = futs[f]
                results[j] = f.result()
    else:
        for j in JUDGES:
            results[j] = call_ollama(j, SYSTEM_PROMPT, user)
    return results


# ============================================================
# Consensus metrics
# ============================================================
def krippendorff_alpha_ordinal(ratings: list[int]) -> float:
    """Simple 1-rater ordinal alpha. Returns 1.0 for unanimous, 0 for chance.

    Here we have 1 scenario x N raters -> compute pairwise squared distance.
    """
    vals = [v for v in ratings if v is not None]
    if len(vals) < 2: return float("nan")
    m = np.mean(vals)
    total_var = np.var(vals, ddof=0)
    if total_var == 0: return 1.0
    # Expected variance if raters picked from uniform {1..4} = var of uniform = (4^2-1)/12 = 1.25
    expected_var = 1.25
    return float(max(0.0, 1.0 - total_var / expected_var))


def jaccard(a: list[str], b: list[str]) -> float:
    a_set = {x.lower().strip() for x in a if isinstance(x, str)}
    b_set = {x.lower().strip() for x in b if isinstance(x, str)}
    if not a_set and not b_set: return 1.0
    if not a_set or not b_set: return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def pairwise_jaccard(lists: list[list[str]]) -> float:
    lists = [l for l in lists if l]
    if len(lists) < 2: return float("nan")
    js = []
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            js.append(jaccard(lists[i], lists[j]))
    return float(np.mean(js)) if js else float("nan")


def aggregate_scenario(jr: dict) -> dict:
    risk_ratings = []
    confs = []
    vulns = []
    mits = []
    latencies = {}
    for j, r in jr.items():
        latencies[j] = r.get("latency_s", 0)
        p = r.get("parsed")
        if not p or not isinstance(p, dict): continue
        risk_ratings.append(RISK_ORDINAL.get(str(p.get("risk_level", "")).upper()))
        if isinstance(p.get("confidence"), (int, float)):
            confs.append(float(p["confidence"]))
        if isinstance(p.get("primary_vulnerabilities"), list):
            vulns.append([str(x) for x in p["primary_vulnerabilities"]])
        if isinstance(p.get("mitigations"), list):
            mits.append([str(x) for x in p["mitigations"]])
    risk_ratings_clean = [r for r in risk_ratings if r is not None]
    if risk_ratings_clean:
        majority = int(np.round(np.median(risk_ratings_clean)))
        rev = {v: k for k, v in RISK_ORDINAL.items()}
        majority_label = rev.get(majority, "UNKNOWN")
    else:
        majority_label = "UNKNOWN"
    return {
        "n_valid_judges": len(risk_ratings_clean),
        "risk_ratings_ordinal": risk_ratings,
        "risk_alpha_ordinal": krippendorff_alpha_ordinal(risk_ratings_clean),
        "risk_majority": majority_label,
        "mean_confidence": float(np.mean(confs)) if confs else None,
        "vulnerabilities_jaccard": pairwise_jaccard(vulns),
        "mitigations_jaccard": pairwise_jaccard(mits),
        "latencies_s": latencies,
    }


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    log.info("R4 Dangerous: 3-judge LLM layer")
    log.info(f"Judges: {JUDGES}")

    # Health check
    try:
        h = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        h.raise_for_status()
        tags = [m["name"].split(":")[0] for m in h.json().get("models", [])]
        missing = [j for j in JUDGES if j.split(":")[0] not in tags]
        if missing:
            log.warning(f"  MISSING judges in Ollama: {missing}")
    except Exception as e:
        log.error(f"Ollama not reachable: {e}")
        return

    scenarios = load_scenarios(n=10)
    log.info(f"Loaded {len(scenarios)} scenarios: {[s['name'] for s in scenarios]}")

    # Judge-first iteration: load each model once, process all 10 scenarios, then swap.
    # This avoids VRAM thrash on 12GB where parallel model calls cause each other to fail.
    out = {"judges": JUDGES, "n_scenarios": len(scenarios),
           "per_scenario": {s["name"]: {"per_judge": {}, "consensus": None} for s in scenarios}}

    for j_idx, judge in enumerate(JUDGES, 1):
        log.info(f"\n=== Judge {j_idx}/{len(JUDGES)}: {judge} ===")
        for s_idx, s in enumerate(scenarios, 1):
            user = USER_TEMPLATE.format(context=s["context"])
            r = call_ollama(judge, SYSTEM_PROMPT, user)
            out["per_scenario"][s["name"]]["per_judge"][judge] = {
                "ok": r["ok"], "latency_s": r["latency_s"],
                "parsed": r["parsed"], "error": r.get("error"),
                "raw_preview": (r.get("raw") or "")[:400],
            }
            status = "OK" if r["ok"] else "FAIL"
            err = f" err={r.get('error','')[:80]}" if not r["ok"] else ""
            log.info(f"  [{s_idx:2d}/{len(scenarios)}] {s['name'][:40]:<40} {status:<4} {r['latency_s']:5.1f}s{err}")

    # Aggregate per scenario
    for s in scenarios:
        jr = {j: out["per_scenario"][s["name"]]["per_judge"].get(j, {}) for j in JUDGES}
        # Rebuild into the shape aggregate_scenario expects
        jr_shaped = {j: {"parsed": v.get("parsed"), "latency_s": v.get("latency_s", 0),
                          "ok": v.get("ok", False)} for j, v in jr.items()}
        out["per_scenario"][s["name"]]["consensus"] = aggregate_scenario(jr_shaped)
        agg = out["per_scenario"][s["name"]]["consensus"]
        log.info(f"\n{s['name']:<40} risk={agg['risk_majority']} alpha={agg['risk_alpha_ordinal']:.3f} "
                 f"vuln_J={agg['vulnerabilities_jaccard']:.3f} mit_J={agg['mitigations_jaccard']:.3f}")

    # Aggregate summary
    alphas = [v["consensus"]["risk_alpha_ordinal"] for v in out["per_scenario"].values()
              if not np.isnan(v["consensus"].get("risk_alpha_ordinal", np.nan))]
    vjs = [v["consensus"]["vulnerabilities_jaccard"] for v in out["per_scenario"].values()
           if not np.isnan(v["consensus"].get("vulnerabilities_jaccard", np.nan))]
    mjs = [v["consensus"]["mitigations_jaccard"] for v in out["per_scenario"].values()
           if not np.isnan(v["consensus"].get("mitigations_jaccard", np.nan))]
    # Mean latency per judge
    lat_per_j = {j: [] for j in JUDGES}
    ok_per_j = {j: 0 for j in JUDGES}
    for v in out["per_scenario"].values():
        for j in JUDGES:
            pj = v["per_judge"].get(j, {})
            lat_per_j[j].append(pj.get("latency_s", 0))
            if pj.get("ok"): ok_per_j[j] += 1
    out["summary"] = {
        "mean_risk_alpha": float(np.mean(alphas)) if alphas else None,
        "mean_vulnerabilities_jaccard": float(np.mean(vjs)) if vjs else None,
        "mean_mitigations_jaccard": float(np.mean(mjs)) if mjs else None,
        "parse_success_rate_per_judge": {j: ok_per_j[j] / len(scenarios) for j in JUDGES},
        "mean_latency_s_per_judge": {j: float(np.mean(lat_per_j[j])) for j in JUDGES},
        "total_elapsed_min": (time.time() - t0) / 60,
    }
    log.info("\n=== SUMMARY ===")
    log.info(f"  mean_risk_alpha       = {out['summary']['mean_risk_alpha']}")
    log.info(f"  mean_vuln_jaccard     = {out['summary']['mean_vulnerabilities_jaccard']}")
    log.info(f"  mean_mitig_jaccard    = {out['summary']['mean_mitigations_jaccard']}")
    for j in JUDGES:
        log.info(f"  {j:<25} success={ok_per_j[j]}/{len(scenarios)}  mean_lat={np.mean(lat_per_j[j]):.1f}s")

    out_path = RESULTS / "R4_DANGEROUS.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info(f"\nSaved: {out_path}  ({out['summary']['total_elapsed_min']:.1f} min)")


if __name__ == "__main__":
    main()
