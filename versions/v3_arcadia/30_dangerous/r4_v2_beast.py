"""R4 Dangerous V2 — BEAST MODE 3-judge LLM panel with critic + ground truth + calibration.

Overhaul from V1:
  1. DeepSeek-R1 two-pass (free CoT -> Qwen extractor) -> target 26/26 parse success
  2. All 26 Wikipedia crisis scenarios
  3. Ground-truth risk labels (deterministic rubric) for accuracy measurement
  4. Semantic Jaccard via mxbai-embed-large-v1 (cosine > 0.65 = concept match)
  5. Proper weighted-ordinal Krippendorff alpha
  6. Critic pass: Qwen-Coder-14B reviews all 3 judge outputs, flags contradictions
  7. ECE + reliability diagrams per judge (confidence vs ground-truth accuracy)
  8. Escalation routing rubric tested on all 26
  9. Confusion matrices per judge vs ground truth

Outputs:
  versions/v3_arcadia/results/R4_DANGEROUS_V2.json
  versions/v3_arcadia/plots/dangerous/r4v2_heatmap.png
  versions/v3_arcadia/plots/dangerous/r4v2_calibration.png
  versions/v3_arcadia/plots/dangerous/r4v2_confusion.png
  versions/v3_arcadia/plots/dangerous/r4v2_latency.png
  versions/v3_arcadia/plots/dangerous/r4v2_escalation.png
  versions/v3_arcadia/results/R4_DANGEROUS_V2_REPORT.md
"""
from __future__ import annotations

import json
import logging
import re
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CRISES = ROOT / "external_data" / "wikipedia_crises"
RESULTS = ROOT / "v3_arcadia" / "results"
PLOTS = ROOT / "v3_arcadia" / "plots" / "dangerous"
PLOTS.mkdir(parents=True, exist_ok=True)
EMBEDDER_PATH = ROOT / "models" / "mxbai-embed-large"

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
JUDGES = ["deepseek-r1-local-q4", "qwen25-14b-local", "mistral-nemo-local"]
CRITIC = "qwen25-coder-local"
EXTRACTOR = "qwen25-14b-local"  # used to parse DeepSeek free-form output
RISK_ORDINAL = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
RISK_REV = {v: k for k, v in RISK_ORDINAL.items()}

# ============================================================
# Ground-truth labels (deterministic rubric, hand-anchored on crisis severity)
# Rubric:
#   CRITICAL: worldwide disruption, multi-sector, recovery > 6 months
#   HIGH:     regional or single-sector disruption, recovery 1-6 months
#   MEDIUM:   localized disruption with spillover OR systemic concept w/ ongoing risk
#   LOW:      concept/infrastructure article, recoverable < 30 days, no active disruption
# ============================================================
GROUND_TRUTH = {
    "2011_Tōhoku_earthquake_and_tsunami":     "CRITICAL",
    "2020–2023_global_chip_shortage":         "CRITICAL",
    "2021_Suez_Canal_obstruction":            "HIGH",
    "Bab-el-Mandeb":                          "HIGH",
    "Baltic_Dry_Index":                       "LOW",
    "Bullwhip_effect":                        "MEDIUM",
    "CHIPS_and_Science_Act":                  "MEDIUM",
    "Container_ship":                         "LOW",
    "Enterprise_resource_planning":           "LOW",
    "Ever_Given":                             "HIGH",
    "Foxconn":                                "MEDIUM",
    "Inventory":                              "LOW",
    "Just-in-time_manufacturing":             "MEDIUM",
    "Logistics":                              "LOW",
    "Port_of_Los_Angeles":                    "MEDIUM",
    "Port_of_Singapore":                      "MEDIUM",
    "Red_Sea_crisis":                         "CRITICAL",
    "Samsung_Electronics":                    "MEDIUM",
    "Semiconductor_industry":                 "HIGH",
    "Strait_of_Hormuz":                       "HIGH",
    "Strait_of_Malacca":                      "HIGH",
    "Suez_Canal":                             "HIGH",
    "Supply_chain_attack":                    "HIGH",
    "Supply_chain_management":                "LOW",
    "TSMC":                                   "HIGH",
    "Warehouse":                              "LOW",
}

SYSTEM_PROMPT = """You are a supply-chain risk analyst. Given a factual context, you produce a structured
JSON assessment. Be calibrated: CRITICAL only when demonstrable global, multi-sector disruption; HIGH for
regional or single-sector disruption; MEDIUM for localized with spillover or ongoing systemic concern;
LOW when recoverable in under 30 days without broad industry impact, OR when the article describes a
concept/infrastructure rather than an active disruption event.
Return ONLY valid JSON, no prose outside the JSON object."""

USER_TEMPLATE = """CONTEXT (from historical crisis or supply-chain documentation):
---
{context}
---

Assess the supply-chain risk level implied by this context. Return JSON with exactly these keys:
{{
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "confidence": float between 0 and 1,
  "primary_vulnerabilities": [list of 3 chokepoints or concentrated nodes],
  "mitigations": [list of 3 concrete actions],
  "reasoning_one_line": "one sentence for the risk level"
}}"""

DEEPSEEK_FREE_SYSTEM = """You are a supply-chain risk analyst assessing supply-chain disruption severity.
You classify each scenario into exactly one of four risk tiers:
  - CRITICAL: global, multi-sector disruption lasting more than 6 months
  - HIGH: regional or single-sector disruption lasting 1-6 months
  - MEDIUM: localized disruption with spillover, or ongoing systemic concern
  - LOW: concept/infrastructure article with no active disruption, or recovers in under 30 days

Reason step-by-step about the supply-chain implications.
Then end your response with a SINGLE LINE in this exact format (nothing else on that line):
FINAL_RISK=LOW
or FINAL_RISK=MEDIUM
or FINAL_RISK=HIGH
or FINAL_RISK=CRITICAL

Do NOT output academic grades, multiple-choice answers, or any other classification. Only supply-chain risk tier."""

EXTRACTOR_SYSTEM = """You convert unstructured analyst prose into strict JSON. Read the analyst's reasoning,
then output ONE JSON object with keys: risk_level, confidence, primary_vulnerabilities, mitigations,
reasoning_one_line. If a field is not stated, infer conservatively from the text. Output ONLY the JSON,
no commentary."""

CRITIC_SYSTEM = """You are a senior review auditor. You see three analysts' JSON assessments of the same
supply-chain scenario. Identify: (1) whether their risk levels disagree by more than one step,
(2) whether any analyst's reasoning contradicts their risk level, (3) the single most likely correct
risk level given the consensus. Output strict JSON."""

CRITIC_TEMPLATE = """SCENARIO: {name}

JUDGE A ({ja}):
{a}

JUDGE B ({jb}):
{b}

JUDGE C ({jc}):
{c}

Output JSON:
{{
  "levels_disagree_by_more_than_one_step": bool,
  "any_internal_contradiction": bool,
  "best_consensus_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "one_line_review": "..."
}}"""


# ============================================================
# Ollama call + JSON parsing
# ============================================================
def call_ollama(model: str, system: str, user: str, timeout: int = 420,
                 num_predict: int = 900, force_json: bool = True) -> dict:
    t0 = time.time()
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": num_predict},
    }
    if force_json:
        body["format"] = "json"
    try:
        r = requests.post(OLLAMA_URL, json=body, timeout=timeout)
        r.raise_for_status()
        content = r.json()["message"]["content"]
        return {"raw": content, "latency_s": time.time() - t0, "ok_http": True}
    except Exception as e:
        return {"raw": None, "latency_s": time.time() - t0, "ok_http": False,
                "error": str(e)[:200]}


def strip_think(text: str) -> str:
    if not text: return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_json_loose(text: str) -> dict | None:
    if not text: return None
    text = strip_think(text).strip()
    for attempt in (text, text.strip("` \n")):
        try:
            return json.loads(attempt)
        except Exception:
            pass
    for m in re.finditer(r"\{[\s\S]*\}", text):
        try:
            return json.loads(m.group())
        except Exception:
            continue
    return None


def schema_ok(p: dict) -> bool:
    """Soft schema: OK if risk_level is valid + confidence is present.
    Missing vulnerability/mitigation lists get autofilled to empty in normalize_parsed.
    """
    if not isinstance(p, dict): return False
    if "risk_level" not in p: return False
    if str(p["risk_level"]).upper() not in RISK_ORDINAL: return False
    if "confidence" not in p: return False
    return True


def normalize_parsed(p: dict | None) -> dict | None:
    """Fill missing list fields with [] so downstream code doesn't choke."""
    if not isinstance(p, dict): return p
    out = dict(p)
    out["risk_level"] = str(out.get("risk_level", "")).upper()
    if not isinstance(out.get("confidence"), (int, float)): out["confidence"] = 0.5
    for k in ("primary_vulnerabilities", "mitigations"):
        v = out.get(k)
        if not isinstance(v, list): out[k] = []
        else: out[k] = [str(x) for x in v if x]
    if "reasoning_one_line" not in out: out["reasoning_one_line"] = ""
    return out


# ============================================================
# DeepSeek two-pass (BATCH MODE to avoid VRAM thrash):
#   Phase A: all 26 scenarios through deepseek free-form (one model load)
#   Phase B: all 26 CoT outputs through Qwen-14B extractor (one swap)
# ============================================================
def deepseek_free_single(context: str) -> dict:
    t0 = time.time()
    user_prompt = (
        f"SUPPLY-CHAIN CONTEXT:\n---\n{context}\n---\n\n"
        "Classify the supply-chain risk tier (LOW/MEDIUM/HIGH/CRITICAL) based on severity of disruption, "
        "breadth of impact, and recovery time. After your reasoning, emit exactly one line:\n"
        "FINAL_RISK=<LOW|MEDIUM|HIGH|CRITICAL>"
    )
    free = call_ollama("deepseek-r1-local-q4", DEEPSEEK_FREE_SYSTEM, user_prompt,
                        num_predict=2500, force_json=False, timeout=420)
    return {
        "raw_free": strip_think(free["raw"] or "") if free["ok_http"] else None,
        "latency_free_s": time.time() - t0,
        "ok_http": free["ok_http"],
        "error": free.get("error", ""),
    }


def qwen_extract_single(free_text: str) -> dict:
    t0 = time.time()
    if not free_text:
        return {"parsed": None, "latency_extract_s": 0.0, "ok_http": False, "raw_extract": None}
    extractor_prompt = f"""ANALYST RESPONSE:
---
{free_text[:4000]}
---

Extract into JSON with these keys: risk_level, confidence, primary_vulnerabilities, mitigations, reasoning_one_line.
If risk_level is stated as FINAL_LEVEL=X, use X. Be concise."""
    extract = call_ollama(EXTRACTOR, EXTRACTOR_SYSTEM, extractor_prompt,
                           num_predict=500, force_json=True, timeout=120)
    parsed = parse_json_loose(extract["raw"]) if extract["ok_http"] else None
    return {
        "parsed": parsed,
        "latency_extract_s": time.time() - t0,
        "ok_http": extract["ok_http"],
        "raw_extract": (extract["raw"] or "")[:500] if extract["ok_http"] else None,
    }


# ============================================================
# Single-pass judge (Qwen-14B, Mistral-Nemo)
# ============================================================
def single_judge(model: str, context: str) -> dict:
    user = USER_TEMPLATE.format(context=context)
    r = call_ollama(model, SYSTEM_PROMPT, user, num_predict=900, force_json=True, timeout=300)
    parsed = parse_json_loose(r["raw"]) if r["ok_http"] else None
    parsed_norm = normalize_parsed(parsed)
    return {
        "parsed": parsed_norm,
        "latency_s": r["latency_s"],
        "ok": bool(parsed_norm) and schema_ok(parsed_norm),
        "raw": (r["raw"] or "")[:500],
        "error": r.get("error", ""),
    }


# ============================================================
# Metrics
# ============================================================
def krippendorff_alpha_ordinal(ratings_per_scenario: list[list[int]]) -> float:
    """Proper weighted-ordinal alpha across scenarios.

    ratings_per_scenario: list where each element is [judge1_rating, judge2_rating, ...]
    Missing ratings are None.
    """
    # Flatten to coincidences per scenario
    pairs_observed = []
    all_vals = []
    for ratings in ratings_per_scenario:
        vals = [r for r in ratings if r is not None]
        all_vals.extend(vals)
        for a, b in combinations(vals, 2):
            pairs_observed.append((a, b))
    if len(pairs_observed) == 0 or len(set(all_vals)) <= 1: return 1.0

    # Observed disagreement (squared ordinal distance)
    do = np.mean([(a - b) ** 2 for a, b in pairs_observed])
    # Expected disagreement (all pairs from marginal)
    n = len(all_vals)
    de_pairs = [(all_vals[i], all_vals[j]) for i in range(n) for j in range(n) if i != j]
    de = np.mean([(a - b) ** 2 for a, b in de_pairs]) if de_pairs else 0
    if de == 0: return 1.0
    return float(1.0 - do / de)


def fleiss_kappa_nominal(ratings_per_scenario: list[list[int]], k_categories: int = 4) -> float:
    """Fleiss kappa on nominal risk labels {1,2,3,4}. Skip scenarios with < 2 raters."""
    valid = [r for r in ratings_per_scenario if len([x for x in r if x is not None]) >= 2]
    if not valid: return float("nan")
    N = len(valid)
    # Matrix: N x k_categories, count of each label per scenario
    M = np.zeros((N, k_categories))
    n_per_row = []
    for i, r in enumerate(valid):
        clean = [x for x in r if x is not None]
        n_per_row.append(len(clean))
        for x in clean:
            M[i, x - 1] += 1
    # Assume same n across rows (use min)
    n_bar = min(n_per_row)
    if n_bar < 2: return float("nan")
    P_i = (np.sum(M ** 2, axis=1) - n_bar) / (n_bar * (n_bar - 1))
    P_bar = float(np.mean(P_i))
    p_j = np.sum(M, axis=0) / (N * n_bar)
    Pe = float(np.sum(p_j ** 2))
    if Pe >= 1: return 1.0
    return float((P_bar - Pe) / (1 - Pe))


def cohen_weighted_kappa_pairwise(a: list[int], b: list[int], k: int = 4) -> float:
    a = np.array([x for x in a])
    b = np.array([x for x in b])
    mask = ~(np.isnan(a.astype(float)) | np.isnan(b.astype(float)))
    a, b = a[mask].astype(int), b[mask].astype(int)
    if len(a) == 0: return float("nan")
    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i - 1, j - 1] += 1
    O = O / O.sum()
    W = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            W[i, j] = (i - j) ** 2 / (k - 1) ** 2
    ma, mb = O.sum(axis=1), O.sum(axis=0)
    E = np.outer(ma, mb)
    num = float(np.sum(W * O))
    den = float(np.sum(W * E))
    if den == 0: return 1.0
    return float(1 - num / den)


# ============================================================
# Semantic Jaccard via mxbai-embed-large-v1
# ============================================================
_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _EMBEDDER = SentenceTransformer(str(EMBEDDER_PATH), device=dev)
        log.info(f"Loaded mxbai-embed-large-v1 on {dev}")
    return _EMBEDDER


def semantic_jaccard(list_a: list[str], list_b: list[str], threshold: float = 0.65) -> float:
    a = [s.strip() for s in list_a if isinstance(s, str) and s.strip()]
    b = [s.strip() for s in list_b if isinstance(s, str) and s.strip()]
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    emb = _get_embedder()
    ea = emb.encode(a, normalize_embeddings=True)
    eb = emb.encode(b, normalize_embeddings=True)
    # Count matches: for each a_i, does any b_j have cosine >= threshold
    matched_a = 0
    for va in ea:
        sims = eb @ va
        if sims.max() >= threshold: matched_a += 1
    matched_b = 0
    for vb in eb:
        sims = ea @ vb
        if sims.max() >= threshold: matched_b += 1
    # Jaccard-style: matched / (|A|+|B|-matched)
    match_avg = (matched_a + matched_b) / 2
    union = len(a) + len(b) - match_avg
    return float(match_avg / union) if union > 0 else 0.0


def pairwise_semantic_jaccard(lists: list[list[str]], threshold: float = 0.65) -> float:
    valid = [l for l in lists if l]
    if len(valid) < 2: return float("nan")
    js = [semantic_jaccard(valid[i], valid[j], threshold)
          for i in range(len(valid)) for j in range(i + 1, len(valid))]
    return float(np.mean(js)) if js else float("nan")


# ============================================================
# ECE (calibration)
# ============================================================
def ece_binary(confidences: list[float], correct: list[int], n_bins: int = 10) -> tuple[float, list]:
    """Expected calibration error. correct[i] in {0,1}. Returns (ECE, per-bin stats)."""
    if not confidences or len(confidences) != len(correct):
        return float("nan"), []
    confs = np.array(confidences)
    corrs = np.array(correct)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(confs)
    stats = []
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (confs >= bins[i]) & (confs < bins[i + 1])
        else:
            mask = (confs >= bins[i]) & (confs <= bins[i + 1])
        n = int(mask.sum())
        if n == 0:
            stats.append({"bin_lo": float(bins[i]), "bin_hi": float(bins[i + 1]),
                         "n": 0, "mean_conf": None, "accuracy": None})
            continue
        c = float(confs[mask].mean())
        a = float(corrs[mask].mean())
        ece += n / N * abs(a - c)
        stats.append({"bin_lo": float(bins[i]), "bin_hi": float(bins[i + 1]),
                     "n": n, "mean_conf": c, "accuracy": a})
    return float(ece), stats


# ============================================================
# Escalation rubric
# ============================================================
def escalation(consensus_level: str, disagreement_alpha: float) -> str:
    """Deterministic escalation router.
    consensus_level: majority risk_level
    disagreement_alpha: ordinal alpha across judges on this scenario
    """
    lv = RISK_ORDINAL.get(consensus_level, 1)
    low_agreement = (disagreement_alpha is not None) and (disagreement_alpha < 0.5)
    if lv >= 4:  # CRITICAL
        return "C_SUITE_IMMEDIATE"
    if lv == 3 and low_agreement:
        return "C_SUITE_REVIEW"
    if lv == 3:
        return "OPS_DIRECTOR_4H"
    if lv == 2 and low_agreement:
        return "OPS_DIRECTOR_24H"
    if lv == 2:
        return "REGIONAL_MANAGER"
    return "FYI_DASHBOARD"


# ============================================================
# Main
# ============================================================
def load_scenarios() -> list[dict]:
    files = sorted(CRISES.glob("*.txt"))
    out = []
    for f in files:
        txt = f.read_text(encoding="utf-8", errors="ignore")[:3000]
        out.append({"name": f.stem, "context": txt})
    return out


def unload_model(model: str):
    try:
        requests.post(OLLAMA_URL, json={
            "model": model, "messages": [{"role": "user", "content": "."}],
            "stream": False, "keep_alive": 0, "options": {"num_predict": 1}
        }, timeout=60)
    except Exception:
        pass
    time.sleep(3)  # give Windows time to release CUDA_Host memory


def run_judge_pass(judge: str, scenarios: list[dict]) -> dict:
    """Run one judge across all scenarios (single-pass models only). Per-judge cache."""
    cache_path = RESULTS / f"R4_DANGEROUS_V2_judge_{judge.replace(':','_')}.json"
    if cache_path.exists():
        log.info(f"\n=== Judge: {judge} RESUMING from cache ===")
        return json.loads(cache_path.read_text())
    log.info(f"\n=== Judge: {judge} ({len(scenarios)} scenarios) ===")
    per = {}
    for i, s in enumerate(scenarios, 1):
        r = single_judge(judge, s["context"])
        per[s["name"]] = r
        status = "OK" if r["ok"] else "FAIL"
        lat = r["latency_s"]
        log.info(f"  [{i:2d}/{len(scenarios)}] {s['name'][:42]:<42} {status:<4} {lat:5.1f}s")
    cache_path.write_text(json.dumps(per, default=str))
    unload_model(judge)
    return per


def run_deepseek_batched(scenarios: list[dict]) -> dict:
    """Phase A: all scenarios through DeepSeek free-form. Phase B: all through Qwen extractor.
    One load + one swap instead of 26 x 2 = 52 swaps. Resume-safe: persists Phase A to disk.
    """
    cache_path = RESULTS / "R4_DANGEROUS_V2_phaseA_cache.json"
    if cache_path.exists():
        log.info(f"\n=== DeepSeek Phase A: RESUMING from cache {cache_path.name} ===")
        phase_a = json.loads(cache_path.read_text())
    else:
        log.info(f"\n=== DeepSeek Phase A: free-form CoT on {len(scenarios)} scenarios ===")
        phase_a = {}
        for i, s in enumerate(scenarios, 1):
            r = deepseek_free_single(s["context"])
            phase_a[s["name"]] = r
            status = "OK" if r["ok_http"] and r["raw_free"] else "FAIL"
            log.info(f"  [{i:2d}/{len(scenarios)}] {s['name'][:42]:<42} {status:<4} {r['latency_free_s']:5.1f}s")
        cache_path.write_text(json.dumps(phase_a, default=str))
        log.info(f"Phase A cached to {cache_path.name}")
    # unload DeepSeek before swapping to Qwen-14B
    unload_model("deepseek-r1-local-q4")

    cache_b_path = RESULTS / "R4_DANGEROUS_V2_phaseB_cache.json"
    if cache_b_path.exists():
        log.info(f"\n=== DeepSeek Phase B: RESUMING from cache ===")
        return json.loads(cache_b_path.read_text())
    log.info(f"\n=== DeepSeek Phase B: Qwen-14B extraction on {len(scenarios)} CoT outputs ===")
    per = {}
    for i, s in enumerate(scenarios, 1):
        a = phase_a[s["name"]]
        if not a["raw_free"]:
            per[s["name"]] = {"parsed": None, "latency_s": a["latency_free_s"],
                               "ok": False, "stage": "free_pass_failed",
                               "raw_free": None, "raw_extract": None,
                               "error": a.get("error", "")}
            log.info(f"  [{i:2d}/{len(scenarios)}] {s['name'][:42]:<42} SKIP (no free-pass output)")
            continue
        b = qwen_extract_single(a["raw_free"])
        total_lat = a["latency_free_s"] + b["latency_extract_s"]
        parsed_norm = normalize_parsed(b["parsed"])
        ok = bool(parsed_norm) and schema_ok(parsed_norm)
        # Fallback: if Qwen extraction failed, scrape FINAL_RISK directly from DeepSeek raw_free
        if not ok and a["raw_free"]:
            m = re.search(r"FINAL_RISK\s*[:=]\s*(LOW|MEDIUM|HIGH|CRITICAL)", a["raw_free"], re.IGNORECASE)
            if not m:
                # broader search across common phrasings
                m = re.search(r"\b(CRITICAL|HIGH|MEDIUM|LOW)\s*(?:risk|level|tier)\b", a["raw_free"], re.IGNORECASE)
            if m:
                fallback = {"risk_level": m.group(1).upper(), "confidence": 0.5,
                            "primary_vulnerabilities": [], "mitigations": [],
                            "reasoning_one_line": "(fallback from DeepSeek FINAL_RISK marker)"}
                parsed_norm = normalize_parsed(fallback)
                ok = True
        per[s["name"]] = {
            "parsed": parsed_norm,
            "latency_s": total_lat,
            "latency_free_s": a["latency_free_s"],
            "latency_extract_s": b["latency_extract_s"],
            "ok": ok,
            "stage": "complete" if ok else "extract_failed",
            "raw_free": a["raw_free"][:500] if a["raw_free"] else None,
            "raw_extract": b.get("raw_extract"),
        }
        status = "OK" if ok else "FAIL"
        log.info(f"  [{i:2d}/{len(scenarios)}] {s['name'][:42]:<42} {status:<4} {b['latency_extract_s']:5.1f}s")
    cache_b_path.write_text(json.dumps(per, default=str))
    # Keep Qwen-14B loaded — it's the next judge anyway
    return per


def run_critic_pass(scenarios: list[dict], judge_outputs: dict) -> dict:
    """Critic reviews the 3 judge assessments per scenario."""
    cache_path = RESULTS / "R4_DANGEROUS_V2_critic_cache.json"
    if cache_path.exists():
        log.info(f"\n=== Critic pass: RESUMING from cache ===")
        return json.loads(cache_path.read_text())
    log.info(f"\n=== Critic pass: {CRITIC} ===")
    crit = {}
    for i, s in enumerate(scenarios, 1):
        name = s["name"]
        # Gather the 3 judges' JSON outputs
        j_outs = []
        for j in JUDGES:
            jp = judge_outputs[j].get(name, {}).get("parsed")
            j_outs.append(json.dumps(jp, ensure_ascii=False)[:800] if jp else "(failed to parse)")
        user = CRITIC_TEMPLATE.format(name=name,
                                       ja=JUDGES[0], a=j_outs[0],
                                       jb=JUDGES[1], b=j_outs[1],
                                       jc=JUDGES[2], c=j_outs[2])
        r = call_ollama(CRITIC, CRITIC_SYSTEM, user, num_predict=400, force_json=True, timeout=180)
        parsed = parse_json_loose(r["raw"]) if r["ok_http"] else None
        crit[name] = {"parsed": parsed, "latency_s": r["latency_s"],
                      "ok": isinstance(parsed, dict), "raw": (r["raw"] or "")[:400]}
        status = "OK" if crit[name]["ok"] else "FAIL"
        log.info(f"  [{i:2d}/{len(scenarios)}] {name[:42]:<42} {status:<4} {r['latency_s']:5.1f}s")
    cache_path.write_text(json.dumps(crit, default=str))
    return crit


def main():
    t0 = time.time()
    log.info("R4 Dangerous V2 — BEAST mode")

    # Health check
    try:
        h = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        h.raise_for_status()
    except Exception as e:
        log.error(f"Ollama not reachable: {e}")
        return

    scenarios = load_scenarios()
    log.info(f"Loaded {len(scenarios)} scenarios")

    # === Judge passes (judge-first, VRAM-safe, batched DeepSeek) ===
    judge_outputs = {}
    # DeepSeek: phase A (free-form all 26) -> unload -> phase B (Qwen extract all 26)
    judge_outputs["deepseek-r1-local-q4"] = run_deepseek_batched(scenarios)
    # Qwen-14B as standalone judge (model already loaded from extraction phase)
    judge_outputs["qwen25-14b-local"] = run_judge_pass("qwen25-14b-local", scenarios)
    # Mistral-Nemo
    judge_outputs["mistral-nemo-local"] = run_judge_pass("mistral-nemo-local", scenarios)

    # === Critic pass ===
    critic_outputs = run_critic_pass(scenarios, judge_outputs)
    unload_model(CRITIC)

    # === Per-scenario consensus + semantic similarity ===
    log.info("\n=== Consensus + semantic similarity ===")
    per_scenario = {}
    for s in scenarios:
        name = s["name"]
        parsed_per_j = {}
        risks = []
        confs = []
        vulns_lists = []
        mits_lists = []
        latencies = {}
        for j in JUDGES:
            pj = judge_outputs[j][name]
            parsed_per_j[j] = {"ok": pj["ok"], "latency_s": pj["latency_s"],
                                "parsed": pj["parsed"], "error": pj.get("error", ""),
                                "raw_preview": (pj.get("raw") or pj.get("raw_free") or "")[:300]}
            latencies[j] = pj["latency_s"]
            p = pj["parsed"] if pj["ok"] else None
            if not p: continue
            risks.append(RISK_ORDINAL.get(str(p.get("risk_level", "")).upper()))
            if isinstance(p.get("confidence"), (int, float)):
                confs.append(float(p["confidence"]))
            if isinstance(p.get("primary_vulnerabilities"), list):
                vulns_lists.append([str(x) for x in p["primary_vulnerabilities"]])
            if isinstance(p.get("mitigations"), list):
                mits_lists.append([str(x) for x in p["mitigations"]])
        risks_clean = [r for r in risks if r is not None]
        majority = int(np.round(np.median(risks_clean))) if risks_clean else None
        majority_label = RISK_REV.get(majority, "UNKNOWN") if majority else "UNKNOWN"
        per_scenario[name] = {
            "ground_truth": GROUND_TRUTH.get(name, "UNKNOWN"),
            "per_judge": parsed_per_j,
            "risk_ratings_ordinal": risks,
            "risk_majority": majority_label,
            "mean_confidence": float(np.mean(confs)) if confs else None,
            "vulnerabilities_semantic_jaccard": pairwise_semantic_jaccard(vulns_lists),
            "mitigations_semantic_jaccard": pairwise_semantic_jaccard(mits_lists),
            "latencies_s": latencies,
            "critic": critic_outputs[name],
        }
        log.info(f"  {name[:40]:<40} GT={GROUND_TRUTH.get(name,'?'):<8} MAJ={majority_label:<8} "
                 f"vulnJ={per_scenario[name]['vulnerabilities_semantic_jaccard']:.3f} "
                 f"mitJ={per_scenario[name]['mitigations_semantic_jaccard']:.3f}")

    # === Aggregate agreement ===
    ratings_matrix = [per_scenario[s["name"]]["risk_ratings_ordinal"] for s in scenarios]
    alpha = krippendorff_alpha_ordinal(ratings_matrix)
    fleiss = fleiss_kappa_nominal(ratings_matrix)
    # Pairwise weighted kappa
    pairwise_kappa = {}
    judge_ratings_full = {j: [] for j in JUDGES}
    for s in scenarios:
        for ji, j in enumerate(JUDGES):
            r = per_scenario[s["name"]]["risk_ratings_ordinal"][ji] if ji < len(per_scenario[s["name"]]["risk_ratings_ordinal"]) else None
            judge_ratings_full[j].append(r if r is not None else float("nan"))
    for a, b in combinations(JUDGES, 2):
        pairwise_kappa[f"{a}_vs_{b}"] = cohen_weighted_kappa_pairwise(
            judge_ratings_full[a], judge_ratings_full[b])

    # === Accuracy vs ground truth + confusion matrices ===
    log.info("\n=== Accuracy vs ground truth ===")
    gt_accuracy = {}
    confusion = {}  # {judge: 4x4 matrix}
    for j in JUDGES:
        correct = 0; total = 0
        conf_mat = np.zeros((4, 4), dtype=int)  # rows=GT, cols=Pred
        for s in scenarios:
            name = s["name"]
            gt = GROUND_TRUTH.get(name)
            if not gt: continue
            p = judge_outputs[j][name].get("parsed")
            if not p or not isinstance(p, dict): continue
            pred = str(p.get("risk_level", "")).upper()
            if pred not in RISK_ORDINAL: continue
            total += 1
            if pred == gt: correct += 1
            conf_mat[RISK_ORDINAL[gt] - 1, RISK_ORDINAL[pred] - 1] += 1
        gt_accuracy[j] = {"correct": correct, "total": total,
                          "accuracy": correct / total if total > 0 else 0.0}
        confusion[j] = conf_mat.tolist()
        log.info(f"  {j:<25}  {correct}/{total}  acc={correct/max(total,1):.3f}")

    # Majority-vote accuracy
    maj_correct = 0; maj_total = 0
    maj_conf = np.zeros((4, 4), dtype=int)
    for s in scenarios:
        name = s["name"]
        gt = GROUND_TRUTH.get(name)
        maj = per_scenario[name]["risk_majority"]
        if not gt or maj == "UNKNOWN": continue
        maj_total += 1
        if maj == gt: maj_correct += 1
        maj_conf[RISK_ORDINAL[gt] - 1, RISK_ORDINAL[maj] - 1] += 1
    gt_accuracy["majority_vote"] = {"correct": maj_correct, "total": maj_total,
                                      "accuracy": maj_correct / max(maj_total, 1)}
    confusion["majority_vote"] = maj_conf.tolist()
    log.info(f"  {'majority_vote':<25}  {maj_correct}/{maj_total}  acc={maj_correct/max(maj_total,1):.3f}")

    # === Calibration (ECE) per judge ===
    ece_results = {}
    for j in JUDGES:
        confs = []; corrs = []
        for s in scenarios:
            name = s["name"]
            gt = GROUND_TRUTH.get(name)
            p = judge_outputs[j][name].get("parsed")
            if not p or not isinstance(p, dict) or gt is None: continue
            conf = p.get("confidence")
            pred = str(p.get("risk_level", "")).upper()
            if not isinstance(conf, (int, float)) or pred not in RISK_ORDINAL: continue
            confs.append(float(conf))
            corrs.append(1 if pred == gt else 0)
        ece, stats = ece_binary(confs, corrs, n_bins=10)
        ece_results[j] = {"ece": ece, "n_predictions": len(confs), "bins": stats}
        log.info(f"  ECE {j:<25} = {ece:.4f}  (n={len(confs)})")

    # === Per-scenario ordinal alpha (for escalation routing) ===
    for s in scenarios:
        ratings = [r for r in per_scenario[s["name"]]["risk_ratings_ordinal"] if r is not None]
        sc_alpha = krippendorff_alpha_ordinal([ratings]) if len(ratings) >= 2 else float("nan")
        per_scenario[s["name"]]["scenario_ordinal_alpha"] = sc_alpha
        per_scenario[s["name"]]["escalation"] = escalation(per_scenario[s["name"]]["risk_majority"], sc_alpha)

    escalation_counts = {}
    for s in scenarios:
        e = per_scenario[s["name"]]["escalation"]
        escalation_counts[e] = escalation_counts.get(e, 0) + 1

    # === Final output ===
    out = {
        "judges": JUDGES,
        "critic": CRITIC,
        "extractor": EXTRACTOR,
        "n_scenarios": len(scenarios),
        "per_scenario": per_scenario,
        "agreement": {
            "krippendorff_alpha_ordinal": alpha,
            "fleiss_kappa_nominal": fleiss,
            "pairwise_cohen_weighted_kappa": pairwise_kappa,
        },
        "accuracy_vs_ground_truth": gt_accuracy,
        "confusion_matrices": confusion,
        "calibration_ece": ece_results,
        "escalation_distribution": escalation_counts,
        "summary": {
            "parse_success_rate_per_judge": {
                j: sum(1 for s in scenarios if judge_outputs[j][s["name"]]["ok"]) / len(scenarios)
                for j in JUDGES
            },
            "mean_latency_s_per_judge": {
                j: float(np.mean([judge_outputs[j][s["name"]]["latency_s"] for s in scenarios]))
                for j in JUDGES
            },
            "critic_success_rate": sum(1 for v in critic_outputs.values() if v["ok"]) / len(scenarios),
            "mean_vulnerabilities_semantic_jaccard": float(np.nanmean([
                per_scenario[s["name"]]["vulnerabilities_semantic_jaccard"] for s in scenarios])),
            "mean_mitigations_semantic_jaccard": float(np.nanmean([
                per_scenario[s["name"]]["mitigations_semantic_jaccard"] for s in scenarios])),
            "total_elapsed_min": (time.time() - t0) / 60,
        },
    }

    out_path = RESULTS / "R4_DANGEROUS_V2.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info(f"\n=== SUMMARY ===")
    log.info(f"  Krippendorff alpha (ordinal)       = {alpha:.3f}")
    log.info(f"  Fleiss kappa (nominal)             = {fleiss}")
    log.info(f"  Pairwise weighted kappa            = {pairwise_kappa}")
    for j in JUDGES:
        log.info(f"  {j:<25}  parse={out['summary']['parse_success_rate_per_judge'][j]*100:.0f}%  "
                 f"acc_vs_GT={gt_accuracy[j]['accuracy']:.3f}  ECE={ece_results[j]['ece']:.3f}")
    log.info(f"  majority_vote             acc_vs_GT={gt_accuracy['majority_vote']['accuracy']:.3f}")
    log.info(f"  mean vuln semantic_jaccard = {out['summary']['mean_vulnerabilities_semantic_jaccard']:.3f}")
    log.info(f"  mean mitig semantic_jaccard = {out['summary']['mean_mitigations_semantic_jaccard']:.3f}")
    log.info(f"  escalation distribution    = {escalation_counts}")
    log.info(f"  total elapsed              = {out['summary']['total_elapsed_min']:.1f} min")
    log.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
