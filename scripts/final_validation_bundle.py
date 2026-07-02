"""final_validation_bundle.py — live-API validation receipt for final submit.

Produces 1 receipt in one execution:
  1. api_keys_live_proof.json      — 4 keys (OPENROUTER, EIA, NASA_FIRMS, GFW)
                                       each makes a real call, hash response

The receipt is mirrored to FINAL_SUBMIT/receipts/ + sha256 stamped.

REMOVED (Wave-4 leftovers sweep, 2026-07-02): two fabricated blocks that this
bundle used to emit were deleted, per CLAUDE.md §0 (zero fakes):
  - process_supervision()  — a hand-crafted 4/5-guess Wordle trajectory with a
    hardcoded uniform baseline, published as a "2735x variance amplification"
    headline (CLAIMS_LEDGER A4, STRUCK). Its receipt was retired by the Wave-3
    evidence-refresh pass; the honest replacement is
    scripts/pass28_killshot_v2.py block 28.F (real _score_guess per-step credit
    → pass28_F_process_super_plot.json).
  - ablation_matrix()      — a "leave-one-out reward ablation" run with a RANDOM
    policy, so ablating reward components never changed the policy's behaviour
    (every trial reported the same 0.27 solve rate). It was theater, not a real
    component ablation; a genuine ablation trains QR-DQN variants (REBUILD_BACKLOG
    R3, P1.3). Block + its stale receipt purged here.
(The former cross_env_transfer block was already removed in an earlier pass; its
receipt is on the RERUN_QUEUE permanently-retired list — feature was fabricated.)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# LIVE API KEY UTILIZATION PROOF
# ---------------------------------------------------------------------------

def api_keys_live_proof() -> dict:
    """Make 1 real call per key, hash response, prove keys actively used."""
    import requests
    from scripts._env import load_env
    load_env()
    out = {"framework": "live-call hash proof",
            "started_at": time.time(),
            "keys": {}}

    # 1. OPENROUTER — quick tiny chat completion
    or_key = os.environ.get("OPENROUTER_API_KEY")
    if or_key:
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {or_key}",
                          "Content-Type": "application/json"},
                json={"model": "openai/gpt-4o-mini",
                       "messages": [{"role": "user", "content": "Reply 'OK'"}],
                       "max_tokens": 5},
                timeout=15,
            )
            ok = r.status_code == 200
            content_hash = hashlib.sha256(r.content[:1000]).hexdigest()
            out["keys"]["OPENROUTER"] = {
                "status_code": r.status_code, "ok": ok,
                "response_hash_first_1k": content_hash,
                "endpoint": "openrouter.ai/api/v1/chat/completions",
                "model": "openai/gpt-4o-mini",
            }
        except Exception as e:  # noqa: BLE001
            out["keys"]["OPENROUTER"] = {"ok": False, "error": str(e)[:200]}
    else:
        out["keys"]["OPENROUTER"] = {"ok": False, "error": "key_not_set"}

    # 2. EIA — real fuel price query
    eia_key = os.environ.get("EIA_API_KEY")
    if eia_key:
        try:
            r = requests.get(
                "https://api.eia.gov/v2/petroleum/pri/spt/data/",
                params={"api_key": eia_key, "frequency": "weekly",
                         "data[0]": "value", "length": 5},
                timeout=15,
            )
            ok = r.status_code == 200
            content_hash = hashlib.sha256(r.content[:1000]).hexdigest()
            out["keys"]["EIA"] = {
                "status_code": r.status_code, "ok": ok,
                "response_hash_first_1k": content_hash,
                "endpoint": "api.eia.gov/v2/petroleum/pri/spt",
                "n_bytes": len(r.content),
            }
        except Exception as e:  # noqa: BLE001
            out["keys"]["EIA"] = {"ok": False, "error": str(e)[:200]}
    else:
        out["keys"]["EIA"] = {"ok": False, "error": "key_not_set"}

    # 3. NASA_FIRMS — real fire data query
    firms_key = os.environ.get("NASA_FIRMS_MAP_KEY")
    if firms_key:
        try:
            r = requests.get(
                f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                f"{firms_key}/MODIS_NRT/world/1",
                timeout=20,
            )
            ok = r.status_code == 200
            content_hash = hashlib.sha256(r.content[:1000]).hexdigest()
            out["keys"]["NASA_FIRMS"] = {
                "status_code": r.status_code, "ok": ok,
                "response_hash_first_1k": content_hash,
                "endpoint": "firms.modaps.eosdis.nasa.gov/api/area/csv",
                "csv_lines": r.text.count("\n") if ok else 0,
            }
        except Exception as e:  # noqa: BLE001
            out["keys"]["NASA_FIRMS"] = {"ok": False, "error": str(e)[:200]}
    else:
        out["keys"]["NASA_FIRMS"] = {"ok": False, "error": "key_not_set"}

    # 4. GFW (Global Fishing Watch) — fishing-vessel real-time
    gfw_key = os.environ.get("GFW_API_TOKEN")
    if gfw_key:
        try:
            r = requests.get(
                "https://gateway.api.globalfishingwatch.org/v3/datasets",
                params={"datasets": "public-global-fishing-effort:latest",
                         "format": "json"},
                headers={"Authorization": f"Bearer {gfw_key}"},
                timeout=15,
            )
            if r.status_code == 422:
                # Retry once with /v3/4wings/stats which only needs auth
                r = requests.get(
                    "https://gateway.api.globalfishingwatch.org/v3/4wings/stats",
                    params={"datasets[0]": "public-global-fishing-effort:latest",
                             "fields": "FLAGS"},
                    headers={"Authorization": f"Bearer {gfw_key}"},
                    timeout=15,
                )
            # ok is TRUE only on a real 2xx data response. A non-2xx status
            # (422 malformed, 503 unavailable) is NOT a success — it is a
            # degraded/failed call. key_authenticated is a separate, weaker
            # signal: the credential passed auth iff we did not get 401/403.
            ok = 200 <= r.status_code < 300
            content_hash = hashlib.sha256(r.content[:1000]).hexdigest()
            out["keys"]["GFW"] = {
                "status_code": r.status_code, "ok": ok,
                "key_authenticated": r.status_code not in (401, 403),
                "response_hash_first_1k": content_hash,
                "endpoint": "gateway.api.globalfishingwatch.org/v3/4wings/stats",
                "n_bytes": len(r.content),
                "note": ("ok=true requires 2xx live data. 422/503 => ok=false "
                          "(degraded); key_authenticated stays true unless 401/403."),
            }
        except Exception as e:  # noqa: BLE001
            out["keys"]["GFW"] = {"ok": False, "error": str(e)[:200]}
    else:
        out["keys"]["GFW"] = {"ok": False, "error": "key_not_set"}

    out["finished_at"] = time.time()
    out["wall_clock_s"] = round(out["finished_at"] - out["started_at"], 2)
    out["n_keys_present"] = sum(1 for k in out["keys"].values()
                                  if k.get("ok") is True or k.get("status_code"))
    out["n_keys_ok_200"] = sum(1 for k in out["keys"].values()
                                 if k.get("ok") is True)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save(name: str, data: dict) -> str:
    receipt = REPO / "tests" / "receipts" / f"{name}.json"
    mirror = REPO / "FINAL_SUBMIT" / "receipts" / f"{name}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    mirror.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(data, indent=2, default=str)
    receipt.write_text(txt, encoding="utf-8")
    mirror.write_text(txt, encoding="utf-8")
    sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    receipt.with_suffix(".sha256").write_text(sha + "\n", encoding="utf-8")
    return sha


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = {}

    logger.info("[1/1] api keys live proof ...")
    r4 = api_keys_live_proof()
    summary["api_keys_live_sha"] = save("api_keys_live_proof", r4)

    summary["headlines"] = {
        "n_keys_ok": r4.get("n_keys_ok_200"),
        "n_keys_total": len(r4.get("keys", {})),
    }
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
