"""Verify mistral-nemo-local Ollama model with 3 tests."""
from __future__ import annotations

import json
import time
from pathlib import Path

import ollama

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "v3_arcadia" / "results" / "mistral_nemo_verify.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

MODEL = "mistral-nemo-local"
tests = [
    ("reasoning", "In 2 sentences: why activate a backup supplier during a typhoon warning?", None, 180),
    ("long_context_test", "Summarize in one sentence: " + ("Supply chain resilience requires diversification, visibility, and proactive risk mitigation. " * 30), None, 120),
    ("json_mode", "Output JSON with keys 'impact' (HIGH/MEDIUM/LOW) and 'action' (one sentence) for: M7.5 earthquake in Taiwan affecting TSMC.", "json", 200),
]

result = {"model": MODEL, "tests": []}
for name, prompt, fmt, predict in tests:
    t0 = time.time()
    try:
        kwargs = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                  "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": predict}}
        if fmt: kwargs["format"] = fmt
        r = ollama.chat(**kwargs)
        elapsed = time.time() - t0
        content = r["message"]["content"]
        detail = {"response": content[:400], "latency_s": round(elapsed, 2)}
        if fmt == "json":
            try:
                obj = json.loads(content)
                detail["json_parsed"] = True
                detail["keys"] = sorted(obj.keys()) if isinstance(obj, dict) else None
            except Exception as e:
                detail["json_parsed"] = False
                detail["parse_error"] = str(e)[:100]
        print(f"[{name}] OK ({elapsed:.1f}s): {content[:120]!r}")
        result["tests"].append({"name": name, "status": "OK", **detail})
    except Exception as e:
        print(f"[{name}] FAIL: {e}")
        result["tests"].append({"name": name, "status": "FAIL", "error": str(e)[:300]})

result["all_ok"] = all(t["status"] == "OK" for t in result["tests"])
OUT.write_text(json.dumps(result, indent=2))
print(f"\nSaved {OUT}  all_ok={result['all_ok']}")
