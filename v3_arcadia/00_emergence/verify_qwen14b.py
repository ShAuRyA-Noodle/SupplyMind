"""3-test verification of qwen25-14b-local in Ollama."""
from __future__ import annotations

import json
import time
from pathlib import Path

import ollama

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "v3_arcadia" / "results" / "qwen14b_verify.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

MODEL = "qwen25-14b-local"
result = {"model": MODEL, "tests": []}

tests = [
    ("factual", "In one sentence, what was Toyota's approximate revenue loss from the 2011 Tohoku earthquake?", None, 120),
    ("reasoning", "List 3 reasons a company should activate a backup supplier during a typhoon warning. Be concise.", None, 200),
    ("json_mode",
     "Output a JSON object with keys 'risk_level' (one of LOW/AMBER/RED) and 'recommendation' (one sentence) for: "
     "cyclone severity 0.85 approaching SUP_TSMC with 2 days inventory.",
     "json", 200),
]

for name, prompt, fmt, predict in tests:
    t0 = time.time()
    try:
        kwargs = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                  "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": predict}}
        if fmt:
            kwargs["format"] = fmt
        r = ollama.chat(**kwargs)
        content = r["message"]["content"]
        elapsed = time.time() - t0
        status = "OK"
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
        result["tests"].append({"name": name, "status": status, **detail})
    except Exception as e:
        print(f"[{name}] FAIL: {e}")
        result["tests"].append({"name": name, "status": "FAIL", "error": str(e)[:300]})

result["all_ok"] = all(t["status"] == "OK" for t in result["tests"])
OUT.write_text(json.dumps(result, indent=2))
print(f"\nSaved {OUT}  all_ok={result['all_ok']}")
