"""Verify qwen25-coder-local Ollama model."""
from __future__ import annotations
import json, time
from pathlib import Path
import ollama

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "v3_arcadia" / "results" / "qwen_coder_verify.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

MODEL = "qwen25-coder-local"
tests = [
    ("code_gen", "Write a 5-line Python function that returns the Fibonacci sequence up to n. No explanations.", None, 250),
    ("code_review", "List 2 problems with this code:\n```python\ndef f(x):\n    for i in range(len(x)):\n        x.append(x[i])\n    return x\n```\nBe concise.", None, 300),
    ("json_mode", "Output JSON: {\"language\":\"python\",\"complexity\":\"O(n)\",\"bugs\":<integer>} for this snippet:\nresult = [x*2 for x in nums if x > 0]", "json", 150),
]

result = {"model": MODEL, "tests": []}
for name, prompt, fmt, predict in tests:
    t0 = time.time()
    try:
        kwargs = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                  "options": {"temperature": 0.2, "num_predict": predict}}
        if fmt: kwargs["format"] = fmt
        r = ollama.chat(**kwargs)
        elapsed = time.time() - t0
        content = r["message"]["content"]
        detail = {"response": content[:400], "latency_s": round(elapsed, 2)}
        if fmt == "json":
            try:
                obj = json.loads(content); detail["json_parsed"] = True
                detail["keys"] = sorted(obj.keys()) if isinstance(obj, dict) else None
            except Exception as e:
                detail["json_parsed"] = False; detail["parse_error"] = str(e)[:100]
        print(f"[{name}] OK ({elapsed:.1f}s): {content[:120]!r}")
        result["tests"].append({"name": name, "status": "OK", **detail})
    except Exception as e:
        print(f"[{name}] FAIL: {e}")
        result["tests"].append({"name": name, "status": "FAIL", "error": str(e)[:300]})

result["all_ok"] = all(t["status"] == "OK" for t in result["tests"])
OUT.write_text(json.dumps(result, indent=2))
print(f"\nSaved {OUT}  all_ok={result['all_ok']}")
