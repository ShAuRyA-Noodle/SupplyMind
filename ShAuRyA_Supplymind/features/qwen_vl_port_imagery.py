"""
qwen_vl_port_imagery.py — G3+F1. Qwen-VL-7B satellite-imagery port-risk scorer.

Runs a vision-language model (Qwen-VL) on satellite imagery of critical ports
(Kaohsiung, Shanghai, Long Beach, Rotterdam, Jebel Ali, Haifa, Hodeidah) and
extracts structured supply-chain risk signals:

    {
        "ship_queue_count": int,
        "container_stack_density": "low|medium|high",
        "smoke_or_fire": bool,
        "flood_indicators": bool,
        "unusual_activity": str,
        "risk_score": float (0-1),
        "confidence": float (0-1),
    }

Modes:
    "ollama" — uses qwen2.5-vl:7b via Ollama HTTP (requires model pulled)
    "local"  — uses transformers + Qwen2VLForConditionalGeneration (requires GPU)
    "heuristic" — deterministic fallback using PIL image stats (no VL model)

Default: attempt ollama -> fall back to heuristic. The heuristic is not random;
it computes color histograms + blob counts so that the integration path is
exercised even without the 15 GB VL model loaded.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OUT_DIR = Path(__file__).resolve().parent / "port_imagery"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Port anchors — (port_id, name, typical baseline tanker count, lat, lon)
PORT_ANCHORS = {
    "KAOHSIUNG": {"name": "Kaohsiung (Taiwan)", "baseline_queue": 18, "lat": 22.62, "lon": 120.27},
    "SHANGHAI": {"name": "Shanghai (China)", "baseline_queue": 45, "lat": 31.23, "lon": 121.47},
    "LONG_BEACH": {"name": "Long Beach (USA)", "baseline_queue": 25, "lat": 33.77, "lon": -118.20},
    "ROTTERDAM": {"name": "Rotterdam (NL)", "baseline_queue": 30, "lat": 51.92, "lon": 4.48},
    "JEBEL_ALI": {"name": "Jebel Ali (UAE)", "baseline_queue": 20, "lat": 25.01, "lon": 55.06},
    "HAIFA": {"name": "Haifa (Israel)", "baseline_queue": 10, "lat": 32.82, "lon": 35.00},
    "HODEIDAH": {"name": "Hodeidah (Yemen)", "baseline_queue": 8, "lat": 14.82, "lon": 42.95},
}


@dataclass
class PortRiskAssessment:
    port_id: str
    port_name: str
    mode: str                        # "ollama" | "local" | "heuristic"
    ship_queue_count: int = 0
    container_stack_density: str = "medium"
    smoke_or_fire: bool = False
    flood_indicators: bool = False
    unusual_activity: str = ""
    risk_score: float = 0.3
    confidence: float = 0.5
    latency_s: float = 0.0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "port_id": self.port_id,
            "port_name": self.port_name,
            "mode": self.mode,
            "ship_queue_count": self.ship_queue_count,
            "container_stack_density": self.container_stack_density,
            "smoke_or_fire": self.smoke_or_fire,
            "flood_indicators": self.flood_indicators,
            "unusual_activity": self.unusual_activity,
            "risk_score": round(self.risk_score, 3),
            "confidence": round(self.confidence, 3),
            "latency_s": round(self.latency_s, 2),
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# Ollama qwen-vl path
# ---------------------------------------------------------------------------


def _ollama_has_vl() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()
        return any("vl" in m.get("name", "").lower() for m in r.get("models", []))
    except Exception:
        return False


def _call_ollama_vl(image_b64: str, prompt: str, model: str = "qwen2.5-vl:7b") -> dict:
    start = time.time()
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 16384},
        },
        timeout=120,
    )
    r.raise_for_status()
    text = r.json()["message"]["content"]
    data = json.loads(text)
    data["_latency_s"] = time.time() - start
    return data


# ---------------------------------------------------------------------------
# Heuristic fallback (no VL model required)
# ---------------------------------------------------------------------------


def _heuristic_from_image(img_bytes: bytes) -> dict:
    """Compute a crude risk signal from basic image statistics.

    Strategy:
        - Red/orange-dominant pixels > 5% -> smoke_or_fire = True (burning risk)
        - Blue saturation abnormally low + brown dominance -> flood_indicators
        - Edge/blob count proxy via std-dev on grayscale intensity
        - Container density via grayscale skew
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return {"ship_queue_count": 0, "container_stack_density": "medium",
                "smoke_or_fire": False, "flood_indicators": False,
                "unusual_activity": "PIL not installed", "risk_score": 0.3,
                "confidence": 0.1}

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    red_dominance = float(((r - g) > 0.2).mean()) + float(((r - b) > 0.2).mean())
    smoke = red_dominance > 0.15
    brown_fraction = float(((r > 0.3) & (g > 0.2) & (g < 0.55) & (b < 0.35)).mean())
    blue_sat = float(b.mean())
    flood = brown_fraction > 0.25 and blue_sat < 0.4

    grey = arr.mean(axis=-1)
    density_score = float(grey.std())
    density_label = ("high" if density_score > 0.20
                     else "medium" if density_score > 0.12 else "low")
    # Rough ship-count proxy: count dark blobs below grey mean threshold
    dark_frac = float((grey < grey.mean() * 0.5).mean())
    ship_count = int(dark_frac * 200)  # calibrated roughly

    risk = 0.2 + 0.3 * float(smoke) + 0.25 * float(flood) + 0.15 * dark_frac
    return {
        "ship_queue_count": ship_count,
        "container_stack_density": density_label,
        "smoke_or_fire": smoke,
        "flood_indicators": flood,
        "unusual_activity": ("smoke detected" if smoke else
                             "possible flooding" if flood else
                             "nominal"),
        "risk_score": min(1.0, risk),
        "confidence": 0.35,    # heuristic is never high-confidence
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


VL_PROMPT = """You are a supply-chain satellite-imagery analyst. Look at the
image of the port and return JSON:
{
  "ship_queue_count": int (0-100, ships visible waiting or moored),
  "container_stack_density": "low" | "medium" | "high",
  "smoke_or_fire": bool (visible smoke plumes or fires),
  "flood_indicators": bool (visible flooding, mud, abnormal water extent),
  "unusual_activity": short string describing anything atypical,
  "risk_score": float 0.0-1.0 (overall supply-chain risk),
  "confidence": float 0.0-1.0 (your confidence in this assessment)
}"""


def assess_port_image(
    image_bytes: bytes,
    port_id: str,
    prefer_mode: str = "auto",
) -> PortRiskAssessment:
    """Main entry. Accepts raw image bytes + port identifier."""
    port_meta = PORT_ANCHORS.get(port_id, {"name": port_id, "baseline_queue": 15})
    start = time.time()

    mode = prefer_mode
    if mode == "auto":
        mode = "ollama" if _ollama_has_vl() else "heuristic"

    if mode == "ollama":
        try:
            b64 = base64.b64encode(image_bytes).decode()
            result = _call_ollama_vl(b64, VL_PROMPT)
            latency = result.pop("_latency_s", 0.0)
            ar = PortRiskAssessment(
                port_id=port_id, port_name=port_meta["name"],
                mode="ollama",
                ship_queue_count=int(result.get("ship_queue_count", 0)),
                container_stack_density=str(result.get("container_stack_density", "medium")),
                smoke_or_fire=bool(result.get("smoke_or_fire", False)),
                flood_indicators=bool(result.get("flood_indicators", False)),
                unusual_activity=str(result.get("unusual_activity", ""))[:200],
                risk_score=float(result.get("risk_score", 0.3)),
                confidence=float(result.get("confidence", 0.5)),
                latency_s=latency,
            )
            return ar
        except Exception as e:  # noqa: BLE001
            logger.warning("Ollama VL failed: %s; falling back to heuristic", e)
            mode = "heuristic"

    # Heuristic path
    data = _heuristic_from_image(image_bytes)
    return PortRiskAssessment(
        port_id=port_id, port_name=port_meta["name"],
        mode=mode,
        ship_queue_count=int(data["ship_queue_count"]),
        container_stack_density=data["container_stack_density"],
        smoke_or_fire=data["smoke_or_fire"],
        flood_indicators=data["flood_indicators"],
        unusual_activity=data["unusual_activity"],
        risk_score=float(data["risk_score"]),
        confidence=float(data["confidence"]),
        latency_s=time.time() - start,
    )


def synthesize_sample_image(port_id: str) -> bytes:
    """Generate a small synthetic RGB PNG for the port (no real satellite
    imagery is required to exercise the pipeline)."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return b""
    rng_seed = sum(ord(c) for c in port_id) % 1_000
    rng = (__import__("numpy").random.default_rng(rng_seed))
    # Blue water + grey docks + small darker blobs (ships)
    h, w = 256, 256
    arr = rng.integers(40, 120, size=(h, w, 3), dtype="uint8")
    # Water (blue dominance) on left half
    arr[:, : w // 2, 2] = rng.integers(120, 200, size=(h, w // 2), dtype="uint8")
    # Land (brown) on right half
    arr[:, w // 2 :, 0] = rng.integers(80, 140, size=(h, w // 2), dtype="uint8")
    arr[:, w // 2 :, 1] = rng.integers(60, 120, size=(h, w // 2), dtype="uint8")
    # Drop some ship-like dark rectangles
    for _ in range(rng.integers(4, 12)):
        x, y = rng.integers(20, w // 2 - 20), rng.integers(20, h - 20)
        arr[y : y + 6, x : x + 14] = 20
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_all_ports(mode: str = "auto") -> dict:
    results = {}
    for pid in PORT_ANCHORS:
        img = synthesize_sample_image(pid)
        ar = assess_port_image(img, pid, prefer_mode=mode)
        results[pid] = ar.to_dict()
        logger.info("[%s] mode=%s risk=%.2f conf=%.2f",
                    pid, ar.mode, ar.risk_score, ar.confidence)
    out = {
        "port_anchors": PORT_ANCHORS,
        "assessments": results,
        "summary": {
            "highest_risk_port": max(results, key=lambda k: results[k]["risk_score"]),
            "any_smoke": any(r["smoke_or_fire"] for r in results.values()),
            "mean_confidence": round(
                sum(r["confidence"] for r in results.values()) / len(results), 3),
        },
    }
    (OUT_DIR / "assessments.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="auto", choices=["auto", "ollama", "heuristic"])
    parser.add_argument("--port", default=None)
    args = parser.parse_args()

    if args.port:
        img = synthesize_sample_image(args.port)
        ar = assess_port_image(img, args.port, prefer_mode=args.mode)
        print(json.dumps(ar.to_dict(), indent=2))
    else:
        out = run_all_ports(mode=args.mode)
        print(json.dumps(out["summary"], indent=2))
