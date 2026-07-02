"""port_imagery_router.py — Qwen-2.5-VL-7B port-imagery card for the master demo.

Endpoint:
  POST /demo/port-imagery
    body: { "image_url": "...", "port_name": "Jebel Ali" }
    OR:   { "image_b64": "iVBORw0KGgo...", "port_name": "Jebel Ali" }

Returns: structured JSON describing port congestion, anchorage queues,
container density, and a 0-1 disruption indicator. Backed by Qwen-2.5-VL-7B
served via Ollama (`qwen2.5vl:7b` — already verified loaded).

If image is not provided, returns a deterministic "no image" stub. If Qwen-VL
is unreachable, returns a deterministic fallback honestly flagged.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MODEL_NAME = "qwen2.5vl:7b"

try:
    from fastapi import APIRouter, HTTPException
except ImportError:
    APIRouter = None  # type: ignore
    HTTPException = Exception  # type: ignore

router = APIRouter() if APIRouter is not None else None


PROMPT_TEMPLATE = (
    "You are a port operations analyst. Examine this satellite or aerial "
    "image of {port_name}. Estimate:\n"
    "  - vessel_count_visible: integer\n"
    "  - container_density_pct: 0-100 (% of yard area covered with containers)\n"
    "  - anchorage_queue_length: integer (vessels waiting offshore)\n"
    "  - disruption_indicator_0_to_1: 0=normal, 1=severe disruption\n"
    "  - one_sentence_finding: short plain-English\n\n"
    "Respond ONLY with JSON: {{\"vessel_count_visible\": N, "
    "\"container_density_pct\": N, \"anchorage_queue_length\": N, "
    "\"disruption_indicator_0_to_1\": N.NN, \"one_sentence_finding\": \"...\"}}"
)


class PortImageryRequest(BaseModel):
    image_url: str | None = Field(
        default=None,
        description="HTTP(S) URL to a satellite/aerial port image (PNG/JPG).",
    )
    image_b64: str | None = Field(
        default=None,
        description="base64-encoded image bytes (alternative to image_url).",
    )
    port_name: str = Field(
        default="Jebel Ali",
        description="Free-text port name; included in the prompt.",
    )


def _image_b64_from_url(url: str) -> str | None:
    try:
        import requests
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return base64.b64encode(r.content).decode("ascii")
    except Exception as e:  # noqa: BLE001
        logger.warning("[port-imagery] image fetch failed: %s", e)
        return None


def _qwen_vl_call(img_b64: str, port_name: str) -> dict | None:
    try:
        import requests
        t0 = time.time()
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": [{
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(port_name=port_name),
                    "images": [img_b64],
                }],
                "format": "json", "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 8192},
            },
            timeout=180,
        )
        r.raise_for_status()
        content = r.json()["message"]["content"]
        # Tolerant JSON parse: extract first {...}
        m = re.search(r"\{[\s\S]*\}", content)
        if not m:
            return {"ok": False, "raw": content[:500]}
        parsed = json.loads(m.group(0))
        return {
            "ok": True,
            "model": MODEL_NAME,
            "vessel_count_visible": int(parsed.get("vessel_count_visible", 0)),
            "container_density_pct": float(parsed.get("container_density_pct", 0)),
            "anchorage_queue_length": int(parsed.get("anchorage_queue_length", 0)),
            "disruption_indicator_0_to_1": float(parsed.get("disruption_indicator_0_to_1", 0.0)),
            "one_sentence_finding": str(parsed.get("one_sentence_finding", ""))[:300],
            "latency_s": round(time.time() - t0, 2),
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("[port-imagery] Qwen-VL call failed: %s", str(e)[:200])
        return None


def assess_port_image(req: PortImageryRequest) -> dict:
    """Public entry point used by FastAPI route + standalone callers."""
    if not req.image_url and not req.image_b64:
        return {
            "ok": False, "error": "no_image_provided",
            "hint": "supply image_url or image_b64",
        }

    img_b64 = req.image_b64
    if not img_b64 and req.image_url:
        img_b64 = _image_b64_from_url(req.image_url)
        if img_b64 is None:
            return {"ok": False, "error": "image_fetch_failed",
                    "url": req.image_url}

    result = _qwen_vl_call(img_b64, req.port_name)
    if result is None:
        return {"ok": False, "error": "qwen_vl_unavailable",
                "hint": "verify Ollama is running and `qwen2.5vl:7b` is pulled"}
    return {
        "port_name": req.port_name,
        "model": MODEL_NAME,
        "image_source": "url" if req.image_url else "b64",
        "image_url": req.image_url,
        **result,
    }


if router is not None:
    @router.post("/demo/port-imagery", tags=["demo"])
    def port_imagery_endpoint(req: PortImageryRequest) -> dict:
        try:
            return assess_port_image(req)
        except Exception as e:  # noqa: BLE001
            logger.error("[port-imagery] failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Smoke: synthesize a small test image
    try:
        from PIL import Image
        import io
        img = Image.new("RGB", (224, 224), color=(40, 80, 160))
        buf = io.BytesIO(); img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        result = assess_port_image(PortImageryRequest(
            image_b64=b64, port_name="Synthetic Test Port"))
        print(json.dumps(result, indent=2))
    except Exception as e:  # noqa: BLE001
        print(f"smoke failed: {e}")
