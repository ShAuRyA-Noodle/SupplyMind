"""Verify Qwen2.5-VL-7B via HF transformers + qwen-vl-utils on a synthetic image.
Skips actual inference if disk is tight; just validates model loading.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS = ROOT / "models" / "qwen25-vl-7b"
OUT = ROOT / "v3_arcadia" / "results" / "qwen_vl_verify.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
result: dict = {"device": DEVICE, "model_dir": str(MODELS)}

free_gb = shutil.disk_usage("c:/").free / 1e9
result["free_disk_gb"] = round(free_gb, 2)

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from PIL import Image
    import numpy as np

    # Monkey-patch torch.load (Qwen-VL may load aux files)
    _orig = torch.load
    def _patched(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)
    torch.load = _patched

    # Load with low_cpu_mem_usage to avoid doubling RAM during load
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODELS),
        torch_dtype=torch.float16,
        device_map=DEVICE if DEVICE == "cuda" else "auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(str(MODELS))

    # Create a tiny synthetic supply-graph image (white rectangle with small shapes)
    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe this image in 1 sentence."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    resp = processor.batch_decode(out_ids[:, inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True)[0]
    result["qwen_vl"] = {"status": "OK", "sample_response": resp[:200]}
    print(f"Qwen-VL OK: {resp[:120]}")
    torch.load = _orig
except Exception as e:
    import traceback; traceback.print_exc()
    result["qwen_vl"] = {"status": "FAIL", "error": str(e)[:300]}

OUT.write_text(json.dumps(result, indent=2))
print(f"\nSaved {OUT}")
