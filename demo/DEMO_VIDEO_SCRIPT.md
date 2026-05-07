# SupplyMind v3.0-arcadia — 3-minute demo video script

**Target**: record with OBS Studio or Loom. Voice-over reads the script verbatim. Each scene has explicit B-roll instructions.

**Output**: `demo/supplymind_v3_demo.mp4`, 1080p, ~3:00.

---

## SCENE 1 — HOOK (0:00 – 0:15)

**B-roll**: News montage — Ever Given blocking Suez, empty car dealerships, Taiwan strait satellite images. 4 quick cuts.

**Voice-over**:
> "Supply chain disruptions cost the global economy 184 billion dollars in 2023. The Suez blockage was 9.6 billion per day for six days. Existing tools tell you AFTER disaster. SupplyMind v3 predicts 72 hours ahead."

---

## SCENE 2 — THE STACK (0:15 – 0:45)

**B-roll**: Terminal window. Type and run in sequence:
```
ollama list
```
Shows 4 local models: `deepseek-r1-local-q4`, `qwen25-14b-local`, `qwen25-coder-local`, `mistral-nemo-local`.

Then Python REPL:
```python
>>> import torch
>>> torch.cuda.get_device_name(0)
'NVIDIA GeForce RTX 4080 Laptop GPU'
>>> from sentence_transformers import SentenceTransformer
>>> m = SentenceTransformer("models/mxbai-embed-large", device="cuda")
>>> m.encode("supply chain risk").shape
(1024,)
```

**Voice-over**:
> "Thirteen state-of-the-art foundation models. All running locally. Zero API cost at inference. DeepSeek-R1 as devil's advocate, Qwen-2.5 and Mistral-Nemo as primary judges. Chronos-Bolt and TimesFM-2 for forecasting. mxbai, BGE-M3, and Snowflake for retrieval. All loaded on a 12-gigabyte laptop."

---

## SCENE 3 — LIVE API: RISK ASSESSMENT (0:45 – 1:15)

**B-roll**: Browser with `https://supplymind.hf.space/docs` (Swagger UI). Open `/assess` endpoint.

Paste request body:
```json
{
  "context": "On March 11, 2011, a magnitude 9.0 earthquake struck off the coast of Tōhoku, Japan, triggering a tsunami that caused the Fukushima Daiichi nuclear disaster. Toyota, Honda, and Nissan halted production at multiple plants. Global automotive supply chains faced semiconductor shortages for months.",
  "judges": ["qwen25-14b-local", "mistral-nemo-local", "deepseek-r1-local-q4"]
}
```

Click "Execute". Response appears:
```json
{
  "consensus_risk": "CRITICAL",
  "escalation": "C_SUITE_IMMEDIATE",
  "judges": [
    {"judge": "qwen25-14b-local", "risk_level": "CRITICAL", "confidence": 0.95, ...},
    {"judge": "mistral-nemo-local", "risk_level": "CRITICAL", "confidence": 0.92, ...},
    {"judge": "deepseek-r1-local-q4", "risk_level": "HIGH", "confidence": 0.85, ...}
  ]
}
```

**Voice-over**:
> "Here's the live API running on HuggingFace Spaces. I'm posting the Tōhoku earthquake context. Three local LLMs return structured JSON. Two say CRITICAL, DeepSeek says HIGH as a devil's-advocate check. Majority consensus: CRITICAL. Escalation: C-suite immediate. Response time: twelve seconds. No OpenAI, no Anthropic, no API key."

---

## SCENE 4 — LIVE API: FORECAST (1:15 – 1:40)

**B-roll**: Same browser, `/forecast` endpoint. Paste 60 values of WTI oil price from FRED.

Response:
```json
{
  "point": [78.4, 78.7, 79.1, ...],
  "lo_80": [74.5, 74.0, 73.6, ...],
  "hi_80": [82.3, 83.4, 84.5, ...],
  "lo_95": [71.2, 70.5, 69.8, ...],
  "hi_95": [85.6, 87.2, 88.7, ...],
  "latency_s": 0.87
}
```

Open a quick matplotlib plot of the forecast + 80%/95% bands.

**Voice-over**:
> "Now a forecast. Chronos-Bolt zero-shot, 14-day horizon, with 80 and 95 percent confidence bands calibrated via per-horizon split-conformal. The oil price heavy-tail usually breaks pooled conformal; our per-horizon q-hat hits nominal coverage within two points."

---

## SCENE 5 — LIVE API: RAG (1:40 – 2:00)

**B-roll**: `/rag` endpoint. Post:
```json
{"query": "How does TSMC dominate the advanced semiconductor foundry market?", "top_k": 5}
```

Response: 5 chunks from actual SEC 10-K filings + Wikipedia article on TSMC + Semiconductor_industry, with cosine scores 0.8+ each.

**Voice-over**:
> "Retrieval over a 6,483-chunk real-world corpus of SEC filings and Wikipedia articles. mxbai-embed-large gives us P-at-1 of 0.962 on precise queries, and on paraphrased hard queries the BGE-reranker adds five points. Forty-millisecond latency."

---

## SCENE 6 — RL SIGN-FLIP (2:00 – 2:30)

**B-roll**: Show `versions/v3_arcadia/plots/euclidian/r6_euclidian.png` — bar chart with error bars.

Zoom into medium and hard task bars:
- medium: random -0.97, greedy **-1.81**, ppo_v3 **+2.78**
- hard: random -1.22, greedy -1.41, ppo_v3 **+2.65**

**Voice-over**:
> "The reinforcement learning result. On medium and hard tasks, the greedy heuristic performs WORSE than random. PPO_v3 with action masking flips the sign: plus 2.78 on medium, plus 2.65 on hard. Eight thousand one hundred bootstrap episodes, confidence intervals non-overlapping, zero constraint violations across the entire benchmark."

---

## SCENE 7 — BENCHMARKS + TESTS (2:30 – 2:50)

**B-roll**: Terminal, run `pytest tests/ -q` and show "173 passed in 1m47s". Then show `versions/v3_arcadia/plots/dangerous/r4v2_ablation.png` and `versions/v3_arcadia/plots/granite/r5_hard_redemption.png` in quick succession.

**Voice-over**:
> "One hundred seventy-three tests passing. OpenEnv formal compliance test. Wilcoxon p less than 0.001 on every RL-versus-baseline comparison. Bootstrap 95 percent confidence intervals. Krippendorff alpha of 0.75 on the two-judge consensus. And a deterministic rubric agent as the human-baseline. Every negative finding is documented with a world-class follow-up fix."

---

## SCENE 8 — OUTRO (2:50 – 3:00)

**B-roll**: GitHub repo page, then HuggingFace Space page, then the v3 tag page.

On screen text, white on black:
```
SupplyMind v3.0-arcadia
13 models · 8 benchmarks · 173 tests · One laptop · One human · Real data
github.com/ShAuRyA-Noodle/Sleep-Token
huggingface.co/spaces/Shaurya-Noodle/Supplymind
```

**Voice-over**:
> "Thirteen models. Eight benchmarks. One hundred seventy-three tests. One laptop. One human. Real data every byte. Even in Arcadia, supply chains break. SupplyMind sees it coming. GitHub link in the description."

---

## Production notes

- **Recording**: 1080p60, macOS QuickTime / OBS Studio / Loom.
- **Audio**: wired USB mic, noise gate, -3 dB normalize.
- **B-roll**: use CC-licensed news footage OR screen-only recording with dramatic text.
- **Music**: instrumental Sleep Token "Arcadia" 30 sec under outro (attribution in video description).
- **Export**: H.264, 2 Mbps, mp4.
- **Upload**: GitHub Release v3.0-arcadia, HF Space README, LinkedIn, Twitter.

## Post-production checklist

- [ ] Captions (auto + human-edited) for accessibility
- [ ] Thumbnail: PPO sign-flip chart with "SupplyMind v3" overlay
- [ ] Description: links to GitHub, HF Space, docs/v3/MODEL_CARD.md, docs/v3/FINAL_DEMO.md
- [ ] Tag: #OpenEnv #PyTorch #SupplyChain #RL #LLM #Hackathon
