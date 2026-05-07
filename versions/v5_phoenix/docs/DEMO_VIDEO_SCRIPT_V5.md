# Demo Video Script — SupplyMind v5 (3 minutes)

Target length: **3:00 exactly**. Target audience: Meta/HF judges + hackathon
voters. Record on Mac (Keynote + ScreenFlow) with terminal font ≥ 18 pt.

---

## 0:00–0:15  COLD OPEN

**Visual**: Terminal on black background. Giant title card over video:
> *SupplyMind · OpenEnv-native supply-chain risk · v5 phoenix*

**Voiceover**:
> "Three months ago I started with an idea: build a production-grade
> OpenEnv environment for supply-chain risk. Real data, real agents,
> real live geopolitics. Here's what it does."

---

## 0:15–0:45  LIVE HORMUZ DEMO

**Visual**: Split screen — terminal on left, JSON response on right.

**Commands**:
```bash
uvicorn versions.v5_phoenix.server.phoenix_app:app --port 8000 &
sleep 3

curl -X POST http://localhost:8000/live/hormuz-closure -d '{
  "scenario_text": "Iran threatens Hormuz closure. Brent $123/bbl.",
  "region": "hormuz"
}' | jq
```

**Voiceover**:
> "This is hitting real 2026 NewsAPI, FRED Brent prices, and our
> 3-judge LLM panel. Risk level: CRITICAL. Top analog: the April 2026
> Iran-US cargo ship seizure. Counterfactual: no-action loss $324 M,
> with-plan loss $65 M — 80 percent savings. Live, on my laptop."

---

## 0:45–1:15  OPENENV ARENA

**Visual**: Terminal upload, then Gradio UI, then leaderboard.

**Commands**:
```bash
curl -X POST http://localhost:8000/arena/run \
  -F "policy=@my_policy.pt" -F "name=demo_agent"
```

**Voiceover**:
> "The hackathon is about OpenEnv. So we built an Arena — judges drop in
> their PyTorch policy and we benchmark it on three tasks with
> bootstrap CI95 reward. This agent ranks between PPO and MaskablePPO.
> The full leaderboard is pre-seeded with our R6 Euclidian 10,800-episode
> baselines."

---

## 1:15–1:45  AUTORESEARCH + DPO-FINE-TUNED JUDGE

**Visual**: Open `lab_notebook.md` in VS Code, then show state.json.

**Commands**:
```bash
cat versions/v5_phoenix/autoresearch_fixed/lab_notebook.md | head -40
python -m versions.v5_phoenix.autoresearch_fixed.rebuild_state
```

**Voiceover**:
> "Karpathy autoresearch: one mutable file, one metric, bootstrap CI95
> accept-reject. Baseline accepted. Higher-entropy experiment accepted
> with +0.051 lift. Three more variants pending — v4 had bugs; v5
> ships the fixes."

**Visual switch**: show `train_dpo_trl.py` + adapter output.

**Voiceover (continued)**:
> "And we DPO-fine-tuned a 3B Qwen judge on our 26 crisis scenarios.
> The adapter is 20 MB and ships to HF Hub."

---

## 1:45–2:15  RECEIPTS + TESTS

**Visual**: Terminal showing reproduce.sh execution + pytest green output.

**Commands**:
```bash
bash versions/v5_phoenix/receipts_v2/R5_GRANITE_mxbai_P1.reproduce.sh
# -> 0.9622

bash versions/v5_phoenix/receipts_v2/R6_MaskingAblation_easy_lift.reproduce.sh
# -> 26.77

pytest tests/ versions/v4_arcadia_live/tests/ versions/v5_phoenix/tests/ -q
# -> 256+ passed
```

**Voiceover**:
> "20 grade-A receipts. Every headline number reproduces in 30 seconds.
> Every test green. We don't ship claims we can't verify."

---

## 2:15–2:45  UPSTREAM PRs

**Visual**: Two browser tabs: github.com/meta-pytorch/openenv and
github.com/alibaba/ROLL, both showing our PR drafts.

**Voiceover**:
> "The hackathon prize is the interview pipeline. The hackathon page
> says 'code ships to Meta-backed projects.' We ship three ways:
> SupplyMind as a reference env on Meta OpenEnv, as an agentic-RL
> training target on Alibaba ROLL, and as a Claude Code skill pack on
> obra's superpowers marketplace."

---

## 2:45–3:00  CLOSING

**Visual**: `README.md` open at the top, then title card.

**Voiceover**:
> "Solo submission. Three months. 256 tests. 20 receipts. No synthetic
> substitution. Happy to answer any question — upload your policy,
> pick any claim, point at any line. SupplyMind v5 phoenix ascensionism.
> Thank you."

**Title card**:
> github.com/ShAuRyA-Noodle/Sleep-Token
> JUDGES_V5.md · 4-minute path

---

## Recording checklist

- [ ] Terminal font size ≥ 18 pt
- [ ] Mac menubar hidden (⌘⇧F)
- [ ] Window at 1920×1080 minimum
- [ ] Ollama warm (qwen2.5:14b, mistral-nemo, deepseek-r1-local-q4)
- [ ] `FORCE_REPLAY=1` set as backup if NewsAPI times out mid-record
- [ ] Demo policy.pt exists at `/tmp/my_policy.pt`
- [ ] `pytest` dry-run passes before recording
- [ ] Mic levels tested (no clipping on excited sentences)
- [ ] 3-minute stopwatch on screen 2 during recording
- [ ] Two takes minimum; use the second unless it's worse

## Post-production

- Cut any pause > 1.5 s
- Add low-volume music bed (30 dB below VO)
- Terminal colors: solarized dark or catppuccin
- Upload to YouTube unlisted, Vimeo, and a direct MP4 at `demo/DEMO_BACKUP_2026_04_24.mp4`
