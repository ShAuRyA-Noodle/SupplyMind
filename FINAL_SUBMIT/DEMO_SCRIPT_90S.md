# 90-second demo recording script

For the Meta OpenEnv × Scaler hackathon Bangalore finals.

## Pre-flight (60 seconds before recording)

1. Quit Slack / Discord / Outlook.
2. `taskkill /F /IM nvcontainer.exe` (Windows) — disable NVIDIA overlay.
3. Browser at 110% zoom, dark mode, hide bookmarks bar (`Ctrl+Shift+B`).
4. Server running: `make demo` or `python -m uvicorn server.app:app --host 0.0.0.0 --port 8000`.
5. Pre-warm 22s: `curl -X POST http://127.0.0.1:8000/demo/hormuz-war-room ...` once so models mmap and FAISS loads.
6. Open `http://127.0.0.1:8000/demo/master` in fullscreen.

## OBS Studio settings

- 1080p60 H.264, CRF 18
- Source: Display Capture or Window Capture (browser only)
- Audio: VB-CABLE or Loopback (mic + system)
- Recording format: MP4 fragmented

## Shot list (target 90s)

| t | Scene | Action | Voiceover prompt |
|---|---|---|---|
| 0:00 | Title | Title card stays for 3 s on `/demo/master` | "Hormuz War Room. Real-time supply-chain risk for India and the Gulf." |
| 0:03 | Master grid | Pan top-bar showing 9 cards live with green LEDs | "Nine subsystems, every one live, every claim sha256-replayable." |
| 0:08 | Headline numbers | Cursor over the 6 stat tiles | "100% risk-band, 100% Brent, 90.01% conformal coverage, 0.567 panel agreement." |
| 0:14 | Click card 1 | `/demo/hormuz-war-room/ui` opens | "Click Hormuz War Room." |
| 0:18 | War-room hero | Pre-typed question visible; tick "6-judge panel" | "If Iran-Israel-US escalation restricts Hormuz, what breaks first?" |
| 0:22 | Click ▶ Run | Stage pills tick green: signals → analogs → judges → sectors → cf → receipt | "32-second pipeline. Every stage on screen." |
| 0:32 | Headline strip | `risk: HIGH`, confidence bar fills, Brent p50, CF savings | "Risk HIGH. Composite confidence 44%. Brent p50 $82. Counterfactual savings positive." |
| 0:40 | Map zoom | Click Saudi East-West edge → bypass popup with IEA citation | "Only 3.5 to 5.5 mb/d can bypass Hormuz. IEA cited." |
| 0:48 | India table | Mouse rank 1 → 7: commercial LPG, urea, ATF, refining, petchem, diesel, household-LPG | "Commercial LPG cuts first. Domestic LPG protected last by MoPNG allocation rule." |
| 0:58 | Gulf table | Hover Qatar LNG → "no real bypass" tooltip | "Qatar LNG, ninety-five percent through Hormuz, no bypass." |
| 1:04 | OpenRouter panel | Show consensus CRITICAL, alpha 1.0, gpt-oss-120b at 0.92 conf | "Six frontier judges cross-check. Four returned. Consensus CRITICAL. Krippendorff α 1.0." |
| 1:14 | Validation | Click ▶ Run on the validation panel; aggregate fills with 100/100/100/100/100% | "Eight historical events. One hundred percent risk band. One hundred percent Brent within thirty percent." |
| 1:24 | Receipt drawer | Click ⊟ open receipt; drawer slides in showing sha256 | "Every claim is one click away from its sha256." |
| 1:32 | Close card | Pan back to master, headline numbers visible | "100 percent risk band. 100 percent Brent. 90 percent action coverage. RAP-XC beats MaskablePPO. CI95 excludes zero." |
| 1:38 | End | Hold on tagline + URL | "SupplyMind. Receipts not vibes." |

## Voiceover script (matched to shot timing)

```
[0:00] Hormuz War Room. Real-time supply-chain risk for India and the Gulf.
[0:03] Nine subsystems, every one live, every claim sha256-replayable.
[0:08] Hundred percent risk-band. Hundred percent Brent. Ninety-point-oh-one percent conformal coverage. Krippendorff alpha point-five-six-seven on twelve frontier judges.
[0:14] Click Hormuz War Room.
[0:18] If Iran-Israel-US escalation restricts the Strait of Hormuz, what breaks first?
[0:22] Thirty-two-second pipeline. Every stage on screen.
[0:32] Risk: HIGH. Brent p50 eighty-two dollars. Counterfactual savings positive.
[0:40] Only three-point-five to five-point-five million barrels a day can bypass Hormuz. IEA cited.
[0:48] Commercial LPG cuts first. Domestic LPG protected last by Ministry allocation rule.
[0:58] Qatar LNG, ninety-five percent through Hormuz, no bypass.
[1:04] Six frontier judges cross-check. Four returned. Consensus CRITICAL. Alpha one-point-zero.
[1:14] Eight historical events. Hundred percent risk band. Hundred percent Brent within thirty percent.
[1:24] Every claim is one click away from its sha256.
[1:32] RAP-XC beats MaskablePPO-v3. CI ninety-five excludes zero.
[1:38] SupplyMind. Receipts not vibes.
```

## What to NOT do on camera

- Don't open the IDE / source code mid-demo. Keep the IDE for a separate "how it works" cut.
- Don't show terminal logs (will distract from the polished UI).
- Don't try to record one continuous 90s take. Multiple cuts are fine, B-roll is fine.
- Don't run with `enable_openrouter_panel=true` if you can't tolerate 60s — the rate limits look slow on camera. Use `false` and screenshot the OpenRouter panel separately.
- Don't oversell. The honesty admissions in `HONEST_LIMITATIONS.md` are credibility, not weakness.

## Post-production

- DaVinci Resolve free, Shotcut, or CapCut.
- Color: keep the dark theme as captured. Maybe add a subtle vignette.
- Add 3 lower-third callouts when a metric flashes:
  - "100% risk-band — 8/8 documented events"
  - "0.9001 conformal coverage — Vovk 2005 split-conformal"
  - "RAP-XC vs MaskablePPO-v3 · CI95 [+0.198, +0.257]"
- End-card link: `github.com/<your-handle>/SupplyMind` and `/demo/master` URL.

## Where the video goes

- Upload as YouTube unlisted.
- Embed link in `FINAL_SUBMIT/README.md` ("90-second demo: ...").
- Also drop the .mp4 in `FINAL_SUBMIT/demo_90s.mp4` for offline judges.
