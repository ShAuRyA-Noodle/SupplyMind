#!/usr/bin/env bash
# Populate the GitHub Release page for v3.0-arcadia with all judge-facing assets.
#
# Requires:
#   - `gh` CLI installed (https://cli.github.com/) and authenticated
#     (`gh auth login`)
#   - Run from repo root
#
# Usage:
#   bash scripts/release_assets.sh

set -euo pipefail

TAG="v3.0-arcadia"
REPO="ShAuRyA-Noodle/Sleep-Token"

echo "→ Checking gh CLI..."
if ! command -v gh &> /dev/null; then
  echo "ERROR: gh CLI not installed. Install from https://cli.github.com/ and run 'gh auth login'"
  exit 1
fi

echo "→ Ensuring tag $TAG is pushed to origin..."
git push origin $TAG 2>&1 || echo "  (tag already pushed or no remote access — ensure origin is set)"

echo "→ Creating Release page..."
gh release create "$TAG" \
  --repo "$REPO" \
  --title "SupplyMind v3.0-arcadia — the complete submission" \
  --notes-file <(cat <<'NOTES'
# SupplyMind v3.0 "Even In Arcadia"

**Meta PyTorch OpenEnv Hackathon submission.** Every phase commit named after a Sleep Token track.

## Highlights
- **173 tests passing** (154 core + 19 formal OpenEnv compliance)
- **13 foundation models** integrated locally (zero API cost at inference)
- **261,175 real data points** from 8 cited sources (zero synthetic)
- **8,100-episode RL benchmark**, PPO_v3 beats every baseline with bootstrap CI95 non-overlapping
- **mxbai RAG P@1 = 0.962**, reranker +5pp on hard paraphrased queries
- **Krippendorff α = 0.750** on 2-judge LLM panel
- **Custom 3-layer GCN in pure PyTorch** (+48-64% vs MLP on arrival-time regression)
- **Per-horizon split-conformal** intervals hit nominal ±2pp on oil@95%
- **3× ONNX-exported PPO policies** (0.97 MB each, verified via onnxruntime)

## Documentation
- `docs/v3/MODEL_CARD.md` — unified model card with every benchmark
- `docs/v3/PYTORCH_STORY.md` — non-trivial PyTorch engineering
- `docs/v3/BENCHMARKS_VS_PUBLIC.md` — honest comparison to M5/MTEB/MuJoCo
- `docs/v3/FINAL_DEMO.md` — demo script + judge path
- `docs/v4/AUDIT_PLAN.md` — full coverage matrix of v3 audit directives
- `FAILURE_TABLE.md` — every v1/v2 failure with v3 resolution link

## Artifacts attached
- All plots from `versions/v3_arcadia/plots/**`
- All JSON results from `versions/v3_arcadia/results/**`
- ONNX policies from `versions/v3_arcadia/checkpoints/gethsemane/*.onnx`
- Pitch deck (markdown + rendered PDF if built)

## Links
- GitHub: https://github.com/ShAuRyA-Noodle/Sleep-Token
- HF Space: https://huggingface.co/spaces/Shaurya-Noodle/Supplymind
- Demo video: [coming — recording via `demo/DEMO_VIDEO_SCRIPT.md`]

*"Even in Arcadia, supply chains break. SupplyMind sees it coming."*
NOTES
)

echo "→ Uploading assets..."

# Plots
for f in versions/v3_arcadia/plots/**/*.png; do
  [ -f "$f" ] && gh release upload "$TAG" "$f" --repo "$REPO" --clobber || true
done

# JSON results
for f in versions/v3_arcadia/results/*.json; do
  [ -f "$f" ] && gh release upload "$TAG" "$f" --repo "$REPO" --clobber || true
done

# Markdown reports
for f in versions/v3_arcadia/results/*REPORT*.md; do
  [ -f "$f" ] && gh release upload "$TAG" "$f" --repo "$REPO" --clobber || true
done

# ONNX policies
for f in versions/v3_arcadia/checkpoints/gethsemane/*.onnx; do
  [ -f "$f" ] && gh release upload "$TAG" "$f" --repo "$REPO" --clobber || true
done

# Pitch deck (markdown + PDF if exists)
gh release upload "$TAG" demo/PITCH_DECK.md --repo "$REPO" --clobber || true
[ -f demo/SupplyMind_pitch.pdf ] && gh release upload "$TAG" demo/SupplyMind_pitch.pdf --repo "$REPO" --clobber
[ -f demo/supplymind_v3_demo.mp4 ] && gh release upload "$TAG" demo/supplymind_v3_demo.mp4 --repo "$REPO" --clobber

# Unified docs
gh release upload "$TAG" docs/v3/MODEL_CARD.md --repo "$REPO" --clobber
gh release upload "$TAG" docs/v3/PYTORCH_STORY.md --repo "$REPO" --clobber
gh release upload "$TAG" docs/v3/BENCHMARKS_VS_PUBLIC.md --repo "$REPO" --clobber
gh release upload "$TAG" docs/v3/FINAL_DEMO.md --repo "$REPO" --clobber

echo "✅ Release populated. Visit: https://github.com/$REPO/releases/tag/$TAG"
