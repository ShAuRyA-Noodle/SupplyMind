#!/usr/bin/env bash
# push_all_upstream.sh — one-command path from `gh auth login` to 4 live PRs.
#
# Prerequisites:
#   1. gh CLI at /c/Users/Dell/bin/gh/bin/gh.exe (already installed per Phoenix push report)
#   2. gh authenticated: `gh auth login` (one-time browser flow)
#   3. Local fork workdirs assembled at ~/Desktop/upstream-workdirs/ (done by Phoenix session)
#
# This script ORCHESTRATES the 4 pushes in the right order. It does NOT fork
# or authenticate — those are one-time user actions.
#
# Order matters: supplymind-skills gets pushed FIRST because the marketplace
# PR's description references its public URL.

set -e

# Put gh on PATH for this script
export PATH="/c/Users/Dell/bin/gh/bin:$PATH"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

die() { echo -e "${RED}error: $*${NC}" >&2; exit 1; }
ok() { echo -e "${GREEN}✓ $*${NC}"; }
info() { echo -e "${YELLOW}>> $*${NC}"; }

# -------------------------------------------------------------------------
# Pre-flight
# -------------------------------------------------------------------------
command -v gh >/dev/null || die "gh CLI not found on PATH. Install or check \$PATH."

gh auth status >/dev/null 2>&1 \
  || die "gh is installed but not authenticated. Run: gh auth login (pick GitHub.com + HTTPS + login via web)"

ok "gh authenticated as $(gh api user --jq .login)"

WORKDIRS="$HOME/Desktop/upstream-workdirs"
[ -d "$WORKDIRS" ] || die "upstream-workdirs not found at $WORKDIRS. Re-run the Phoenix assembly scripts first."

for d in supplymind-skills openenv-fork ROLL-fork marketplace-fork; do
    [ -d "$WORKDIRS/$d" ] || die "$WORKDIRS/$d missing — Phoenix assembly incomplete."
done

ok "all 4 workdirs present"

# -------------------------------------------------------------------------
# 1. supplymind-skills — create + push + tag + release
# -------------------------------------------------------------------------
info "1/4  Create + push ShAuRyA-Noodle/supplymind-skills (NEW repo)"

# Check if repo already exists; if not, create it
if gh repo view ShAuRyA-Noodle/supplymind-skills >/dev/null 2>&1; then
    ok "repo already exists — skipping creation"
else
    gh repo create ShAuRyA-Noodle/supplymind-skills \
      --public \
      --description "3 ML-hackathon-tested Claude Code skills: benchmark-runner, autoresearch-experiment, live-demo-orchestrator" \
      --homepage "https://github.com/ShAuRyA-Noodle/Sleep-Token"
    ok "created repo ShAuRyA-Noodle/supplymind-skills"
fi

cd "$WORKDIRS/supplymind-skills"
git remote | grep -q origin \
  || git remote add origin https://github.com/ShAuRyA-Noodle/supplymind-skills.git
git push -u origin main
ok "pushed main"

# Tag + release
if ! git tag | grep -q v1.0.0; then
    git tag v1.0.0 -m "v1.0.0 initial — 3 skills battle-tested at Meta PyTorch OpenEnv Hackathon 2026"
    git push origin v1.0.0
fi
gh release create v1.0.0 \
  --notes "v1.0.0 initial release: 3 skills (benchmark-runner, autoresearch-experiment, live-demo-orchestrator) battle-tested during Meta PyTorch OpenEnv Hackathon 2026." \
  2>/dev/null || ok "release already exists"

# -------------------------------------------------------------------------
# 2. obra/superpowers-marketplace PR
# -------------------------------------------------------------------------
info "2/4  Open PR on obra/superpowers-marketplace"

cd "$WORKDIRS/marketplace-fork"
# Ensure fork exists
gh repo view ShAuRyA-Noodle/superpowers-marketplace >/dev/null 2>&1 \
  || gh repo fork obra/superpowers-marketplace --remote=false --clone=false

git remote | grep -q origin \
  || git remote add origin https://github.com/ShAuRyA-Noodle/superpowers-marketplace.git
git push -u origin add-supplymind-skills

gh pr create --repo obra/superpowers-marketplace \
  --head "ShAuRyA-Noodle:add-supplymind-skills" \
  --title "Add supplymind-skills@1.0.0 — 3 ML-hackathon-tested skills" \
  --body "Adds https://github.com/ShAuRyA-Noodle/supplymind-skills as a curated entry. Three skills (benchmark-runner, autoresearch-experiment, live-demo-orchestrator), derived from obra/superpowers methodology with full attribution. Battle-tested during Meta PyTorch OpenEnv Hackathon 2026." \
  2>&1 | tail -3

# -------------------------------------------------------------------------
# 3. meta-pytorch/openenv PR
# -------------------------------------------------------------------------
info "3/4  Open PR on meta-pytorch/openenv"

cd "$WORKDIRS/openenv-fork"
gh repo view ShAuRyA-Noodle/openenv >/dev/null 2>&1 \
  || gh repo fork meta-pytorch/openenv --remote=false --clone=false

git remote | grep -q origin \
  || git remote add origin https://github.com/ShAuRyA-Noodle/openenv.git
git push -u origin add-supplymind-env

gh pr create --repo meta-pytorch/openenv \
  --head "ShAuRyA-Noodle:add-supplymind-env" \
  --title "Add envs/supplymind_env — supply-chain risk RL environment" \
  --body-file "$HOME/Desktop/Sleep-Token/versions/v5_phoenix/upstream_prs/meta_openenv/PR.md" \
  2>&1 | tail -3

# -------------------------------------------------------------------------
# 4. alibaba/ROLL PR
# -------------------------------------------------------------------------
info "4/4  Open PR on alibaba/ROLL"

cd "$WORKDIRS/ROLL-fork"
gh repo view ShAuRyA-Noodle/ROLL >/dev/null 2>&1 \
  || gh repo fork alibaba/ROLL --remote=false --clone=false

git remote | grep -q origin \
  || git remote add origin https://github.com/ShAuRyA-Noodle/ROLL.git
git push -u origin add-supplymind-crisis-env

gh pr create --repo alibaba/ROLL \
  --head "ShAuRyA-Noodle:add-supplymind-crisis-env" \
  --title "Add examples/supplymind_crisis — agentic RL for supply-chain risk" \
  --body-file "$HOME/Desktop/Sleep-Token/versions/v5_phoenix/upstream_prs/alibaba_roll/PR.md" \
  2>&1 | tail -3

echo ""
ok "ALL 4 PUSHES COMPLETE"
echo ""
echo "Review the PRs:"
echo "  https://github.com/ShAuRyA-Noodle/supplymind-skills"
echo "  https://github.com/obra/superpowers-marketplace/pulls"
echo "  https://github.com/meta-pytorch/openenv/pulls"
echo "  https://github.com/alibaba/ROLL/pulls"
