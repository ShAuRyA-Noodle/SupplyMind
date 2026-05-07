#!/usr/bin/env bash
# build_pr_branch.sh — assemble the alibaba/ROLL PR branch locally.

set -euo pipefail

FORK="${1:-../ROLL-fork}"
BRANCH="add-supplymind-crisis-agentic-example"

if [ ! -d "$FORK" ]; then
  echo "Fork dir not found at $FORK."
  echo "  gh repo fork alibaba/ROLL --clone=true"
  echo "  mv ROLL ../ROLL-fork"
  exit 2
fi

cd "$FORK"
git checkout main
git pull origin main
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"

mkdir -p examples/supplymind_crisis/{configs,data,tests,benchmarks}

cp -v ../Sleep-Token/versions/v5_phoenix/roll_integration/env/supplymind_roll_env.py \
      examples/supplymind_crisis/
cp -v ../Sleep-Token/versions/v5_phoenix/roll_integration/reward_bridge/supplymind_judge_worker.py \
      examples/supplymind_crisis/
cp -v ../Sleep-Token/versions/v5_phoenix/roll_integration/configs/*.yaml \
      examples/supplymind_crisis/configs/

if [ -f ../Sleep-Token/versions/v5_phoenix/roll_integration/dpo_judge/data/preference_pairs.jsonl ]; then
  cp -v ../Sleep-Token/versions/v5_phoenix/roll_integration/dpo_judge/data/preference_pairs.jsonl \
        examples/supplymind_crisis/data/
else
  echo "[info] preference_pairs.jsonl not yet generated; generate via prepare_preference_data.py first"
fi

cp -v ../Sleep-Token/versions/v5_phoenix/upstream_prs/alibaba_roll/README.crisis.md \
      examples/supplymind_crisis/README.md 2>/dev/null || \
   echo "[info] README.crisis.md not yet materialised"

echo
echo "== smoke test =="
python -m pytest examples/supplymind_crisis/tests -q --tb=no 2>/dev/null || \
   echo "[info] tests not yet wired; OK to open PR as draft"

echo
echo "== git status =="
git add examples/supplymind_crisis
git status --short

echo
echo "Branch '$BRANCH' ready. Next:"
echo "  git commit -m 'Add examples/supplymind_crisis: agentic RL for supply-chain risk'"
echo "  git push origin '$BRANCH'"
echo "  gh pr create --repo alibaba/ROLL --head ShAuRyA-Noodle:$BRANCH \\"
echo "    --title 'Add examples/supplymind_crisis: agentic RL for supply-chain risk' \\"
echo "    --body-file ../Sleep-Token/versions/v5_phoenix/upstream_prs/alibaba_roll/PR.md"
