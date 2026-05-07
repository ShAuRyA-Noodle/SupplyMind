#!/usr/bin/env bash
# build_pr_branch.sh — assemble the meta-pytorch/openenv PR branch locally.
#
# Run from the Sleep-Token repo root. Assumes you've already:
#   1. Forked meta-pytorch/openenv to your account
#   2. Cloned that fork to ../openenv-fork (sibling dir)
#
# What this does:
#   - Creates branch add-supplymind-reference-env in the fork
#   - Copies only the files listed in PR.md (no extra cruft)
#   - Runs a smoke test to confirm OpenEnv compliance passes
#   - Leaves you with a clean, PR-ready branch

set -euo pipefail

FORK="${1:-../openenv-fork}"
BRANCH="add-supplymind-reference-env"

if [ ! -d "$FORK" ]; then
  echo "Fork dir not found at $FORK. Clone it first:"
  echo "  gh repo fork meta-pytorch/openenv --clone=true"
  echo "  mv openenv ../openenv-fork"
  exit 2
fi

cd "$FORK"
git checkout main
git pull origin main
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"

mkdir -p examples/supplymind/{benchmarks,policies,tasks,tests}

cp -v ../Sleep-Token/versions/v3_arcadia/results/R6_EUCLIDIAN.json         examples/supplymind/benchmarks/
cp -v ../Sleep-Token/versions/v3_arcadia/results/R6_GETHSEMANE.json        examples/supplymind/benchmarks/
cp -vr ../Sleep-Token/versions/v3_arcadia/checkpoints/onnx_bundle/*         examples/supplymind/policies/ 2>/dev/null || \
    echo "[warn] onnx_bundle missing; skipping policies (host on HF Hub instead)"
cp -v ../Sleep-Token/server/supply_environment.py                 examples/supplymind/
cp -vr ../Sleep-Token/server/tasks                                 examples/supplymind/
cp -v ../Sleep-Token/tests/test_openenv_compliance.py              examples/supplymind/tests/
cp -v ../Sleep-Token/openenv.yaml                                  examples/supplymind/
cp -v ../Sleep-Token/Dockerfile                                    examples/supplymind/
cp -v ../Sleep-Token/docker-compose.yml                            examples/supplymind/

# Write the examples/supplymind/README.md from the PR.md body excerpt
cp -v ../Sleep-Token/versions/v5_phoenix/upstream_prs/meta_openenv/README.supplymind.md \
      examples/supplymind/README.md 2>/dev/null || echo "[info] README.supplymind.md not yet materialised"

echo
echo "== smoke test =="
python -m pytest examples/supplymind/tests -q --tb=no || {
  echo "[fail] compliance tests did not pass in fork"
  echo "       fix before pushing the branch"
  exit 3
}

echo
echo "== git status =="
git add examples/supplymind
git status --short

echo
echo "Branch '$BRANCH' ready. Next:"
echo "  git commit -m 'Add SupplyMind reference environment (3 tasks, Pydantic v2, Docker)'"
echo "  git push origin '$BRANCH'"
echo "  gh pr create --repo meta-pytorch/openenv --head ShAuRyA-Noodle:$BRANCH \\"
echo "    --title 'Add SupplyMind: real-data supply-chain risk environment' \\"
echo "    --body-file ../Sleep-Token/versions/v5_phoenix/upstream_prs/meta_openenv/PR.md"
