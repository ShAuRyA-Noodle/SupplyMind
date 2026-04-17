#!/bin/bash
# Wait for Phase N to complete (all 4 offline agent checkpoints present), then launch orchestrator.

cd "$(dirname "$0")"

echo "Waiting for Phase N (Chokehold) to finish..."
while true; do
    if [ -f "rl/checkpoints/bc_v2.pt" ] && [ -f "rl/checkpoints/cql_v2.pt" ] && [ -f "rl/checkpoints/iql_v2.pt" ] && [ -f "rl/checkpoints/td3bc_v2.pt" ]; then
        # All four ckpts present; wait 30s to ensure training finalized
        sleep 30
        # Check no active Python training process (heuristic)
        if ! pgrep -f "train_phase_n.py" > /dev/null; then
            echo "Phase N complete. Launching orchestrator..."
            break
        fi
    fi
    sleep 30
done

# Commit Phase N checkpoints first
git add -f rl/checkpoints/bc_v2.pt rl/checkpoints/cql_v2.pt rl/checkpoints/iql_v2.pt rl/checkpoints/td3bc_v2.pt phase_n.log 2>/dev/null
git commit -m "Phase N Chokehold: 4 offline agents trained 300K steps on v2 buffer" 2>/dev/null
git push origin main 2>/dev/null

# Launch orchestrator (runs O -> Omega)
python vessel_orchestrator.py 2>&1 | tee vessel_orchestrator.log
