"""Re-run Phase Q (BCE fix) and Phase S (MC Dropout) after orchestrator finishes."""

import subprocess
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd, name):
    print(f"\n=== RETRY {name} ===")
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, cwd=ROOT)
    print(f"=== RETRY {name} done ({(time.time()-t0)/60:.1f} min, exit {r.returncode}) ===")
    return r.returncode == 0


def git(cmd, check=False):
    return subprocess.run(f"git -C {ROOT} {cmd}", shell=True)


def main():
    # Re-run Q
    ok_q = run("python train_phase_q.py 2>&1 | tee phase_q_retry.log", "Q Alkaline retry")
    git("add -A")
    git('commit -m "Phase Q retry: BCE -> BCEWithLogits fix, world model + RSSM on v2 buffer"')
    git("push origin main")

    # Re-run S
    ok_s = run("python train_phase_s.py 2>&1 | tee phase_s_retry.log", "S Aqua Regia retry")
    git("add -A")
    git('commit -m "Phase S retry: MC Dropout full 27K test + reliability plot"')
    git("push origin main")

    print(f"Q ok={ok_q}, S ok={ok_s}")


if __name__ == "__main__":
    main()
