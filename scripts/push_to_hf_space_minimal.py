# -*- coding: utf-8 -*-
"""Push MINIMAL submission to HF Space (under 1GB limit).

Includes only files essential for hackathon submission. Skips all model checkpoints,
training artifacts, and non-essential dirs.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_TOKEN = os.environ.get('HF_TOKEN') or ''  # set HF_TOKEN env var
REPO_ID = 'Shaurya-Noodle/Supplymind'

if not HF_TOKEN:
    print('ERROR: no HF_TOKEN'); sys.exit(1)

from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)

# Allowed list — explicit allowlist of what gets uploaded (under 1GB total)
ALLOW_PATTERNS = [
    # Env runtime (essential)
    'server/**',
    'models.py',
    'openenv.yaml',
    'requirements.txt',
    'Dockerfile',
    '.dockerignore',
    'README.md',
    'LICENSE',
    'pyproject.toml',
    # Master notebook + key submission notebooks
    'notebooks/08_HACKATHON_FOOLPROOF.ipynb',
    'notebooks/09_LLAMA_GRPO_FOOLPROOF.ipynb',
    'notebooks/10_PRO_COLAB_KILLSHOT.ipynb',
    'notebooks/11_REAL_DATA_INGEST.ipynb',
    'notebooks/12_FRED_BRENT_REFIT.ipynb',
    'notebooks/13_MASTER_HACKATHON_FINAL.ipynb',
    # FINAL_SUBMIT — all docs + receipts + small plots
    'FINAL_SUBMIT/**.md',
    'FINAL_SUBMIT/**.html',
    'FINAL_SUBMIT/receipts/**.json',
    'FINAL_SUBMIT/plots/**.png',
    'FINAL_SUBMIT/CITATIONS.bib',
    'FINAL_SUBMIT/REPRODUCE_ONE_BASH.sh',
    # Wordle env (small)
    'versions/v5_phoenix/wordle_env/**',
    'versions/v5_phoenix/__init__.py',
    # Crisis library code (skip large embeddings)
    'versions/v4_arcadia_live/realtime/*.py',
    'versions/v4_arcadia_live/__init__.py',
    'versions/v4_arcadia_live/scenarios/*.json',
    # Scripts (training scripts judges can rerun)
    'scripts/pass23_colab_local_smoke.py',
    'scripts/pass27_killshot.py',
    'scripts/pass27_reasoning_gym_alt_env.py',
    'scripts/pass27_scenario_extractor.py',
    'scripts/pass28_killshot_v2.py',
    'scripts/pass28_keys_ingest.py',
    'scripts/push_to_hf_space.py',
    'scripts/push_to_hf_space_minimal.py',
    'scripts/patch_nb13*.py',
    # Small data
    'data/**.json',
    'data/**.yaml',
    # Tests
    'tests/**.py',
    'tests/**.json',
]

# Hard ignore — never upload these even if matched by allow
IGNORE_PATTERNS = [
    '.git/**', '.github/**', '.venv/**', 'venv/**',
    '__pycache__/**', '*.pyc', '.pytest_cache/**',
    'catboost_info/**', '_dump/**', '.tmp_pytest/**', '.source_cache/**',
    # Large model checkpoints
    '**/*.pkl', '**/*.npz', '**/*.zip', '**/*.bin',
    '**/*.gguf', '**/*.safetensors', '**/*.pth', '**/*.pt',
    '**/*.h5', '**/*.tar.gz', '**/*.parquet', '**/*.xlsx',
    'wgidataset*.xlsx',
    # Frontend node_modules
    'frontend/**', 'dashboard/**',
    # Old training data
    'rl/data/**', 'rl/checkpoints/**',
    'versions/v3_arcadia/checkpoints/**', 'versions/v3_arcadia/logs/**',
    'rl/analysis/trained/**',
    'external_data/**',
    'models/*.pth', 'models/*.pt', 'models/*.bin',
    'plots/v3/**',
    # Large docs
    'wgidataset*',
]

# Verify auth
for attempt in range(3):
    try:
        me = api.whoami()
        print(f'Logged in as: {me.get("name")}')
        break
    except Exception as e:
        if '429' in str(e) and attempt < 2:
            time.sleep(30)
        else:
            print(f'AUTH error: {e}')
            sys.exit(1)

print(f'\nMINIMAL push to https://huggingface.co/spaces/{REPO_ID}')
print(f'(Allowlist: env code + 6 notebooks + FINAL_SUBMIT/ + scripts + Wordle env)')
print(f'(Skipping: all model checkpoints, .pkl/.npz/.pth/.zip/.gguf etc)\n')

t0 = time.time()
try:
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=REPO_ID,
        repo_type='space',
        commit_message='pass 28 final submission · OpenEnv India 2026 · nb13 + FINAL_SUBMIT/ + scripts (minimal under 1GB)',
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )
    elapsed = time.time() - t0
    print(f'\n[OK] Upload complete in {elapsed:.0f}s')
    print(f'\n=== URLS FOR SUBMISSION FORM ===')
    print(f'Field 1 (HF Space):   https://huggingface.co/spaces/{REPO_ID}')
    print(f'Field 2 (Notebook):   https://huggingface.co/spaces/{REPO_ID}/blob/main/notebooks/13_MASTER_HACKATHON_FINAL.ipynb')
    print(f'Field 3 (YouTube):    https://www.youtube.com/watch?v=0Jy78rg_0BQ')
    print(f'Field 4 (URL):        same as Field 3')
    print(f'\nVerify in Incognito browser before submitting!')
except Exception as e:
    print(f'\n[FAIL] {type(e).__name__}: {str(e)[:500]}')
    raise
