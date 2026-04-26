"""Push entire submission to HuggingFace Space via huggingface_hub upload_folder.

Handles LFS automatically server-side. No local git LFS needed.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_TOKEN = os.environ.get('HF_TOKEN') or ''  # SCRUBBED � set HF_TOKEN env var
REPO_ID = 'Shaurya-Noodle/Supplymind'

if not HF_TOKEN:
    print('ERROR: no HF_TOKEN'); sys.exit(1)

from huggingface_hub import HfApi

api = HfApi(token=HF_TOKEN)

# Verify auth (with retry on rate limit)
for attempt in range(3):
    try:
        me = api.whoami()
        print(f'Logged in as: {me.get("name")}')
        break
    except Exception as e:
        if '429' in str(e) and attempt < 2:
            print(f'Rate-limited, waiting 30s...'); time.sleep(30)
        else:
            print(f'AUTH error: {e}'); break

print(f'\nPushing entire submission to https://huggingface.co/spaces/{REPO_ID}')
print(f'(LFS handled automatically server-side)\n')

# Files/folders to skip (large historical data not needed for HF Space deploy)
IGNORE = [
    '.git/**', '.github/**', '.venv/**', 'venv/**',
    '__pycache__/**', '*.pyc', '.pytest_cache/**',
    'catboost_info/**', '_dump/**', '.tmp_pytest/**',
    '.source_cache/**',
    # Large historical training artifacts (not needed for env deploy)
    'rl/data/*.npz', 'rl/data/*.csv',
    'rl/checkpoints/**',
    'rl/analysis/trained/v3/**',  # 17-41MB pkl files
    'v3_arcadia/checkpoints/**',  # large model checkpoints
    'v3_arcadia/logs/**',
    'plots/v3/**.npy',
    'external_data/**',
    'models/*.pth', 'models/*.pt', 'models/*.bin',
    # Lockfiles + binary artifacts
    'uv.lock', '*.whl',
    # Large vision model files
    '*.gguf', '*.safetensors',
    # Notebooks 1-7 (legacy, not used for submission — only 8-13 needed)
    'notebooks/01_*.ipynb', 'notebooks/02_*.ipynb', 'notebooks/03_*.ipynb',
    'notebooks/04_*.ipynb', 'notebooks/05_*.ipynb', 'notebooks/06_*.ipynb',
    'notebooks/07_*.ipynb',
    # Skip wgidataset_with_sourcedata-2025.xlsx if it's >10MB
    'wgidataset*.xlsx',
    # Frontend node_modules
    'frontend/node_modules/**', 'dashboard/node_modules/**',
    'frontend/.next/**', 'dashboard/.next/**',
]

t0 = time.time()
try:
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=REPO_ID,
        repo_type='space',
        commit_message='pass 28 final submission · OpenEnv India 2026 · nb13 master + 128 receipts + 13 plots + blog + 7 patches',
        ignore_patterns=IGNORE,
    )
    elapsed = time.time() - t0
    print(f'\n[OK] Upload complete in {elapsed:.0f}s')
    print(f'\nHF Space URL: https://huggingface.co/spaces/{REPO_ID}')
    print(f'Repo URL:     https://huggingface.co/spaces/{REPO_ID}/tree/main')
    print(f'Notebook:     https://huggingface.co/spaces/{REPO_ID}/blob/main/notebooks/13_MASTER_HACKATHON_FINAL.ipynb')
    print(f'Blog:         https://huggingface.co/spaces/{REPO_ID}/blob/main/FINAL_SUBMIT/THE_SUPPLYMIND_STORY.md')
except Exception as e:
    print(f'\n[FAIL] {type(e).__name__}: {str(e)[:500]}')
    raise
