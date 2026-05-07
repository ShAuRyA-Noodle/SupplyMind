"""Phase 4c pass 3: final cleanup — broaden file-extension scope + slash-form refs.

Targets:
  - .sh, .receipt, .yaml configs that may have hardcoded ShAuRyA_X/path
  - master.html alert strings double-prefixed by pass 1
  - bare ShAuRyA_X followed by `/` in any text file
"""
from __future__ import annotations
import re, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path("c:/Users/Dell/Desktop/Sleep-Token")

SKIP_PREFIXES = (
    "_dump/", ".venv/", ".git/", "vendor/", "models/", "external_data/", "wandb/",
    ".pytest_cache/", ".tmp_pytest/", ".source_cache/", ".openrouter_cache/",
    ".agents/", ".claude/",
    "versions/v5_phoenix/.venv-roll/",   # vendored .venv inside Phoenix
)
# Treat ALL files as candidates if size ok and decode utf-8 ok
MAX_BYTES = 5_000_000

PATH_RENAMES = [
    # Slash form first — broad coverage
    ("ShAuRyA_Phoenix/",     "versions/v5_phoenix/"),
    ("ShAuRyA_Supplymind/",  "versions/v4_arcadia_live/"),
    # Backslash variants
    ("ShAuRyA_Phoenix\\",    "versions/v5_phoenix/"),
    ("ShAuRyA_Supplymind\\", "versions/v4_arcadia_live/"),
    # Module form (dotted)
    ("ShAuRyA_Phoenix.",     "versions.v5_phoenix."),
    ("ShAuRyA_Supplymind.",  "versions.v4_arcadia_live."),
]

# Fix double-prefix bug from pass 1 (versions/versions/v3_arcadia → versions/v3_arcadia)
DOUBLE_PREFIX_FIX = [
    ("versions/versions/v3_arcadia",  "versions/v3_arcadia"),
    ("versions/versions/v5_phoenix",  "versions/v5_phoenix"),
    ("versions/versions/v4_arcadia_live", "versions/v4_arcadia_live"),
    ("versions.versions.v5_phoenix", "versions.v5_phoenix"),
    ("versions.versions.v4_arcadia_live", "versions.v4_arcadia_live"),
]

stats = {"files_touched": 0, "subs": 0, "double_fixes": 0}

for p in ROOT.rglob("*"):
    if not p.is_file(): continue
    rel = p.relative_to(ROOT).as_posix()
    if any(rel.startswith(s) for s in SKIP_PREFIXES): continue
    try:
        size = p.stat().st_size
    except OSError:
        continue
    if size > MAX_BYTES: continue
    try:
        text = p.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        continue
    orig = text
    n = 0
    for old, new in DOUBLE_PREFIX_FIX:
        if old in text:
            c = text.count(old)
            text = text.replace(old, new)
            stats["double_fixes"] += c
            n += c
    for old, new in PATH_RENAMES:
        if old in text:
            c = text.count(old)
            text = text.replace(old, new)
            n += c
            stats["subs"] += c
    if text != orig:
        p.write_bytes(text.encode("utf-8"))
        stats["files_touched"] += 1
        if n:
            print(f"  {rel}  +{n}")

print()
print("STATS:", stats)
