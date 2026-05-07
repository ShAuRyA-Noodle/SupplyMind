"""Phase 4c: rewrite Python imports + string paths + parents[N] indices after version dir rename.

Renames done in phase 4b:
  v3_arcadia/             -> versions/v3_arcadia/
  ShAuRyA_Supplymind/     -> versions/v4_arcadia_live/
  ShAuRyA_Phoenix/        -> versions/v5_phoenix/
  ROLL-main/ROLL-main/    -> vendor/ROLL/    (phase 4a)

Targets:
  1. Python imports (from X / import X)
  2. Dynamic-import string paths inside .py files
  3. File-path references in any text file (md, json, yml, sh, py strings)
  4. parents[N] indices in Phoenix/Supplymind files that climbed to repo root
"""
from __future__ import annotations
import re, sys, io, os
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path("c:/Users/Dell/Desktop/Sleep-Token")

SKIP_PREFIXES = (
    "_dump/", ".venv/", ".git/", "vendor/", "models/", "external_data/", "wandb/",
    ".pytest_cache/", ".tmp_pytest/", ".source_cache/", ".openrouter_cache/",
    ".agents/", ".claude/", "rl/data/", "rl/checkpoints/", "rl/analysis/",
    # skip auto-generated experiment data inside Phoenix
    "versions/v5_phoenix/experiments/",
)
TEXT_EXTS = {".py", ".md", ".json", ".yml", ".yaml", ".toml", ".sh", ".html", ".bat",
             ".ps1", ".txt", ".cfg", ".ini", ".env", ".example", ".bib", ".ipynb",
             ".jsonl", ".log", ".disabled", ".Modelfile"}
MAX_BYTES = 10_000_000

# Module-style replacements (Python import syntax). Order matters — longest first.
IMPORT_RENAMES = [
    ("ShAuRyA_Phoenix",     "versions.v5_phoenix"),
    ("ShAuRyA_Supplymind",  "versions.v4_arcadia_live"),
]
# Path-style replacements (slash form).
PATH_RENAMES = [
    ("ShAuRyA_Phoenix/",    "versions/v5_phoenix/"),
    ("ShAuRyA_Supplymind/", "versions/v4_arcadia_live/"),
    ("v3_arcadia/",         "versions/v3_arcadia/"),
    # Backslash variants for Windows-style paths in some scripts
    ("ShAuRyA_Phoenix\\",   "versions/v5_phoenix/"),
    ("ShAuRyA_Supplymind\\","versions/v4_arcadia_live/"),
    ("v3_arcadia\\",        "versions/v3_arcadia/"),
    # ROLL vendor move
    ("ROLL-main/ROLL-main/", "vendor/ROLL/"),
    ("ROLL-main/ROLL-main",  "vendor/ROLL"),
    ("ROLL-main\\ROLL-main\\","vendor/ROLL/"),
]

stats = {"py_imports": 0, "py_string_modules": 0, "fs_paths": 0, "parents_bumps": 0,
         "files_touched": 0, "files_scanned": 0, "binary_skipped": 0}

def should_skip(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    return any(rel.startswith(p) for p in SKIP_PREFIXES)

def adjust_parents_in_versions(rel: str, text: str) -> tuple[str, int]:
    """For files now under versions/{v4_arcadia_live,v5_phoenix}/<x>/<y>...,
       depth from repo root increased by 1. parents[N] that previously climbed
       to repo root must become parents[N+1]. We can't know N exactly without
       semantic analysis, but we can compute: old_depth = (depth_in_old_layout),
       and if a parents[N] equals (old_depth - 1), we bump by 1.
    """
    rel_posix = rel.replace("\\", "/")
    if not (rel_posix.startswith("versions/v5_phoenix/") or
            rel_posix.startswith("versions/v4_arcadia_live/")):
        return text, 0
    if not rel_posix.endswith(".py"):
        return text, 0

    # New depth from repo root: count slashes in rel
    new_depth = rel_posix.count("/") + 1  # versions/v5_phoenix/X/Y.py -> 4
    # Old depth was new_depth - 1 (we added "versions/" layer).
    old_depth = new_depth - 1
    # Index that climbed to repo root in old layout = old_depth - 1
    target_old_idx = old_depth - 1

    bumps = 0
    def repl(m: re.Match) -> str:
        nonlocal bumps
        n = int(m.group(1))
        # Bump only if it equals target_old_idx (climb-to-root pattern)
        if n == target_old_idx:
            bumps += 1
            return f"parents[{n+1}]"
        return m.group(0)

    new_text = re.sub(r"parents\[(\d+)\]", repl, text)
    return new_text, bumps

def rewrite_file(path: Path, rel: str) -> None:
    stats["files_scanned"] += 1
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size > MAX_BYTES:
        return
    if path.suffix.lower() not in TEXT_EXTS and path.suffix not in TEXT_EXTS:
        # also accept extensionless Modelfiles
        if not path.name.endswith(".Modelfile"):
            return

    try:
        raw = path.read_bytes()
    except OSError:
        return
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            stats["binary_skipped"] += 1
            return

    orig = text
    n_py_imp = 0
    n_py_str = 0
    n_fs = 0

    # 1. Python imports — `from X.` / `from X ` / `import X.` / `import X\n`
    if path.suffix == ".py" or path.suffix == ".ipynb":
        for old, new in IMPORT_RENAMES:
            # Patterns: must be word-boundary, must be at module-position
            before = text
            text = re.sub(r"(\bfrom\s+)" + re.escape(old) + r"(?=[\.\s])", r"\g<1>" + new, text)
            text = re.sub(r"(\bimport\s+)" + re.escape(old) + r"(?=[\.\s,])", r"\g<1>" + new, text)
            text = re.sub(r"(\bimport\s+)" + re.escape(old) + r"$", r"\g<1>" + new, text, flags=re.MULTILINE)
            n_py_imp += (text != before) * 1  # rough — count later if needed

        # 2. Dynamic-import string paths: "ShAuRyA_Phoenix.X.router", etc.
        for old, new in IMPORT_RENAMES:
            before = text
            # quoted dotted path
            text = re.sub(r'(["\'])' + re.escape(old) + r'(\.[A-Za-z_][\w\.]*)\1',
                          lambda m: m.group(1) + new + m.group(2) + m.group(1), text)
            n_py_str += (text != before) * 1

    # 3. Filesystem-style path references everywhere (md/yml/json/sh/py strings)
    for old, new in PATH_RENAMES:
        if old in text:
            count = text.count(old)
            text = text.replace(old, new)
            n_fs += count

    # 4. parents[N] adjustment for files now under versions/v{4,5}_*
    text, n_par = adjust_parents_in_versions(rel, text)

    if text != orig:
        path.write_bytes(text.encode("utf-8"))
        stats["files_touched"] += 1
        stats["py_imports"] += n_py_imp
        stats["py_string_modules"] += n_py_str
        stats["fs_paths"] += n_fs
        stats["parents_bumps"] += n_par
        if n_par or n_py_imp or n_py_str:
            print(f"  {rel}  imp={n_py_imp} strmod={n_py_str} fs={n_fs} par={n_par}")

def main():
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if should_skip(rel):
            continue
        rewrite_file(p, rel)

    print()
    print("STATS:", stats)

if __name__ == "__main__":
    main()
