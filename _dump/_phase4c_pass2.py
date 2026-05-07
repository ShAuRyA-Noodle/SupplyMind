"""Phase 4c pass 2: catch bare references missed by pass 1."""
from __future__ import annotations
import re, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path("c:/Users/Dell/Desktop/Sleep-Token")

SKIP_PREFIXES = (
    "_dump/", ".venv/", ".git/", "vendor/", "models/", "external_data/", "wandb/",
    ".pytest_cache/", ".tmp_pytest/", ".source_cache/", ".openrouter_cache/",
    ".agents/", ".claude/", "rl/data/", "rl/checkpoints/", "rl/analysis/",
    # NOT skipping experiments now — phase_a.sh + configs need rewrite
)
TEXT_EXTS = {".py", ".md", ".json", ".yml", ".yaml", ".toml", ".sh", ".html", ".bat",
             ".ps1", ".txt", ".cfg", ".ini", ".env", ".example", ".bib", ".ipynb",
             ".jsonl", ".disabled", ".Modelfile"}
MAX_BYTES = 10_000_000

# Bare module names: ShAuRyA_Phoenix → versions.v5_phoenix when followed by `.`, `:`, `"`, or word-end
# Bare path names: ShAuRyA_Phoenix → versions/v5_phoenix when in path context (after `/`, `cd `, etc.)
BARE_MODULE = [
    ("ShAuRyA_Phoenix",     "versions.v5_phoenix"),
    ("ShAuRyA_Supplymind",  "versions.v4_arcadia_live"),
]
BARE_PATH = [
    ("ShAuRyA_Phoenix",     "versions/v5_phoenix"),
    ("ShAuRyA_Supplymind",  "versions/v4_arcadia_live"),
]

stats = {"module_form": 0, "path_form": 0, "files_touched": 0, "parents_bumps": 0}

def adjust_parents(rel: str, text: str) -> tuple[str, int]:
    if not rel.startswith("versions/v5_phoenix/") and not rel.startswith("versions/v4_arcadia_live/"):
        return text, 0
    if not rel.endswith(".py"):
        return text, 0
    new_depth = rel.count("/") + 1
    target_old_idx = (new_depth - 1) - 1  # old root parent index
    bumps = 0
    def repl(m):
        nonlocal bumps
        n = int(m.group(1))
        if n == target_old_idx:
            bumps += 1
            return f"parents[{n+1}]"
        return m.group(0)
    return re.sub(r"parents\[(\d+)\]", repl, text), bumps

def process(p: Path, rel: str) -> None:
    if p.suffix not in TEXT_EXTS and not p.name.endswith(".Modelfile"):
        return
    try:
        size = p.stat().st_size
    except OSError:
        return
    if size > MAX_BYTES: return
    try:
        text = p.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return
    orig = text
    n_mod = 0
    n_path = 0

    # MODULE form: bare name followed by `.` (dotted module ref) or `:` (uvicorn app spec) or `,)` etc
    # Use regex: \bShAuRyA_Phoenix\b NOT followed by `/` (which is path form)
    for old, new in BARE_MODULE:
        # only match where NOT followed by '/' or '\' (path forms handled separately)
        pat = re.compile(r"\b" + re.escape(old) + r"(?=[\.:])")
        new_text, n = pat.subn(new, text)
        n_mod += n
        text = new_text

    # PATH form: bare name followed by space, `"`, `'`, end of line, or quote+space
    # OR preceded by `/` (path) or in command like `cd FOO`
    for old, new in BARE_PATH:
        # bare name followed by `"` or `'` or whitespace or `)` or end (path-like contexts)
        pat = re.compile(r"\b" + re.escape(old) + r"(?=[\"\'\s\)\]\,]|$)", re.MULTILINE)
        new_text, n = pat.subn(new, text)
        n_path += n
        text = new_text

    # parents[N] bumps
    text, n_par = adjust_parents(rel, text)

    if text != orig:
        p.write_bytes(text.encode("utf-8"))
        stats["files_touched"] += 1
        stats["module_form"] += n_mod
        stats["path_form"] += n_path
        stats["parents_bumps"] += n_par
        if n_par or n_mod or n_path:
            print(f"  {rel}  mod={n_mod} path={n_path} par={n_par}")

for p in ROOT.rglob("*"):
    if not p.is_file(): continue
    rel = p.relative_to(ROOT).as_posix()
    if any(rel.startswith(s) for s in SKIP_PREFIXES): continue
    process(p, rel)

print()
print("STATS:", stats)
