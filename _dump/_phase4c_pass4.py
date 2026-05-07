"""Pass 4: catch v3_arcadia/ in receipts + Dockerfile + scripts that were skipped."""
import sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path("c:/Users/Dell/Desktop/Sleep-Token")

SKIP = ("_dump/", ".venv/", ".git/", "vendor/", "models/", "external_data/",
        "wandb/", ".pytest_cache/", ".tmp_pytest/", ".source_cache/",
        ".openrouter_cache/", ".agents/", ".claude/",
        "versions/v5_phoenix/.venv-roll/",
        # Skip v3_arcadia internal files — their internal refs are correct as-is
        "versions/v3_arcadia/")

# Substitute non-prefixed v3_arcadia/ → versions/v3_arcadia/
# Use a safe replacement that won't double-prefix existing versions/v3_arcadia/
SUBS = [
    ("\"v3_arcadia/", "\"versions/v3_arcadia/"),
    ("'v3_arcadia/", "'versions/v3_arcadia/"),
    (" v3_arcadia/", " versions/v3_arcadia/"),
    ("(v3_arcadia/", "(versions/v3_arcadia/"),
    ("[v3_arcadia/", "[versions/v3_arcadia/"),
    ("/v3_arcadia/", "/versions/v3_arcadia/"),  # path-internal, but excludes versions/v3_arcadia/ which has versions/ prefix
    # Markdown link / line-start
]
# Apply only when not already prefixed by `versions/`
def safe_replace(text, old, new, sentinel="versions/v3_arcadia/"):
    # Plain replace would handle "versions/v3_arcadia/" → "versions/versions/v3_arcadia/" if old contains v3_arcadia/.
    # Protect existing sentinel first.
    text = text.replace("versions/v3_arcadia/", "\x00VPROT\x00")
    text = text.replace(old, new)
    text = text.replace("\x00VPROT\x00", "versions/v3_arcadia/")
    return text

stats = {"files": 0, "subs": 0}
for p in ROOT.rglob("*"):
    if not p.is_file(): continue
    rel = p.relative_to(ROOT).as_posix()
    if any(rel.startswith(s) for s in SKIP): continue
    try:
        size = p.stat().st_size
    except OSError: continue
    if size > 5_000_000: continue
    try:
        text = p.read_bytes().decode("utf-8")
    except UnicodeDecodeError: continue
    orig = text
    n = 0
    for old, new in SUBS:
        if old in text:
            cnt = text.count(old) - text.count(new)
            text = safe_replace(text, old, new)
            n += cnt
    # Line-start v3_arcadia/ (rare)
    new_text = []
    for line in text.split("\n"):
        if line.startswith("v3_arcadia/"):
            line = "versions/" + line
            n += 1
        new_text.append(line)
    text = "\n".join(new_text)
    if text != orig:
        p.write_bytes(text.encode("utf-8"))
        stats["files"] += 1
        stats["subs"] += n
        if n: print(f"  {rel}  +{n}")
print()
print("STATS:", stats)
