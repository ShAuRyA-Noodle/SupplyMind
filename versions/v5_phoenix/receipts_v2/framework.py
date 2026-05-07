"""framework.py — grade-A receipt generator and verifier.

Usage:

    from versions.v5_phoenix.receipts_v2.framework import Receipt

    r = Receipt(
        claim_id="R5_GRANITE_mxbai_P1",
        claim="mxbai-embed-large P@1 on 53 precise queries equals 0.9622",
        command="python -m v3_arcadia.40_granite.r5_rag_beast --out /tmp/r5.json",
        extraction="jq '.pipelines.P2_mxbai_bi.p1' /tmp/r5.json",
        expected="0.9622",
        comparator="==",
    )
    r.run()           # executes command + extraction; fills actual, stdout, exit_code, match
    r.save("receipts_v2/R5_GRANITE_mxbai_P1")   # writes .receipt.yaml + .reproduce.sh
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_INLINE_STDOUT = 8192  # truncate beyond this (full hash still recorded)
MAX_STDERR_TAIL = 40      # last N lines


@dataclass
class Receipt:
    claim_id: str
    claim: str
    command: str                # shell command producing output
    extraction: str = ""        # optional pipeline to extract numeric value
    expected: str = ""
    comparator: str = "=="      # "==", ">=", "<=", "in_range", "regex"
    expected_range: list[float] | None = None   # used when comparator == "in_range"
    expected_regex: str = ""    # used when comparator == "regex"

    actual: str = ""
    exit_code: int = -1
    stdout_inline: str = ""
    stdout_sha256: str = ""
    stderr_tail: str = ""
    stdout_bytes: int = 0
    match: bool = False
    comparator_note: str = ""
    runtime_s: float = 0.0
    timestamp_utc: str = ""
    hardware: str = ""
    python_version: str = ""
    platform: str = ""
    env_notes: dict[str, str] = field(default_factory=dict)

    def run(self, cwd: Path | None = None, timeout: int = 600) -> None:
        self.timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.hardware = _describe_hardware()
        self.python_version = platform.python_version()
        self.platform = f"{platform.system()} {platform.release()}"

        start = time.time()
        cmd_full = f"{self.command}"
        if self.extraction:
            cmd_full = f"{self.command} && {self.extraction}"
        logger.info("[receipt] running: %s", cmd_full)
        try:
            proc = subprocess.run(
                cmd_full, shell=True, cwd=str(cwd) if cwd else None,
                capture_output=True, text=True, timeout=timeout,
            )
            self.exit_code = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except subprocess.TimeoutExpired as e:
            self.exit_code = -9
            stdout = e.stdout or ""
            stderr = (e.stderr or "") + f"\n[receipt] command timed out after {timeout}s"
        self.runtime_s = round(time.time() - start, 2)

        self.stdout_bytes = len(stdout.encode("utf-8"))
        self.stdout_sha256 = hashlib.sha256(stdout.encode("utf-8")).hexdigest()
        self.stdout_inline = stdout if len(stdout) <= MAX_INLINE_STDOUT else stdout[:MAX_INLINE_STDOUT] + "\n...[truncated]"
        stderr_lines = stderr.splitlines()
        self.stderr_tail = "\n".join(stderr_lines[-MAX_STDERR_TAIL:])

        # Extract the "actual" value — last non-empty stdout line by convention
        lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
        self.actual = lines[-1] if lines else ""

        self.match, self.comparator_note = _compare(self.actual, self.expected,
                                                    self.comparator, self.expected_range,
                                                    self.expected_regex)

    def save(self, stem: Path | str) -> tuple[Path, Path]:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        yaml_path = stem.with_suffix(".receipt.yaml")
        sh_path = stem.with_suffix(".reproduce.sh")
        yaml_path.write_text(_to_yaml(asdict(self)))
        sh_path.write_text(_to_shell(self))
        try:
            sh_path.chmod(0o755)
        except Exception:
            pass
        return yaml_path, sh_path


def _compare(actual: str, expected: str, op: str,
             expected_range: list[float] | None,
             expected_regex: str) -> tuple[bool, str]:
    op = (op or "==").strip()
    if op == "regex":
        rx = re.compile(expected_regex or expected)
        return (rx.search(actual) is not None,
                f"regex /{rx.pattern}/ over actual={actual!r}")
    if op == "in_range":
        if not expected_range or len(expected_range) != 2:
            return False, "expected_range missing"
        try:
            a = float(actual)
            lo, hi = float(expected_range[0]), float(expected_range[1])
            return lo <= a <= hi, f"{lo} <= {a} <= {hi}"
        except Exception:
            return False, f"could not parse actual={actual!r} as float"
    # numeric comparators
    try:
        a = float(actual)
        e = float(expected) if expected != "" else float("nan")
        if op == "==":
            ok = abs(a - e) < 1e-6 or (str(a) == str(e))
        elif op == ">=":
            ok = a >= e
        elif op == "<=":
            ok = a <= e
        elif op == ">":
            ok = a > e
        elif op == "<":
            ok = a < e
        else:
            ok = False
        return ok, f"actual={a} {op} expected={e}"
    except Exception:
        # fall back to string equality
        return actual.strip() == expected.strip(), f"string cmp: {actual!r} == {expected!r}"


def _describe_hardware() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return f"{props.name} {props.total_memory // (1024**3)}GB VRAM"
    except Exception:
        pass
    return f"{platform.processor()} (CPU only)"


def _to_yaml(d: dict) -> str:
    """Tiny hand-rolled YAML writer (no PyYAML dep). Handles our known schema."""
    lines = []
    for k, v in d.items():
        if v is None or v == "":
            lines.append(f"{k}: ''")
        elif isinstance(v, bool):
            lines.append(f"{k}: {str(v).lower()}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {v}")
        elif isinstance(v, list):
            lines.append(f"{k}: {json.dumps(v)}")
        elif isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {json.dumps(vv)}")
        else:
            if "\n" in str(v) or ":" in str(v) or "#" in str(v):
                # block scalar, literal
                block = str(v).replace("\r\n", "\n")
                lines.append(f"{k}: |")
                for bl in block.splitlines():
                    lines.append(f"  {bl}")
            else:
                lines.append(f"{k}: {v}")
    return "\n".join(lines) + "\n"


def _to_shell(r: Receipt) -> str:
    extraction_line = r.extraction or 'echo "(no extraction stage)"'
    return f"""#!/usr/bin/env bash
# Auto-generated by Phoenix v5 receipts framework.
# Claim:    {r.claim}
# Expected: {r.expected!r} ({r.comparator})
# Hardware at last run: {r.hardware}
# Runtime:  {r.runtime_s}s
set -euo pipefail
echo "[{r.claim_id}] command:"
echo '> {r.command}'
{r.command}
echo
echo "[{r.claim_id}] extraction:"
echo '> {extraction_line}'
{extraction_line}
echo
echo "[{r.claim_id}] expected: {r.expected}"
echo "[{r.claim_id}] comparator: {r.comparator}"
"""


def load(stem: Path | str) -> Receipt:
    """Load a saved receipt. Strict YAML subset matching _to_yaml output."""
    stem = Path(stem)
    yaml_path = stem.with_suffix(".receipt.yaml") if not str(stem).endswith(".yaml") else stem
    text = yaml_path.read_text(encoding="utf-8")
    d = _tiny_yaml_parse(text)
    return Receipt(**{k: d.get(k) for k in Receipt.__dataclass_fields__})


def _tiny_yaml_parse(text: str) -> dict:
    result: dict[str, Any] = {}
    current_key = None
    current_block: list[str] | None = None
    for line in text.splitlines():
        if current_block is not None:
            if line.startswith("  "):
                current_block.append(line[2:])
                continue
            result[current_key] = "\n".join(current_block)  # type: ignore[index]
            current_block = None
            current_key = None
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip()
        v = v.strip()
        if v == "|":
            current_key = k
            current_block = []
            continue
        if v.startswith("[") and v.endswith("]"):
            try:
                result[k] = json.loads(v)
                continue
            except Exception:
                pass
        if v in ("true", "false"):
            result[k] = (v == "true")
            continue
        try:
            if "." in v or "e" in v.lower():
                result[k] = float(v)
            else:
                result[k] = int(v)
            continue
        except ValueError:
            pass
        if v.startswith("'") and v.endswith("'"):
            v = v[1:-1]
        result[k] = v
    if current_block is not None and current_key:
        result[current_key] = "\n".join(current_block)
    return result
