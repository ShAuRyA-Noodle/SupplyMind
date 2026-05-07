"""Grade-A reproducibility receipts — Phoenix v5 upgrade over v4.

v4 receipts emitted a single number (e.g. `0.9622`).
v5 receipts emit a full audit packet:

    command:       exact shell command
    extraction:    how we distilled the numeric claim
    expected:      the value committed to in the paper / README / docs/v4/JUDGES.md
    actual:        what we observed when we last ran
    exit_code:     process exit code
    stdout:        full stdout (or truncated w/ sha256)
    stderr_tail:   last 10 lines of stderr
    match:         true iff `actual` satisfies the comparator against `expected`
    comparator:    "==", ">=", "<=", "in_range", "regex", etc
    hardware:      RTX 4080 Laptop 12GB VRAM, 15.7GB RAM, CUDA 12.1
    timestamp:     ISO8601 UTC
    runtime_s:     wall-clock seconds

This is the obra/superpowers "verification-before-completion" pattern
productized. Every receipt is one YAML + one bash script.
"""
