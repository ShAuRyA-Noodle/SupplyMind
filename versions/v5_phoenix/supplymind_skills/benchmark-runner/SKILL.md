---
name: benchmark-runner
description: Use when you need to measure a performance change (latency, memory, throughput, accuracy, reward) and claim an improvement. Enforces paired baseline/post-change measurement, fresh output capture, and a receipt that an external reviewer can re-run.
---

# Benchmark Runner

## The iron law

NO PERFORMANCE CLAIM WITHOUT A PAIRED BENCHMARK RECEIPT.

If you're about to say "this is 3x faster" / "accuracy improved by X pp" /
"VRAM usage dropped" — stop. Run this skill.

## When to invoke

- After any change that touches a hot path (training loop, inference, data
  pipeline, retrieval, forecasting)
- Before writing up a result in a PR description or README
- Before shipping a receipt (this skill is the receipt's backbone)
- Before replying "done" on a perf-related task

## When NOT to invoke

- For correctness-only changes where you've already written tests
- For exploratory profiling (no claim yet, just investigation)
- For docs / comment / typo changes

## Required inputs

1. **Claim being made** (one sentence, specific numbers if known):
   "mxbai-P@1 improves by ≥0.01 after adding HyDE rewriting."
2. **Benchmark command** (exact, including flags):
   `python -m v3_arcadia.40_granite.r5_rag_beast --pipeline hyde_rewrite --out tmp.json`
3. **Metric extraction command** (exact):
   `jq '.pipelines.P8_hyde_rrf_rerank.p1' tmp.json`

## The RED → GREEN cycle for benchmarks

This is TDD applied to performance:

### Stage 1 — RED (baseline)

```bash
# Check out the commit BEFORE your change (or use git stash)
git stash
python -m v3_arcadia.40_granite.r5_rag_beast --out baseline.json
jq '.pipelines.P2_mxbai_bi.p1' baseline.json > baseline_p1.txt
```

Record `baseline_p1.txt` — paste full content into your receipt.

### Stage 2 — Implement (your change)

```bash
git stash pop
# Apply your change
```

### Stage 3 — GREEN (post-change)

```bash
python -m v3_arcadia.40_granite.r5_rag_beast --out post.json
jq '.pipelines.P8_hyde_rrf_rerank.p1' post.json > post_p1.txt
```

### Stage 4 — Verify claim

```bash
paste -d' - ' post_p1.txt baseline_p1.txt | bc
```

**If delta doesn't match claim, STOP. Either the change didn't do what you
thought, or the claim is wrong.** Don't rationalize — iterate.

### Stage 5 — Receipt

Write a receipt file `receipts_v2/<claim_id>.receipt.yaml`:

```yaml
claim_id: V5_HYDE_REWRITE_p1_delta
claim: mxbai P@1 improves by >=0.01 after HyDE
command: python -m v3_arcadia.40_granite.r5_rag_beast --pipeline hyde_rewrite --out /tmp/post.json
extraction: jq '.pipelines.P8_hyde_rrf_rerank.p1' /tmp/post.json
baseline_command: python -m v3_arcadia.40_granite.r5_rag_beast --pipeline mxbai_bi --out /tmp/base.json
baseline_extraction: jq '.pipelines.P2_mxbai_bi.p1' /tmp/base.json
baseline_expected: "0.9622"
expected: ">=0.9722"
actual_baseline: "0.9622"
actual_post: "0.9747"
delta: "+0.0125"
match: true
exit_code: 0
hardware: RTX 4080 Laptop, 12GB VRAM
runtime_s: 430
timestamp: 2026-04-22T03:00:00Z
```

Drop a `reproduce.sh` next to it so an external reviewer can replay.

## Anti-patterns (stop if you catch yourself doing this)

- "It feels faster" — feelings aren't numbers
- "I ran it once, got 3x" — run it 3x, report median
- "The graph shows improvement" — is the graph generated fresh or cached?
- "Small noise, probably fine" — publish the std dev, let the reader decide
- "Everything passed" — which version? what commit?

## Integration with verification-before-completion

This skill IS verification-before-completion for performance claims. If the
receipt's `match: false`, you don't ship the claim. Period.
