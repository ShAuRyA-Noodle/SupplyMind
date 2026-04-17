# R5 Granite — RAG SOTA Benchmark

- **Corpus**: 6483 chunks across 48 documents
- **Queries**: 53 (each with 1–2 gold doc IDs, derived from 26 crisis articles)
- **Pipelines**: 8 configurations (3 bi-encoders, 3 with reranker, RRF ensemble, HyDE)
- **Total runtime**: 8.1 min

## Corpus composition

- wiki_crisis: 564 chunks
- sec_10k: 5790 chunks
- policy: 129 chunks
- world_bank: 0 chunks

## Pipeline results (sorted by MRR)

| Pipeline | P@1 | P@3 | P@5 | MRR | nDCG@10 | Latency |
|----------|-----|-----|-----|-----|---------|---------|
| P2_mxbai_bi | 0.962 | 0.925 | 0.857 | 0.978 | 0.961 | 0.04s |
| P3_snowflake_bi | 0.943 | 0.899 | 0.883 | 0.972 | 0.958 | 0.03s |
| P1_bge_m3_bi | 0.925 | 0.912 | 0.875 | 0.962 | 0.958 | 0.05s |
| P4_bge_m3_rerank | 0.925 | 0.868 | 0.811 | 0.959 | 0.938 | 1.33s |
| P5_mxbai_rerank | 0.925 | 0.862 | 0.819 | 0.959 | 0.939 | 1.14s |
| P6_snowflake_rerank | 0.925 | 0.855 | 0.800 | 0.959 | 0.935 | 1.86s |
| P7_rrf_ensemble_rerank | 0.925 | 0.868 | 0.808 | 0.959 | 0.936 | 1.43s |
| P8_hyde_rrf_rerank | 0.925 | 0.862 | 0.819 | 0.959 | 0.938 | 1.19s |

## Key findings

- **Best pipeline**: **P2_mxbai_bi** with MRR 0.978, P@1 0.962, latency 0.04s
- On this corpus, **bi-encoder alone outperforms rerank variants** by 3.8 pp on P@1 — the reranker's chunk-level scoring can actively demote relevant chunks from the gold document when the bi-encoder retrieval is already near-ceiling.
- All 3 embedders (bge_m3, mxbai, snowflake) achieve P@1 ≥ 0.925, showing modern dense retrievers are highly competitive on well-curated corpora.
- HyDE + RRF ensemble did **not** improve over bare bi-encoders here because queries are already explicit and matched to gold doc vocabulary. HyDE's benefit is typically on vague/open queries where LLM-expansion bridges the lexical gap.

## vs V3 Block 4 baseline (1,111 chunks, loose-phrase queries)

| Config | V3 Block 4 | R5 Granite |
|--------|------------|-----------|
| mxbai bi P@1 | 0.52 | **0.962** |
| mxbai+rerank P@1 | 0.54 | 0.925 |
| mxbai bi MRR | 0.537 | **0.978** |