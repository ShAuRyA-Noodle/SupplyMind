# MOVED OUT OF REPO — see archive

The following heavy, gitignored, runtime-unused artifacts were relocated during P0.4 de-bloat
(2026-07-02) to keep the working tree lean:

| What | Was here | Now at | Size | Why |
|---|---|---|---|---|
| `gguf_out/` | `versions/v3_arcadia/gguf_out/` | `C:\Users\Dell\Desktop\Sleep-Token-ARCHIVE\versions\v3_arcadia\gguf_out\` | 28.7 GB | 4 local GGUF Q4_K_M judge quants; gitignored; no live endpoint loads them |
| `tools/llama.cpp/` | `versions/v3_arcadia/tools/llama.cpp/` | `C:\Users\Dell\Desktop\Sleep-Token-ARCHIVE\versions\v3_arcadia\tools\llama.cpp\` | 295 MB | local llama.cpp build/toolchain; gitignored; not imported at runtime |

These are historical inference assets for the v3 GGUF judge panel. The result JSONs and judge
caches they produced remain in `versions/v3_arcadia/results/`. See ARCHIVE/MANIFEST.md.
