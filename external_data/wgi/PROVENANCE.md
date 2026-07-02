# WGI Dataset — Provenance

- File: `wgidataset_with_sourcedata-2025.xlsx`
- Source: World Bank — Worldwide Governance Indicators (WGI), 2025 release (with source data).
- SHA256: `25A2F9EABB90B0092973392C0B31571AA58B691CC5786292E504B52F693E1EB8`
- Size: 10,423,344 bytes (9.9 MB)
- Rescued from `_dump/data_legacy/` on 2026-07-02 during P0.4 de-bloat and relocated here so
  `rl/analysis/trained_models.py` and `rl/data/build_unified_buffer_v2.py` resolve it at
  `external_data/wgi/`.
- Tracking: intentionally NOT tracked in git (covered by the `external_data/` .gitignore rule);
  the file lives on disk only. Re-download from the World Bank WGI portal if missing.
