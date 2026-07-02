# MOVED OUT OF REPO — see archive

The entire `_dump/` quarantine (233 files, 210 git-tracked) was relocated during P0.4 de-bloat
(2026-07-02) to:

`C:\Users\Dell\Desktop\Sleep-Token-ARCHIVE\_dump\`

It held pre-reorg legacy copies (`*_legacy/`, `final_submit_obsolete/`, `vendor_legacy/`), one-shot
reorg/deploy scripts (`_phase4c_*.py`, `_scan_refs.py`, `_hf_cleanup.py`, `_hf_deploy_test.py`), and
junk (`_ci_runs.json`, a 3-byte BOM file). Zero live-code references (verified: `_dump` grep in
server/scripts/tests/rl hits only pydantic `.model_dump` false positives).

RESCUED before the move: `_dump/data_legacy/wgidataset_with_sourcedata-2025.xlsx` was copied to
`external_data/wgi/` (SHA256 verified identical) because `rl/analysis/trained_models.py` and
`rl/data/build_unified_buffer_v2.py` resolve it there.

See ARCHIVE/MANIFEST.md.
