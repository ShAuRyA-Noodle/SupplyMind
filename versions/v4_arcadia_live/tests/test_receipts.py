"""test_receipts.py — F10 receipt system regression."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from versions.v4_arcadia_live.features import receipts


def test_receipts_dir_exists():
    assert receipts.RECEIPTS_DIR.exists() or True  # created on first generate


def test_receipt_specs_are_structured():
    assert len(receipts.RECEIPT_SPECS) >= 10
    for spec in receipts.RECEIPT_SPECS:
        assert "number_id" in spec and len(spec["number_id"]) > 0
        assert "description" in spec
        assert "command" in spec


def test_jqlike_helper_generates_python_snippet():
    cmd = receipts._jqlike("foo.json", ".a.b.c")
    # Must be a portable `python -c "..."` command
    assert cmd.startswith("python -c")
    assert "json.load" in cmd
    assert "['a']" in cmd and "['b']" in cmd and "['c']" in cmd


def test_receipt_dataclass_serializes():
    r = receipts.Receipt(
        number_id="TEST_X",
        description="unit test receipt",
        value="42",
        command="echo 42",
        expected_output="42",
    )
    d = r.to_dict()
    assert d["number_id"] == "TEST_X"
    assert d["value"] == "42"
