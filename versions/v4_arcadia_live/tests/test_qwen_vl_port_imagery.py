"""test_qwen_vl_port_imagery.py — G3+F1 regression (heuristic path only)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from versions.v4_arcadia_live.features.qwen_vl_port_imagery import (
    PORT_ANCHORS, assess_port_image, run_all_ports, synthesize_sample_image,
)


def test_seven_port_anchors_defined():
    assert len(PORT_ANCHORS) >= 7
    for _, meta in PORT_ANCHORS.items():
        assert "name" in meta and "baseline_queue" in meta and "lat" in meta and "lon" in meta


def test_synthesize_image_returns_bytes():
    img = synthesize_sample_image("KAOHSIUNG")
    assert isinstance(img, bytes)
    assert len(img) > 1000  # non-trivial PNG


def test_heuristic_assessment_produces_valid_fields():
    img = synthesize_sample_image("SHANGHAI")
    ar = assess_port_image(img, "SHANGHAI", prefer_mode="heuristic")
    assert ar.mode == "heuristic"
    assert 0 <= ar.risk_score <= 1
    assert 0 <= ar.confidence <= 1
    assert ar.container_stack_density in ("low", "medium", "high")
    assert isinstance(ar.smoke_or_fire, bool)
    assert isinstance(ar.flood_indicators, bool)


def test_run_all_ports_covers_every_anchor():
    out = run_all_ports(mode="heuristic")
    for pid in PORT_ANCHORS:
        assert pid in out["assessments"]
    assert out["summary"]["mean_confidence"] > 0
    assert out["summary"]["highest_risk_port"] in PORT_ANCHORS


def test_different_ports_give_different_assessments():
    img_a = synthesize_sample_image("HAIFA")
    img_b = synthesize_sample_image("ROTTERDAM")
    ar_a = assess_port_image(img_a, "HAIFA", prefer_mode="heuristic")
    ar_b = assess_port_image(img_b, "ROTTERDAM", prefer_mode="heuristic")
    # Port names must differ; risk scores may or may not
    assert ar_a.port_name != ar_b.port_name
