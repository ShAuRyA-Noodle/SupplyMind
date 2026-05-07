"""
Create and register custom Ollama models for SupplyMind.

This module covers the full local-model lineage:
  - supplymind-analyst:v1 through supplymind-analyst:v5
  - qwen25-14b-local
  - qwen25-coder-local
  - mistral-nemo-local
  - deepseek-r1-local-q4

The analyst versions are prompt, format, and calibration upgrades built from
committed Modelfiles. They are local Ollama models, not hidden API calls.
Creation forces OLLAMA_MAX_LOADED_MODELS=1 to avoid VRAM contention on the
12 GB demo machine.

Usage:
    python -m rl.lora.create_ollama_model --version v5
    python -m rl.lora.create_ollama_model --all
    ollama run supplymind-analyst:v5 "Assess: Hormuz closure, Brent +3.5%"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

MODELFILE_PATH = Path(__file__).resolve().parent / "Modelfile"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ANALYST_MODELS: dict[str, tuple[str, Path]] = {
    "v1": ("supplymind-analyst:v1", Path(__file__).resolve().parent / "Modelfile"),
    "v2": ("supplymind-analyst:v2", Path(__file__).resolve().parent / "Modelfile.v2"),
    "v3": ("supplymind-analyst:v3", Path(__file__).resolve().parent / "Modelfile.v3"),
    "v4": ("supplymind-analyst:v4", Path(__file__).resolve().parent / "Modelfile.v4"),
    "v5": (
        "supplymind-analyst:v5",
        _PROJECT_ROOT / "versions/v4_arcadia_live" / "features" / "Modelfile.analyst_v5",
    ),
}

LOCAL_WRAPPER_MODELS: dict[str, tuple[str, Path]] = {
    "qwen25-14b-local": (
        "qwen25-14b-local",
        _PROJECT_ROOT / "v3_arcadia" / "00_emergence" / "qwen25-14b.Modelfile",
    ),
    "qwen25-coder-local": (
        "qwen25-coder-local",
        _PROJECT_ROOT / "v3_arcadia" / "00_emergence" / "qwen25-coder-14b.Modelfile",
    ),
    "mistral-nemo-local": (
        "mistral-nemo-local",
        _PROJECT_ROOT / "v3_arcadia" / "00_emergence" / "mistral-nemo.Modelfile",
    ),
    "deepseek-r1-local-q4": (
        "deepseek-r1-local-q4",
        _PROJECT_ROOT / "v3_arcadia" / "00_emergence" / "deepseek-r1.Modelfile",
    ),
}


def build_system_prompt() -> str:
    """Build the v1 system prompt from committed calibration data.

    The committed versioned Modelfiles are the source of truth for v2-v5.
    This builder remains for regenerating the original v1 from the 225
    real environment rollouts in rl/data/lora_training_data.json.
    """
    taiwan_data = {}
    red_sea_data = {}
    try:
        taiwan_path = DATA_DIR / "taiwan_strait_calibration.json"
        if taiwan_path.exists():
            taiwan_data = json.loads(taiwan_path.read_text(encoding="utf-8"))
        red_sea_path = DATA_DIR / "red_sea_calibration.json"
        if red_sea_path.exists():
            red_sea_data = json.loads(red_sea_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Calibration sidecar load failed; using baked defaults.", exc_info=True)

    examples = ""
    lora_data_path = DATA_DIR / "lora_training_data.json"
    if lora_data_path.exists():
        try:
            data = json.loads(lora_data_path.read_text(encoding="utf-8"))
            for sample in data[:5]:
                text = sample.get("text", "")
                if text:
                    examples += f"\n{text}\n---\n"
        except Exception:
            logger.warning("Training examples load failed; v1 will omit few-shots.", exc_info=True)

    return f"""You are SupplyMind Analyst, an AI supply chain risk management expert.
You explain RL agent decisions for the SupplyMind environment.

DOMAIN KNOWLEDGE:
- TSMC holds {taiwan_data.get('semiconductor_concentration', {}).get('tsmc_global_foundry_share', 0.54)*100:.0f}% global foundry market share, {taiwan_data.get('semiconductor_concentration', {}).get('tsmc_advanced_node_share_sub7nm', 0.92)*100:.0f}% of advanced (<7nm) nodes
- Red Sea reroute via Cape of Good Hope adds {red_sea_data.get('route_data', {}).get('additional_transit_days', 10)} transit days, {red_sea_data.get('cost_impact', {}).get('fuel_cost_increase_pct', 25)}% fuel cost increase
- Container rates spike 200-300% during maritime disruptions
- Action costs: backup qualification $150K, air expedite 10x sea freight, hedge premium 6%
- SLA penalty: $25K/day after 3-day grace period
- Risk thresholds: score >=0.8 = RED, >=0.5 = AMBER, >=0.3 = YELLOW

ENVIRONMENT:
- 7 actions: do_nothing, activate_backup_supplier, reroute_shipment, increase_safety_stock, expedite_order, hedge_commodity, issue_supplier_alert
- Dense reward: revenue_preservation(35%), proactive_bonus(15%), cost_penalty(10%), stockout_penalty(25%), unnecessary_action(5%), health_maintenance(5%), sla_compliance(5%)
- Episode: 30-60 days, budget $5-10M

WHEN EXPLAINING AN ACTION:
1. State the specific risk factors driving the decision (node names, severity, financials)
2. Quantify the cost-benefit tradeoff (action cost vs projected loss avoided)
3. Explain why this action beats alternatives
4. Reference real-world precedents when relevant
5. Keep explanations to 2-4 sentences, precise and data-driven

{f'EXAMPLES FROM TRAINING DATA:{examples}' if examples else ''}"""


def create_modelfile(base_model: str = "qwen2.5:14b") -> Path:
    """Regenerate the v1 Ollama Modelfile from current calibration files."""
    system_prompt = build_system_prompt()
    modelfile_content = f"""FROM {base_model}

SYSTEM \"\"\"
{system_prompt}
\"\"\"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_predict 256
PARAMETER num_ctx 8192
"""
    MODELFILE_PATH.write_text(modelfile_content, encoding="utf-8")
    logger.info("Modelfile created at %s", MODELFILE_PATH)
    return MODELFILE_PATH


def _ollama_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
    return env


def create_ollama_model(
    model_name: str = "supplymind-analyst:v1",
    modelfile: Path | None = None,
) -> bool:
    """Register one model with Ollama."""
    if modelfile is None:
        modelfile = create_modelfile()
    if not modelfile.exists():
        logger.error("Modelfile not found: %s", modelfile)
        return False

    logger.info("Creating Ollama model '%s' from %s", model_name, modelfile)
    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile)],
            capture_output=True,
            text=True,
            timeout=120,
            env=_ollama_env(),
        )
        if result.returncode == 0:
            logger.info("Ollama model '%s' created successfully.", model_name)
            return True
        logger.error("Ollama create failed: %s", result.stderr)
        return False
    except FileNotFoundError:
        logger.error("ollama CLI not found. Is Ollama installed?")
        return False
    except subprocess.TimeoutExpired:
        logger.error("Ollama create timed out after 120s")
        return False


def create_version(version: str) -> bool:
    """Create one analyst version from its committed Modelfile."""
    if version not in ANALYST_MODELS:
        raise ValueError(f"Unknown analyst version: {version}. Expected {sorted(ANALYST_MODELS)}")
    name, modelfile = ANALYST_MODELS[version]
    return create_ollama_model(name, modelfile)


def create_local_wrapper(wrapper_key: str) -> bool:
    """Create one base-model wrapper such as qwen25-coder-local."""
    if wrapper_key not in LOCAL_WRAPPER_MODELS:
        raise ValueError(f"Unknown wrapper: {wrapper_key}. Expected {sorted(LOCAL_WRAPPER_MODELS)}")
    name, modelfile = LOCAL_WRAPPER_MODELS[wrapper_key]
    return create_ollama_model(name, modelfile)


def create_all() -> dict[str, bool]:
    """Create every committed analyst model and local wrapper."""
    results: dict[str, bool] = {}
    for _, (name, modelfile) in ANALYST_MODELS.items():
        results[name] = create_ollama_model(name, modelfile)
    for _, (name, modelfile) in LOCAL_WRAPPER_MODELS.items():
        results[name] = create_ollama_model(name, modelfile)
    return results


def test_model(model_name: str = "supplymind-analyst:v5") -> str | None:
    """Smoke-test a created model. v5 is additionally JSON-schema checked."""
    try:
        import ollama as ollama_pkg

        response = ollama_pkg.chat(
            model=model_name,
            messages=[{
                "role": "user",
                "content": (
                    "STATE: Day 3/30. TSMC Fab 14 offline (risk=0.85). "
                    "Health: 72/100. Budget: $4.2M/$5M. P95 loss: $2.1M. "
                    "Active disruption: tropical_cyclone (warning phase). "
                    "ACTION: activate_backup_supplier targeting SUP_TSMC, backup=SUP_SAMSUNG. "
                    "Return the required SupplyMind decision object."
                ),
            }],
            options={"temperature": 0.0},
        )
        content = response["message"]["content"]
        if model_name.endswith(":v5"):
            start, end = content.index("{"), content.rindex("}") + 1
            parsed = json.loads(content[start:end])
            required = {
                "decision",
                "evidence",
                "counterfactual",
                "precedent",
                "risk_level",
                "confidence",
            }
            missing = required - set(parsed)
            if missing:
                raise ValueError(f"v5 JSON missing keys: {sorted(missing)}")
        logger.info("Model test response:\n%s", content)
        return content
    except Exception as e:  # noqa: BLE001
        logger.error("Model test failed: %s", e)
        return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Create SupplyMind Ollama models")
    parser.add_argument("--version", choices=sorted(ANALYST_MODELS), default="v5")
    parser.add_argument("--wrapper", choices=sorted(LOCAL_WRAPPER_MODELS), default=None)
    parser.add_argument("--all", action="store_true", help="Create every analyst version and base wrapper")
    parser.add_argument("--test", action="store_true", help="Run one smoke prompt after creation")
    args = parser.parse_args()

    if args.all:
        results = create_all()
        print(json.dumps(results, indent=2))
        if args.test and results.get("supplymind-analyst:v5"):
            test_model("supplymind-analyst:v5")
        return

    if args.wrapper:
        success = create_local_wrapper(args.wrapper)
        model_name = LOCAL_WRAPPER_MODELS[args.wrapper][0]
    else:
        success = create_version(args.version)
        model_name = ANALYST_MODELS[args.version][0]

    if success and args.test:
        test_model(model_name)


if __name__ == "__main__":
    main()
