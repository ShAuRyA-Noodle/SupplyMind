"""adversarial_gauntlet.py — the REAL adversarial gauntlet for SupplyMind.

Executes every attack in ``attack_corpus.build_corpus()`` (>300 distinct
attacks) against the ACTUAL system and decides "blocked" with a single crisp
assertion on the real return value. There is no both-branches counter and no
phantom padding: each attack is run once through a real code path and its
verdict is justified by captured evidence stored in the receipt.

Real code paths exercised:
  * ``server.openenv_mcp_wrapper.SupplyMindMCP.step`` — the OpenEnv MCP action
    entrypoint (schema gate + engine ``_validate_action`` + graph/financial
    gates). Imported READ-ONLY; this module never modifies the wrapper.
  * ``SupplyMindMCP`` MCP tools (node status, RAG, financial, recent events).
  * ``server.supply_environment`` grade path (via the wrapper's env).
  * ``supplymind.phoenix.wordle_env`` schema/format/dictionary gates + grader.
  * ``supplymind.warroom.crisis_library.find_analogs`` RAG retrieval.

Honesty rules (CLAUDE.md §0):
  * blocked == the system verifiably rejected, sanitized, demoted, or bounded
    the attack — proven by the recorded evidence, never assumed.
  * A non-blocked attack is a BREACH and is listed loudly, never hidden.
  * Controls (legitimate inputs) are scored separately to expose false
    positives — a defense that blocks everything by refusing all input is not
    a defense.

Known limitation, stated up front: with ``OPENROUTER_API_KEY`` revoked there is
no live LLM in the loop, so prompt-injection is tested only against the
DATA-HANDLING surfaces the agent can influence (RAG retrieval, MCP tool string
args, string action fields). Whether a live analyst LLM resists instruction
override is BLOCKED-ON-KEY and is reported as such, not faked.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

# Run fully offline: the crisis-library RAG tool defaults to an mxbai embedding
# model that would otherwise hit HuggingFace. Forcing offline makes the gauntlet
# deterministic, fast, and survivable on no-WiFi (falls back to TF-IDF if the
# model is not cached locally). Set before any crisis_library import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from supplymind.warroom.attack_corpus import build_corpus, build_controls  # noqa: E402

# Fixed seed for every reset so reward/budget comparisons are deterministic.
FIXED_SEED = 12345
PASSIVE_GRADE_CAP = 0.70  # a fully-passive episode must not exceed this grade

# MCP tools that legitimately require an active episode (must reject pre-reset).
_EPISODE_SCOPED_TOOLS = {
    "tool_sm_get_node_status",
    "tool_sm_get_financial_state",
    "tool_sm_explain_disruption",
}


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


class Gauntlet:
    """Runs the corpus against the real system. One shared MCP instance is used
    for step/tool attacks; each attack that reaches the engine resets it first
    so state cannot bleed between attacks."""

    def __init__(self) -> None:
        from server.openenv_mcp_wrapper import SupplyMindMCP  # read-only import
        self._MCP = SupplyMindMCP
        self.mcp = SupplyMindMCP()
        self.schema_notes: list[dict] = []

    # ---- MCP step reset helper ----
    def _reset(self, task: str = "easy_typhoon_response", seed: int = FIXED_SEED):
        self.mcp.reset(task_id=task, seed=seed)

    def _lar(self, out: dict) -> dict:
        obs = out.get("observation") or {}
        return (obs.get("last_action_result") or {}) if isinstance(obs, dict) else {}

    def _budget(self, out: dict):
        obs = out.get("observation") or {}
        fin = obs.get("financials") or {} if isinstance(obs, dict) else {}
        return fin.get("budget_remaining"), fin.get("budget_total")

    # =====================================================================
    # Surface handlers — each returns (blocked, reason, evidence)
    # =====================================================================

    def _h_mcp_step(self, atk: dict):
        expect = atk["expect"]
        task = atk.get("task", "easy_typhoon_response")
        payload = atk["payload"]

        if expect == "reject":
            # Malformed body → wrapper must return invalid_action_format, no obs.
            out = self.mcp.step(payload) if isinstance(payload, dict) else \
                self._step_nondict(payload)
            info = out.get("info") or {}
            obs = out.get("observation")
            blocked = (obs is None and info.get("error") == "invalid_action_format")
            return blocked, ("schema_rejected" if blocked else "schema_bypassed"), {
                "observation_is_none": obs is None,
                "error": info.get("error"),
                "detail": str(info.get("detail", ""))[:160],
                "reward": out.get("reward"),
            }

        if expect == "field_ignored":
            junk_key = next((k for k in payload if k != "action_type"), None)
            self._reset(task)
            clean = self.mcp.step({"action_type": payload["action_type"]})
            self._reset(task)
            got = self.mcp.step(payload)
            cb, ct = self._budget(clean)
            gb, gt = self._budget(got)
            from supplymind.contracts import SupplyMindAction
            validated = SupplyMindAction(**payload).model_dump()
            key_dropped = junk_key not in validated
            reward_same = got.get("reward") == clean.get("reward")
            budget_same = gb == cb
            done_same = ((got.get("observation") or {}).get("done")
                         == (clean.get("observation") or {}).get("done"))
            blocked = key_dropped and reward_same and budget_same and done_same
            return blocked, ("extra_field_dropped" if blocked else "extra_field_took_effect"), {
                "junk_key": junk_key, "key_dropped_by_schema": key_dropped,
                "reward": got.get("reward"), "clean_reward": clean.get("reward"),
                "budget_remaining": gb, "clean_budget_remaining": cb,
            }

        if expect == "budget_gate":
            self._reset(task)
            out = self.mcp.step(payload)
            lar = self._lar(out)
            gb, gt = self._budget(out)
            success = lar.get("success")
            reward = out.get("reward")
            blocked = (success is False and gb == gt
                       and _finite(reward) and _finite(gb))
            self.schema_notes.append({
                "attack": atk["name"],
                "note": "pydantic accepted the notional (inf/huge); the effect "
                        "was blocked at the financial budget gate (defense-in-depth).",
            })
            return blocked, ("budget_gate_rejected" if blocked else "over_budget_action_applied"), {
                "action_success": success,
                "message": str(lar.get("message", ""))[:140],
                "reward": reward, "budget_remaining": gb, "budget_total": gt,
            }

        if expect == "string_field_inert":
            self._reset(task)
            out = self.mcp.step(payload)
            lar = self._lar(out)
            gb, _ = self._budget(out)
            reward = out.get("reward")
            cost = lar.get("cost")
            amt = payload.get("hedge_amount_usd", 0.0)
            blocked = (_finite(reward) and _finite(gb) and gb >= 0
                       and _finite(cost) and 0 <= float(cost) <= float(amt) + 1e-6)
            return blocked, ("string_treated_as_data" if blocked else "string_field_amplified"), {
                "action_success": lar.get("success"), "cost": cost,
                "reward": reward, "budget_remaining": gb, "requested_notional": amt,
            }

        if expect == "bad_target":
            self._reset(task)
            out = self.mcp.step(payload)
            info = out.get("info") or {}
            obs = out.get("observation")
            # Defense layer 1 — schema/model-validator rejection (e.g. an empty
            # required target_node_id) surfaces as invalid_action_format, obs None.
            if obs is None and info.get("error") == "invalid_action_format":
                return True, "schema_rejected_target", {
                    "layer": "schema", "error": info.get("error"),
                    "detail": str(info.get("detail", ""))[:120],
                }
            lar = self._lar(out)
            reward = out.get("reward")
            success = lar.get("success")
            msg = str(lar.get("message", ""))
            demoted = msg.strip() == "No action taken."
            # Defense layer 2 — the bogus target was rejected by the graph
            # (success False) or safely demoted to a no-op; NOT an illegitimate
            # success on a fake node.
            blocked = (success is False or demoted) and _finite(reward)
            return blocked, ("target_rejected_or_demoted" if blocked else "acted_on_bogus_target"), {
                "layer": "graph", "action_success": success,
                "demoted_to_do_nothing": demoted, "message": msg[:140], "reward": reward,
            }

        if expect == "reroute_safe":
            self._reset(task)
            t0 = time.time()
            out = self.mcp.step(payload)
            elapsed = time.time() - t0
            lar = self._lar(out)
            gb, _ = self._budget(out)
            reward = out.get("reward")
            blocked = _finite(reward) and _finite(gb) and gb >= 0 and elapsed < 5.0
            return blocked, ("reroute_bounded" if blocked else "reroute_unbounded_or_crash"), {
                "action_success": lar.get("success"),
                "message": str(lar.get("message", ""))[:120],
                "reward": reward, "budget_remaining": gb,
                "elapsed_s": round(elapsed, 4),
            }

        raise ValueError(f"unknown mcp_step expect: {expect}")

    def _step_nondict(self, payload):
        """Feed a non-dict body straight to the wrapper's step (it must catch)."""
        try:
            return self.mcp.step(payload)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001 — a raw crash here IS a finding
            return {"observation": None, "reward": None,
                    "info": {"error": "harness_uncaught", "detail": f"{type(e).__name__}: {e}"}}

    def _h_crisis_rag(self, atk: dict):
        from supplymind.warroom.crisis_library import find_analogs
        text = atk["payload"]["text"]
        expect = atk["expect"]
        analogs = find_analogs(text, k=3, mode="tfidf")
        names = [a.name for a in analogs]
        sims = [a.similarity for a in analogs]
        # The payload is safe iff the output is drawn from the real library (not
        # the attacker string) and all similarities are finite numbers.
        payload_echoed = any(text.strip() and text.strip() in (n or "") for n in names)
        well_formed = len(analogs) >= 1 and all(_finite(s) for s in sims)
        if expect == "accept":
            blocked = well_formed  # a control: "blocked" here means "worked"
            return blocked, ("rag_returned_analogs" if blocked else "rag_failed"), {
                "n": len(analogs), "top": names[0] if names else None,
                "top_sim": round(sims[0], 4) if sims else None,
            }
        blocked = well_formed and not payload_echoed
        return blocked, ("injection_treated_as_query" if blocked else "injection_leaked_or_crashed"), {
            "n": len(analogs), "top": names[0] if names else None,
            "top_sim": round(sims[0], 4) if sims else None,
            "payload_echoed_as_result": payload_echoed,
        }

    def _h_mcp_tool_rag(self, atk: dict):
        self._reset()
        res = self.mcp.tool_sm_query_crisis_library(
            text=atk["payload"]["text"], k=atk["payload"].get("k", 3))
        ok = res.get("ok")
        # Blocked = clean, well-formed handling: either analogs returned (data)
        # or a clean ok:False error string — never an exception, never a leak.
        analogs = res.get("analogs") or []
        blocked = isinstance(res, dict) and (
            (ok is True and isinstance(analogs, list))
            or (ok is False and isinstance(res.get("error"), str)))
        return blocked, ("tool_handled_cleanly" if blocked else "tool_crashed_or_leaked"), {
            "ok": ok, "n_results": res.get("n_results"),
            "error": str(res.get("error", ""))[:120],
        }

    def _h_mcp_tool_node(self, atk: dict):
        self._reset()
        res = self.mcp.tool_sm_get_node_status(node_id=atk["payload"]["node_id"])
        ok = res.get("ok")
        # Bogus/oversized node ids must resolve to a clean not-found, no crash.
        blocked = isinstance(res, dict) and ok is False and isinstance(res.get("error"), str)
        return blocked, ("node_not_found_cleanly" if blocked else "node_tool_misbehaved"), {
            "ok": ok, "error": str(res.get("error", ""))[:120],
        }

    def _h_wordle_schema(self, atk: dict):
        from supplymind.phoenix.wordle_env.env import (
            WordleAction, WordleResetRequest, reset, step)
        from pydantic import ValidationError
        guess = atk["payload"]["guess"]
        expect = atk["expect"]
        try:
            action = WordleAction(guess=guess)
        except ValidationError:
            if expect == "accept":
                return False, "schema_rejected_valid_word", {"schema_gate": "raised"}
            return True, "schema_length_gate", {"schema_gate": "raised"}
        state, _ = reset(WordleResetRequest(seed=999, target_word="brain"))
        state, obs, br = step(state, action)
        rejected = state.history[-1].get("rejected") if state.history else None
        reward = float(br.get("reward", obs.reward))
        if expect == "accept":
            # A legitimate word must score as a real guess (not rejected).
            blocked = rejected is None and _finite(reward)
            return blocked, ("valid_word_scored" if blocked else "valid_word_rejected"), {
                "rejected": rejected, "reward": round(reward, 4),
            }
        blocked = rejected in ("format_invalid", "non_dictionary")
        return blocked, (rejected or "scored_as_valid_word"), {
            "rejected": rejected, "reward": round(reward, 4),
        }

    def _h_wordle_reward(self, atk: dict):
        from supplymind.phoenix.wordle_env.env import (
            WordleAction, WordleResetRequest, reset, step)
        from pydantic import ValidationError
        spec = atk["payload"]
        kind = spec["kind"]
        target = spec["target"]

        if kind == "repeat_nonsolve":
            state, _ = reset(WordleResetRequest(seed=1, target_word=target))
            total = 0.0
            for _ in range(6):
                if state.won or state.lost:
                    break
                state, obs, br = step(state, WordleAction(guess=spec["word"]))
                total += float(br.get("reward", 0.0))
            blocked = (not state.won) and total <= 0.5
            return blocked, ("nonsolve_bounded" if blocked else "nonsolve_farmed"), {
                "total_reward": round(total, 4), "won": state.won,
                "guesses_used": 6 - state.guesses_remaining,
            }

        if kind == "solved_repeat":
            state, _ = reset(WordleResetRequest(seed=1, target_word=target))
            state, _o1, br1 = step(state, WordleAction(guess=target))
            solve_reward = float(br1.get("reward", 0.0))
            state, _o2, br2 = step(state, WordleAction(guess=target))
            extra = float(br2.get("reward", 0.0))
            blocked = state.won and extra == 0.0
            return blocked, ("episode_done_gate" if blocked else "post_solve_reward_gained"), {
                "solve_reward": round(solve_reward, 4),
                "extra_reward_after_solve": round(extra, 4), "won": state.won,
            }

        if kind == "spam_guess":
            state, _ = reset(WordleResetRequest(seed=1, target_word=target))
            schema_rejects = 0
            step_rewards: list[float] = []
            for _ in range(spec["n"]):
                if state.won or state.lost:
                    break
                try:
                    action = WordleAction(guess=spec["guess"])
                except ValidationError:
                    schema_rejects += 1
                    break  # schema gate rejects this shape every time
                state, obs, br = step(state, action)
                step_rewards.append(float(br.get("reward", 0.0)))
            max_reward = max(step_rewards) if step_rewards else -1.0
            blocked = (not state.won) and max_reward <= 0.0
            return blocked, ("invalid_guess_penalized" if blocked else "invalid_guess_rewarded"), {
                "schema_rejects": schema_rejects,
                "max_step_reward": round(max_reward, 4),
                "n_scored_steps": len(step_rewards), "won": state.won,
            }

        raise ValueError(f"unknown wordle_reward kind: {kind}")

    def _h_sm_reward(self, atk: dict):
        spec = atk["payload"]
        kind = spec["kind"]
        task = spec["task"]
        seed = spec.get("seed", FIXED_SEED)

        if kind == "all_do_nothing":
            self.mcp.reset(task_id=task, seed=seed)
            steps, cum, done = 0, 0.0, False
            while not done and steps < 300:
                out = self.mcp.step({"action_type": "do_nothing"})
                cum += out.get("reward") or 0.0
                done = out.get("done", False)
                steps += 1
            grade = self.mcp._env.grade()
            score = grade.get("score")
            blocked = _finite(score) and float(score) < PASSIVE_GRADE_CAP
            return blocked, ("passive_scores_low" if blocked else "passive_scores_high"), {
                "passive_grade_score": score, "cumulative_reward": round(cum, 4),
                "steps": steps, "cap": PASSIVE_GRADE_CAP,
            }

        if kind == "invalid_action_spam":
            self.mcp.reset(task_id=task, seed=seed)
            rewards = []
            for _ in range(spec["n"]):
                out = self.mcp.step({"action_type": "totally_invalid_action"})
                rewards.append(out.get("reward"))
            numeric = [r for r in rewards if _finite(r)]
            max_reward = max(numeric) if numeric else 0.0
            blocked = all(_finite(r) for r in rewards) and max_reward <= 0.0
            return blocked, ("malformed_never_positive" if blocked else "malformed_rewarded"), {
                "max_reward": max_reward, "n": spec["n"],
                "sample": rewards[:3],
            }

        if kind == "budget_never_negative":
            self.mcp.reset(task_id=task, seed=seed)
            expensive = [
                {"action_type": "hedge_commodity", "commodity": "chips",
                 "hedge_amount_usd": 900000.0},
                {"action_type": "increase_safety_stock",
                 "target_node_id": "SUP_TSMC", "additional_stock_days": 60},
                {"action_type": "expedite_order", "target_node_id": "SUP_TSMC",
                 "expedite_mode": "air"},
            ]
            min_budget, all_finite, done, i = float("inf"), True, False, 0
            while not done and i < 300:
                out = self.mcp.step(expensive[i % len(expensive)])
                i += 1
                gb, _ = self._budget(out)
                if not _finite(gb):
                    all_finite = False
                    break
                min_budget = min(min_budget, gb)
                done = out.get("done", False)
            blocked = all_finite and min_budget >= 0.0
            return blocked, ("budget_floor_held" if blocked else "budget_went_negative"), {
                "min_budget": None if min_budget == float("inf") else round(min_budget, 2),
                "all_finite": all_finite, "steps": i,
            }

        raise ValueError(f"unknown sm_reward kind: {kind}")

    def _h_session(self, atk: dict):
        spec = atk["payload"]
        kind = spec["kind"]

        if kind == "step_before_reset":
            m = self._MCP()
            out = m.step(spec["action"])
            obs, info = out.get("observation"), (out.get("info") or {})
            blocked = obs is None and info.get("error") in ("step_failed", "invalid_action_format")
            return blocked, ("rejected_pre_reset" if blocked else "ran_without_reset"), {
                "info": info,
            }

        if kind == "tool_before_reset":
            m = self._MCP()
            tool = spec["tool"]
            args = {
                "tool_sm_get_node_status": {"node_id": "SUP_TSMC"},
                "tool_sm_explain_disruption": {"signal_id": "SIG_1"},
                "tool_sm_query_crisis_library": {"text": "hormuz"},
                "tool_sm_query_recent_events": {},
                "tool_sm_get_financial_state": {},
                "tool_sm_describe_action_space": {},
            }[tool]
            res = getattr(m, tool)(**args)
            if tool in _EPISODE_SCOPED_TOOLS:
                blocked = isinstance(res, dict) and res.get("ok") is False
                reason = "episode_scoped_tool_refused" if blocked else "leaked_without_episode"
            else:
                # Episode-independent tools may answer with static/global data;
                # blocked = clean dict response, no exception, no episode data.
                blocked = isinstance(res, dict) and ("ok" in res)
                reason = "static_tool_ok" if blocked else "tool_misbehaved"
            return blocked, reason, {"tool": tool, "ok": res.get("ok"),
                                     "keys": sorted(res.keys())[:6]}

        if kind == "grade_before_reset":
            m = self._MCP()
            try:
                m._env.grade()
                return False, "graded_without_episode", {"raised": False}
            except RuntimeError as e:
                return True, "grade_raised_clean", {"raised": True, "msg": str(e)[:100]}

        if kind == "state_before_reset":
            m = self._MCP()
            st = m.state()
            blocked = isinstance(st, dict) and st.get("task") is None
            return blocked, ("no_episode_state_leaked" if blocked else "leaked_prior_state"), {
                "task": st.get("task"),
            }

        if kind == "two_instance":
            m1, m2 = self._MCP(), self._MCP()
            m1.reset(task_id=spec["task_a"], seed=1)
            task_a_before = m1._env.state.task_id
            m2.reset(task_id=spec["task_b"], seed=2)
            task_a_after = m1._env.state.task_id
            blocked = task_a_after == task_a_before == spec["task_a"]
            return blocked, ("instances_isolated" if blocked else "cross_instance_bleed"), {
                "m1_task_before": task_a_before, "m1_task_after": task_a_after,
                "m2_task": spec["task_b"],
            }

        if kind == "cross_task_reset":
            m = self._MCP()
            obs_a = m.reset(task_id=spec["task_a"], seed=1)
            nodes_a = {n["node_id"] for n in (obs_a.get("node_statuses") or [])}
            obs_b = m.reset(task_id=spec["task_b"], seed=1)
            nodes_b = {n["node_id"] for n in (obs_b.get("node_statuses") or [])}
            task_after = m._env.state.task_id
            # Full replacement: the new graph is loaded and the task label swaps.
            blocked = task_after == spec["task_b"] and nodes_a != nodes_b
            return blocked, ("graph_fully_replaced" if blocked else "old_graph_leaked"), {
                "n_nodes_a": len(nodes_a), "n_nodes_b": len(nodes_b),
                "task_after": task_after,
            }

        if kind == "query_recent":
            m = self._MCP()
            res = m.tool_sm_query_recent_events(**spec["args"])
            n = res.get("n_events")
            blocked = isinstance(res, dict) and (res.get("ok") in (True, False)) \
                and (n is None or (isinstance(n, int) and n >= 0))
            return blocked, ("query_bounds_handled" if blocked else "query_crashed"), {
                "ok": res.get("ok"), "n_events": n,
                "error": str(res.get("error", ""))[:100],
            }

        if kind == "reset_invalid_task":
            m = self._MCP()
            try:
                m.reset(task_id=spec["task_id"])
                return False, "invalid_task_accepted", {"raised": False}
            except (ValueError, KeyError) as e:
                return True, "invalid_task_rejected", {"raised": True,
                                                       "exc": type(e).__name__}

        if kind == "reset_seed":
            m = self._MCP()
            obs = m.reset(task_id="easy_typhoon_response", seed=spec["seed"])
            blocked = isinstance(obs, dict) and "current_day" in obs
            return blocked, ("hostile_seed_handled" if blocked else "seed_broke_reset"), {
                "seed": spec["seed"], "current_day": obs.get("current_day"),
            }

        raise ValueError(f"unknown session kind: {kind}")

    def _h_replay(self, atk: dict):
        spec = atk["payload"]
        kind = spec["kind"]
        task = spec["task"]

        if kind == "post_done":
            self.mcp.reset(task_id=task, seed=spec["seed"])
            done, guard = False, 0
            while not done and guard < 300:
                out = self.mcp.step({"action_type": "do_nothing"})
                done = out.get("done", False)
                guard += 1
            post_rewards, all_done = [], True
            for _ in range(spec["n_after"]):
                out = self.mcp.step(spec["action"])
                post_rewards.append(out.get("reward"))
                all_done = all_done and out.get("done", False)
            total_extra = sum(r for r in post_rewards if _finite(r))
            blocked = all_done and total_extra == 0.0 and all(_finite(r) for r in post_rewards)
            return blocked, ("post_done_no_reward" if blocked else "post_done_farmed"), {
                "post_done_reward_sum": round(total_extra, 6),
                "all_steps_done": all_done, "n_after": spec["n_after"],
            }

        # Budget-invariant replay probes: budget must never increase (no refund
        # exploit) and never go negative; rewards stay finite.
        actions = self._replay_actions(spec)
        self.mcp.reset(task_id=task, seed=spec.get("seed", FIXED_SEED))
        budgets, rewards, done, i = [], [], False, 0
        for act in actions:
            if done:
                break
            out = self.mcp.step(act)
            gb, _ = self._budget(out)
            budgets.append(gb)
            rewards.append(out.get("reward"))
            done = out.get("done", False)
            i += 1
        numeric_b = [b for b in budgets if _finite(b)]
        non_increasing = all(numeric_b[j] >= numeric_b[j + 1] - 1e-6
                             for j in range(len(numeric_b) - 1))
        never_negative = all(b >= 0 for b in numeric_b)
        rewards_finite = all(_finite(r) for r in rewards)
        blocked = (len(numeric_b) == len(budgets) and non_increasing
                   and never_negative and rewards_finite)
        return blocked, ("replay_bounded_no_gain" if blocked else "replay_refund_or_blowup"), {
            "kind": kind, "steps": i,
            "budget_non_increasing": non_increasing,
            "budget_never_negative": never_negative,
            "min_budget": round(min(numeric_b), 2) if numeric_b else None,
            "rewards_finite": rewards_finite,
        }

    @staticmethod
    def _replay_actions(spec: dict) -> list[dict]:
        kind = spec["kind"]
        if kind == "dup_activate_backup":
            return [{"action_type": "activate_backup_supplier",
                     "target_node_id": spec["target"],
                     "backup_supplier_id": spec["backup"]} for _ in range(spec["n"])]
        if kind == "dup_safety_stock":
            return [{"action_type": "increase_safety_stock",
                     "target_node_id": spec["node"],
                     "additional_stock_days": spec["days"]} for _ in range(spec["n"])]
        if kind == "free_alert_spam":
            return [{"action_type": "issue_supplier_alert",
                     "target_node_id": spec["node"]} for _ in range(spec["n"])]
        if kind == "replay_hedge_broke":
            return [{"action_type": "hedge_commodity", "commodity": "chips",
                     "hedge_amount_usd": 500000.0} for _ in range(spec["n"])]
        if kind == "replay_do_nothing":
            return [{"action_type": "do_nothing"} for _ in range(spec["n"])]
        if kind == "replay_reroute":
            return [{"action_type": "reroute_shipment",
                     "target_node_id": spec["port"],
                     "reroute_via": spec["via"]} for _ in range(spec["n"])]
        raise ValueError(f"unknown replay kind: {kind}")

    # =====================================================================
    _DISPATCH = {
        "mcp_step": "_h_mcp_step",
        "crisis_rag": "_h_crisis_rag",
        "mcp_tool_rag": "_h_mcp_tool_rag",
        "mcp_tool_node": "_h_mcp_tool_node",
        "wordle_schema": "_h_wordle_schema",
        "wordle_reward": "_h_wordle_reward",
        "sm_reward": "_h_sm_reward",
        "session": "_h_session",
        "replay": "_h_replay",
    }

    def execute_one(self, atk: dict) -> dict:
        handler = getattr(self, self._DISPATCH[atk["surface"]])
        t0 = time.time()
        try:
            blocked, reason, evidence = handler(atk)
            err = None
        except Exception as e:  # noqa: BLE001 — an uncaught crash IS a finding
            blocked, reason = False, "harness_or_system_exception"
            evidence = {"exception": f"{type(e).__name__}: {str(e)[:200]}",
                        "trace": traceback.format_exc().splitlines()[-3:]}
            err = str(e)[:200]
        return {
            "id": atk["id"], "category": atk["category"], "surface": atk["surface"],
            "name": atk["name"], "expect": atk["expect"], "desc": atk["desc"],
            "blocked": bool(blocked), "reason": reason, "evidence": evidence,
            "error": err, "elapsed_s": round(time.time() - t0, 4),
        }


def run() -> dict:
    """Execute the full corpus + controls and return the honest report."""
    started = time.time()
    g = Gauntlet()

    corpus = build_corpus()
    controls = build_controls()

    results = [g.execute_one(a) for a in corpus]

    # Controls are executed through the same handlers (expect == "accept").
    control_results = []
    for c in controls:
        surf = c["surface"]
        if surf == "mcp_step":
            g._reset()
            out = g.mcp.step(c["payload"])
            lar = g._lar(out)
            obs = out.get("observation")
            accepted = obs is not None and lar.get("success") is True
            control_results.append({**{k: c[k] for k in ("id", "name", "surface", "desc")},
                                    "accepted": accepted,
                                    "evidence": {"success": lar.get("success"),
                                                 "message": str(lar.get("message", ""))[:100]}})
        elif surf == "wordle_schema":
            r = g._h_wordle_schema(c)
            control_results.append({**{k: c[k] for k in ("id", "name", "surface", "desc")},
                                    "accepted": bool(r[0]), "evidence": r[2]})
        elif surf == "crisis_rag":
            r = g._h_crisis_rag(c)
            control_results.append({**{k: c[k] for k in ("id", "name", "surface", "desc")},
                                    "accepted": bool(r[0]), "evidence": r[2]})

    # Aggregate per category (attacks only).
    per_cat: dict[str, dict] = defaultdict(lambda: {"n": 0, "blocked": 0})
    breaches = []
    for r in results:
        c = per_cat[r["category"]]
        c["n"] += 1
        c["blocked"] += int(r["blocked"])
        if not r["blocked"]:
            breaches.append(r)
    for c, d in per_cat.items():
        d["block_rate_pct"] = round(100 * d["blocked"] / max(1, d["n"]), 2)

    n_total = len(results)
    n_blocked = sum(r["blocked"] for r in results)
    n_controls_accepted = sum(cr["accepted"] for cr in control_results)
    n_controls = len(control_results)

    report = {
        "gauntlet": "adversarial_gauntlet_REAL",
        "framework": "R7 — real attacks executed against the real system; "
                     "blocked == verified rejection/sanitization/demotion/bound "
                     "(single assertion per attack, no both-branches counting)",
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "started_at": started,
        "elapsed_s": round(time.time() - started, 2),
        "n_executed": n_total,
        "n_blocked": n_blocked,
        "block_rate_pct": round(100 * n_blocked / max(1, n_total), 2),
        "per_category": {c: dict(d) for c, d in sorted(per_cat.items())},
        "n_breaches": len(breaches),
        "breaches": breaches,
        "controls": {
            "n": n_controls, "n_accepted": n_controls_accepted,
            "false_positive_rate_pct": round(
                100 * (n_controls - n_controls_accepted) / max(1, n_controls), 2),
            "results": control_results,
        },
        "schema_notes": g.schema_notes,
        "limitations": [
            "OPENROUTER_API_KEY revoked: prompt-injection is tested only against "
            "data-handling surfaces (RAG retrieval, MCP tool string args, string "
            "action fields). Live-LLM instruction-override resistance is "
            "BLOCKED-ON-KEY and not claimed here.",
        ],
        "results": results,
    }
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(run()["per_category"], indent=2))
