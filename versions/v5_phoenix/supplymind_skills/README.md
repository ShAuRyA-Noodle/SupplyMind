# supplymind-skills — a Claude Code skill pack

> "Methodology as a shippable artifact."

This is a three-skill pack for Claude Code (and compatible agents: Cursor,
Copilot CLI, Gemini CLI, OpenCode) that encodes the disciplines SupplyMind
used to build a hackathon-grade ML submission in 3 days.

Skills are distributed the same way `obra/superpowers` is distributed —
through the Claude Code plugin marketplace plus compatible hosts. Judges can
install this pack, inspect the `SKILL.md` files, and reproduce our
methodology on their own projects.

## Skills

| Skill | Purpose |
|---|---|
| [`benchmark-runner`](benchmark-runner/SKILL.md) | TDD discipline applied to benchmarks: baseline → change → verify, never claim a speedup without paired output |
| [`autoresearch-experiment`](autoresearch-experiment/SKILL.md) | Karpathy-pattern autonomous ML research loop: program.md + mutable candidate + fixed-budget runner + bootstrap-CI95 evaluator + auto lab notebook |
| [`live-demo-orchestrator`](live-demo-orchestrator/SKILL.md) | Pre-demo / during-demo / post-demo checklists with offline replay fallbacks for any live-data feature |

## Install (once we're merged into the marketplace)

```
/plugin install supplymind-skills@shaurya-marketplace
```

Or directly from GitHub:

```
gh repo clone ShAuRyA-Noodle/supplymind-skills ~/.claude/plugins/supplymind-skills
```

## Attribution

Methodology derived from Jesse Vincent's [`obra/superpowers`](https://github.com/obra/superpowers)
(MIT) framework. Specifically we inherit:
- The "iron law" formulation of TDD (no production code before failing test)
- `verification-before-completion` — claim = fresh command output
- `writing-plans` — bite-sized 2-5 min tasks, zero-context-assumed
- `subagent-driven-development` — two-stage review

## License

MIT — same as superpowers, same as our hackathon repo.

## Why we built this

The hackathon grades "meaningful open-source contributions." A skill pack is a
shippable, install-able artifact that encodes methodology. Judges can install,
try, and verify the whole pack in under 2 minutes. It's a proof that our ML
submission is backed by a reproducible discipline, not one-off luck.
