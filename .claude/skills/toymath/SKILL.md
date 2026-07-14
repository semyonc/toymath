---
name: toymath
key: core
description: Core verified-derivation protocol and algebra/checking tactics used by every ToyMath workflow.
---

# ToyMath verified derivations

You choose the strategy; ToyMath executes narrow named tactics and checks each
transformation independently. The ledger—not prose—is the derivation artifact.

## Invocation

```bash
python toymath_cli.py <tactic> ... [--session ledger.json]
```

Expressions are quoted LaTeX strings. Use `--session` for a multi-step
derivation and feed each returned `result` verbatim into the next tactic.

Discover the progressively separated subject interfaces instead of loading a
single exhaustive command table:

```bash
python toymath_cli.py skills
python toymath_cli.py tactics --skill core
python toymath_cli.py describe expand
```

Subject workflows live under `domains/*/SKILL.md`. Read only the relevant one.

## Core workflow

- Use `apply` plus `expand` to solve equations one visible move at a time.
- Use `collect` before dividing by a symbolic coefficient.
- Use `substitute` plus `evaluate` to check candidate solutions.
- Use `rewrite` only with a registered lemma; use `lemmas` to discover names.
- Use the two named factoring tactics instead of asking for a general factor.
- Load the `equations` skill when a workflow needs a complete rational root
  result for a quadratic; candidate checks alone are not a solution record.
- Use `equal` for a query; `unknown` is not verification.
- There is deliberately no `solve`, `simplify`, or general `factor` tactic.

For CLI proofs, record a relation with `claim`, pass `--goal cN` to every
transforming tactic, then `conclude` with the ordered closing step ids. Replay
the session at the end.

## Trust rules

- A transforming record has `{ok, op, args, input, result, assumptions, check}`.
- Continue only from `agree` or `exact`; a refusal or `disagree` requires a new
  strategy. `domain-differs` requires an explicit domain qualification.
- Surface every recorded assumption. Record assumptions; do not claim they
  were established.
- Results are mechanically checked, never proved.
- Narrative comments and plots are unverified and never justify a result.
