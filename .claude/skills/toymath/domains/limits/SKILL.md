---
name: toymath-limits
key: limits
description: Limits by verified rewrite, continuity, tables, l'Hopital, linearity, assembly, and squeeze.
---

# Limits

Use one tactic per mathematical justification:

- `limit_rewrite` carries the binder only after core equality confirms the
  proposed body.
- `limit_substitute` is continuity substitution at a finite point.
- `limit_table` handles constants, named zero limits, rational behavior at
  infinity, geometric decay, and supported root-power decay.
- `limit_lhopital` performs one step only after probes observe an indeterminate
  quotient and records the theorem premises.
- `limit_linearity` splits a top-level sum. Close every piece and cite the
  ordered value-step ids in `limit_assemble`.
- `limit_squeeze` requires recorded lower- and upper-bound limit steps reaching
  the same value. Pass the full target limit, lower body, upper body, and the
  two source step ids. Ordering is spot-checked and recorded as an assumption.

Never type the final linear combination into core `expand`; use assembly so
the cited branch provenance remains replayable. One-sided and infinite
approach points are supported. Non-converged approach sampling is oracle
ignorance, not a counterexample.

If the body contains an ellipsis or a finite sum/product, load
`finite_operators` first, then return to this skill for the outer limit.
