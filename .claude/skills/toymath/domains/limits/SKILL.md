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

When no direct rule closes the limit but the body is bounded by simpler
sequences, propose the bounds yourself and squeeze. This is the standard
route for bounded products: a Wallis-type
`\lim_{n \to \infty} \prod_{k=1}^{n} \frac{2k-1}{2k}` has no table rule,
but `0` and `\frac{1}{\sqrt{2n+1}}` bound it (root-power decay closes the
upper bound). Close each bound's `\lim` as its own step first, then cite
both step ids in `limit_squeeze`. Do not chase factorial or binomial
rewrites of a product body — those forms have no grammar here.

Never type the final linear combination into core `expand`; use assembly so
the cited branch provenance remains replayable. One-sided and infinite
approach points are supported. Non-converged approach sampling is oracle
ignorance, not a counterexample.

If the body contains an ellipsis or a finite sum/product, load
`finite_operators` first, then return to this skill for the outer limit.
