---
name: toymath-limits
key: limits
description: Limits by verified rewrite, continuity, tables, l'Hopital, linearity, assembly, squeeze, and one-sided combination.
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
- `limit_from_sides` closes a two-sided limit from recorded left and right
  one-sided limit steps of the same body that reached the same value. Cite the
  two step ids; each side may itself be closed by any tactic, including a
  one-sided squeeze.
- `limit_evaluate` certifies a limit VALUE you propose, checked by the
  independent directional approach oracle. It is the route for a
  composite no named rule spells — the standard example is an endpoint
  behaviour you can see, `\lim_{x \to \pi/2^{-}} \arctan(\frac{a}{b}
  \tan x)`. Its check is sampled, so reach for the named rules first
  when one binds (they carry exact or stronger evidence), and propose a
  spelling that holds for EVERY parameter sign — here
  `\frac{\pi}{2|ab|}`-style absolute values, not a sign-assuming form;
  a wrong proposal is refused with the oracle's witness.

When no direct rule closes the limit but the body is bounded by simpler
sequences, propose the bounds yourself and squeeze. This is the standard
route for bounded products: a Wallis-type
`\lim_{n \to \infty} \prod_{k=1}^{n} \frac{2k-1}{2k}` has no table rule,
but `0` and `\frac{1}{\sqrt{2n+1}}` bound it (root-power decay closes the
upper bound). Close each bound's `\lim` as its own step first, then cite
both step ids in `limit_squeeze`. Do not chase factorial or binomial
rewrites of a product body — those forms have no grammar here.

If a two-sided squeeze ordering fails because the natural bounds flip sign
with the variable (a witness near `0^-` while the ordering holds on the
right), either use absolute-value bounds — `(-|x|)` and `(|x|)`,
parenthesized — or split by direction: squeeze `\lim_{x \to a^{+}}` and
`\lim_{x \to a^{-}}` as their own steps, then cite both step ids in
`limit_from_sides`. A leading `-` in any limit body or bound must be
parenthesized: `\lim_{x \to 0} (-x)`, never `\lim_{x \to 0} -x`.

Bare `|...|` bars accept a whole expression, so `|\cos\frac{1}{x}|` and
`\sqrt{|\cos\frac{1}{x}|}` are writable bounds. Absolute value, `\lfloor
\rfloor`, and `\lceil \rceil` are opaque bracket operators: algebra keeps
them whole while the oracle evaluates them, so they are safe inside bounds
but never simplify away.

Never type the final linear combination into core `expand`; use assembly so
the cited branch provenance remains replayable. One-sided and infinite
approach points are supported. Non-converged approach sampling is oracle
ignorance, not a counterexample.

If the body contains an ellipsis or a finite sum/product, load
`finite_operators` first, then return to this skill for the outer limit.
