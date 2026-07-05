# Agent-Scoped Primitives

Prototype of the verified-derivation direction: the agent decides strategy,
toymath executes and verifies each step, the ledger is the artifact. A
capable model already solves most math problems on its own; what it cannot
do alone is prove each step is valid. These primitives give it that: every
move is a narrow, named, mechanically checked transformation, and the chain
of steps is the product. New code lives beside the old engine and does not
touch the fixed-point rewrite rules.

## Modules

| Module | Purpose |
|---|---|
| `engine/polyrat.py` | Canonical core for the rational fragment: sparse `Poly` (`{monomial: Fraction}`), `RatFunc` with cancellation in the constructor (monomial GCD, univariate Euclidean GCD, sign/content normalization), `to_ratfunc` / `ratfunc_to_notation` |
| `engine/primitives.py` | The seven primitives + `equal_exprs` checker + numeric oracle |
| `engine/ledger.py` | Step ledger: JSON persistence, assumption accumulation, replay verification, chain-continuity detection |
| `toymath_cli.py` | Agent-facing CLI; one deterministic JSON object per call |
| `engine/unittests_primitives.py` | 62 tests |

## The primitives

Every transforming primitive returns
`{ok, op, args, input, result, assumptions, check}` where `check` is the
independent numeric spot-check (`agree` / `disagree` / `skipped`).

1. **substitute(expr, var, value)** — notation-graph replacement, wraps
   values in parens so juxtaposition never corrupts (`2x`, x:=3 → `2(3)`,
   never `23`). Checked by evaluating input-with-binding vs output.
2. **apply_both_sides(eq, op, arg)**, op ∈ {`+`,`-`,`*`,`/`,`^`} — the spine
   of equation solving. Division by a non-constant records the assumption
   `arg ≠ 0` in the step (record, don't prove). Refuses `*0` and `/0`.
   Each new side is oracle-checked against `op(old side, arg)`.
3. **expand(expr)** — `to_ratfunc` → canonical, degree-ordered output.
   Accepts equations (expands each side), which doubles as the
   "simplify both sides" move after `apply_both_sides`.
4. **collect(expr, var)** — polynomial grouped by powers of `var`.
   Accepts equations.
5. **evaluate(expr)** — exact `Fraction` arithmetic when no free variables
   remain; on an equation reports `holds: true/false` (solution checking).
   Falls back to float for non-rational constants.
6. **differentiate(expr, var)** — exact `RatFunc` derivative on the rational
   fragment; ~20 mechanical rules (trig, exp, ln, sqrt, chain/product/
   quotient, `f^n(u)`) outside it. Verified by central differences.
7. **rewrite(expr, lemma, direction)** — registered equality lemmas
   (`diff_squares`, `square_of_sum`, `cube_of_sum`, `diff_cubes`, …) matched
   at the root; the lemma library is the extensibility point.

**equal_exprs(e1, e2)** → verdict `yes`/`no`/`unknown`. Canonical `RatFunc`
comparison decides the rational fragment; outside it the numeric oracle
answers probabilistically (Schwartz–Zippel style) and says so in `method`.
Equations compare per-side.

Deliberately absent: `solve` and `simplify` (either would collapse the
derivation into one opaque move — the visible steps are the product),
autonomous `integrate` (tactics the agent picks are the honest shape), and
general `factor` (specialized named factorings instead).

## Two independent trust legs

1. Each primitive is a narrow, tested implementation.
2. The numeric oracle shares nothing with the symbolic code: random rational
   sample points, assumption-aware (skips points where an assumed-nonzero
   divisor vanishes), fixed seed for reproducibility.

A bug in either leg is caught by the other. During development the oracle
caught three real bugs (function symbols treated as polynomial variables,
`\left(...\right)` vgroups unevaluated, wrong function-argument binding in
products) — the design works.

## The ledger

```
s1#dff1198 [ok] apply_both_sides: 2x + 3 = 7  ==>  {2}x+{3} - {3} = {7} - {3}
s2#43bdac6 [ok] expand: {2}x+{3} - {3} = {7} - {3}  ==>  {2}x = {4}
...
```

- steps carry id, content hash, op, args, result, assumptions, check;
- assumptions accumulate in the header — the derivation is honestly
  conditional;
- `continues` flags whether a step's input matches the previous result
  (semantic comparison), so branches are visible;
- `replay` re-runs every step and confirms recorded results — cheap
  verification of the whole derivation.

## Known limitations (v1)

- `rewrite` matches at the root only; the agent extracts subterms.
- `collect` handles polynomials, not rational functions.
- Multivariate polynomial GCD is monomial-level only (univariate is full
  Euclidean).
- `apply_both_sides` supports `=` only (inequalities need direction logic).
- Function-argument binding in products uses the convention: a parenthesized
  group right after the function is the whole argument; otherwise the run of
  tight factors up to the next function/group binds (`\sin 2x` = sin(2x),
  `\cos(x) y` = cos(x)·y).
