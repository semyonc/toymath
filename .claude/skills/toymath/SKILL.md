---
name: toymath
description: Verified-derivation math primitives. Use when the user wants a machine-checked, step-by-step derivation (solve an equation showing every move, expand/collect polynomials, differentiate, verify two expressions are equal). You plan the strategy; each toymath call executes one narrow step and independently spot-checks it numerically. The session ledger is the proof artifact.
---

# ToyMath verified derivations

You are the strategist; toymath is the mechanical checker. Never do algebra
in your head when a primitive can do it verified. Every call prints one JSON
record; `check.status: "agree"` means an independent numeric oracle confirmed
the step at random sample points.

## Invocation

```bash
python toymath_cli.py <command> ... [--session ledger.json]
```

Run from the repo root (or set `PYTHONPATH` to `engine/`). Expressions are
LaTeX strings — quote them. Use `--session` for any multi-step derivation:
it appends each verified step to a replayable ledger.

## Commands

| Command | Example | Meaning |
|---|---|---|
| `apply EQ OP ARG` | `apply "2x + 3 = 7" - 3` | op ∈ `+ - * / ^` to both sides. Multiplying or dividing an equation by a symbolic expression records the assumption `arg ≠ 0`. Inequalities (`<`, `>`, `\le`, `\ge`, …) work too: the relation flips under `*`/`/` by a negative constant, and symbolic-sign factors are refused |
| `expand EXPR` | `expand "(x+1)(x-2)"` | distribute products/powers; also simplifies each side of an equation. Outside the rational fragment it canonicalizes over opaque atoms: `2\sin x + 3\sin x` → `5 \sin x`, `(\sin x)^2 - \sin^2 x` → `0` (both power spellings are one atom), and it assembles tactic results like `x(-\cos x) - (-\sin x + C)`. FRACTIONAL powers of a plain variable fold (records `x > 0`): `(x^{1/6})^4` → `x^{2/3}`, `\frac{(x^{1/6})^4}{x x^{1/6}+x}` → `\frac{1}{x^{1/2}+x^{1/3}}`; fractional powers of composite bases (`(x+1)^{1/2}`) and negative fractional exponents stay opaque — verify those with `equal` or substitute `u = x^{1/n}` |
| `collect EXPR VAR` | `collect "ax + 2x" x` | group by powers of VAR |
| `substitute EXPR VAR VAL` | `substitute "x^2+1" x 3` | replace a variable |
| `evaluate EXPR` | `evaluate "2(2)+3 = 7"` | exact arithmetic; on an equation reports `holds: true/false` |
| `diff EXPR VAR` | `diff "x \sin x" x` | derivative, checked by central differences |
| `rewrite EXPR LEMMA [--direction backward]` | `rewrite "x^2-y^2" diff_squares` | apply a registered identity at the root, or at the first matching subterm (reported as `at`). Power terms also match perfect-power constants and monomials: `x^2-4` → `(x+2)(x-2)`, `4x^2-9` → `(2x+3)(2x-3)`, `x^3-8` via `diff_cubes` (binding reported as `numeric`) |
| `integrate_power_rule EXPR VAR` | `integrate_power_rule "3x^2+2x" x` | term-by-term power rule; accepts `\int ... dx` wrappers; refuses the `1/x` case (that is the table's log rule) |
| `integrate_table EXPR VAR` | `integrate_table "2\cos x" x` | sin, cos, e^x, sinh, cosh, `1/x`→`ln x` (records `x > 0`); closed under sums and constant factors |
| `integrate_by_parts EXPR VAR U DV` | `integrate_by_parts "x \sin x" x "x" "\sin x"` | verifies `u·dv` equals the integrand, returns `u v - \int v du`; feed `remaining_integral` to the next tactic |
| `integrate_substitute EXPR VAR U_EXPR U_VAR NEW_INTEGRAND` | `integrate_substitute "2x \cos(x^2)" x "x^2" u "\cos(u)"` | u-substitution: you supply u and the integrand rewritten in u; toymath verifies `f(u(x))·u'(x)` equals the integrand and returns `\int f(u) du`; back-substitute with `substitute` after integrating |
| `integrate_rewrite EXPR VAR NEW_INTEGRAND` | `integrate_rewrite "\frac{x^2}{(1-x^2)^3}" x "<partial fractions>"` | congruence under the integral sign: you propose an equivalent integrand (partial fractions, algebraic massage); toymath verifies the two integrands are mechanically equal, then rewrites the integral |
| `integrate_linearity EXPR VAR` | `integrate_linearity "\int (x + \sin x) \, dx" x` | exact sum rule: split the integral of a top-level sum into a signed sum of integrals; attack each piece separately, assemble with `expand` |
| `factor_gcd EXPR` | `factor_gcd "6x^2+9x=3"` | pull out common factors on applicable expression/relation sides → `3x(2x+3)=3` |
| `factor_quadratic EXPR VAR` | `factor_quadratic "x^2+6x+9=4" x` | rational-root factorization on applicable expression/relation sides → `(x+3)^2=4`; reports roots; refuses irrational cases |
| `equal E1 E2` | `equal "(x+1)^2" "x^2+2x+1"` | verdict yes / no / unknown. A `no` with method `domain mismatch` means the sides agree in value but one is defined where the other is not (`\ln(x^2)` vs `2\ln x`) — equality may hold on a restricted domain; a `yes` may carry a `note` that it was checked only on the common domain |
| `lemmas` | | list rewrite lemmas |
| `show --session f [--format md]` | | render the ledger (Markdown with `--format md`) |
| `replay --session f` | | re-verify every recorded step |

## How to solve an equation (there is deliberately no `solve`)

Drive it move by move; the visible chain of steps IS the product:

```bash
python toymath_cli.py apply "2x + 3 = 7" - 3 --session d.json
python toymath_cli.py expand "{2}x+{3} - {3} = {7} - {3}" --session d.json   # -> 2x = 4
python toymath_cli.py apply "{2}x = {4}" / 2 --session d.json
python toymath_cli.py expand "\frac{{2}x}{{2}} = \frac{{4}}{{2}}" --session d.json  # -> x = 2
python toymath_cli.py replay --session d.json
```

Feed each step's `result` verbatim as the next input (the ledger flags
discontinuities). Check candidate solutions with `substitute` + `evaluate`.

## How to integrate (there is deliberately no autonomous `integrate`)

Pick the tactic; chain until no integral remains; verify the assembled
answer by differentiating it:

```bash
python toymath_cli.py integrate_by_parts "\int x \sin x \, dx" x "x" "\sin x"
python toymath_cli.py integrate_table "\int \left(-\cos\left(x\right)\right) \, d x" x
# assemble: x(-cos x) - (-sin x + C)  =>  -x cos x + sin x + C, then verify:
python toymath_cli.py diff "-x \cos(x) + \sin(x)" x
python toymath_cli.py equal "<that derivative>" "x \sin x"    # -> yes
```

Antiderivatives carry `+ C` (a fresh symbol, reported as `constant`); the
oracle checks them by central-difference differentiation against the
integrand.

For rational integrands beyond a single power in the denominator
(e.g. `\int \frac{x^2}{(1-x^2)^3} dx`): propose the partial-fraction
decomposition yourself and let `integrate_rewrite` verify it, split with
`integrate_linearity`, then per piece use `integrate_substitute`
(u = the linear factor) + `integrate_power_rule`/`integrate_table` +
`substitute` back. Pin each piece's constant to 0 with `substitute`
(`C := 0`) so the pieces assemble cleanly, and add one `+ C` at the end.

## Matrices and vectors (literal, phase 1)

`\begin{pmatrix} a & b \\ c & d \end{pmatrix}` and `\begin{matrix}…\end{matrix}`
parse (normalized to plain-TeX `\pmatrix{a & b \cr c & d}`), round-trip, and
`substitute` reaches the cells. `expand`/`collect` do scalar-linear algebra
over matrix terms: `M + M → 2M`, `2\vec v + 3\vec v → 5\vec v`, `xA − Ax → 0`
(scalars commute). **Ordered products never commute**: a product of two or
more matrix-valued factors is treated as one opaque word, so `AB − BA` does
NOT collapse and `(A+B)^2` stays unexpanded. For literal matrices the oracle
does real matrix arithmetic — it verifies these steps at sample points and
`equal` can disprove commutation (`equal "AB" "BA"` → no, with both evaluated
matrices as witness). Honest refusals: `evaluate` rejects matrix-valued
expressions, division by a matrix stays opaque (`A^{-1}A` is not `1`), and
there is no `mat_mul`/`det` yet — do not multiply matrices by hand; say the
tactic does not exist.

## Rules

- If `check.status` is `"disagree"`, the step is wrong — do not use its
  result; report the discrepancy.
- If `check.status` is `"domain-differs"`, the result changed where the
  expression is defined (a definedness witness point is reported) — treat
  it as valid only with an explicit domain assumption, and surface that.
- Surface recorded `assumptions` (e.g. `y ≠ 0`) to the user; the derivation
  is conditional on them.
- `equal` verdict `unknown` means unverified — say so honestly.
- Results are "mechanically checked", not "proved" — the oracle rules out
  hallucinated algebra, not implementation bugs.
