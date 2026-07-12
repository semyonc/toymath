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
| `limit_rewrite EXPR NEW_BODY` | `limit_rewrite "\lim_{x \to 1} \frac{x^2-1}{x-1}" "x+1"` | replace the body only after `equal` verifies the proposal |
| `limit_substitute EXPR` | `limit_substitute "\lim_{x \to 2}(x^2+1)"` | continuity substitution at a finite point; records continuity and requires converged approach samples |
| `limit_table EXPR` | `limit_table "\lim_{x \to 0}\frac{\sin x}{x}"` | standard zero-limit/constant rules, finite rational leading-coefficient limits at infinity, and geometric decay at `+\infty` (`r^n` with numeric `0<r<1`, or division by `s^n` with numeric `s>1`, times constants — rewrite `e^{-cn}` into that form first) |
| `limit_lhopital EXPR` | `limit_lhopital "\lim_{x \to 0}\frac{e^x-1}{x}"` | one conditional l'Hopital step after the oracle observes `0/0` or `infinity/infinity`; records theorem premises |
| `limit_linearity EXPR` | `limit_linearity "\lim_{x \to 0}(\frac{\sin x}{x}+x^2)"` | split a top-level sum and record that every piece limit exists |
| `limit_assemble EXPR VALUES...` | `limit_assemble "<original limit>" 1 0` | CLI form: independently check ordered piece values and signed assembly; in do! pass the linearity step id and value-step ids so provenance is replayable |
| `sum_from_ellipsis EXPR SUM_FORM` | `sum_from_ellipsis "\frac{1}{1 \cdot 2}+\frac{1}{2 \cdot 3}+\ldots+\frac{1}{n(n+1)}" "\sum_{k=1}^{n} \frac{1}{k(k+1)}"` | interpret an ellipsis sum (optionally inside a `\lim`) as an explicit finite `\sum`: every displayed term is checked against the proposed summand at its index (≥ 2 leading terms required); the pattern continuation is recorded as an assumption |
| `sum_rewrite EXPR NEW_SUMMAND` | `sum_rewrite "\sum_{k=1}^{n} \frac{1}{k(k+1)}" "\frac{1}{k} - \frac{1}{k+1}"` | replace the summand (optionally inside a `\lim`) only after `equal` verifies the proposal |
| `sum_telescope EXPR TERM` | `sum_telescope "\sum_{k=1}^{n} \frac{1}{k(k+1)}" "\frac{1}{k}"` | collapse `\sum_{k=a}^{b} (f(k)-f(k+1))` to `f(a)-f(b+1)` for your proposed `f`; `equal` gates the summand and a literal finite-sum evaluation independently confirms the closed form |
| `integrate_power_rule EXPR VAR` | `integrate_power_rule "3x^2+2x" x` | term-by-term power rule; accepts `\int ... dx` wrappers; refuses the `1/x` case (that is the table's log rule) |
| `integrate_table EXPR VAR` | `integrate_table "2\cos x" x` | sin, cos, e^x, sinh, cosh, `1/x`→`ln x` (records `x > 0`); closed under sums and constant factors |
| `integrate_by_parts EXPR VAR U DV` | `integrate_by_parts "x \sin x" x "x" "\sin x"` | verifies `u·dv` equals the integrand, returns `u v - \int v du`; feed `remaining_integral` to the next tactic |
| `integrate_substitute EXPR VAR U_EXPR U_VAR NEW_INTEGRAND` | `integrate_substitute "2x \cos(x^2)" x "x^2" u "\cos(u)"` | u-substitution: you supply u and the integrand rewritten in u; toymath verifies `f(u(x))·u'(x)` equals the integrand and returns `\int f(u) du`; back-substitute with `substitute` after integrating |
| `integrate_rewrite EXPR VAR NEW_INTEGRAND` | `integrate_rewrite "\frac{x^2}{(1-x^2)^3}" x "<partial fractions>"` | congruence under the integral sign: you propose an equivalent integrand (partial fractions, algebraic massage); toymath verifies the two integrands are mechanically equal, then rewrites the integral |
| `integrate_linearity EXPR VAR` | `integrate_linearity "\int (x + \sin x) \, dx" x` | exact sum rule: split the integral of a top-level sum into a signed sum of integrals; attack each piece separately |
| `integrate_assemble LINEARITY_STEP PIECE_STEPS` | `integrate_assemble s4 [s8,s12]` | do!-only provenance move: retrieve recorded piece results in linearity order, verify every derivative, apply the signs, and add one fresh constant |
| `factor_gcd EXPR` | `factor_gcd "6x^2+9x=3"` | pull out common factors on applicable expression/relation sides → `3x(2x+3)=3` |
| `factor_quadratic EXPR VAR` | `factor_quadratic "x^2+6x+9=4" x` | rational-root factorization on applicable expression/relation sides → `(x+3)^2=4`; reports roots; refuses irrational cases |
| `equal E1 E2` | `equal "(x+1)^2" "x^2+2x+1"` | verdict yes / no / unknown. A `no` with method `domain mismatch` means the sides agree in value but one is defined where the other is not (`\ln(x^2)` vs `2\ln x`) — equality may hold on a restricted domain; a `yes` may carry a `note` that it was checked only on the common domain |
| `lemmas` | | list rewrite lemmas |
| `claim STATEMENT --session f` | `claim "\lim_{n \to \infty} 1/n = 0" --session f` | record an OPEN root claim (use `--parent c1` for a subclaim) |
| `conclude CLAIM STEPS... --session f` | `conclude c1 s1 s2 --session f` | mechanically close a claim from an ordered, goal-owned chain; verdict is established or conditional |
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

## How to establish a claim

Record a top-level relation as the claim first, then tag every transforming
CLI step with its goal:

```bash
python toymath_cli.py claim "\lim_{n \to \infty} \frac{1}{n} = 0" --session p.json
python toymath_cli.py limit_table "\lim_{n \to \infty} \frac{1}{n}" --goal c1 --session p.json
python toymath_cli.py conclude c1 s1 --session p.json
python toymath_cli.py replay --session p.json
```

`conclude` accepts only `agree`/`exact` steps owned by that goal, verifies
chain continuity (or explicit assembly provenance), and mechanically checks
that the endpoint closes the claim. A relation endpoint must itself hold, so
reformatting `x=2` cannot establish `x=2`. An OPEN claim is visibly unfinished
even if an unrelated step reaches the same value. In the notebook, `prove!`
records the root claim automatically and suppresses a chainable final value
until the agent calls `conclude` successfully.

## How to integrate (there is deliberately no autonomous `integrate`)

Pick the tactic and chain until no integral remains. For a by-parts result,
verify the assembled answer by differentiating it:

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
(`C := 0`) so the pieces assemble cleanly. In do! mode, finish with
`integrate_assemble(linearity_step, antiderivative_steps)`, passing the
recorded step ids in the exact order returned by `integrate_linearity`.
It applies the recorded signs and adds the single `+ C`; do not type the
sum into `expand`, because that checks only the expression you typed, not
whether it contains the right piece results.

## How to take a limit (there is deliberately no autonomous `limit`)

Use one narrow move at a time. Algebraic cancellation belongs in
`limit_rewrite`; continuity belongs in `limit_substitute`; standard forms in
`limit_table`; and l'Hopital applies only one conditional derivative-quotient
step. One-sided `a^+`/`a^-` and infinite approach points are supported. For a
sum, call `limit_linearity`, solve each returned limit, and in do! finish with
`limit_assemble(linearity_step, value_steps)`. Do not type the final sum into
`expand`: that would verify only the typed arithmetic, not which branch values
were used. Approach sampling uses Richardson extrapolation; non-convergence is
oracle ignorance and must never be presented as a counterexample.

## Ellipsis sums and telescoping (series limits)

An ellipsis (`\ldots`) has no mechanical semantics — every primitive rejects
it except `sum_from_ellipsis`, which turns the displayed pattern into an
explicit finite `\sum` (recording the continuation as an assumption). Never
split an ellipsis sum with `limit_linearity`: the number of terms depends on
the variable. The chain for `\lim_{n \to \infty} (t_1 + t_2 + \ldots + t_n)`:

```bash
python toymath_cli.py sum_from_ellipsis "\lim_{n \to \infty}\left[\frac{1}{1 \cdot 2}+\frac{1}{2 \cdot 3}+\ldots+\frac{1}{n(n+1)}\right]" "\sum_{k=1}^{n} \frac{1}{k(k+1)}"
python toymath_cli.py sum_telescope "\lim_{n \to \infty} \sum_{k=1}^{n} \frac{1}{k(k+1)}" "\frac{1}{k}"   # -> \lim_{n \to \infty} \frac{n}{n+1}
python toymath_cli.py limit_table "\lim_{n \to \infty} \frac{n}{n+1}"   # -> 1
```

`sum_rewrite` reshapes a summand in place (e.g. explicit partial fractions)
when the telescoping `f` is not immediately visible.

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
