---
name: toymath-integration
key: integration
description: Tactic-shaped indefinite and definite integration - substitutions, by-parts, linearity, provenance-aware assembly.
---

# Integration

There is no autonomous integrate tactic. Pick one narrow move at a time and
continue until no integral remains. Feed every returned `result` string
verbatim into the next call: provenance linkage is structural, and retyping
a result (even a harmless-looking reformatting) disconnects the checked
chain that lets the cell accept your final value.

## Order of attack

1. Try `integrate_power_rule` for powers with integer or rational
   literal exponents (`x^{1/2}` works directly; a fractional exponent
   records `x > 0`), or `integrate_table` for basic functions of the
   bare variable (sin, cos, e^x, sinh, cosh), constant multiples, fully
   constant integrands, `1/x` (gives `\ln x`, records `x > 0`), and the
   arctan family `1/(a x^2 + b)` with positive literals `a`, `b` — a
   constant numerator (even an irrational one like `\sqrt{5}`, in the
   numerator or as a denominator factor) is handled with it, and any
   var-free constant factor peels automatically, symbolic ones included
   (`\frac{1}{ab(v^2+1)}` closes as `\frac{1}{ab}\arctan(v)` directly —
   no rewrite needed to pull the constant out first). Symbolic
   coefficients ON the variable (`1/(a^2u^2+b^2)`) still need your
   substitution (`v = \frac{a}{b}u`). A quadratic
   denominator with a linear term is one completing-the-square
   `integrate_rewrite` plus shift substitution away from this rule.
2. Use `integrate_substitute` when you can propose `u`, a new variable, and
   the rewritten integrand. ToyMath checks `f(u(x))u'(x)` against the
   original. The new integrand is an expression in the new variable ONLY -
   never append the differential (propose `\cos(u)`, not `\cos(u) du`).
   After the u-space integral closes, map back with core `substitute`
   (u := the expression in x).
3. Use `integrate_by_parts` for one proposed `u·dv` split. Continue from its
   `remaining_integral` handle.
4. For algebraic massage, propose the new integrand with
   `integrate_rewrite`; core equality must confirm it. For PARTIAL
   FRACTIONS, do not propose the finished decomposition — a rejected guess
   records nothing, and guessing the coefficients one sign at a time is how
   a run exhausts its turns. Load the `equations` skill and derive them:
   clear the denominators, `match_coefficients` in the variable, solve the
   small system, then propose the decomposition once with
   `integrate_rewrite` knowing it will pass.
5. Split a top-level sum with `integrate_linearity`, solve every returned
   integral independently, pin per-piece constants to zero with core
   `substitute` (C := 0), then call `integrate_assemble` citing the PINNED
   per-piece step ids (the C := 0 results), in the exact piece order the
   linearity step returned.

## Roots of the variable inside a fraction

Pure power terms like `x^{1/2}` integrate directly with
`integrate_power_rule`. When roots appear inside a fraction the standard
move is the root substitution `u = x^{1/n}` with `n` the least common
multiple of the exponent denominators. Worked shape (mixed roots):

- `\int \frac{dx}{x^{1/2}+x^{1/3}}`: substitute `u = x^{1/6}` proposing the
  new integrand `\frac{6u^3}{u+1}`; divide out with `integrate_rewrite`
  (`6u^2-6u+6-\frac{6}{u+1}`); split with `integrate_linearity`; close the
  polynomial pieces with `integrate_power_rule` and the last piece by
  substituting `v = u+1` into the table's `1/v` rule; pin constants,
  assemble, then `substitute` u := `x^{1/6}` back. Core `expand` folds the
  resulting fractional powers (records `x > 0`).

## Rational functions of sin and cos

The Weierstrass substitution closes `\int \frac{dx}{a \sin x + b \cos x + c}`
end to end. Worked shape:

- `\int \frac{dx}{2\sin x - \cos x + 5}`: substitute `t = \tan(\frac{x}{2})`
  proposing the rational integrand (`\frac{1}{3t^2+2t+2}` here); complete
  the square with `integrate_rewrite`; `integrate_substitute` the shifted
  variable (`u = t + \frac{1}{3}`); the table's arctan rule closes
  `\frac{1}{3u^2+\frac{5}{3}}` in one step; back-substitute with core
  `substitute` in reverse order. Do NOT hand-simplify the arctan argument
  between steps — feed each recorded result forward exactly.

## Definite integrals

A definite integral `\int_a^b f \, dx` is closed in exactly two moves, and
no other route is accepted: derive the ANTIDERIVATIVE first with the
indefinite tactics above (the integrand alone, no bounds — the bounded
spelling is refused there), then call `integrate_definite` with the whole
definite integral and the ledger step id of the recorded antiderivative.
It substitutes both bounds itself, the `+ C` cancels in a follow-up core
`expand`, and the independent check re-integrates the integrand
numerically. Substituting the bounds by hand records steps about the
antiderivative, not about the definite integral — the cell will refuse
that chain as its value.

If the antiderivative's SPELLING is singular at a bound (the classic
`\arctan(\frac{a}{b}\tan x)` at `x = \pi/2`), do not substitute there:
the honest endpoint value is a one-sided limit. Load the `limits` skill
and record `\lim_{x \to b^{-}} (F)` — the FULL antiderivative, its
`+ C` included, approaching the bound from inside the interval
(`a^{+}` for the lower bound); `limit_evaluate` certifies your proposed
value when no named rule reaches it, and a value that holds for every
parameter sign (absolute-value spelling, `\frac{\pi}{2|ab|}+C`) needs
no case split. Then cite both recorded steps:
`integrate_definite EXPR VAR ANTIDERIVATIVE_STEP --upper-limit-step
LIMIT_STEP` (or `--lower-limit-step`). The limit step must be about
exactly that antiderivative or it is refused.

Continuity of the integrand on `[a, b]` is recorded as an assumption; a
pole strictly inside the bounds makes the check refuse (the integral is
improper there). Truly improper integrals — infinite bounds or an
integrand singularity inside or at the interval — still have no closing
tactic: report the verified stopping point with the open outcome
instead of forcing a value.

Never type an assembled sum into core `expand`: that checks only the
expression you typed, not whether it contains the recorded pieces. Assembly
provenance is part of the mechanical certificate.

Antiderivatives carry a fresh `+ C`. The independent leg differentiates the
candidate; `1/x -> ln x` records the positive-domain assumption used by the
current table rule.
