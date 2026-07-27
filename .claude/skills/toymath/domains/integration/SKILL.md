---
name: toymath-integration
key: integration
description: Tactic-shaped indefinite integration, substitutions, by-parts, linearity, and provenance-aware assembly.
---

# Indefinite integration

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
   numerator or as a denominator factor) is handled with it. A quadratic
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
4. For algebraic massage or partial fractions, propose the new integrand with
   `integrate_rewrite`; core equality must confirm it.
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

Never type an assembled sum into core `expand`: that checks only the
expression you typed, not whether it contains the recorded pieces. Assembly
provenance is part of the mechanical certificate.

Antiderivatives carry a fresh `+ C`. The independent leg differentiates the
candidate; `1/x -> ln x` records the positive-domain assumption used by the
current table rule.
