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

## Hyperbolic reciprocals

There is no sech/csch vocabulary and no table rule for
`\frac{1}{\cosh x}` — these close by substitution, one move each:

- `\int \frac{dx}{\cosh x}`: substitute `u = \sinh x` proposing
  `\frac{1}{1+u^{2}}` (since `\cosh^2 = 1+\sinh^2`); the arctan table
  rule closes it; back-substitute.
- `\int \frac{dx}{\cosh^{2} x}`: substitute `u = \tanh x` proposing the
  constant `1`; back-substitute gives `\tanh x + C`.
- Higher powers `\frac{1}{\cosh^{m} x}` for literal `m`: peel one
  `\frac{1}{\cosh^{2} x}` factor and use `integrate_by_parts`, or state
  the family's recurrence with `integrate_reduction`.

`\cosh(x)^2` and `\cosh^{2} x` are the same expression (the power
applies to the application); the continental `\operatorname{ch}`,
`\operatorname{sh}`, `\operatorname{th}` spellings are the
`\cosh`/`\sinh`/`\tanh` heads.

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
improper there).

## Improper integrals (endpoint singularity or an infinite bound)

An integrand singular at ONE finite bound
(`\int_0^1 \frac{dx}{(2-x)\sqrt{1-x}}` blows up at `x = 1`), or an
integral with ONE infinite bound (`\int_0^{+\infty} \frac{dx}{\cosh
x}`), closes by its definition — the limit of the truncated
integrals — in four moves:

1. Derive the ANTIDERIVATIVE of the integrand with the indefinite
   tactics above.
2. Evaluate the TRUNCATED integral: replace the improper bound with a
   fresh variable (`\int_0^t`, keeping the other bound verbatim) and
   call `integrate_definite` citing the antiderivative step. The
   symbolic bound is fine — the check samples it.
3. Load the `limits` skill and record the limit of that step's FULL
   result at the replaced bound: one-sided from inside for a singular
   bound (`t \to 1^{-}` for an upper bound, `t \to 0^{+}` for a
   lower), plain `t \to \infty` (or `t \to -\infty`) for an infinite
   one. Spell the limit body exactly as the truncated step returned
   it; `limit_evaluate` certifies your proposed value when no named
   rule reaches it.
4. Call `integrate_improper` with the ORIGINAL integral, citing the
   truncated step and the limit step. It records the definitional
   reading and the half-open continuity assumption, and an independent
   truncation-ladder quadrature re-derives the value from the
   integrand alone.

If the limit in move 3 does not exist, nothing can certify it and the
integral has no finite value in evidence: report the verified stopping
point with the open outcome — a refusal is never evidence of
divergence. A singularity strictly INSIDE the interval, improperness
at BOTH ends (two singular ends, two infinite ends, or one of each),
and a singular integrand UNDER an infinite bound still have no closing
tactic; use the open outcome there too.

## Reduction formulas (a parameterized family)

When the task is a family `I_n` rather than one integral, the honest
complete answer is a RECURRENCE plus recorded base cases — three
separate statements (`I_n = \frac{n-1}{n} I_{n-2}` assuming `n > 1`,
then `I_0`, then `I_1`), never one conjunction. State the recurrence
with `integrate_reduction`: the relation (left side ONE integral;
right side a coefficient times the SAME integral with the parameter
shifted — anything else is refused), the variable, the parameter, the
integer shift, and `--assuming` with the TIGHTEST parameter domain you
can state. The check samples parameters only inside that domain, and
its agreement is evidence about the sampled region, not a proof —
overclaiming the domain weakens the record's honesty, never its
greenness. Close each base case as its own cell through the definite
or improper recipe above. A divergent side stalls the check's ladder
and certifies nothing — a refusal is never evidence of divergence.

Never type an assembled sum into core `expand`: that checks only the
expression you typed, not whether it contains the recorded pieces. Assembly
provenance is part of the mechanical certificate.

Antiderivatives carry a fresh `+ C`. The independent leg differentiates the
candidate; `1/x -> ln x` records the positive-domain assumption used by the
current table rule.
