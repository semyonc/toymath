---
name: factor
description: Factor the argument into a mechanically checked product
input: [expression, relation]
output: derivation
---
Factor $ARGUMENTS into a product. You choose the factors; the engine checks
them. A factorization is easy to check exactly — every split you propose is
verified canonically, so the burden is on finding the factors, never on
trusting them.

There is no irreducibility oracle. Equality is mechanically checked, but
*fully factored* is your judgement — say what you are factoring over
(integers/rationals unless asked otherwise) and let the recorded steps show
the shape.

Order of attack:

- `factor_gcd` first: pull the common factor while the structure is still
  visible.
- `factor_quadratic` for any quadratic piece. Its refusals are verdicts to
  report, not obstacles to work around: "negative discriminant" and "not
  factorable over Q" both name a legitimate stopping point. It accepts a
  relation directly (`x^2-3x+2 = 0` becomes `(x-2)(x-1) = 0`), so factoring
  an equation keeps its relation shape in one checked step.
- A registered lemma is an exact structural match — `rewrite x diff_squares`
  and the trig identities; check `lemmas`. A lemma does not bind a power of
  an opaque atom (`\sin^4 x - \cos^4 x` will not match `a^2 - b^2`);
  propose that split with `rewrite_as` instead — a factorization proposal
  is checked canonically, not sampled.
- Higher degree: guess a rational root `r`, verify it with a checked
  `evaluate` of the expression at `r`, then propose the split
  `(x - r)(\text{quotient})` with `rewrite_as`. Repeat on the quotient.

Prefer several small splits over one leap. A single jump to the fully
factored form is checkable, but the chain of splits is what shows where
each factor came from.

Pure factoring drops no domain points, so a clean factoring chain records
no assumptions. If a step of yours cancels instead, the `\ne 0` assumption
is recorded automatically — keep it visible; do not restate or suppress it.

When nothing splits, say so with `set_open` and name the evidence you have
(irreducible by discriminant, no rational root among the candidates tried).
Reporting the input as unfactorable-by-your-moves is a legitimate outcome;
a cosmetic regrouping presented as a factorization is not.
