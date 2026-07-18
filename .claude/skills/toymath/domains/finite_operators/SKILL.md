---
name: toymath-finite-operators
key: finite_operators
description: Finite sums/products, honest ellipsis interpretation, summand rewrite, telescoping, checked closed forms, and infinite series as partial-sum limits.
---

# Finite sums and products

An ellipsis has no mechanical semantics. Before any other tactic:

- use `sum_from_ellipsis` for displayed `+` terms;
- use `prod_from_ellipsis` for displayed factors;
- propose the explicit finite operator and accept the recorded pattern-
  continuation assumption;
- parenthesize a composite trailing product factor such as `(2n)` because a
  flat product otherwise parses it as two factors.

Infinite bounds are refused by every finite tactic. Apply
`series_partial_sums` first: it rewrites `\sum_{k=a}^{\infty} f` (or the
`\prod` form) as `\lim_{n \to \infty} \sum_{k=a}^{n} f` — exact by
definition, with "value = limit of partial sums, if it exists" recorded.
Then drive the finite tactics under the `\lim`.

Choosing the finite move:

- `sum_expand` / `prod_expand` — both bounds are small integer literals:
  writes the terms out and folds them; nothing to propose.
- `sum_rewrite` — swap the summand for a mechanically equal one
  (partial fractions before telescoping is the standard use).
- `sum_telescope` — the summand has the form `f(k) - f(k+1)` for an `f`
  you propose; strongest check (summand identity via equal? plus literal
  summation). An increasing telescope needs the negated `f`.
- `sum_closed_form` / `prod_closed_form` — you can propose the value of
  the whole operator as a formula in the upper-bound variable (factorial
  ratios, Faulhaber forms, shifted sums). The literal accumulation loop
  checks it at several integer bounds including the empty case, and the
  integer domain of the bound is recorded as an assumption. Use this for
  product↔factorial identities: general equal? samples real values and
  honestly answers `unknown` there.

These tactics preserve an enclosing limit binder: apply the door to the
whole `\lim` expression, never to the inner body alone — a step recorded on
the bare body cannot anchor a claim about the limit. After interpreting,
telescoping, or closing a partial sum/product, load `limits` to close the
remaining limit.

## Walkthrough: geometric series end to end

Goal `\sum_{k=0}^{\infty} (\frac{1}{2})^k`:

1. `series_partial_sums` → `\lim_{n \to \infty} \sum_{k=0}^{n} (\frac{1}{2})^k`.
2. `sum_closed_form` with `2 - (\frac{1}{2})^{n}` (or `sum_telescope`
   with `\frac{2}{2^k}`) → `\lim_{n \to \infty} (2 - (\frac{1}{2})^{n})`.
3. Load `limits`: `limit_linearity`, `limit_table` on each piece
   (constant; geometric decay), `limit_assemble` → `2`.

## Convergence

Convergence has two honest routes, in order of preference:

1. **Value route**: `series_partial_sums`, then telescope or a closed
   form, then the `limits` skill. Exhibiting the value establishes
   convergence outright.
2. **Comparison route**: `series_converges` with a dominating summand you
   propose — geometric `c r^k` with literal `0 < r < 1` (also spelled
   `c/s^k` or `s^{k-1}`) or p-series `c/k^p` with literal `p > 1`. The
   recorded step is the bound relation `\sum |f| \le T` for the family's
   closed tail bound; domination for every index is a recorded,
   spot-checked assumption, and the verdict is *absolute* convergence.
   Example: `\sum 1/n!` is certified by proposing `\frac{1}{2^{n-1}}`.

A parametric series is refused — substitute parameter values first. There
is no divergence tactic and no conditional-convergence test: a refusal is
never evidence of divergence, and an alternating series that does not
converge absolutely needs the value route or stays honestly open.

## Walkthrough: product to factorial ratio

Goal: express `\prod_{k=1}^{n} \frac{2k-1}{2k}` in factorials. Propose the
ratio directly: `prod_closed_form` with `\frac{(2n)!}{2^{2n}(n!)^2}`. The
check compares literal products against the proposal at integer `n` and
records `n \in \mathbb{Z}, n \ge 0`. Do not argue the identity in prose and
do not expect `equal` to certify it — the bound is integer-valued, which is
exactly what this named tactic records.
