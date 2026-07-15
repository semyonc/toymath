---
name: toymath-finite-operators
key: finite_operators
description: Finite sums/products, honest ellipsis interpretation, summand rewrite, and telescoping.
---

# Finite sums and products

An ellipsis has no mechanical semantics. Before any other tactic:

- use `sum_from_ellipsis` for displayed `+` terms;
- use `prod_from_ellipsis` for displayed factors;
- propose the explicit finite operator and accept the recorded pattern-
  continuation assumption;
- parenthesize a composite trailing product factor such as `(2n)` because a
  flat product otherwise parses it as two factors.

For a finite sum, use `sum_rewrite` to propose a mechanically equal summand.
Use `sum_telescope` with the proposed `f(k)` when the summand has the form
`f(k)-f(k+1)`. The independent leg literally accumulates several finite sums.

The finite-product oracle likewise evaluates literal bounded products, but no
product rewrite or ratio-telescope tactic exists yet. Do not perform that move
in prose.

These tactics preserve an enclosing limit binder: apply the door to the
whole `\lim` expression, never to the inner body alone — a step recorded on
the bare body cannot anchor a claim about the limit. After interpreting or
telescoping a partial sum/product, load `limits` to close the remaining limit.
Infinite operator bounds are refused; rewrite them as limits of partial
operators through a named tactic when such a tactic becomes available.
