---
name: toymath-equations
key: equations
description: Narrow equation and root-finding tactics that return complete solution records for supported classes.
---

# Equations and roots

Use this skill when the task asks where an expression is zero or needs the
complete rational roots of a quadratic. There is deliberately no general
`solve` tactic.

Feed `quadratic_roots` the established quadratic expression verbatim; a plain
expression means `expr = 0`. It also accepts an equality and moves its sides
together internally. The result records every rational root and checks both
the candidates and completeness independently. Irrational or complex roots
are refused rather than approximated.

For stationary points, first load differentiation and call `diff`, then load
this skill and pass the derivative result verbatim to `quadratic_roots`. To
mark points on the original graph, substitute each returned root into the
original function and evaluate before plotting. The plot remains an
unverified illustration.

An equation being solved is not a claim that the equation is universally
true. Do not call `claim` for it; designate the root result with `set_result`.
