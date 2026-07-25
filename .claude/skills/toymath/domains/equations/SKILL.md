---
name: toymath-equations
key: equations
description: Narrow equation and root-finding tactics that return complete solution records for supported classes.
---

# Equations and roots

Use this skill when the task asks where an expression is zero or needs the
complete rational roots of a quadratic. There is deliberately no general
`solve` tactic.

Comma-separated relations such as `x+y=3, x-y=1` are a system container;
one-column `\cases{... \cr ...}` is the equivalent display form. Use
`substitute` on the whole container to replace a variable in every relation,
or `apply` when the same both-sides operation belongs on every relation. Each
relation is checked independently and the checks are merged. Mixed scalar
comma lists and ordinary two-column piecewise `\cases` are not systems.

These moves do not choose an elimination strategy or combine rows. Keep that
strategy explicit: isolate a value, substitute it through the system, then use
ordinary algebra tactics on the resulting relation as needed.

Feed `quadratic_roots` the established quadratic expression verbatim; a plain
expression means `expr = 0`. It also accepts an equality and moves its sides
together internally. The result records every rational root and checks both
the candidates and completeness independently. Irrational or complex roots
are refused rather than approximated.

For stationary points, first load differentiation and call `diff`, then load
this skill and pass the derivative result verbatim to `quadratic_roots`. That
records where the derivative vanishes — the x-values, not the points.

When the task asks for the points themselves, finish with `points_assemble`.
Substitute each recorded root into the ORIGINAL function and `evaluate` it, so
every value is its own checked step, then call

```text
points_assemble ORIGINAL_FUNCTION VAR ROOTS_STEP VALUE_STEP...
```

with the step id of the root result and one value step id per root, in the
order the roots are written. It re-derives every root-to-value association
independently and returns the complete collection `\{(x_1,y_1),...\}`, which
is the answer to designate with `set_result`. Never type that collection
yourself: an assembled answer is a claim about where each number came from,
and only this tactic can check it. If a value has not been recorded as a step
yet, record it first — the tactic reads ledger ids, not typed numbers.

To mark the points on a graph, plot after assembling. The plot remains an
unverified illustration.

An equation being solved is not a claim that the equation is universally
true. Do not call `claim` for it; designate the complete result with
`set_result`.
