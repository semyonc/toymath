---
name: toymath-differentiation
key: differentiation
description: Differentiation tactics and derivative-verification workflow.
---

# Differentiation

Use this skill when the goal requires a derivative or when an independently
computed derivative is useful for checking a candidate expression.

`diff` handles the rational fragment exactly and common functions through
mechanical product, quotient, chain, and table rules. Its result is checked by
independent central differences. Unsupported operators are refused rather than
treated as constants.

A variable-bound definite integral closes in one `diff` call: for
`\int_{a(x)}^{b(x)} f(t) dt` the FTC bound rule builds
`f(b(x)) b'(x) - f(a(x)) a'(x)`, records continuity of the integrand between
the bounds as an assumption, and is checked independently by quadrature plus a
central difference — do not substitute bounds or differentiate pieces by hand.
The Leibniz spelling `\frac{d}{d x} (...)` is accepted and names the variable
itself. Two honest refusals to expect: an integrand that depends on the
differentiation variable (differentiation under the integral sign is not in
the rule set), and an unevaluated indefinite integral (integrate first, then
differentiate the result).

For a derivative-only goal, call `diff`, inspect its check, then designate the
returned result. For an identity involving a derivative, use core `equal` or a
closed relation through `evaluate` as appropriate.

If the task asks where the derivative vanishes, load the `equations` skill
after `diff` and feed the derivative result verbatim to `quadratic_roots` when
it is quadratic. Check graph-point coordinates through core substitution and
evaluation before drawing an unverified plot.
