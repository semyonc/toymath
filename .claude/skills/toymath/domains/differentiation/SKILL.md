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

For a derivative-only goal, call `diff`, inspect its check, then designate the
returned result. For an identity involving a derivative, use core `equal` or a
closed relation through `evaluate` as appropriate.
