---
name: int
description: Apply symbolic integration to the argument, step by step
expr: true
fresh: C
---
Apply symbolic integration for $ARGUMENTS.

Pick the right tactic — integrate_power_rule, integrate_table,
integrate_by_parts, or integrate_substitute — and drive it move by move.
Show every step, keep the fresh `+ C`, and record every assumption (for
example `x > 0` on the `1/x -> ln x` rule). Do not claim an antiderivative
you have not obtained through the verified integrate_* tactics.
