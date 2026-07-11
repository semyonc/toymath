---
name: int
description: Apply symbolic integration to the argument, step by step
expr: true
fresh: C
---
Apply symbolic integration for $ARGUMENTS.

Pick the right tactic — integrate_power_rule, integrate_table,
integrate_by_parts, integrate_substitute, integrate_rewrite, or
integrate_linearity — and drive it move by move. After a linearity split,
pin each piece's constant to 0, then call integrate_assemble with the
linearity step id and the completed piece step ids in the returned order.
Never type that final sum into expand: expand cannot check piece provenance.
Show every step, keep the one fresh `+ C`, and record every assumption (for
example `x > 0` on the `1/x -> ln x` rule). Do not claim an antiderivative
you have not obtained through the verified integrate_* tactics.
