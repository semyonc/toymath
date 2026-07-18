---
name: lim
description: Evaluate the limit move by move via the verified limit tactics
expr: true
---
Evaluate the limit $ARGUMENTS using the verified limit tactics only.

Load the `limits` skill and follow it with the whole `\lim` expression. If the
body contains an ellipsis or a `\sum`/`\prod` (finite or infinite), load
`finite_operators` first and interpret it before returning to the limit
workflow. Preserve assembly/squeeze provenance. If no loaded tactic chain
closes the limit, report the last verified form and missing move; never bridge
the gap with prose or a plot.
