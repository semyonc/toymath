---
name: int
description: Apply symbolic integration to the argument, step by step
expr: true
fresh: C
input: expression
output: expression
---
Apply symbolic integration for $ARGUMENTS.

Load the `integration` skill and follow its order of attack. Keep every move
in the ledger, use provenance-aware assembly after a split, retain one fresh
`+ C`, and surface every assumption. If the loaded tactics do not close the
integral, report the verified stopping point and missing move.
