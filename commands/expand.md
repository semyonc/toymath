---
name: expand
description: Expand/normalize the argument (zero-token, verified expand primitive)
direct: expand
---
Expand $ARGUMENTS.

Direct command: no agent run - polynomial/rational normalization over
opaque atoms via the verified `expand` primitive, spot-checked by the
numeric oracle. This is the verified replacement for a procedural
`mul!`/`add!` inside an expression: `{expand! (1+x)(1-x)}` multiplies and
cancels with an oracle-checked ledger step.
