---
name: diff
description: Differentiate the argument (zero-token, verified differentiate primitive)
direct: differentiate
input: expression
output: expression
---
Differentiate $ARGUMENTS with respect to its single free variable.

Direct command: no agent run - the argument goes straight to the verified
`differentiate` primitive (polyrat fast path + rules, spot-checked by the
numeric oracle). The variable is inferred as the argument's single plain
free variable; constants minted by fresh: commands in the same cell (an
inner {int! ...}'s + C) are excluded from inference. A multi-variable
argument is refused, never guessed - use a do! cell to pick the variable
explicitly.
