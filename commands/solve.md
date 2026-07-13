---
name: solve
description: Solve the equation move by move, recording every assumption
---
Solve $ARGUMENTS, showing every algebraic move as a separate verified step.

Use `apply` to add, subtract, multiply, or divide both sides, and use the core
expand/collect/named-factor tactics to normalize. Record each assumption they
surface (for example `a \ne 0` when dividing by a symbolic factor). When you
reach a candidate, check it by substitute + evaluate rather than asserting
it. The visible chain of steps is the answer — never collapse it.
