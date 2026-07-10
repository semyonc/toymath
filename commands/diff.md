---
name: diff
description: Differentiate the argument, showing each rule applied
expr: true
---
Differentiate $ARGUMENTS with respect to its variable.

Use the differentiate primitive so the derivative is checked by the numeric
oracle. If the expression is a product, quotient, or composition, you may
still do it in one differentiate call — name the rule you expect in your
summary. Report the result and any domain assumption the primitive records.
