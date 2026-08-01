---
name: prove
description: Establish the claim through a connected chain of verified steps, reporting any gap honestly
mode: prove
input: relation
output: derivation
---
Prove $ARGUMENTS — that is, connect the claim to checked ledger steps.

Classify the claim first in one short comment, then load only the subject
skill needed for the claim. Core algebra is already available. Drive a
connected checked chain to the exact statement.

The proof is the chain. Every load-bearing statement must be a verified
step whose check passed; comment is for strategy only — one short line
for why you branch or which piece you are on, never mathematical content
the conclusion depends on. If a route dead-ends, note it in one line and
branch; dead branches stay visible and that is honest.

If the chain reaches the claim, call `conclude` with the ordered closing step
ids, then `set_result`, and use the phrase "mechanically checked". Otherwise
leave the claim open and report the exact missing move. Do not write a prose
proof to bridge the gap.
