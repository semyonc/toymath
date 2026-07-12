---
name: prove
description: Establish the claim through a connected chain of verified steps, reporting any gap honestly
mode: prove
---
Prove $ARGUMENTS — that is, connect the claim to checked ledger steps.

Classify the claim first and say so in one comment. An identity: drive
one side to the other with expand/collect/rewrite/factor_*, or settle it
with equal? and then exhibit the transforming chain. A solution claim:
substitute the candidate and evaluate both sides. A limit claim: drive
the limit_* tactics to the exact stated value. A derivative or
antiderivative claim: diff / integrate_* toward the stated form.

The proof is the chain. Every load-bearing statement must be a verified
step whose check passed; comment is for strategy only — one short line
for why you branch or which piece you are on, never mathematical content
the conclusion depends on. If a route dead-ends, note it in one line and
branch; dead branches stay visible and that is honest.

If the verified chain reaches the claim, finish with set_result on the
established value and a one-line summary saying "mechanically checked"
(never "proved"). If it does not, stop and report the gap exactly: state
which verified form you reached and which move is missing. Do NOT write
a prose proof to bridge the gap — an unverified argument is not a
result, and presenting one as the conclusion is the failure mode this
command exists to prevent.
