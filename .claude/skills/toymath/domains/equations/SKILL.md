---
name: toymath-equations
key: equations
description: Narrow equation and root-finding tactics that return complete solution records for supported classes.
---

# Equations and roots

Use this skill when the task asks where an expression is zero or needs the
complete rational roots of a quadratic. There is deliberately no general
`solve` tactic.

Comma-separated relations such as `x+y=3, x-y=1` are a system container;
one-column `\cases{... \cr ...}` is the equivalent display form. Use
`substitute` on the whole container to replace a variable in every relation,
or `apply` when the same both-sides operation belongs on every relation. Each
relation is checked independently and the checks are merged. Mixed scalar
comma lists and ordinary two-column piecewise `\cases` are not systems.

These moves do not choose an elimination strategy or combine rows. Keep that
strategy explicit: isolate a value, substitute it through the system, then use
ordinary algebra tactics on the resulting relation as needed.

## Matching an ansatz

When an ansatz has to hold for every value of a variable — a partial-fraction
decomposition, an undetermined-coefficient guess — do not try to read the
coefficients off by eye and propose the whole decomposition at once. Clear the
denominators first so both sides are polynomials, then equate like powers:

```text
match_coefficients EXPR VAR

# from \frac{x^2}{(1-x^2)^3}, having cleared denominators:
match_coefficients "x^2 = A(x+1) + B(x-1)" x     ->  1 = A+B, 0 = A-B
```

It is not a solver. You say which variable to match in; it reports one
equation per power, and the values still come from your own later steps —
feed that system to `system_assemble` once each unknown has a recorded value.
The coefficients are recovered a second time by evaluation alone, so a
misread coefficient is refused rather than recorded.

A degree that cannot balance shows up honestly as an impossible equation such
as `1 = 0`: that means the ansatz itself is wrong, so fix the ansatz rather
than the arithmetic. Guessing coefficients and testing them one at a time
with `integrate_rewrite` or `expand` is the slow path — it burns the run's
turns and records nothing when it fails.

## Answers with several unknowns

A task like "find A, B and C" — matching an ansatz, or solving a system — ends
in several statements at once, not in one expression. Derive each unknown
however the problem allows until a step records `unknown = value`, one recorded
step per unknown, then bind them into the answer:

```text
system_assemble TARGET STEP...

# matching \frac{1}{x^2-1} = \frac{A}{x-1}+\frac{B}{x+1}, with s6 recording
# A = 1/2 and s10 recording B = -1/2:
system_assemble "\frac{1}{x^2-1} = \frac{A}{x-1}+\frac{B}{x+1}" s6 s10
```

`TARGET` is the equality (or comma system) the values have to satisfy — the
ansatz being matched, the system being solved, or the defining equation of a
single unknown. Never the answer: a target that already states a value
substitutes to `value = value` and is refused. The tactic puts every recorded
value back into the target, checks what comes out is an identity, and
independently re-derives the whole association by binding each unknown to its
own value, so a swapped pair of values is refused instead of assembled. The
result is the answer `A=…,B=…`, and that is what `set_result` designates.

It reads ledger step ids, never typed values. A value that still mentions
another unknown is a half-solved system and is refused, and every assembled
unknown must be one the target names. The record says the assignment satisfies
the target; it never says it is the only one that does.

A single unknown's value can also be stated as a claim. Such a claim can never
be decided on its own — asking whether `A` equals its value IS the question —
so it closes CONDITIONAL on the first input of the chain that derived it, and
the ledger shows that premise beside the verdict. Give `conclude` the chain
that starts at the equation the unknown came from.

## Inequalities and sign cases

An inequality only keeps its direction when the factor moved across it has a
known sign. A literal decides itself; anything symbolic needs the case stated:

```text
apply "\frac{1}{x} \lt 2" "*" "x" --assuming "x > 0"     # keeps \lt
apply "\frac{1}{x} \lt 2" "*" "x" --assuming "x < 0"     # flips to \gt
```

The hypothesis is recorded, not established, and the checks sample only where
it holds — including a second, independent check that the input and output
relations are true at exactly the same points, which is what catches a wrong
flip. State it about the factor itself (`x - 3 > 0` or the equivalent `x > 3`);
a hypothesis about something else does not pin the sign and is refused. Only
strict hypotheses (`<`, `>`) and `\ne` are accepted: a region with no interior
is one no check can live in.

Record every case, each as its own step from the same starting relation, and
keep them apart afterwards. Nothing holds under two opposite hypotheses at
once: a claim whose chain mixes them is refused, and the ledger lists such
hypotheses as alternatives rather than as one condition. When the cases end in
different answers, finish with `cases_assemble`: propose their union as a
`\lor` disjunction and pass the endpoint step of each case, in the order the
union writes them. A disjunct may itself be a `\land` conjunction when one
case needs both its hypothesis and endpoint; write the textbook chained form
when possible (`-1 \lt x \lt 1`) and pass that case's one endpoint step. Each
conjunction member is checked separately against that recorded endpoint or
hypothesis. A lone conjunction is still an assembly even though there is no
`\lor`: it combines two recorded relations into one bounded answer.

State each hypothesis in the spelling the answer should read — `x \gt 3`,
not `x - 3 \gt 0` (both pin the factor's sign) — because a disjunct may only
restate the endpoint or the hypothesis exactly as recorded. Use the endpoint
when it stayed inside its hypothesis, or the hypothesis when the endpoint
outgrew it (deriving `x \lt \frac{1}{2}` under `x \lt 0` solves that case as
`x \lt 0`; assembling the raw endpoint there is a coverage error the check
reports with a witness point). The union is checked to hold at exactly the
points the stated relation does, and a successful assembly discharges the case
hypotheses: the union is unconditional, and it is the step to designate with
`set_result`. If the available checked cases still cannot express the complete
answer, close the run with the open outcome rather than passing one case off as
the result.

For the bounded prototype “(x-1)(x+1) < 0”, state the useful case as
“assuming x < 1” before dividing by x-1, not as “x-1 < 0”. Its endpoint is
“x > -1”, so those two recorded relations assemble directly as the textbook
answer “-1 < x < 1”. Call cases_assemble once and designate that step; spot
checks afterward are optional illustrations, not a reason to assemble again.

Feed `quadratic_roots` the established quadratic expression verbatim; a plain
expression means `expr = 0`. It also accepts an equality and moves its sides
together internally. The result records every rational root and checks both
the candidates and completeness independently. Irrational or complex roots
are refused rather than approximated.

For stationary points, first load differentiation and call `diff`, then load
this skill and pass the derivative result verbatim to `quadratic_roots`. That
records where the derivative vanishes — the x-values, not the points.

When the task asks for the points themselves, finish with `points_assemble`.
Substitute each recorded root into the ORIGINAL function and `evaluate` it, so
every value is its own checked step, then call

```text
points_assemble ORIGINAL_FUNCTION VAR ROOTS_STEP VALUE_STEP...
```

with the step id of the root result and one value step id per root, in the
order the roots are written. It re-derives every root-to-value association
independently and returns the complete collection `\{(x_1,y_1),...\}`, which
is the answer to designate with `set_result`. Never type that collection
yourself: an assembled answer is a claim about where each number came from,
and only this tactic can check it. If a value has not been recorded as a step
yet, record it first — the tactic reads ledger ids, not typed numbers.

To mark the points on a graph, plot after assembling. The plot remains an
unverified illustration.

An equation being solved is not a claim that the equation is universally
true. Do not call `claim` for it; designate the complete result with
`set_result`.
