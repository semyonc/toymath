---
name: toymath
description: Verified-derivation math primitives. Use when the user wants a machine-checked, step-by-step derivation (solve an equation showing every move, expand/collect polynomials, differentiate, verify two expressions are equal). You plan the strategy; each toymath call executes one narrow step and independently spot-checks it numerically. The session ledger is the proof artifact.
---

# ToyMath verified derivations

You are the strategist; toymath is the mechanical checker. Never do algebra
in your head when a primitive can do it verified. Every call prints one JSON
record; `check.status: "agree"` means an independent numeric oracle confirmed
the step at random sample points.

## Invocation

```bash
python toymath_cli.py <command> ... [--session ledger.json]
```

Run from the repo root (or set `PYTHONPATH` to `engine/`). Expressions are
LaTeX strings — quote them. Use `--session` for any multi-step derivation:
it appends each verified step to a replayable ledger.

## Commands

| Command | Example | Meaning |
|---|---|---|
| `apply EQ OP ARG` | `apply "2x + 3 = 7" - 3` | op ∈ `+ - * / ^` to both sides. Multiplying or dividing an equation by a symbolic expression records the assumption `arg ≠ 0`. Inequalities (`<`, `>`, `\le`, `\ge`, …) work too: the relation flips under `*`/`/` by a negative constant, and symbolic-sign factors are refused |
| `expand EXPR` | `expand "(x+1)(x-2)"` | distribute products/powers; also simplifies each side of an equation |
| `collect EXPR VAR` | `collect "ax + 2x" x` | group by powers of VAR |
| `substitute EXPR VAR VAL` | `substitute "x^2+1" x 3` | replace a variable |
| `evaluate EXPR` | `evaluate "2(2)+3 = 7"` | exact arithmetic; on an equation reports `holds: true/false` |
| `diff EXPR VAR` | `diff "x \sin x" x` | derivative, checked by central differences |
| `rewrite EXPR LEMMA [--direction backward]` | `rewrite "x^2-y^2" diff_squares` | apply a registered identity at the root, or at the first matching subterm (reported as `at`) |
| `factor_gcd EXPR` | `factor_gcd "6x^2+9x"` | pull out the common factor → `3x(2x+3)` |
| `factor_quadratic EXPR VAR` | `factor_quadratic "x^2-5x+6" x` | rational roots → `(x-2)(x-3)`; reports roots; refuses irrational cases |
| `equal E1 E2` | `equal "(x+1)^2" "x^2+2x+1"` | verdict yes / no / unknown |
| `lemmas` | | list rewrite lemmas |
| `show --session f` | | render the ledger |
| `replay --session f` | | re-verify every recorded step |

## How to solve an equation (there is deliberately no `solve`)

Drive it move by move; the visible chain of steps IS the product:

```bash
python toymath_cli.py apply "2x + 3 = 7" - 3 --session d.json
python toymath_cli.py expand "{2}x+{3} - {3} = {7} - {3}" --session d.json   # -> 2x = 4
python toymath_cli.py apply "{2}x = {4}" / 2 --session d.json
python toymath_cli.py expand "\frac{{2}x}{{2}} = \frac{{4}}{{2}}" --session d.json  # -> x = 2
python toymath_cli.py replay --session d.json
```

Feed each step's `result` verbatim as the next input (the ledger flags
discontinuities). Check candidate solutions with `substitute` + `evaluate`.

## Rules

- If `check.status` is `"disagree"`, the step is wrong — do not use its
  result; report the discrepancy.
- Surface recorded `assumptions` (e.g. `y ≠ 0`) to the user; the derivation
  is conditional on them.
- `equal` verdict `unknown` means unverified — say so honestly.
- Results are "mechanically checked", not "proved" — the oracle rules out
  hallucinated algebra, not implementation bugs.
