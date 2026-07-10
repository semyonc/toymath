# ToyMath project overview

This document is the detailed system tour. For the short statement of what
makes ToyMath unusual, start with the [README](../README.md). For the exact
verified-primitive contract, see [PRIMITIVES.md](../doc/PRIMITIVES.md).

## System at a glance

ToyMath is a Jupyter kernel and console application with LaTeX as its input and
output language. Its current agentic workflow sits on top of the original
symbolic engine:

```text
Natural-language goal
        |
        v
Agent chooses a named tactic
        |
        v
Deterministic primitive -> fresh notation graph -> LaTeX result
        |                                      |
        +---------- independent oracle <-------+
                               |
                               v
                    replayable step ledger
```

The layers have deliberately different responsibilities:

- **Strategy:** an LLM decides what mathematical move to try next.
- **Mechanism:** deterministic code parses and transforms the expression.
- **Verification:** a separate numeric evaluator checks the proposed result at
  reproducible sample points.
- **Record:** the ledger stores tool name, arguments, input, result,
  assumptions, check status, and a content hash.

Only successful transforming tool calls become ledger steps. Model prose and
plots are outside the artifact.

## The `do!` notebook endpoint

A Jupyter cell beginning with `do!` sends the remainder to an agent backed by
the OpenAI Agents SDK over OpenRouter:

```text
do! solve 2x + 3 = 7 for x and verify the candidate
```

The agent can act only through ToyMath's function tools. As each call returns:

- the checked step streams into the output cell;
- assumptions are accumulated;
- the step joins the notebook-wide ledger;
- a later cell can refer to the selected result as `[[n]]`.

A typical rendered step has the shape:

```text
s2#43bdac6 [ok] expand: 2x+3-3 = 7-3  ==>  2x = 4
```

The agent can designate which established expression should become the cell's
chainable value. This matters when the last tool call verifies an earlier
answer—for example, substitution may end at `7 = 7`, while the useful value is
still `x = 2`.

All `do!` cells in a notebook share one ledger, but each output renders only
the steps added by that cell. Ledger replay re-executes every stored operation
and checks its recorded result.

### Plotting

When Deno is installed, the agent may receive a plotting tool. Its Python code
runs in Pyodide WASM under Deno's deny-by-default permissions:

- no environment access;
- no project-filesystem access;
- network access only to the package CDNs;
- matplotlib images returned as data, not executable notebook content.

Plots are captioned as illustrations, never machine-checked evidence. They are
not ledger steps and replay ignores them. Set `TOYMATH_SANDBOX=off` to disable
the tool.

## Verified primitives

Every transforming primitive returns a record shaped like:

```text
{ok, op, args, input, result, assumptions, check}
```

The principal operations are:

| Primitive | Role |
|---|---|
| `substitute(expr, var, value)` | notation-graph replacement with binding-safe grouping |
| `apply_both_sides(eq, op, arg)` | `+ - * / ^` on equations and inequalities; records nonzero assumptions and flips inequalities for negative constants |
| `expand(expr)` | canonical rational algebra; outside it, combines like opaque atoms such as `\sin x` |
| `collect(expr, var)` | groups polynomial and rational expressions by powers of a variable |
| `evaluate(expr)` | exact arithmetic; reports whether a closed relation holds |
| `differentiate(expr, var)` | exact rational derivative plus mechanical rules for common functions |
| `rewrite(expr, lemma, direction)` | applies a registered identity at the root or a matching subterm |
| `factor_gcd(expr)` | pulls out a common monomial/numeric factor, including applicable relation sides |
| `factor_quadratic(expr, var)` | factors rational-root quadratics, including applicable relation sides |
| `integrate_power_rule` | termwise power-rule tactic |
| `integrate_table` | table tactic for basic functions and `1/x` |
| `integrate_by_parts(u, dv)` | verifies the supplied split and returns one by-parts step |
| `integrate_substitute(...)` | verifies a user-supplied substitution before changing variables |
| `equal?(e1, e2)` | exact/canonical comparison where possible, otherwise an honest numeric verdict |

There is no general `solve`, `simplify`, `factor`, or autonomous `integrate`.
The agent must expose its strategy as a sequence of narrower operations.

### Two checking paths

ToyMath does not treat one implementation as its own proof of correctness:

1. The symbolic path constructs the transformed notation graph.
2. The numeric oracle independently evaluates the input and output at fixed,
   reproducible sample points.

For rational expressions, canonical comparison is exact. Outside that
fragment, oracle agreement is probabilistic and is reported as such. Domain
differences, unsupported evaluation, and assumptions are surfaced rather than
silently converted into equality.

Full details—including opaque atoms, noncommutative matrix words, domain-aware
checks, and integration verification—are in
[doc/PRIMITIVES.md](../doc/PRIMITIVES.md).

## Named notebook commands

A notebook command is a reusable `do!` instruction stored as Markdown in
`commands/`. Discovery is automatic: add a valid file and the command becomes
available without editing a registry.

```markdown
---
name: int
description: Apply symbolic integration step by step
expr: true
---
Apply symbolic integration for $ARGUMENTS ...
```

This creates `int!`. `commands!` lists the discovered commands. The shipped
templates include `solve!`, `int!`, and `diff!`.

The frontmatter fields are:

| Field | Meaning |
|---|---|
| `name` | command name; defaults to the Markdown filename |
| `description` | discovery/help text |
| `expr` | when true, permits inline `{name! ...}` composition |

The body must contain `$ARGUMENTS`, which is replaced by the cell argument.
The template adds no new authority: the resulting agent still has only the
verified primitive tools.

## Inline command composition

Expression-capable commands can appear inside LaTeX:

```text
{diff! {int! x^3}}
{int! x^2} + {int! x^2}
2 {diff! x^2} - 1
```

The expression resolver:

1. parses the cell into the notation graph;
2. resolves nested commands from the inside out;
3. memoizes identical command/argument pairs within the cell;
4. splices each checked command result back into the graph with safe grouping;
5. sends the combined expression through `expand` so the arithmetic glue gets
   its own oracle check.

Only commands marked `expr: true` can appear inline. Legacy commands or unknown
commands are refused rather than mixed into an apparently verified composite.
A per-cell call cap prevents accidental unbounded agent expansion.

## Command-line interface

`toymath_cli.py` exposes the same primitive layer. Each invocation prints one
deterministic JSON object; `--session` appends transforming results to a ledger.

```bash
python toymath_cli.py apply "2x + 3 = 7" - 3 --session derivation.json
python toymath_cli.py expand "2x+3 - 3 = 7 - 3" --session derivation.json
python toymath_cli.py factor_quadratic "x^2+6x+9=4" x --session derivation.json
python toymath_cli.py show --session derivation.json --format md
python toymath_cli.py replay --session derivation.json
```

The Claude Code skill in `.claude/skills/toymath/` documents the same protocol.
The `do!` prompt is generated from that skill file so the two interfaces share
one operational description.

## The classic kernel

The original ToyMath is a fixed-point symbolic LaTeX kernel:

```text
LaTeX -> parser -> notation graph (DAG) -> processor -> LaTeX
```

- `notation.py` defines `Symbol`, `Func`, and the graph relations.
- `LatexParser.py` and `lexer.py` build the notation graph.
- `LatexWriter.py` renders it back to LaTeX.
- `processor.py` repeatedly walks the graph until the output reaches a fixed
  point.
- `cmd_*.py` modules are auto-discovered procedural commands.

Numeric constants evaluate automatically. Classic symbolic transformations are
requested with commands such as:

```latex
\frac{1}{2} + \frac{1}{3}      % -> 5/6
mul! x^{-1}x                   % -> 1
mul! (a/b)^2                   % -> a^2/b^2
add! \frac{a}{b}+\frac{c}{d}   % -> (ad+bc)/(bd)
add! {mul! (a+b)(c+d)} + x     % nested procedural commands
```

The verified primitives now carry most agentic algebra. The classic engine
remains important as the LaTeX parser, notation-DAG substrate, fixed-point
runtime, and command-node grammar reused by inline agent commands.

Other classic commands include Prolog-style `goal`/`rules`, notation and trace
tools such as `dump`/`track`/`debug`, and session controls such as
`echo-on`/`echo-off`/`clear`.

Console mode without Jupyter is available through `python console.py`.

## Installation and configuration

Requirements:

- Python 3.11 or newer;
- [uv](https://docs.astral.sh/uv/);
- JupyterLab, installed from `requirements.txt`;
- optionally [Deno](https://deno.com) for sandboxed plotting.

```bash
git clone https://github.com/semyonc/toymath.git
cd toymath

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

jupyter kernelspec install kernel_spec --user
```

Agent configuration lives in `.env` at the repository root:

```dotenv
OPEN_ROUTER=sk-or-...
OPENROUTER_MODEL=anthropic/claude-sonnet-5
TOYMATH_SANDBOX=auto
```

| Variable | Meaning |
|---|---|
| `OPEN_ROUTER` | OpenRouter API key required for agent-backed commands |
| `OPENROUTER_MODEL` | OpenRouter model slug |
| `TOYMATH_SANDBOX` | `auto`, `pyodide`, or `off` for the plotting backend |

Launch JupyterLab and select the **Toy Math** kernel:

```bash
jupyter lab
```

## Tests

The normal suite is offline. It uses a scripted fake model to exercise the
complete agent/tool/ledger loop without an API call.

```bash
pytest                                  # all offline suites
pytest engine/unittests.py              # classic engine
pytest engine/unittests_frac.py         # fractions
pytest engine/unittests_primitives.py   # verified primitives and oracle
pytest engine/unittests_do.py           # agent endpoint and commands
```

Live probes are opt-in:

```bash
TOYMATH_LIVE_TESTS=1 pytest engine/unittests_do.py
TOYMATH_PLOT_TESTS=1 pytest engine/unittests_do.py
```

The first uses the configured OpenRouter model. The second exercises the live
Deno/Pyodide sandbox.

## Repository map

```text
toymath/
|- engine/
|  |- primitives.py       verified primitives and numeric oracle
|  |- polyrat.py          canonical rational-function core
|  |- ledger.py           record, render, and replay derivations
|  |- agent_do.py         do! agent endpoint and function tools
|  |- prompt_commands.py  discovery of commands/*.md templates
|  |- expr_commands.py    inline {name! expression} resolver
|  |- plot_sandbox.py     plotting backend seam
|  |- notation.py         notation graph structures
|  |- processor.py        classic fixed-point processor
|  |- mathShell.py        notebook cell routing
|  `- cmd_*.py            classic procedural commands
|- commands/              solve/int/diff prompt templates
|- doc/                   primitive and notation internals, images
|- docs/OVERVIEW.md       this system tour
|- toymath_cli.py         JSON command-line interface
|- toymathkernel.py       Jupyter kernel entry point
`- .claude/skills/        agent-facing ToyMath protocol
```

## Further documentation

- [Verified primitives](../doc/PRIMITIVES.md)
- [Notation graph internals](../doc/NOTATION.md)
- [Developer guide](../CLAUDE.md)
- [Repository agent guidance](../AGENTS.md)

ToyMath is released under the MIT License.
