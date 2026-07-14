# ToyMath project overview

ToyMath is a LaTeX-native symbolic mathematics system: LaTeX in, LaTeX out,
with expressions stored as a notation DAG. It runs as a Jupyter kernel, a
console application, and a deterministic JSON CLI.

The project contains two deliberately different execution paths:

- the current verified-derivation layer, where an agent chooses narrow tactics
  and every transformation becomes a mechanically checked ledger step;
- the classic fixed-point engine, which still evaluates ordinary notebook and
  console math cells.

For the exact trust and extension contract, read
[PRIMITIVES.md](PRIMITIVES.md). For notation internals, read
[NOTATION.md](NOTATION.md).

## Verified derivations at a glance

```text
natural-language goal
        |
        v
small core prompt + domain skill catalog
        |
        +--> load only the needed subject skill
        |
        v
agent chooses a registered tactic
        |
        v
deterministic transformation ----> fresh notation graph
        |                                  |
        +------ independent oracle <-------+
                         |
                         v
                 replayable ledger step
```

The responsibilities are separated:

- **Strategy:** the model chooses the next named tactic.
- **Mechanism:** deterministic code performs one narrow transformation.
- **Verification:** an independent numeric evaluator spot-checks the move.
- **Record:** the ledger stores the operation, arguments, input, result,
  assumptions, check, hash, goal, and any source provenance.

Only successful transforming tactic calls enter the artifact. Model prose,
skill text, comments, and plots cannot establish a mathematical result.

## `do!` and progressive skills

A notebook cell beginning with `do!` sends the rest of the cell to an
OpenRouter-backed agent through the OpenAI Agents SDK:

```text
do! solve 2x + 3 = 7 for x and verify the candidate
```

The runtime no longer exposes one function schema per mathematical tactic.
It has a stable small surface:

- `load_skill` progressively loads a relevant subject workflow;
- `run_tactic` invokes an allowlisted registry entry;
- `comment`, `claim`, `conclude`, and `set_result` manage the ledger/result;
- `plot` appears only when the optional sandbox is available.

Core algebra/checking guidance is always present. Differentiation, equations,
integration, limits, and finite-operator workflows are loaded only when the
problem needs them. The virtual loader accepts canonical subjects, bounded
subject aliases, or an exact tactic name and resolves all of them through the
static registry. A derivation may load another skill when it crosses a subject
boundary. This keeps the prompt and tool schemas bounded while the tactic
library grows.

Each successful call streams a checked step such as:

```text
s2#43bdac6 [ok] expand: 2x+3-3 = 7-3  ==>  2x = 4
```

All agent cells in a notebook share one ledger. Each cell renders only its new
slice, and a selected established result becomes addressable as `[[n]]` by a
later cell.

`prove!` creates a root claim before the model runs. The claim remains visibly
open until `conclude` receives a connected, goal-owned checked chain. Prose
cannot substitute for a missing step.

## Notebook command tiers

Saved notebook commands are Markdown templates in `commands/`:

```markdown
---
name: int
description: Apply symbolic integration step by step
expr: true
---
Apply symbolic integration for $ARGUMENTS ...
```

They form three execution tiers:

1. **Direct primitive:** `expand!` and `diff!` run one verified operation with
   no model call.
2. **Tactic template:** `int!`, `lim!`, and `solve!` seed a focused agent run;
   the agent loads the relevant skill and records each move.
3. **Whole derivation:** `do!` accepts an unrestricted natural-language goal;
   `prove!` adds the claim-closure requirement.

Expression-capable commands compose inline:

```text
{diff! {int! x^3}}
2 {diff! x^2} - 1
```

The resolver works inside-out, splices only verified results with safe
grouping, and sends the combined expression through `expand` so the glue gets
its own oracle check. Certificates compose locally: each sub-command keeps its
own steps, while the final check certifies only the splice arithmetic.

## Command-line interface

`toymath_cli.py` is the stable external interface. Existing tactic commands
remain positional and backward compatible:

```bash
python toymath_cli.py apply "2x + 3 = 7" - 3 --session work.json
python toymath_cli.py expand "2x+3-3 = 7-3" --session work.json
python toymath_cli.py replay --session work.json
```

The parser and dispatch are generated from the same registry used by `do!` and
ledger replay. Discovery is generated too:

```bash
python toymath_cli.py skills
python toymath_cli.py tactics --skill integration
python toymath_cli.py describe integrate_by_parts
```

Ledger-control commands (`claim`, `conclude`, `show`, `replay`) remain explicit
CLI operations rather than math tactics.

## Plotting

When Deno is installed, plots run in Pyodide WASM under deny-by-default Deno
permissions: no environment access, no project-filesystem access, and network
access only to package CDNs. Images are rendered as unverified illustrations,
never ledger evidence. Set `TOYMATH_SANDBOX=off` to disable the tool.

## The classic kernel

Ordinary math cells use the original fixed-point pipeline:

```text
LaTeX -> parser -> notation DAG -> processor fixed point -> LaTeX
```

`processor.py` auto-discovers `cmd_*.py` commands such as `mul!` and `add!`.
That rewrite layer is retained for existing calculator behavior but is not the
extension point for new polynomial/rational capabilities. Canonical work
belongs in `polyrat.py`; agentic mathematical strategy belongs in registered
verified tactics.

Notebook prefix commands and `cmd_*.py` rewrite commands share the `name!`
surface but are different systems. A prompt command must never shadow a
registered rewrite command.

## Installation and configuration

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
jupyter kernelspec install kernel_spec --user
jupyter lab
```

Agent configuration is read from `.env`:

```dotenv
OPEN_ROUTER=sk-or-...
OPENROUTER_MODEL=anthropic/claude-sonnet-5
TOYMATH_SANDBOX=auto
```

The normal test suite is offline; live OpenRouter and plot probes are opt-in:

```bash
.venv/bin/python -m pytest -q
TOYMATH_LIVE_TESTS=1 .venv/bin/python -m pytest engine/unittests_do.py -q
TOYMATH_PLOT_TESTS=1 .venv/bin/python -m pytest engine/unittests_do.py -q
```

## Repository map

```text
engine/primitives.py       shared notation infrastructure and numeric oracle
engine/tactics/*.py        static tactic implementations by subject skill
engine/tactic_registry.py  authoritative tactic schemas and dispatch
engine/tactic_skills.py    progressive SKILL.md discovery/rendering
engine/ledger.py           record, render, claim closure, and replay
engine/agent_do.py         do! runtime and stable model tool surface
engine/polyrat.py          canonical rational-function core
engine/expr_commands.py    inline command composition
engine/prompt_commands.py  commands/*.md discovery
engine/plot_sandbox.py     plotting backend seam
engine/processor.py        classic fixed-point engine
toymath_cli.py             generated tactic CLI + ledger controls
```

Further reading: [verified-derivation contract](PRIMITIVES.md),
[notation DAG](NOTATION.md), and [agent/developer workflow](../AGENTS.md).
