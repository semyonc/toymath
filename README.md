# ToyMath

ToyMath is a LaTeX-native symbolic mathematics system with an agentic twist:
an LLM plans the mathematical strategy, ToyMath executes every move as a
narrow, named, **mechanically checked** transformation, and the session
ledger — the full chain of verified steps — is the product. Ask for a
derivation in plain language in a Jupyter cell, and get back a proof-style
trace where every step was validated by code, not by the model's confidence:

```
do! differentiate x^3 - 3x, find where the derivative is zero, plot with those points marked
```

<img src="doc/Pic1.png" alt="A do! cell: six verified ledger steps, a sandboxed plot of the critical points, and the chainable result" width="850">

Underneath sits the original ToyMath: a Jupyter kernel that parses LaTeX,
represents expressions as a notation graph (DAG), and transforms them with a
fixed-point rewrite engine — see [The classic kernel](#the-classic-kernel)
below.

## The idea: derivations you can trust

A capable model already solves most school and early-undergraduate math on
its own. What it cannot do alone is **prove each step is valid** —
model-generated math comes with model-generated confidence, sometimes right,
sometimes wrong, with no mechanical check.

ToyMath inverts the usual CAS design:

- **The agent decides strategy.** It picks the next move: subtract 3 from
  both sides, expand, factor the quadratic, integrate by parts with this
  particular u and dv.
- **ToyMath executes and verifies.** Each primitive is a small, narrow,
  battle-tested transformation. Every result is additionally spot-checked by
  a **numeric oracle** that shares nothing with the symbolic code — random
  sample points, Schwartz–Zippel style. A bug in either leg is caught by the
  other.
- **The ledger is the artifact.** Every step is recorded with its hash,
  arguments, result, assumptions, and check status. `replay` re-runs the
  whole derivation and confirms it. The model's prose can never enter the
  ledger — only tool executions can — so a hallucinated step is structurally
  impossible.

Side conditions are **recorded, not proved**: dividing both sides by
`a + b` stamps the step with *assumes `a + b ≠ 0`*, and the accumulated
assumptions travel with the derivation — honestly conditional, the way a
good teacher writes it on the board.

There is deliberately **no `solve`, no `simplify`, no autonomous
`integrate`, no general `factor`**. Any of them would collapse the visible
chain of steps into one opaque move — and the visible chain is the point.
Smart operations are split into named tactics the agent chooses between.

Honest wording: results are *mechanically checked*, not *proved*. The
ledger eliminates model hallucination, not implementation bugs — which is
why every primitive is verified twice, symbolically and numerically.

## The `do!` agent endpoint

A cell starting with `do!` hands the rest of the cell to an agent (OpenAI
Agents SDK over OpenRouter) whose only way to do math is calling the
verified primitives:

- Steps stream into the result cell as they are verified, each rendered as
  `sN#hash [ok] op: input ⟹ result` with its per-step assumptions;
  failed checks show in red.
- `[[n]]` references the result of cell *n* — plain math cells and `do!`
  cells chain freely, so a derivation can span the notebook.
- All `do!` cells share one notebook-wide ledger; each cell renders the
  steps it added, and a replay verifies the entire chain.
- The agent designates the cell's final value (validated by parsing), which
  becomes what later cells see through `[[n]]`.
- **Plotting**: the agent can render matplotlib/seaborn figures — but the
  code runs *outside* the kernel, in a Pyodide WASM sandbox under Deno's
  deny-by-default permissions (no filesystem, no environment, network only
  to the package CDNs). Figures appear inline captioned *"illustration, not
  machine-checked"*: plots are illustrations, never evidence, and never
  ledger steps.

Configuration (`.env` in the repo root):

| Variable | Meaning |
|---|---|
| `OPEN_ROUTER` | OpenRouter API key (required for `do!`) |
| `OPENROUTER_MODEL` | model slug, default `anthropic/claude-sonnet-5` |
| `TOYMATH_SANDBOX` | `auto` (default) / `pyodide` / `off` — plot sandbox |

Plotting needs [Deno](https://deno.com) (`brew install deno`); without it
the plot tool simply isn't offered to the agent.

## The verified primitives

Every transforming primitive returns one JSON record
`{ok, op, args, input, result, assumptions, check}` where `check` is the
independent numeric spot-check. The same primitives back the `do!` agent,
the CLI, and the Claude Code skill.

| Primitive | What it does |
|---|---|
| `substitute(expr, var, value)` | replace a variable, juxtaposition-safe |
| `apply_both_sides(eq, op, arg)` | `+ - * / ^` on both sides of `=` or an inequality; records `arg ≠ 0` when dividing by a symbol; flips the relation when multiplying an inequality by a negative |
| `expand(expr)` | distribute and canonicalize; outside the rational fragment it works over *opaque atoms*, so `2\sin x + 3\sin x → 5\sin x` |
| `collect(expr, var)` | group by powers of a variable — `x\sin x + x\cos x → (\sin x + \cos x)x` |
| `evaluate(expr)` | exact arithmetic; on an equation reports `holds: true/false` |
| `differentiate(expr, var)` | exact on the rational fragment, ~20 mechanical rules beyond it; verified by central differences |
| `rewrite(expr, lemma, direction)` | apply a registered identity (`diff_squares`, …) — the lemma library is the extensibility point |
| `factor_gcd` / `factor_quadratic` | named factorings in place of a general `factor`; applicable relation sides are transformed and checked independently |
| `integrate_power_rule` / `integrate_table` / `integrate_by_parts(u, dv)` / `integrate_substitute(u, u_var, new_integrand)` | integration as agent-chosen tactics; antiderivatives verified by differentiating back |
| `equal?(e1, e2)` | the checker: verdict **yes / no / unknown** — canonical on the rational fragment, numeric oracle beyond it, honest about undecidability |

The most under-appreciated one is `apply_both_sides`: mainstream CAS hide
it behind `solve`. For verified derivations it is the whole game — it makes
equation solving transparent, step-by-step, and mechanically sound.

### CLI

The same workflow scripts from a terminal (one deterministic JSON object
per call, `--session` appends to a replayable ledger):

```bash
python toymath_cli.py apply "2x + 3 = 7" - 3 --session d.json
python toymath_cli.py expand "2x+3 - 3 = 7 - 3" --session d.json   # -> 2x = 4
python toymath_cli.py collect "x \sin x + x \cos x" x              # -> (\sin x + \cos x)x
python toymath_cli.py equal "\sin(2x)" "2 \sin x \cos x"           # -> yes
python toymath_cli.py replay --session d.json                      # re-verify everything
```

A Claude Code skill (`.claude/skills/toymath/`) teaches an agent the same
protocol — and the `do!` endpoint generates its instructions from that very
skill file, so the two can't drift apart.

See [doc/PRIMITIVES.md](doc/PRIMITIVES.md) for the full design: the trust
model, opaque atoms, the ledger format, and what is deliberately absent.

## Commands, and how they compose

A `do!` cell is free-form. A **notebook command** is a `do!` you can name and
reuse: a Markdown file in `commands/` with a `$ARGUMENTS` placeholder becomes
a `name!` cell prefix.

```markdown
---
name: int
description: Apply symbolic integration, step by step
expr: true
---
Apply symbolic integration for $ARGUMENTS ...
```

Now `int! x^3` runs the agent with that focused instruction, and `commands!`
lists everything available. Discovery mirrors the classic kernel's `cmd_*`
auto-registration — drop a file in `commands/`, and the command exists.

Mark a command `expr: true` and it is no longer just a whole-cell prefix: it
**composes inside an expression**, alongside plain math and other commands.

```
{diff! {int! x^3}}          →  x^3        integrate, then differentiate
{int! x^2} + {int! x^2}     →  ⅔x³ + 2C   one call (memoised); sum checked
2 {diff! x^2} - 1           →  4x - 1
```

Each `{…!}` is resolved inner-to-outer into a verified sub-derivation
(identical sub-expressions cost a single agent call). The arithmetic **glue**
between the results is then handed to `expand`, so the composition is
confirmed by the numeric oracle — **not by another LLM call**. The cell's
ledger is the union of every sub-derivation and the final combine, and
`replay` verifies the whole composite.

### Three layers, no blind trust

This is the shape of the whole system, and composite commands are where it
becomes literal:

- **Strategy — the LLM.** Chooses the next move, or the whole plan.
- **Mechanism — deterministic algebra.** Computes it: the verified
  primitives (`expand`, `differentiate`, …) and, underneath them, the classic
  fixed-point rewrite engine.
- **Verification — the numeric oracle.** Shares nothing with the mechanism
  and re-checks every result at random sample points.

No layer trusts another on faith. The model's prose can never enter the
ledger — only tool calls can — so it cannot hallucinate a step; and any bug
in the algebra is caught by the oracle from the opposite side. A composite
like `{diff! {int! x^3}}` is just the point where all three meet in one
expression: the LLM resolves each command, the procedural `expand` combines
the pieces, and the oracle signs off on the combination. Strategy is cheap
and fallible, mechanism is fast but not self-verifying, verification is
independent and deliberately dumb — and the interesting behaviour is in how
they check each other.

## The classic kernel

Underneath all of this is the original ToyMath: a Jupyter kernel with LaTeX
as both input and output.

```
Input (LaTeX) → Parser → Notation Graph (DAG) → Processor → Output (LaTeX)
```

- **`notation.py`** — expressions live in a notation graph with structural
  sharing (Symbol, Func, Notation)
- **`processor.py`** — a fixed-point iteration engine: transformations are
  applied until nothing changes
- **`LatexParser.py` / `LatexWriter.py`** — LaTeX in, LaTeX out
- **`cmd_*.py`** — short `!`-commands, auto-discovered

Numeric constants evaluate automatically; symbolic transformations happen
when you ask for them with a `!`-command:

```latex
\frac{1}{2} + \frac{1}{3}      % → 5/6 (automatic numeric evaluation)
mul! x^{-1}x                   % → 1 (power cancellation)
mul! (a/b)^2                   % → a²/b² (power expansion)
add! \frac{a}{b}+\frac{c}{d}   % → \frac{ad+bc}{bd} (fraction addition)
add! {mul! (a+b)(c+d)} + x     % commands compose; nesting defers a step
```

These `mul!` / `add!` commands were the original way ToyMath did algebra, and
they no longer carry much of the weight. The verified primitives have
superseded them for real work: `expand` already multiplies, distributes, and
cancels, and every result it returns is oracle-checked — which the classic
rewrite rules are not. So read the classic engine as the **substrate**
ToyMath grew out of: the LaTeX parser, the notation graph, and the
command-node machinery. That last part still earns its keep — the parser has
always turned `{name! arg}` into a command node, and that is exactly the
syntax the composite `int!` / `diff!` commands reuse. The old command layer
and the new agentic one share one grammar; the classic `add! {mul! …}`
nesting above and the verified `{diff! {int! …}}` are the same idea, a
generation apart.

| Command | Role |
|---------|------|
| `mul!` / `add!` | the original symbolic algebra — superseded by the verified `expand` / `collect`, kept as the substrate |
| `do!` | the free-form agent endpoint |
| `int!` / `diff!` / `commands!` | notebook & composite commands (above) |
| `goal` / `rules` | Prolog-style logic layer: goals and transformation rules |
| `dump` / `track` / `debug` | introspection: notation graph, tracing |
| `echo-on` / `echo-off` / `clear` | session control |

Console mode without Jupyter: `python console.py`.

## Installation

Requirements: Python ≥ 3.11, [uv](https://docs.astral.sh/uv/),
JupyterLab ≥ 4.2 (installed by the requirements), optionally
[Deno](https://deno.com) for `do!` plotting.

```bash
git clone https://github.com/semyonc/toymath.git && cd toymath

# environment
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# register the kernel
jupyter kernelspec install kernel_spec --user

# for the do! endpoint: put the OpenRouter key in .env
echo 'OPEN_ROUTER=sk-or-...' >> .env

# launch
jupyter lab        # pick the "Toy Math" kernel
```

## Running tests

```bash
pytest                                  # all suites
pytest engine/unittests.py              # classic engine
pytest engine/unittests_frac.py         # fractions
pytest engine/unittests_primitives.py   # verified-derivation primitives
pytest engine/unittests_do.py           # do! endpoint (offline, scripted agent)

TOYMATH_LIVE_TESTS=1 pytest engine/unittests_do.py   # + live OpenRouter round-trip
TOYMATH_PLOT_TESTS=1 pytest engine/unittests_do.py   # + live Deno/Pyodide sandbox
```

The offline suites include a scripted fake model, so the whole agent loop
is tested without network access.

## Project structure

```
toymath/
├── engine/
│   ├── polyrat.py          # canonical core: Poly/RatFunc for the rational fragment
│   ├── primitives.py       # the verified primitives + numeric oracle
│   ├── ledger.py           # step ledger: record, render, replay
│   ├── agent_do.py         # do! endpoint: agent, tools, prompt from the skill
│   ├── prompt_commands.py  # discoverable name! commands from commands/*.md
│   ├── expr_commands.py    # inline composition: {diff! {int! x^3}} resolver
│   ├── plot_sandbox.py     # sandboxed plotting (Pyodide under Deno)
│   ├── pyodide_runner.mjs  # vendored Deno runner for the plot sandbox
│   ├── notation.py         # notation graph structures
│   ├── processor.py        # classic fixed-point engine
│   ├── mathShell.py        # kernel cell dispatch (math, do!, name!, composite)
│   ├── cmd_*.py            # classic kernel commands
│   └── unittests*.py       # test suites
├── commands/               # notebook & composite command templates (int.md, …)
├── toymath_cli.py          # agent-facing CLI
├── toymathkernel.py        # Jupyter kernel entry point
├── .claude/skills/toymath/ # Claude Code skill (also the do! prompt source)
├── doc/                    # PRIMITIVES.md, NOTATION.md, images
└── examples/               # example notebooks
```

## Documentation

- [doc/PRIMITIVES.md](doc/PRIMITIVES.md) — the verified-derivation design
- [doc/NOTATION.md](doc/NOTATION.md) — notation graph internals
- [CLAUDE.md](CLAUDE.md) — developer guide

## License

MIT License

## Author

Semyon C
