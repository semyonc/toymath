# AGENTS.md

Guidance for coding agents working in this repository. `CLAUDE.md` imports
this file — keep this the single source of truth.

## Project Overview

ToyMath is a LaTeX-native symbolic mathematics system: LaTeX in, LaTeX out,
expressions stored as a notation DAG (not a syntax tree). It runs as a
Jupyter kernel, a console app, and an agent-facing CLI.

Two layers coexist:

1. **Agent-scoped verified-derivation primitives** — the current focus
   (developed on topic branches). The agent (or user) decides strategy; each
   primitive executes one narrow, named, mechanically checked step; an
   independent numeric oracle spot-checks every transformation; the session
   ledger is the replayable artifact. Read `docs/PRIMITIVES.md` and the
   relevant domain `SKILL.md` before touching this layer.
2. **Legacy fixed-point engine** — the original autonomous simplifier
   (`processor.py` + `cmd_*.py` rewrite commands). Still powers plain math
   cells in the kernel/console. Kept working, deliberately not extended
   (see guardrails below).

## Key Files

| File | Purpose |
|------|---------|
| `engine/primitives.py` | Shared parse/write, binder/substitution, result-record, and independent numeric-oracle infrastructure |
| `engine/tactics/*.py` | Static tactic implementations grouped by owning skill (`core`, differentiation, equations, integration, limits, finite operators) |
| `engine/tactic_registry.py` | Single allowlist/schema for tactic invocation, CLI generation, replay dispatch, provenance validation, and skill ownership |
| `engine/tactic_skills.py` | Discovery and progressive rendering of committed domain skills |
| `engine/polyrat.py` | Canonical core for the rational fragment: sparse `Poly`, `RatFunc` with cancellation, `to_ratfunc`/`ratfunc_to_notation` |
| `engine/ledger.py` | Step ledger: JSON persistence, assumption accumulation, replay verification |
| `toymath_cli.py` | Agent-facing CLI; one deterministic JSON object per call |
| `engine/agent_do.py` | `do!` Jupyter endpoint: small stable tool surface (`load_skill`, `run_tactic`, ledger controls) over the tactic registry |
| `engine/expr_commands.py`, `engine/prompt_commands.py` | Composite/inline command resolution; notebook prompt-commands loaded from `commands/*.md` |
| `engine/plot_sandbox.py` | Sandboxed figure backends for `do!`: Python under Pyodide/Deno (`pyodide_runner.mjs`), TeX under node-tikzjax (`tikz_runner.mjs`) |
| `engine/processor.py` | MathProcessor, Calculator — legacy fixed-point iteration engine |
| `engine/notation.py` | Symbol, Func, Notation — DAG representation |
| `engine/replicator.py` | Base class for graph walking (visitor pattern) |
| `engine/LatexParser.py`, `engine/lexer.py` | LaTeX → notation (ply-based) |
| `engine/LatexWriter.py` | Notation → LaTeX |
| `engine/comparer.py` | Structural pattern matching (used by lemmas/rewrite) |
| `engine/cmd_mul.py`, `engine/cmd_add.py` | Legacy rewrite commands: fraction/power rules |
| `engine/frac_utils.py`, `engine/value.py` | Fraction utilities; IntegerValue/FracValue/FloatValue |
| `engine/prolog.py` | Legacy logic layer — do not build new features on it |
| `engine/mathShell.py` | Kernel shell: cell dispatch (`do!`, notebook commands, math cells) |

## Architectural Guardrails

- **Do not grow the fixed-point rewrite rules** (`cmd_mul.py`/`cmd_add.py`)
  to fix polynomial or rational defects — canonical forms in `polyrat.py`
  are the answer there. The rewrite layer is frozen except for genuine bugs.
- The primitives layer never gains `solve`, `simplify`, autonomous
  `integrate`, or a general `factor`. Smart operations are split into named
  tactics the agent chooses between (`integrate_by_parts`, `factor_gcd`,
  `factor_quadratic`, …).
- **Record assumptions, don't prove them** — e.g. dividing both sides by a
  non-constant records `arg ≠ 0` in the step.
- Every transforming primitive returns
  `{ok, op, args, input, result, assumptions, check}` and is spot-checked
  by the numeric oracle (or marked exact). An unchecked transformation is a
  regression.
- The numeric oracle must share **nothing** with the symbolic code path —
  the two independent trust legs are the design.
- Never mutate an input notation; build results into fresh/cloned ones.
- Say "mechanically checked", never "proved".
- Skill Markdown guides tactic choice; it never grants execution authority.
  Only the allowlisted registry can invoke code or admit replayable steps.
- Core code is fair game: when a fix belongs in the parser grammar, lexer,
  writer, comparer, or replicator, make it there rather than layering
  workarounds. After grammar changes: regenerate the tables, check
  `grep -c conflict parser.out` stays 0, and commit the regenerated
  `engine/parsetab.py` (it is tracked; `parser.out` is not).

## Two Command Systems (don't conflate)

1. `engine/cmd_*.py` rewrite commands (`mul!`, `add!`) run **during the
   notation walk**, inside expressions, auto-discovered by
   `processor.register_actions()`.
2. Notebook commands (`do!`, `int!`, `diff!`, …) are **cell prefixes**
   handled in `mathShell.exec()` before the LaTeX path, loaded from
   `commands/*.md` by `prompt_commands.load_commands()`; inline
   `{name! …}` composition is resolved by `expr_commands.ExprResolver`.

Never give a `commands/*.md` file the same name as a registered `cmd_*`
action (e.g. `mul`) — it would silently reroute every cell containing that
command away from the fixed-point engine.

## Extending the Verified Tactic Layer

Do not add another function tool to `engine/agent_do.py`, another handwritten
CLI dispatch branch, or another replay lambda. The model-facing tool set must
stay constant as subject coverage grows.

For a new tactic:

1. Implement the narrow transformation in its owning module under
   `engine/tactics/`, with the standard result record and an independent
   oracle check (or `exact`). Put only genuinely cross-domain infrastructure
   in `engine/primitives.py`.
2. Add one `TacticSpec` to `engine/tactic_registry.py`: stable public name,
   stable ledger `op`, owning skill, ordered arguments, summary, and callable.
   The registry automatically supplies CLI parsing/dispatch, transforming-op
   classification, do! dispatch, discovery, and replay.
3. If the tactic consumes earlier ledger results, add its session adapter and
   replay provenance validator to that same registry entry. Persist source ids
   in the result; provenance is load-bearing, not decorative.
4. Put strategy, ordering, examples, and pitfalls in the owning domain file
   under `.claude/skills/toymath/domains/*/SKILL.md`. Do not copy exact
   signatures into Markdown: they are rendered from the registry. Create a
   new domain skill only for a coherent subject workflow, not per tactic.
5. Add primitive/oracle tests plus registry, CLI-compatibility, do! dispatcher,
   skill-gating, and replay tests as applicable. Run the full offline suite.

Keep `docs/OVERVIEW.md` as the system tour and `docs/PRIMITIVES.md` as the
trust/extension contract. They link to skills and registry discovery rather
than duplicating an exhaustive tactic manual. `python toymath_cli.py skills`,
`tactics [--skill NAME]`, and `describe TACTIC` are the generated reference.

## Running

```bash
source .venv/bin/activate            # or PYTHONPATH=. to avoid import errors

python console.py                    # console mode
jupyter notebook                     # kernel mode (LaTeX kernel, registered by toymathkernel.py)
python toymath_cli.py expand "(x+1)^2" --session s.json   # agent CLI

uv pip install -r requirements.txt   # dependencies
```

## Testing

```bash
.venv/bin/python -m pytest -q                       # full suite
pytest engine/unittests.py                          # legacy core
pytest engine/unittests_frac.py                     # fractions
pytest engine/unittests_primitives.py               # verified-derivation primitives
pytest engine/unittests_do.py                       # do! endpoint (offline scripted agent)
TOYMATH_LIVE_TESTS=1 pytest engine/unittests_do.py  # + live OpenRouter test
```

## Legacy Engine: Fixed-Point Iteration

Plain math cells flow: LaTeX → `LatexParser` → notation DAG → MathProcessor
fixed-point loop → `LatexWriter` → LaTeX.

```python
while True:
    calculator = Calculator(notation, output_notation, actions, model)
    result = calculator(expression)
    if s_equal(result, output_notation, expression, notation):
        break  # Converged
    notation, expression = output_notation, result
    output_notation = Notation()
```

Principles: transformations must be **monotonic** (no oscillation) and
**idempotent** (applying twice = applying once); nested commands process
across iterations via `chainexpr()`; each iteration builds into a fresh
notation graph — never mutate the input notation, always build into
`processor.output_notation`.

### Adding a rewrite command

Create `engine/cmd_yourname.py`:

```python
class YourCommand(object):
    arity = 1

    def exec(self, processor, sym, f):
        # f.args[1] = arguments tuple
        # build the result into processor.output_notation
        # use chainexpr() to defer nested commands to the next iteration
        pass

def create_actions():
    return {'yourname': YourCommand()}
```

## Notation Structures

| Type | Structure | Example |
|------|-----------|---------|
| INDEX | `(base, (sub, sup_l, power, sup_r))` | `x^2` → `(x, (None, None, 2, None))` |
| FACTORIAL | `(operand,)` | `n!` → `(n,)` |
| BINOM | `(upper, lower)` | `\binom{n}{k}` → `(n, k)` |
| P_LIST | `(factor1, factor2, ...)` | `xy` → `(x, y)` |
| S_LIST | `(term1, +term2, ...)` | `x+y` → `(x, +y)` |
| GROUP | `(inner,)` with `br` prop | `{x}` → `(x,)` br="{}" |
| FUNC | `(name, args)` | `\frac{a}{b}` |

## Fractions and Powers (legacy rules)

Dual-path system: numeric `\frac{1}{2}` becomes `FracValue` via the
Preprocessor and is evaluated immediately; symbolic `\frac{a}{b}` stays a
graph and is handled by rules in `cmd_mul.py`/`cmd_add.py`
(fraction × fraction, scalar × fraction, distribution, cross-multiplied
addition). Shared helpers live in `frac_utils.py`: `is_frac`,
`get_numerator`/`get_denominator`, `normalize_frac`.

Power handling (`cmd_mul.py`): `(\frac{a}{b})^n` expands to a fraction of
powers (inverted for negative `n`), `x^{-n}` → `\frac{1}{x^n}`. Negative
and zero exponents are detected by direct INDEX inspection —
`NotationParam.N` only matches positive integers, so pattern matching
cannot see them.

## Development Patterns

Unwrap GROUP/MINUS wrappers before type checks:

```python
def unwrap(notation, sym):
    f = notation.getf(sym, Notation.GROUP)
    if f: sym = f.args[0]
    f = notation.getf(sym, Notation.MINUS)
    if f: sym = f.args[0]
    return sym
```

Defer nested evaluation to the next iteration:

```python
from cmd_mul import chainexpr, Mul
nested = chainexpr(Mul.MUL, notation, plist, None)
```

Guard against re-expansion (idempotence):

```python
if notation.getf(x, Notation.INDEX) or notation.getf(y, Notation.INDEX):
    return None
```

Extract numeric values through wrappers:

```python
from processor import get_value
n = get_value(power, notation)
if isinstance(n, IntegerValue): ...
```

## Important Notes

- Symbols use name-based equality: `Symbol('x') == Symbol('x')`. Relation
  nodes all share the name `comp` — always check `f.sym.props['op']`.
- `!` is a dedicated postfix token. A bang-word becomes a command token only
  when its name is in `MathLexer.KNOWN_COMMANDS` (or is supplied by the live
  command registry); keep the static table's registry-parity test green when
  adding committed commands.
- Use a Replicator to copy expressions between notation contexts.
- `parser.out` is a gitignored cache; `engine/parsetab.py` is tracked —
  commit it whenever the grammar changes.
- Rewrite commands are auto-discovered via `register_actions()`; notebook
  commands via `prompt_commands.load_commands()`.
- The `do!` agent endpoint requires `OPEN_ROUTER` in `.env` (model via
  `OPENROUTER_MODEL`, default `anthropic/claude-sonnet-5`).
- `do!` figures need Deno (`brew install deno`) — toggle both off with
  `TOYMATH_SANDBOX=off`. Two backends, deliberately separate processes
  with different grants (see `plot_sandbox.py`):
  - `plot` runs agent **Python** (matplotlib/seaborn/plotly) under
    Pyodide. Since the agent's code executes here and Pyodide's `js`
    bridge is gated only by Deno's flags, it gets no `--allow-env`.
    Wheels cache in `TOYMATH_WHEEL_CACHE` (default
    `~/.cache/toymath/wheels`).
  - `tikz` renders agent **TeX** to SVG via node-tikzjax, offline. No
    agent code runs there — TeX executes in a wasm engine over an
    in-memory filesystem — which is the only reason it can afford
    `--allow-sys`/`--allow-env`. Never render agent Python in it.
  Both spawn with a scrubbed env, so a widened grant still reaches no
  secrets. Figure kinds are `png`/`html`/`svg`; only `html` (plotly) needs
  the network at view time.

## Reference Reading

- `docs/PRIMITIVES.md` — the primitives, trust legs, ledger, `do!` endpoint,
  notebook/composite/direct command tiers
- `docs/NOTATION.md` — notation DAG and walking-pattern details
- `docs/OVERVIEW.md` — system tour: architecture, interfaces, repository map
- `engine/cmd_*.py` — legacy command implementations and examples
