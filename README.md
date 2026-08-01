# ToyMath

ToyMath is a LaTeX-native symbolic mathematics system built around a simple
idea: let an LLM choose the strategy, but never ask it to certify its own
algebra.

The agent works through narrow, named transformations. ToyMath executes each
move, checks it independently, and records the derivation in a replayable
ledger.

```text
do! differentiate x^3 - 3x, find where the derivative is zero, plot with those points marked
```

See this cell run in the [intro notebook](examples/Into.ipynb) — six checked
ledger steps, a sandboxed plot with the critical points marked, and a
chainable result — alongside a stepwise integral and a machine-checked
limit proof.

## What is different?

Most tools combine mathematical strategy and execution in one place. ToyMath
separates three roles:

1. **Strategy — the agent.** It decides whether to expand, factor, substitute,
   split a task, or use an integration tactic.
2. **Mechanism — deterministic transformations.** ToyMath performs the
   requested algebra on its notation graph.
3. **Verification — an independent numeric oracle.** The result is checked by
   a path that shares no symbolic implementation with the transformation.

| | Conventional CAS | LLM-only math | ToyMath |
|---|---|---|---|
| Strategy | hidden inside large operations | visible but unconstrained | chosen by the agent |
| Intermediate steps | often reconstructed afterward | generated as prose | produced by named tools |
| Checking | trusts the symbolic engine | usually self-checks | independent oracle per move |
| Artifact | final answer | conversation | replayable derivation ledger |
| Side conditions | often implicit | easy to omit | recorded with the step |

There is deliberately no general `solve`, `simplify`, `factor`, or autonomous
`integrate`. Those operations would hide the chain that ToyMath is designed to
preserve. Instead, the agent composes smaller tactics such as
`apply_both_sides`, `expand`, `factor_quadratic`, `integrate_by_parts`, and
`substitute`.

This restraint is not only an interface choice — it is what the mathematics
allows. By [Richardson's theorem](https://en.wikipedia.org/wiki/Richardson%27s_theorem)
(1968), once expressions are built from a variable, rational constants and π,
arithmetic, `sin`, `exp`, and absolute value, deciding whether an expression
is identically zero is undecidable — and with it, deciding equality. A
complete, always-correct `simplify` cannot exist; every CAS trades quietly
between incompleteness and heuristics. ToyMath draws the boundary explicitly
instead: exact canonical algebra on the decidable polynomial/rational
fragment, an independent probabilistic oracle beyond it, and an honest
`unknown` where neither applies. Strategy — the part no algorithm can
complete — goes to the agent.

Results are described as **mechanically checked**, not proved. Canonical
algebra is exact where supported; the independent oracle is reproducible but
probabilistic outside that fragment. Assumptions such as `a + b \ne 0` remain
attached to the derivation.

## Surfaces

- **Jupyter:** `do!` accepts a free-form instruction; `[[n]]` chains results
  across cells; steps stream into a notebook-wide ledger. `backend!` and
  `model!` set notebook-local routing — the model catalog follows the active
  backend (`engine/models.yaml` and `model! MODEL, PROVIDER...` for
  OpenRouter, the account's own list for Codex); native completion opens
  after `model! ` and the toolbar title tracks the active selection.
- **Named commands:** `name!` loads reusable instructions from `commands/*.md`.
- **Inline composition:** `{name! expression}` embeds checked sub-derivations
  inside a larger expression.
- **CLI:** `toymath_cli.py` exposes the same primitives as deterministic JSON
  commands and can save or replay a ledger.
- **Classic kernel:** the original fixed-point LaTeX rewrite engine remains the
  parser, notation-DAG, and command substrate beneath the agentic layer.

Figures are optional illustrations, never evidence: they do not enter the
ledger and replay ignores them. Agent-generated figure code runs outside the
kernel in deny-by-default Deno sandboxes — Python (matplotlib/seaborn/plotly)
under Pyodide, and TeX/TikZ through a WebAssembly TeX engine that renders to
SVG with no network access at all.

### A cell reads as the mathematics it is, not as its source

A dense formula is hard to read as LaTeX source. In JupyterLab, a ToyMath cell
shows its formula typeset while you are not editing it, and its source the
moment you are — the bargain a markdown cell makes, applied to code cells. One
click on the rendered input opens the editor with the cursor placed; leaving
the cell renders it again. *Render ToyMath Cell Input* in the command palette
turns it off for a notebook.

The kernel decides what a cell renders as, because the kernel owns the parser:
what you see is what the **engine** understood, not a second reading of the
source by the frontend. `\int \frac {dx} (x+1)` — the ToyMath dialect, taking a
parenthesised second operand — renders as the fraction it actually parses to,
which MathJax on the raw source would get wrong. A `[[n]]` backreference
renders as the formula it stands for, and a cell the engine cannot read keeps
its source, so an unparsable cell is visible as one before you run it.

A cell is read as a whole formula first, and only failing that as prose with
formulas buried in it. So `int! \int x^2 dx` renders as one formula, while

```text
do! differentiate x^3 - 3x, find where the derivative is zero
```

renders as that sentence with `x^3 - 3x` typeset inside it — no `$…$`
required, since nobody writes them in a prompt. Only a command that hands its
argument to the agent is read this way; a plain cell and a rewrite action are
one expression.

Every rendered fragment must parse back to the same expression as the
characters it replaces, so a cell can never show something other than what it
runs.

## Agent backends

`do!` runs on one of two providers, selected per notebook with `backend!`:

```text
backend!             # show the effective backend and why it was selected
backend! openrouter  # a shared OpenRouter key
backend! codex       # your own ChatGPT subscription (experimental)
backend! auto        # default: resolve from what is configured
```

Both drive the *same* tool surface. Names, descriptions, schemas, and
handlers come from one canonical definition, so the model sees an identical
set of tactics whichever provider is running, and a parity test fails if the
two adapters ever diverge. The trust boundary does not move either: the
tactic registry, the independent oracle, and the ledger remain the only
authority for a checked result.

A run **never fails over**. A Codex rate limit will not silently create
OpenRouter charges, and an OpenRouter outage will not silently consume a
Codex allowance. Switching providers is always an explicit act.

### No OpenRouter account? Use the one you already pay for

The OpenRouter backend needs an API key with credits — usage is billed per
token on top of whatever you already subscribe to. The **Codex backend
removes that second bill**: it runs `do!` through the Codex app-server against
your personal ChatGPT account, so the work draws on the Plus/Pro subscription
you already have, under that plan's own quota and rate limits. There is no
ToyMath-side key, no shared credential, and no per-token charge.

```bash
uv pip install ".[codex]"      # pins the Codex runtime ToyMath was validated against
```

```text
login!               # managed ChatGPT sign-in, opens your browser
login! device        # device-code flow, for headless machines
login! status        # auth mode and plan type
login! logout
```

Then `backend! codex`, or leave `backend! auto` — with no OpenRouter key
configured, a signed-in account is selected automatically.

`model!` lists what your account offers rather than the OpenRouter catalog —
`gpt-5.6-sol`, `-terra`, `-luna`, and older releases with the currently
pinned runtime. ToyMath hardcodes no Codex model: with no explicit choice the
runtime's own default is used, so the two catalogs never mix.

What ToyMath does and does not touch:

- **Authentication is managed by the app-server.** ToyMath receives a sign-in
  URL, a completion notification, and an account status. It never handles,
  stores, logs, or displays a ChatGPT token, and it keeps no account
  identifier or email. The general `~/.codex` home and its `auth.json` are
  never read or copied.
- **Only a managed ChatGPT account runs a derivation.** The runtime also
  reports API-key and Amazon Bedrock accounts as signed in; those bill a
  different — possibly organizational — credential, so they are refused with
  an explanation rather than spent on a math run.
- **A sign-in challenge never reaches the saved notebook.** Jupyter persists
  cell output into the `.ipynb`, so the browser flow hands its one-time URL
  to the OS browser and prints no link; the device code, which has to be
  readable while you type it, is cleared from the cell as soon as the flow
  ends.
- **Each run is contained.** A fresh ephemeral thread, an empty working
  directory outside the repository, a read-only sandbox, no approvals, and
  every capability switch off, in a dedicated Codex home separate from your
  CLI one.

## Quick start

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and JupyterLab.
[Deno](https://deno.com) is optional, and enables the sandboxed `plot` and
`tikz` figure tools. No LaTeX installation is needed for TikZ.

```bash
git clone https://github.com/semyonc/toymath.git
cd toymath
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install --no-deps .             # installs the prebuilt Lab extension
jupyter kernelspec install kernel_spec --user --name toymath --replace

jupyter lab
```

Pick the **Toy Math** kernel. Console mode is also available with
`python console.py`.

Plain math cells work with no credentials at all. `do!` and prompt commands
need an agent backend — pick one:

```bash
# A: OpenRouter — an API key with credits, billed per token
echo 'OPEN_ROUTER=sk-or-...' >> .env

# B: your own ChatGPT subscription — no OpenRouter account needed
uv pip install ".[codex]"     # then run `login!` in a notebook cell
```

See [Agent backends](#agent-backends) for what each one costs and contains.

## Read more

- [Project overview](docs/OVERVIEW.md) — architecture, interfaces, commands,
  setup, testing, and repository map
- [JupyterLab extension](jupyterlab-extension/README.md) — build and install
  the native `model!` completion and live model-title integration.
- [Verified primitives](docs/PRIMITIVES.md) — trust model, oracle, tactics, and
  ledger format
- [Notation graph](docs/NOTATION.md) — DAG representation and traversal
- [Developer guide](AGENTS.md) — classic engine and extension details

Run the offline test suite with `pytest`. Live OpenRouter and sandbox probes
are opt-in; the commands are documented in the project overview.

## Environment variables

All are optional — ToyMath runs plain math cells with none of them set. Those
marked *(secret)* belong in `.env`, which is git-ignored; nothing here is ever
written into a notebook, a ledger, or a trace.

### Agent backends

| Variable | Default | Purpose |
|---|---|---|
| `OPEN_ROUTER` | — | OpenRouter API key *(secret)*. Enables the OpenRouter backend, and selects it under `backend! auto`. |
| `OPENROUTER_MODEL` | `openai/gpt-5.6-luna` | Default OpenRouter model. `model!` overrides it per notebook. |
| `TOYMATH_AGENT_BACKEND` | unset | Forces `openrouter` or `codex`. Outranked only by an explicit `backend!` in the notebook. |
| `TOYMATH_CODEX_MODEL` | the account's own default | Default model for the Codex backend. ToyMath hardcodes none — the runtime chooses (`gpt-5.6-sol` with the currently pinned one). `model!` overrides it per notebook. |
| `TOYMATH_CODEX_HOME` | `~/.toymath/codex-home` | ToyMath's dedicated Codex home. Must not be the general one; a directory ToyMath did not create is refused rather than overwritten. |
| `CODEX_HOME` | `~/.codex` | Read only to know which home to stay out of. ToyMath never reads its contents or its `auth.json`. |

Backend resolution under `backend! auto`, in order: an explicit `backend!` in
this notebook → `TOYMATH_AGENT_BACKEND` → OpenRouter if `OPEN_ROUTER` is set →
Codex if a managed ChatGPT account is signed in. If nothing is configured, the
cell says so instead of guessing.

### Figure sandboxes

| Variable | Default | Purpose |
|---|---|---|
| `TOYMATH_SANDBOX` | `auto` | `auto` \| `pyodide` \| `off`. `off` unregisters the `plot` and `tikz` tools entirely. |
| `TOYMATH_WHEEL_CACHE` | `~/.cache/toymath/wheels` | Wheel cache for the Pyodide sandbox; without it seaborn and plotly re-download on every call. |

### Observability (off by default)

| Variable | Default | Purpose |
|---|---|---|
| `TOYMATH_OBSERVABILITY` | `off` | Set `on` to send one Langfuse trace per `do!` run. Never touches the ledger or oracle; any failure downgrades to a warning. |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse credential *(secret)*, required when tracing is on. |
| `LANGFUSE_SECRET_KEY` | — | Langfuse credential *(secret)*, required when tracing is on. |
| `LANGFUSE_BASE_URL` | `cloud.langfuse.com` | Langfuse host. `LANGFUSE_HOST` is accepted as a fallback. |

### Opt-in tests

| Variable | Purpose |
|---|---|
| `TOYMATH_LIVE_TESTS=1` | Adds a live OpenRouter derivation test (spends credits). |
| `TOYMATH_CODEX_LIVE_TESTS=1` | Adds a live Codex derivation test (needs `login!` first). |
| `TOYMATH_PLOT_TESTS=1` | Adds the Deno figure-sandbox probes. |

MIT License · Semyon C
