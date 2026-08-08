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

## What a checked derivation actually asserts

Results are described as **mechanically checked**, never *proved*, and the
distinction is precise rather than modest.

A ledger is a finite sequence of steps `s₁ … sₙ`. Each `sᵢ` records an input
`eᵢ`, a result `e'ᵢ`, an operation drawn from a fixed allowlist, and a set of
side conditions `Aᵢ`. Write `P` for the derivation's **premises**: the inputs
that no earlier step produced. What the artifact asserts is

```text
    for each i:   A₁ … Aᵢ  ⊢  eᵢ ≡ e'ᵢ        (mechanically checked)
    conclusion holds relative to P
```

and three qualifications come with it.

**1. Per step, and conditional on `P`.** Each step is checked on its own; the
ledger checks *transformations*, so an input nothing derived is never checked
at all. A derivation must start somewhere, so premises are legitimate — but
they are exactly where verification stops, and every view lists them. A green
ledger over a mis-stated premise is honestly green and materially wrong. Read
the premise list before the answer.

**2. Two verification strengths, not one.** On the polynomial/rational
fragment, `e ≡ e'` is decided by canonical form: a decision procedure, exact,
no sampling. Outside it, the oracle evaluates both sides at reproducible
sample points. The design intent is the Schwartz–Zippel bound — for a nonzero
polynomial of total degree `d` and a finite sample set `S`,
`Pr[p(r) = 0] ≤ d/|S|` — but the implemented evaluator works in floating
point, so that bound is indicative rather than attained, and the oracle
additionally measures its own numerical noise before reporting a difference.
Agreement over sampled points is *quantified evidence*, not a proof.

**3. Side conditions are recorded, not discharged.** Dividing by `a + b`
records `a + b ≠ 0`; it does not prove it. The derivation is honestly
conditional, and the conditions travel with it.

### Where a wrong answer can still survive

A guarantee is only worth what its failure modes are. There are three, and
the second and third are **immune to oracle independence by construction** —
independence governs how consequences are *computed*, never what is *assumed*
or what the notation *means*:

| | failure | defended by |
|---|---|---|
| 1 | a step's transformation is wrong and both legs miss it | the two independent legs; the oracle shares no algebra with the symbolic path |
| 2 | every step is valid but rests on an asserted premise | premises are derived and displayed — visibility, not prevention |
| 3 | both legs read the input the same wrong way | nothing; unambiguous notation is the only defence |

The design target is (1), and that is where the two-leg structure earns its
cost. (2) and (3) are boundaries of the method, not defects to be patched
away, and they are documented in [docs/PRIMITIVES.md](docs/PRIMITIVES.md).

The honest summary: **ToyMath removes hallucinated algebra from the chain. It
does not certify the chain's starting point, and it does not know what you
meant.**

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
backend! codex       # recommended — your own ChatGPT subscription (experimental)
backend! openrouter  # fallback — an OpenRouter API key, billed per token
backend! auto        # default: resolve from what is configured
```

**Codex is the recommended path, and OpenRouter is the fallback.** The reason
is not backend quality — the two are equivalent by construction. Both drive
the *same* tool surface: names, descriptions, schemas, and handlers come from
one canonical definition, so the model sees an identical set of tactics
whichever provider is running, and a parity test fails if the two adapters
ever diverge. The trust boundary does not move either; the tactic registry,
the independent oracle, and the ledger remain the only authority for a checked
result.

What differs is **which models you will realistically run**, and that matters
more here than in most agent tools.

- **A weak model does not merely fail more often — it fails worse.** The
  characteristic failure is not giving up; it is asserting a starting point and
  then computing confidently past it. Every recorded step checks out, and the
  conclusion is wrong because its premise was. That is the second failure mode
  in the table above, and it is the expensive one: a refusal costs a retry,
  while a green ledger over a mis-stated premise costs your trust in the
  artifact. Read the premise line, whatever backend you use.
- **A strong model on OpenRouter is not cheap.** Billing is per token on top of
  whatever you already subscribe to, and a derivation is many turns — the
  reduction-formula example below runs to roughly thirty checked steps.

The Codex backend removes that second bill: it runs `do!` through the Codex
app-server against your personal ChatGPT account, so the work draws on the
Plus/Pro subscription you already have, under that plan's own quota and rate
limits, with access to that account's strong models. There is no ToyMath-side
key, no shared credential, and no per-token charge.

**Reach for OpenRouter when** you have no ChatGPT subscription, when you need a
specific model that account does not offer, or when you are running headless or
in CI where an interactive sign-in is impractical.

A run **never fails over**. A Codex rate limit will not silently create
OpenRouter charges, and an OpenRouter outage will not silently consume a
Codex allowance. Switching providers is always an explicit act.

### Setting up the recommended path

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
# A (recommended): your own ChatGPT subscription — no per-token bill
uv pip install ".[codex]"     # then run `login!` in a notebook cell

# B (fallback): OpenRouter — an API key with credits, billed per token
echo 'OPEN_ROUTER=sk-or-...' >> .env
```

See [Agent backends](#agent-backends) for what each one costs and contains,
and why the model you end up running matters more than the provider.

## Read more

- [Project overview](docs/OVERVIEW.md) — architecture, interfaces, commands,
  setup, testing, and repository map
- [JupyterLab extension](jupyterlab-extension/README.md) — build and install
  the native `model!` completion and live model-title integration.
- [Verified primitives](docs/PRIMITIVES.md) — trust model, oracle, tactics, and
  ledger format
- [Notation graph](docs/NOTATION.md) — DAG representation and traversal
- [Developer guide](AGENTS.md) — classic engine and extension details

Run the offline test suite with `pytest`. Live provider probes (OpenRouter and
Codex have separate opt-in switches) and sandbox probes are off by default; the
commands are documented in the project overview.

## Environment variables

All are optional — ToyMath runs plain math cells with none of them set. Those
marked *(secret)* belong in `.env`, which is git-ignored; nothing here is ever
written into a notebook, a ledger, or a trace.

### Agent backends

| Variable | Default | Purpose |
|---|---|---|
| `OPEN_ROUTER` | — | OpenRouter API key *(secret)*. Enables the OpenRouter backend, and selects it under `backend! auto`. |
| `OPENROUTER_MODEL` | `openai/gpt-5.6-luna` | Default model for this backend. `model!` overrides it per notebook — set it when the endpoint below is not OpenRouter. |
| `TOYMATH_OPENAI_BASE_URL` | OpenRouter | Points the same backend at any OpenAI-compatible endpoint — Ollama (`http://localhost:11434/v1`), vLLM, LM Studio, a gateway. Also selects the backend under `backend! auto`. Provider order does not apply there. |
| `TOYMATH_OPENAI_API_KEY` | a placeholder | That endpoint's own credential *(secret)*, if it wants one. `OPEN_ROUTER` is never sent to a redirected endpoint. |
| `TOYMATH_AGENT_BACKEND` | unset | Forces `openrouter` or `codex`. Outranked only by an explicit `backend!` in the notebook. |
| `TOYMATH_CODEX_MODEL` | the account's own default | Default model for the Codex backend. ToyMath hardcodes none — the runtime chooses (`gpt-5.6-sol` with the currently pinned one). `model!` overrides it per notebook. |
| `TOYMATH_CODEX_HOME` | `~/.toymath/codex-home` | ToyMath's dedicated Codex home. Must not be the general one; a directory ToyMath did not create is refused rather than overwritten. |
| `CODEX_HOME` | `~/.codex` | Read only to know which home to stay out of. ToyMath never reads its contents or its `auth.json`. |

Backend resolution under `backend! auto`, in order: an explicit `backend!` in
this notebook → `TOYMATH_AGENT_BACKEND` → the OpenAI-compatible backend if
`TOYMATH_OPENAI_BASE_URL` or `OPEN_ROUTER` is set → Codex if a managed ChatGPT
account is signed in. If nothing is configured, the cell says so instead of
guessing.

Note that this order predates the recommendation above: with **both**
configured, `auto` selects OpenRouter and starts billing per token. If you keep
an `OPEN_ROUTER` key around as a fallback, make the recommended path explicit —
`backend! codex` in the notebook, or `TOYMATH_AGENT_BACKEND=codex` in `.env` —
rather than relying on `auto`.

Running a local model, for example, is two variables and a `model!`:

```bash
echo 'TOYMATH_OPENAI_BASE_URL=http://localhost:11434/v1' >> .env
echo 'OPENROUTER_MODEL=qwen3:14b' >> .env      # or `model! qwen3:14b`
```

`do!` asks for 7–9 tools and a multi-turn tactic sequence, so a local model
needs solid tool-calling and a context window that fits ~4k tokens of prompt
and schemas before a skill is even loaded — with Ollama, raise `num_ctx`.

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

A live test decides its own configuration: it always runs on
`gpt-5.6-luna`, on the backend its class names, with tracing and the figure
sandboxes off. `OPENROUTER_MODEL`, `TOYMATH_CODEX_MODEL`,
`TOYMATH_AGENT_BACKEND`, `TOYMATH_OBSERVABILITY`, and `TOYMATH_SANDBOX` in
your `.env` change what `do!` does, never what a live test measures — only
the credential is taken from the environment.

MIT License · Semyon C
