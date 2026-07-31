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

A notebook cell beginning with `do!` sends the rest of the cell to an agent
backend — OpenRouter through the OpenAI Agents SDK, or the user's own Codex
account (experimental; see *Choosing an agent backend*):

```text
do! solve 2x + 3 = 7 for x and verify the candidate
```

The runtime no longer exposes one function schema per mathematical tactic.
It has a stable small surface:

- `load_skill` progressively loads a relevant subject workflow;
- `run_tactic` invokes an allowlisted registry entry;
- `comment`, `claim`, `conclude`, and `set_result` manage the ledger/result;
  giving `comment` an earlier `from_step` records an exploration-only resume
  marker rather than an ordinary note. Its next checked step persists the
  presentation edge, while `set_result` persists the selected final spine;
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

All agent cells in a notebook share one ledger. Each cell normally renders its
new slice (an exploration fold may repeat earlier checked steps named by a new
marker), and a selected established result becomes addressable as `[[n]]` by a
later cell. Abandoned exploration routes remain in the append-only artifact
but collapse behind expandable source/reason summaries; their assumptions do
not condition the selected final spine.

`prove!` creates a root claim before the model runs. The claim remains visibly
open until `conclude` receives a connected, goal-owned checked chain. Prose
cannot substitute for a missing step.

### Stopping a run

Jupyter's Stop button interrupts a `do!` cell. Cancellation is part of
correctness, not polish: the provider run is cancelled, the session is closed
to further writes within a bounded grace period, and no tool call started
after the interrupt can append anything.

What was already mechanically checked is kept. Steps committed before the
stop stay in the notebook ledger and still replay, and a verified value the
run had reached is shown as a labelled `partial_result`. What a cancelled
cell never does is act finished: it designates no result and creates no
`[[n]]` backreference, because nothing confirmed that value answers the
instruction. A harness limit (tool-call or wall-clock budget) uses the same
machinery but reports `budget_exhausted`, so a user's Stop and a safety limit
stay distinguishable.

### Choosing an agent backend

`backend!` selects which provider runs the agent, for this notebook kernel
only:

```text
backend!             # show the effective backend and why it was selected
backend! auto
backend! openrouter
backend! codex       # experimental; needs login!
```

`auto` resolves once before each run, in this order: an explicit selection in
this notebook, `TOYMATH_AGENT_BACKEND`, OpenRouter when `OPEN_ROUTER` is
configured, then Codex when a managed ChatGPT account is already signed in.
An existing OpenRouter installation therefore behaves exactly as before. If
nothing is configured, the cell says so instead of guessing.

A run never fails over. A Codex rate limit will not silently create
OpenRouter charges, and an OpenRouter outage will not silently consume a
Codex allowance; switching providers is always an explicit act.

#### The experimental personal-Codex backend

The Codex backend is opt-in and local: each user authenticates their own
account and consumes their own entitlement. Install the extra and sign in:

```bash
uv pip install ".[codex]"
```

```text
login!               # managed ChatGPT sign-in in the browser
login! device        # device-code flow
login! status        # auth mode and plan type
login! logout
```

Authentication is managed by the Codex app-server: ToyMath receives a
sign-in URL, a completion notification, and an account status, and never
handles, stores, or displays a ChatGPT token. It keeps no account identifier
or email either. Signing in never changes the *selected* backend — those are
separate operations — though on `auto` it can change the *effective* one,
which the status line and toolbar then republish. Interrupting a pending
`login!` cancels it on the app-server, on a deadline of its own; a runtime
that will not acknowledge is replaced rather than reused, because an
unacknowledged cancellation means the challenge is still open.

Only a managed ChatGPT account runs a derivation. The runtime also reports
`apiKey` and `amazonBedrock` accounts as signed in, and those bill a
different — possibly organizational — credential, so they are refused with an
explanation rather than spent on a math run.

A sign-in challenge is short-lived, but Jupyter persists cell output into the
saved `.ipynb`, so it is kept out of the file: the browser flow hands its
one-time URL to the OS browser and prints no link, and the device code —
which has to be readable while you type it — is cleared from the cell as soon
as the flow ends, whether it succeeded, failed, or was interrupted. On a
headless machine where no browser can be opened, the link is shown instead,
labelled as one-time, and cleared the same way.

ToyMath uses its own Codex home (`~/.toymath/codex-home`, or
`TOYMATH_CODEX_HOME`), never `~/.codex`, because the general CLI home carries
MCP servers, plugins, apps, and project instructions that would leak
capabilities into a math agent. It marks that home as its own and refuses to
adopt a directory it did not create — it rewrites `AGENTS.md` there, and a
mistyped `TOYMATH_CODEX_HOME` must not silently overwrite somebody else's
instructions or inherit their configuration. Each run gets a fresh ephemeral
thread with an empty working directory outside the repository, a read-only
sandbox, no approvals, and every capability switch off.

Both the home's `AGENTS.md` and the thread's developer instructions reach the
model, so only one of them may enumerate tools — the thread. Which tools
exist is a property of the session (`plot` and `tikz` appear only when the
figure sandboxes resolved), while the home file is written once, by whichever
command first starts the kernel's runtime. Baking names into it produced a
global instruction listing seven tools while the thread offered nine, and a
model reading both obeyed the stricter one: it declined to draw a diagram it
had a working `tikz` tool for. The durable file now carries the role and the
prohibitions — which never vary — and defers the list to the thread.

MCP is contained separately, because the runtime has no global switch for it:
measured against the pinned binary, one configured server adds four
model-visible tools, and a whole-table config override merges rather than
replaces. So ToyMath enumerates the home's `mcp_servers` at startup and
disables each by name; a server whose name cannot be addressed that way is a
hard error rather than one left enabled. The contract test runs against a
home that carries a server, so the enforcement is measured and not asserted.

The honest limitation: the pinned Codex runtime still leaves three native
tools visible to the model — `update_plan`, `request_user_input`, and
`view_image` — and offers no endpoint reporting the effective tool list.
ToyMath accepts that residual exposure for this experimental backend and does
not claim those tools are disabled. A contract test captures the outgoing
request from the real runtime and fails if the tool set is anything other
than the ToyMath tools plus those three; at run time, any tool call outside
the ToyMath surface is refused, and a run that reaches for a native
capability is interrupted and reports `capability_violation` with no
chainable result. The same applies to server-initiated requests: a ToyMath
tool call is the only one answered normally, and every other method in the
runtime's protocol — approvals, elicitations, a token refresh, or something
this version has never seen — gets that method's own refusal shape and
invalidates the run.

Whatever the backend, the trust boundary is the same: the model chooses
strategy and calls tools, while the tactic registry, the independent numeric
oracle, and the ledger remain the only authority for a mechanically checked
result. Model prose can never become a designated value.

### Selecting the notebook model

`model!` changes the agent model for subsequent agent-backed cells in the
current notebook kernel. It does not mutate `.env` or affect another running
notebook:

```text
model! z-ai/glm-5.2
model! z-ai/glm-5.2, Cerebras, Fireworks
```

Comma-separated provider names set OpenRouter's provider order and disable
fallbacks. With only a model name, the optional provider order comes from
`engine/models.yaml`; unlisted model names are also accepted and use
OpenRouter's default routing. In JupyterLab, type `model! ` (including the
trailing space) to open the native completion menu populated from that file.
After a model and comma, the same menu offers its configured providers. The
extension opens the menu automatically; <kbd>Tab</kbd> or
<kbd>Ctrl</kbd>+<kbd>Space</kbd> invokes it manually. Running bare `model!`
shows the current selection and this shortcut.

The selected routing is used by `do!` and model-backed named/inline commands;
direct primitive commands still make no model call. The notebook toolbar's
kernel button shows `Toy Math · MODEL` (with the backend, as
`Toy Math · codex · MODEL`, when it is not the default one) and updates
immediately when the selection changes. Model selection and its title are
local to that notebook's kernel.

The editable configuration shape is:

```yaml
models:
  - model: anthropic/claude-sonnet-5
  - model: z-ai/glm-5.2
    providers: [Cerebras, Fireworks]
```

The catalog is backend-aware. `models.yaml` and provider ordering describe
OpenRouter; with Codex selected, `model!` lists the models that account
offers and takes no provider argument. With no explicit choice, Codex uses
the account's own default rather than a model id hard-coded into ToyMath.

### Reading a cell as mathematics

A cell holds LaTeX, and a dense formula is hard to read as source. In
JupyterLab, a ToyMath cell shows its formula typeset while it is not being
edited, and its source the moment it is — the bargain a markdown cell makes,
applied to code cells. Double-click (or <kbd>Enter</kbd>) opens the source;
leaving the cell renders it again. *Render ToyMath Cell Input* in the command
palette turns this off for a notebook.

The kernel decides what a cell renders as, because the kernel owns the
parser: the rendered formula is what the engine understood, not a second
reading of the source by the frontend. `\int \frac {dx} (x+1)` — the ToyMath
dialect, taking a parenthesised second operand — renders as the fraction it
parses to. A `[[n]]` backreference renders as the formula it stands for.

A cell is read as a whole formula first, and only failing that as prose with
formulas in it. So `int! \int x^2 dx` renders as one formula, while

```text
do! differentiate x³−3x, find where the derivative is zero
```

renders as that sentence with `x³−3x` typeset inside it — no `$…$` required,
since nobody writes them in a prompt. Only a command that hands its argument
to the agent is read this way; a plain cell and a rewrite action are one
expression, and describing them as a sentence would misdescribe them.

Every rendered formula parses back to the same expression as the characters
it replaces, so a cell can never show something other than what it runs. The
prose scan adds one thing that check cannot cover — it *guesses where a
formula starts and ends* — so it is tuned for precision: a formula left as
prose is invisible, whereas prose swallowed into a formula is glaring. What
the guess gives back stays visible as prose, so no part of a prompt can
disappear from the view. Fragments the engine cannot read keep their source,
which makes an unparsable cell visible as one before it is run.

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

Ledger-control commands (`claim`, `conclude`, `branch`, `show`, `replay`)
remain explicit CLI operations rather than math tactics.

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
uv pip install --no-deps .
jupyter kernelspec install kernel_spec --user --name toymath --replace
jupyter lab
```

The project install places the prebuilt `@toymath/model-ui` extension in the
active Jupyter environment; no Node.js build is needed by users. Restart a
running JupyterLab server after installing or upgrading it. Extension
developers can rebuild the committed frontend assets with:

```bash
cd jupyterlab-extension
npm install
npm run build
cd ..
uv pip install --reinstall --no-deps .
```

The experimental personal-Codex backend is an optional extra:

```bash
uv pip install ".[codex]"
```

It pins the exact Codex SDK/runtime pair the capability-containment contract
was validated against. Without it, `backend! codex` reports the install
command and OpenRouter behaviour is untouched.

Agent configuration is read from `.env`:

```dotenv
OPEN_ROUTER=sk-or-...
OPENROUTER_MODEL=anthropic/claude-sonnet-5
TOYMATH_AGENT_BACKEND=auto
# optional, for the experimental Codex backend. Unset, the runtime's own
# default is used (gpt-5.6-sol with the pinned one); this overrides it.
TOYMATH_CODEX_MODEL=gpt-5.6-luna
TOYMATH_CODEX_HOME=~/.toymath/codex-home
TOYMATH_SANDBOX=auto
# optional Langfuse tracing of do! runs (off unless set to on/1/true)
TOYMATH_OBSERVABILITY=off
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

With `TOYMATH_OBSERVABILITY=on`, each do! run is exported to Langfuse as one
trace. How much detail it carries depends on the backend:

- **OpenRouter** — agent turns, LLM generations with token usage and latency,
  and every trusted-primitive tool call, via the OpenInference instrumentor.
- **Codex** — the run span (instruction, final text, backend, model, status,
  tool counts) plus one child observation per ToyMath tool call. The
  instrumentor only sees Agents-SDK runs, so these come from ToyMath's own
  backend-neutral seam; the tracing context is carried explicitly across the
  transport's worker threads so they nest under the run rather than starting
  orphan traces. Per-turn LLM generations are not available on this path.

Tracing is observability only — it never touches the ledger or the numeric
oracle, and a Langfuse outage is downgraded to a logged warning so the
derivation still runs.

The Codex runtime has its own OpenTelemetry pipeline — three exporters,
each `none | statsig | otlp-http | otlp-grpc`. ToyMath pins the two that
would otherwise start on their own (`otel.exporter` for logs, and
`otel.metrics_exporter`, which defaults to `statsig` and would resolve to a
built-in endpoint) to `none` for its threads, so a derivation ships no
runtime telemetry anywhere unless that is an explicit, separate decision.
`otel.trace_exporter` is left unpinned on purpose, because that separate
decision is `TOYMATH_OBSERVABILITY` itself: with tracing on, ToyMath writes
an OTLP block into the home's own `config.toml` pointing that exporter at
the same Langfuse, and removes the block when tracing is off. It goes in
the file rather than a `--config` override so the credential never becomes
a process argument. The runtime's spans then nest inside the `do!` trace
rather than arriving beside it — each JSON-RPC request carries a W3C
`traceparent`, which the app-server uses to parent that request's spans.
A parent-based sampler (`OTEL_TRACES_SAMPLER`) keeps the export to spans
that descend from a traced run; without it the runtime exports every span
it raises, at any level.

The normal test suite is offline; live provider and plot probes are opt-in
and separately gated:

```bash
.venv/bin/python -m pytest -q
TOYMATH_LIVE_TESTS=1 .venv/bin/python -m pytest engine/unittests_do.py -q
TOYMATH_CODEX_LIVE_TESTS=1 .venv/bin/python -m pytest engine/unittests_do.py -q
TOYMATH_PLOT_TESTS=1 .venv/bin/python -m pytest engine/unittests_do.py -q
```

The Codex live tests require an already authenticated account and skip rather
than starting an interactive login. The tool-set contract test needs only the
installed runtime: it captures the real outgoing request against a loopback
server, with no account and no network.

## Repository map

```text
engine/primitives.py       shared notation infrastructure and numeric oracle
engine/tactics/*.py        static tactic implementations by subject skill
engine/tactic_registry.py  authoritative tactic schemas and dispatch
engine/tactic_skills.py    progressive SKILL.md discovery/rendering
engine/ledger.py           record, render, claim closure, and replay
engine/agent_do.py         do! session, canonical tool bindings, finalizer
engine/agent_config.py     notebook-local AgentRoute and backend resolution
engine/agent_backends/     provider seam: cancellation, OpenRouter, Codex
engine/model_config.py     model! configuration loading and validation
engine/models.yaml         selectable OpenRouter models/provider orders
jupyterlab-extension/      TypeScript source for completion/title/rendered input
labextension/              committed prebuilt JupyterLab extension
engine/polyrat.py          canonical rational-function core
engine/expr_commands.py    inline command composition
engine/prompt_commands.py  commands/*.md discovery
engine/plot_sandbox.py     figure backend seam (Pyodide plots, TikZ SVG)
engine/processor.py        classic fixed-point engine
toymath_cli.py             generated tactic CLI + ledger controls
```

Further reading: [verified-derivation contract](PRIMITIVES.md),
[notation DAG](NOTATION.md), and [agent/developer workflow](../AGENTS.md).
