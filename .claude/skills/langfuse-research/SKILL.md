---
name: langfuse-research
description: Research do! agent problems (stuck runs, wrong turns, blown token/turn budgets) by pulling their Langfuse traces down to the terminal with the official langfuse CLI. Covers install/auth against the local self-hosted instance, the list→get loop, and the diagnostic signals worth reading in a trace.
---

# Research do! agent problems in Langfuse

The `do!` endpoint can emit one Langfuse trace per run (agent turns, LLM
generations with token usage/latency, and every trusted-primitive tool call).
When a run misbehaves — stalls, takes a wrong turn, blows its turn/token
budget — pull that trace back to the terminal and read it instead of guessing.
This is **read-only research tooling**: it never touches the ledger or the
numeric oracle, the same trust boundary as the tracing leg itself
(`engine/observability.py`).

## Produce the trace

1. Re-run the failing case with tracing on:
   `TOYMATH_OBSERVABILITY=on` in the environment (or `.env`). The wiring lives
   in `engine/observability.py`; it is off by default and any Langfuse
   failure is downgraded to a warning, so the derivation still runs.
2. The trace lands in the local self-hosted Langfuse at
   `http://localhost:3100`.

## The CLI

Official tool: `langfuse-cli` (installed globally here as `langfuse`; `npx
langfuse-cli …` also works with no install). Usage is
`langfuse api <resource> <action>`.

**Auth — the one gotcha:** the CLI does *not* auto-load `.env`. Pass it
explicitly as a **top-level** flag before `api`, or the call fails with
`Missing --username for basic auth`:

```bash
langfuse --env .env api traces list --limit 10
```

It reads `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` and
`LANGFUSE_BASE_URL` (or `LANGFUSE_HOST`, or `--host`) — the same creds the run
traced to, already in the project `.env`. Discover the surface with
`langfuse api __schema` and `langfuse api <resource> --help`.
`langfuse get-skill` prints Langfuse's own upstream agent skill from GitHub if
you want the full API-wrangling reference.

## The loop: list → get

```bash
# Find the run (traces are named "do!"; filter/sort as needed)
langfuse --env .env api traces list --name 'do!' --limit 10 --json

# Pull the whole tree in one call — the do! root plus every AGENT / CHAIN /
# GENERATION / TOOL observation and any scores are embedded in the trace.
langfuse --env .env api traces get <trace-id> --json
```

`traces get` is the drill-down you want; you rarely need a separate
observations query. If you do want a single observation by id, use
`langfuse --env .env api legacy-observations-v1s get <observation-id>` — the v2
`observations list --trace-id …` returns 404 against this instance (it needs
Langfuse v4 write mode).

**`--json` envelope:** `--json` wraps the reply as `{ok, status, body}`, so the
trace itself is under `.body` (`.body.observations`, `.body.metadata`,
`.body.output`). The default (no `--json`) pretty form returns the raw API body
directly — nicer to eyeball, but script against `--json` and remember the
`.body` unwrap. Add `--curl` to any call to preview the HTTP request instead of
sending it.

## What to look for

A `do!` trace's shape maps directly onto how a run can go wrong:

- **Trace `output` is `null` ⇒ the run never called `set_result`.** The run
  ended without committing a result — incomplete or failed. A concluded run
  carries the final answer here.
- **An `ERROR`-level observation is the terminal failure.** The AGENT
  observation's `statusMessage` names it, e.g.
  `Max turns exceeded: {'max_turns': 64}` — the run ran out of turns before
  closing the chain. Scan `level == "ERROR"` first.
- **`run_tactic` outcome mix is the root-cause signal.** Each `run_tactic`
  TOOL observation's `output` carries the check verdict. Count
  `agree`/`exact` (progress) against `error`/`disagree`/`refus`/`domain-differs`
  (fumbles). A high error rate means the agent is misinvoking tactics and
  burning turns — e.g. one real stall here was 15 of 45 `run_tactic` calls
  erroring, which exhausted `max_turns`.
- **TOOL `name` is the do! stable tool surface** — `claim`, `comment`,
  `load_skill`, `run_tactic`, `conclude`, `set_result`. Reading them in order
  reconstructs the agent's strategy, including dead branches it (honestly)
  left visible.
- **GENERATION `usage` / `latency`** expose token spend and slow turns; a turn
  with huge `input` tokens points at prompt bloat.
- **Root `metadata`** carries `mode` (prove/explore/…), `model`, and
  `max_turns` — the knobs to compare a failing run against a passing one, or
  the same case across models.

### One-shot triage

```bash
langfuse --env .env api traces get <trace-id> --json | python3 -c '
import sys, json, collections
d = json.load(sys.stdin)["body"]
m = d.get("metadata", {})
obs = d.get("observations", [])
print("mode/model:", m.get("mode"), "/", m.get("model"), "| max_turns:", m.get("max_turns"))
print("concluded:", d.get("output") is not None)   # False => never called set_result
for e in obs:
    if e.get("level") == "ERROR":
        print("ERROR:", e.get("type"), e.get("name"), "-", e.get("statusMessage"))
rt = [x for x in obs if x.get("name") == "run_tactic"]
kw = collections.Counter()
for x in rt:
    s = json.dumps(x.get("output"))
    for k in ("agree", "exact", "disagree", "refus", "error", "domain-differs"):
        if k in s: kw[k] += 1
print("run_tactic:", len(rt), "calls", dict(kw))
'
```

## Trust boundary

Everything here is inspection only. A trace is a record of what happened, not a
verified artifact — the ledger and its independent numeric-oracle checks remain
the only source of trust. Never let something read out of a trace stand in for
a checked step.
