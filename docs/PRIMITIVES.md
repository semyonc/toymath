# Agent-Scoped Primitives

Prototype of the verified-derivation direction: the agent decides strategy,
toymath executes and verifies each step, the ledger is the artifact. A
capable model already solves most math problems on its own; what it cannot
do alone is prove each step is valid. These primitives give it that: every
move is a narrow, named, mechanically checked transformation, and the chain
of steps is the product. New code lives beside the old engine and does not
touch the fixed-point rewrite rules.

## Modules

| Module | Purpose |
|---|---|
| `engine/polyrat.py` | Canonical core for the rational fragment: sparse `Poly` (`{monomial: Fraction}`), `RatFunc` with cancellation in the constructor (monomial GCD, univariate Euclidean GCD, multivariate exact trial division, sign/content normalization), `to_ratfunc` / `ratfunc_to_notation` |
| `engine/primitives.py` | The named primitives + `equal_exprs` checker + numeric oracle |
| `engine/ledger.py` | Step ledger: JSON persistence, assumption accumulation, replay verification, chain-continuity detection |
| `toymath_cli.py` | Agent-facing CLI; one deterministic JSON object per call |
| `engine/agent_do.py` | The `do!` Jupyter endpoint: OpenRouter-backed agent (OpenAI Agents SDK) whose only way to do math is calling the primitives |
| `engine/plot_sandbox.py` | Sandboxed plotting backends for the do! agent (backend seam; v1: Pyodide under Deno) |
| `engine/pyodide_runner.mjs` | Vendored Deno script: runs agent plot code in the Pyodide WASM sandbox, captures matplotlib figures as base64 PNGs |
| `engine/unittests_primitives.py` | Focused primitive/oracle/ledger tests |
| `engine/unittests_do.py` | do! endpoint tests (scripted fake model; live OpenRouter test behind `TOYMATH_LIVE_TESTS=1`) |

## The primitives

Every transforming primitive returns
`{ok, op, args, input, result, assumptions, check}` where `check` is the
independent numeric spot-check (`agree` / `disagree` / `skipped`).

1. **substitute(expr, var, value)** — notation-graph replacement, wraps
   values in parens so juxtaposition never corrupts (`2x`, x:=3 → `2(3)`,
   never `23`). Big-operator binders are respected: substituting a bound
   variable, or inserting a value that would be captured by one, is refused.
   Checked by evaluating input-with-binding vs output.
2. **apply_both_sides(eq, op, arg)**, op ∈ {`+`,`-`,`*`,`/`,`^`} — the spine
   of equation solving. Multiplication/division by a non-constant records
   the assumption `arg ≠ 0` in the step (record, don't prove). Refuses `*0`
   and `/0`. Inequalities are supported: `*`/`/` by a negative constant
   flips the relation, unknown-sign factors are refused, `^` is `=`-only.
   Each new side is oracle-checked against `op(old side, arg)`.
3. **expand(expr)** — `to_ratfunc` → canonical, degree-ordered output.
   Accepts equations (expands each side), which doubles as the
   "simplify both sides" move after `apply_both_sides`. Outside the
   rational fragment it canonicalizes over *opaque atoms*: maximal
   non-fragment subtrees (`\cos x`, `e^x`, unevaluated `\int ...`) become
   opaque variables, the same polyrat engine runs, and the subtrees are
   substituted back — so like terms merge (`2\sin x + 3\sin x → 5 \sin x`),
   cancellations happen, and tactic-chain assemblies finish inside
   the ledger. No new rewrite rules; atom identity ignores transparent
   grouping (`\sin(x)` ≡ `\sin x`). Exact rational subexpressions
   inside an opaque atom are canonicalized recursively, so
   `\ln(4+(x^2)^2)` prints as `\ln(x^4+4)` and shares an atom with that
   spelling. Indexed big operators keep their entire scoped body in one
   atom, so factors cannot commute out of a limit, sum, product, or bounded
   integral. Function powers enter the
   polynomial layer (`\sin^2 x` is `atom(\sin x)^2`, so
   `(\sin x)^2 - \sin^2 x → 0` and `\sin^2 x \cdot \sin x → \sin^3 x`;
   `\sin^{-1}` keeps its arcsin reading and stays opaque).
4. **collect(expr, var)** — polynomial grouped by powers of `var`.
   Accepts equations and rational functions (numerator and denominator
   collected separately). Outside the rational fragment it collects over
   the same opaque atoms as `expand`: `x\sin x + x\cos x` →
   `(\sin x + \cos x)x`, the natural setup step before dividing both
   sides by the collected coefficient. Opaque subexpressions are not
   entered — collecting `\sin x + \cos x` by `x` is refused honestly.
5. **evaluate(expr)** — exact `Fraction` arithmetic when no free variables
   remain; on an equation reports `holds: true/false` (solution checking).
   Falls back to float for non-rational constants.
6. **differentiate(expr, var)** — exact `RatFunc` derivative on the rational
   fragment; ~20 mechanical rules (trig, exp, ln, sqrt, chain/product/
   quotient, `f^n(u)`) outside it. Verified by central differences.
7. **rewrite(expr, lemma, direction)** — registered equality lemmas
   (`diff_squares`, `square_of_sum`, `cube_of_sum`, `diff_cubes`, …) matched
   at the root or, failing that, at the first matching subterm (reported as
   `at`, with the total match count); the lemma library is the
   extensibility point. Power terms in a pattern also bind perfect-power
   constants and monomials when the structural match fails: `x^2 - 4` →
   `(x+2)(x-2)`, `4x^2 - 9` → `(2x+3)(2x-3)`, `x^3 - 8` via `diff_cubes`
   (the derived binding is reported as `numeric`); imperfect powers
   (`x^2 - 3`) are still refused, and rewriting inside an equation is
   oracle-checked per side.
8. **factor_gcd(expr)** / **factor_quadratic(expr, var)** — named
   factorings in place of a general `factor`. `factor_quadratic` factors
   over Q via the discriminant, reports the roots, and honestly refuses
   irrational/complex cases. Both tactics accept relations and factor each
   applicable side independently, leaving non-applicable sides unchanged;
   the result records which sides changed and is mechanically checked per
   side.
9. **integrate_power_rule / integrate_table / integrate_by_parts(u, dv) /
   integrate_substitute(u_expr, u_var, new_integrand)** —
   tactic-shaped integration in place of an autonomous `integrate`. All
   accept a bare integrand or an `\int ... d<var>` wrapper (definite
   integrals refused). Antiderivatives carry a fresh `+ C` (collision-safe,
   reported as `constant`) and are verified by central-difference
   differentiation against the integrand. `by_parts` verifies the agent's
   split (`u·dv` must equal the integrand via `equal?`), computes `du`
   with the trusted differentiator and `v` from the table, and returns
   `u v - \int v du` plus a `remaining_integral` handle for the next
   step; its pieces are verified individually since the result still
   contains an integral. `integrate_substitute` takes the agent's `u =
   u_expr` and the integrand rewritten in `u_var`, mechanically verifies
   `f(u(x))·u'(x)` equals the original integrand via `equal?`, and returns
   `\int f(u) du` with a `back_substitute` handle. The `1/x → ln x` rule
   records the assumption `x > 0`.
10. **integrate_rewrite(new_integrand)** — congruence under the integral
   sign: the agent proposes an equivalent integrand (typically a
   partial-fraction decomposition); `equal?` must answer a plain `yes`
   before the integral is rewritten. The equality is the checked content;
   the tactic itself adds nothing. **integrate_linearity** — the exact sum
   rule: `\int (f_1 ± f_2 ± …) dx → \int f_1 dx ± \int f_2 dx ± …`, one
   unevaluated integral per top-level term (`|…|` never splits).
   **integrate_assemble** is the provenance-closing companion in `do!`:
   given the linearity ledger step and one completed ledger step per piece,
   it retrieves those results rather than accepting a retyped sum, checks
   each derivative against its corresponding integrand, applies the stored
   signs, and adds one fresh constant. Replay revalidates the source step
   references. Together these make rational integrands with factored
   denominators derivable without trusting the agent's final arithmetic:
   propose partial fractions, split, solve each piece, then assemble the
   recorded results.
11. **limit_rewrite / limit_substitute / limit_linearity / limit_table /
   limit_lhopital / limit_assemble** — tactic-shaped limits in place of an
   autonomous evaluator. `limit_rewrite` carries the binder only after
   `equal?` verifies the proposed body. `limit_substitute` records continuity
   and requires an independent approach-sampling oracle to converge.
   `limit_table` handles constants, six standard zero limits, and finite
   rational leading-coefficient limits at infinity. `limit_lhopital` performs
   one step only after numeric probes observe `0/0` or `infinity/infinity`;
   it records differentiability/nonzero-derivative/existence premises and
   checks both derivatives plus the original/transformed approach values.
   One-sided binders (`a^+`, `a^-`) are parsed and sampled directionally.
   `limit_linearity` records existence assumptions for every piece, and
   `limit_assemble` retrieves/checks the ordered piece values so a retyped
   wrong sum cannot enter the ledger.

**equal?** additionally compares canonically over *shared opaque atoms*
when both sides leave the fragment: syntactic atom equality is conclusive
for "yes" (`\sin(x)·x ≡ x \sin x` decides canonically), but atom
*inequality* is never trusted — distinct atoms may still be related (e.g.
`sin²x + cos²x` vs `1`), so those fall through to the numeric oracle.

**equal_exprs(e1, e2)** → verdict `yes`/`no`/`unknown`. Canonical `RatFunc`
comparison decides the rational fragment; outside it the numeric oracle
answers probabilistically (Schwartz–Zippel style) and says so in `method`.
Equations compare per-side.

Deliberately absent: `solve` and `simplify` (either would collapse the
derivation into one opaque move — the visible steps are the product),
autonomous `integrate` or `limit` (tactics the agent picks are the honest
shape), and general `factor` (specialized named factorings instead).

## Two independent trust legs

1. Each primitive is a narrow, tested implementation.
2. The numeric oracle shares nothing with the symbolic code: random rational
   sample points, assumption-aware (skips points where an assumed-nonzero
   divisor vanishes), fixed seed for reproducibility.

A bug in either leg is caught by the other. During development the oracle
caught three real bugs (function symbols treated as polynomial variables,
`\left(...\right)` vgroups unevaluated, wrong function-argument binding in
products) — the design works.

## The goal-aware ledger

```
s1#dff1198 [ok] apply_both_sides: 2x + 3 = 7  ==>  {2}x+{3} - {3} = {7} - {3}
s2#43bdac6 [ok] expand: {2}x+{3} - {3} = {7} - {3}  ==>  {2}x = {4}
...
```

- ledger v2 records parseable top-level relations as root claims and
  subclaims, separately from the step stream; each starts `open`;
- steps carry id, content hash, op, args, result, assumptions, check, and an
  optional `goal` claim id;
- `conclude(claim, steps)` accepts only goal-owned `agree`/`exact` steps,
  checks chain continuity (or explicit assembly provenance), and checks that
  the endpoint closes the claim. A relation-valued endpoint must itself be
  mechanically true, so a no-op rewrite of `x=2` cannot establish `x=2`;
  equality claims may instead close by a checked left-to-right or
  right-to-left chain. The verdict is `established` without assumptions or
  `conditional` with them;
- replay re-runs both the primitive steps and conclusion provenance. A v1
  session is upgraded in memory and remains replayable with no claims;
- assumptions accumulate in the header — the derivation is honestly
  conditional;
- `continues` flags whether a step's input matches the previous result
  (semantic comparison), so branches are visible;
- `replay` re-runs every step and confirms recorded results — cheap
  verification of the whole derivation;
- `comment` entries are narrative notes the agent leaves between steps
  ("splitting by partial fractions; now piece 3/6"). They are unverified
  prose: no input/result, skipped by `replay`, never provenance for a
  final value, and transparent to `continues` chaining.
- Markdown/text renderers lead with visible claim banners. An `OPEN` claim
  stays visibly unfinished even if an unrelated checked step happens to end
  at the same value.

## Known limitations

- **Fractional powers fold in `expand` only for plain-variable bases.**
  `expand` substitutes `t = x^{1/q}` (q = lcm of the exponent
  denominators), canonicalizes with ordinary integer-power machinery,
  and maps back: `(x^{1/6})^4 → x^{2/3}`,
  `\frac{(x^{1/6})^4}{x\,x^{1/6}+x} → \frac{1}{x^{1/2}+x^{1/3}}`.
  Every fold records `x > 0`; a fold that extends the domain (e.g.
  `(x^{1/6})^6 → x`) is flagged `domain-differs` by the oracle.
  Remaining boundary: composite bases (`(x+1)^{1/2}`, and deliberately
  `(x^2)^{1/2}`, which is `|x|`) stay opaque atoms; negative fractional
  exponents and relation sides skip the fold; `collect` and the
  canonical leg of `equal?` do not fold (the numeric oracle still
  decides such identities).
- Multivariate cancellation covers monomial factors and the
  one-side-divides-the-other case via exact trial division
  (`(x²-y²)/(x+y)` → `x-y`, `(\sin x+\cos x)x / (\sin x+\cos x)` → `x`);
  *partial* multivariate common factors still print uncancelled
  (univariate is full Euclidean). `equal?` compensates by
  cross-multiplying, so verdicts stay exact either way.
- `apply_both_sides` on inequalities refuses `*`/`/` by expressions of
  unknown sign (no case splitting) and `^` entirely.
- `rewrite` rewrites the first matching subterm (innermost, parse order)
  and reports it as `at`; it does not offer a position selector yet.
- Function-argument binding in products uses the convention: a parenthesized
  group right after the function is the whole argument; otherwise the run of
  tight factors up to the next function/group binds (`\sin 2x` = sin(2x),
  `\cos(x) y` = cos(x)·y).
- Absolute value `|...|` is an opaque atom, not a transparent bracket: like
  terms collect (`2|x| + |x|` → `3|x|`), `equal?` distinguishes `|x|` from
  `x`, and the oracle computes real `|·|`, but no `|.|`-specific tactic is
  provided and `differentiate` refuses it. Bare `|expr|` parses only around a
  single scalar (`|x|`, `|x^2|`); wrap products/sums as `\left|x-1\right|` or
  `|{x-1}|`. `\lfloor·\rfloor`/`\lceil·\rceil` are not yet modelled.
- Sums, products, and bounded integrals are scope-safe opaque atoms, not yet
  evaluable tactics. Limits have the named tactic chain above, but table
  coverage is intentionally narrow and the approach oracle may return
  `skipped` when Richardson estimates do not converge. Free-symbol discovery
  excludes bound variables; identical binders compare canonically and
  alpha-renamed binders remain honestly `unknown`. `\infty` is accepted in an
  operator bound but is never sampled or treated as a cancellable ring
  variable elsewhere.

## The `do!` endpoint (Jupyter)

A cell starting with `do!` hands the rest of the cell to an agent as a
natural-language instruction:

```
do! solve [[3]] for x, recording every assumption
```

- `[[n]]` references resolve to the rendered result of cell *n* before
  the agent runs (undefined references fail fast, before any API call).
- The agent (OpenAI Agents SDK over OpenRouter; `OPEN_ROUTER` key in
  `.env`, model from `OPENROUTER_MODEL`, default
  `anthropic/claude-sonnet-5`) gets one tool per primitive and its
  instructions are generated from the committed skill file at load time.
  **Only tool executions write ledger steps** — the model's prose cannot
  enter the artifact, so a hallucinated step is structurally impossible.
- Steps stream into the result cell as they are verified, each rendered
  `sN#hash [ok] op: input ⟹ result` with per-step assumptions;
  `disagree`/refused steps show in red; the run ends with the agent's
  one-line summary, the accumulated assumptions, and the final value.
- The agent designates the cell's final value with `set_result` (falling
  back to the last step's result). The value must be mechanically equivalent
  to a result anywhere in the shared ledger, so the agent may select an
  earlier answer after later verification but cannot synthesize a detached
  conclusion. A run with no transforming steps may still return a query-only
  value, which is rendered explicitly as **unverified**. Final values are
  stored in execution history so later cells can chain on them with `[[n]]`.
- A `prove!` prompt-command is different at the harness boundary: its raw
  argument is recorded as root claim `c1` before the model starts, every new
  step is goal-tagged, and `set_result` is refused until `conclude(c1, ...)`
  records a closing checked chain. If the claim remains open, no last-step
  fallback becomes a chainable value. The final agent prose is capped and
  explicitly labelled unverified so it cannot visually substitute for the
  ledger artifact.
- All `do!` cells in a notebook share one session ledger; each cell
  renders only the steps it added, and a replay verifies the whole
  notebook's derivation chain.
- **Plotting** (optional): when Deno is installed (`brew install deno`),
  the agent gets a `plot(code, caption)` tool. Its matplotlib/seaborn
  code runs OUTSIDE the kernel, in a Pyodide WASM sandbox under Deno's
  deny-by-default permissions (read/write only the wheel cache, network
  only to the package CDNs, **no environment access** — the agent cannot
  reach the OpenRouter key or the filesystem even through the `js`
  bridge). Figures stream inline captioned *"illustration, not
  machine-checked"*: plots are illustrations, never evidence — they do
  not become ledger steps and `replay` ignores them. The model receives
  only figure counts, never image bytes. Disable with
  `TOYMATH_SANDBOX=off`; the tool (and its prompt section) simply isn't
  registered when unavailable. The backend seam
  (`run_plot(code, timeout)`) is container-agnostic, so a Docker /
  llm-sandbox backend can be added later.

## Notebook commands (`int!`, `diff!`, `solve!`, …)

`do!` takes a free-form instruction; a **notebook command** is a saved,
named instruction template. Each is a Markdown file in `commands/` with a
SKILL-compatible YAML frontmatter and a body containing the `$ARGUMENTS`
placeholder:

```markdown
---
name: int
description: Apply symbolic integration to the argument, step by step
---
Apply symbolic integration for $ARGUMENTS ...
```

A cell that starts with `int!` renders the template (`$ARGUMENTS` → the rest
of the cell, after `[[n]]` back-references resolve) and runs it through the
same `do!` agent. Discovery mirrors the `cmd_*.py` auto-registration
(`prompt_commands.load_commands()` globs the directory); `commands!` (or
`help!`) lists what is available and reloads the registry so a newly-added
file goes live without a kernel restart. Only a *registered* name at the
start of a cell diverts, so ordinary math such as `n!` (factorial) is
untouched.

A command adds **no new authority**: it only seeds the agent, which can
still only call the oracle-checked primitives, and only tool executions
write ledger steps. So `int!` is not an autonomous `integrate` — it is a
convenience shortcut for a `do!` run, and a hallucinated step stays
structurally impossible. The template can *steer* (which tactics to prefer,
what to record) but cannot bypass verification.

## Composite commands — the LLM / procedural bridge

A command whose frontmatter carries `expr: true` may appear **inside** an
expression and compose with plain math and with other commands:

```
{diff! {int! x^3}}          →  x^3            (two verified sub-derivations)
{int! x^2} + {int! x^2}     →  ⅔x³ + C + C₁   (one call, memoised, glue checked)
{int! x^2} - {int! x^2}     →  C - C₁         (an arbitrary constant, NOT 0)
2 {diff! x^2} - 1           →  4x - 1
```

`ExprResolver` (`engine/expr_commands.py`) walks the parsed tree — the
parser already builds a command node for every `{name! …}` — and replaces
each `expr` command with the verified `final_result` of its `do!` run,
**inner-to-outer** (a command's argument is resolved before the command
runs, so `{diff! {int! x^3}}` integrates first, then differentiates).
Identical sub-expressions are memoised, so they cost one call, and a
per-cell cap bounds the number of agent runs. Query-only/unverified final
values are refused at this boundary and cannot be laundered through the
final glue check.

The arithmetic **glue** between results is then handed to the `expand`
primitive, so the composition is checked by the **numeric oracle** — not by
another LLM. This is the whole point: `{int! x^2} + {int! x^2}` is combined
and its result is spot-checked against random sample points,
deterministically, at no token cost. The cell's ledger is the union of each
command's sub-derivation plus the final `expand` step, so `replay` verifies
the entire composite.

A command that **mints** symbols is effectful, not a pure function.
`int!` declares its integration constant in frontmatter (`fresh: C`), and
the resolver renames the minted symbol per splice on collision
(`C → C₁, C₂, …`): two independent antiderivatives never share one
constant, so `{int! f} - {int! f}` yields the honest `C - C₁` instead of
silently collapsing to 0, and a `C` the user wrote in the cell is never
captured. A fresh name that also occurs in the command's own *argument* is
bound to that argument, not minted, and keeps its name. Subscripted names
like `C₁`, `x₁` are atomic variables — independent of their base symbol —
in both trust legs: the symbolic path treats them as opaque atoms and the
numeric oracle samples them as ordinary free variables.

Only `expr` commands compose; a plain or unknown `name!` inside a composite
cell is refused, so **everything in the cell is either agent-verified or
expand-verified**. The glue is deliberately the verified `expand` (not the
legacy `cmd_*` rewrite rules) — `expand` multiplies and cancels, so you
rarely need a procedural `mul!` at all. Verification is never bypassed; the
bridge only lets the strategic (LLM) and mechanical (oracle-checked
algebra) layers interleave at the level of a single expression.

### The zero-token tier: `direct:` commands

A command whose frontmatter carries `direct: <primitive>` **is** one
verified primitive — no agent run, no tokens. The argument goes straight to
the primitive and the oracle-checked record becomes a normal, replayable
ledger step. `diff!` and `expand!` ship this way, so

```
{expand! (1+x)(1-x)} + 2x^2   →  x² + 1     (zero agent calls)
{diff! {int! x^3}}            →  x³         (one agent call: int! only)
```

Three tiers fall out naturally: **direct procedural primitive** (`expand!`,
`diff!` — instant, free) < **LLM tactic** (`int!` — the agent picks an
integration tactic, every move oracle-checked) < **whole-cell derivation**
(`solve!` — the visible chain of steps is the answer). Reserve the agent
for commands that genuinely need strategy.

Whitelisted primitives: `expand`, `collect`, `differentiate`, `factor_gcd`,
`factor_quadratic`, `evaluate`. For primitives that take a variable, the
resolver infers the argument's **single** plain free variable — constants
minted by `fresh:` commands in the same cell are excluded (that is why
`{diff! {int! x^3}}` infers `x`, not `C`) — and refuses ambiguity rather
than guessing. Two escape hatches:

- **Explicit variable**: `diff! [x] <expr>` (also inline,
  `{diff! [x] …}`) names the variable in brackets and skips inference
  entirely — the natural way to differentiate a chained antiderivative:
  `diff! [x] [[3]]`.
- **Session-constant provenance**: integration steps record the constant
  they mint (`constant: C` on the ledger step), and a two-way ambiguity
  resolves against recorded constants — so a bare `diff! [[3]]` on an
  antiderivative derived earlier in the session infers `x` over its `C`.
  A user-written `C` with no such provenance still refuses.

`direct` implies `expr`, and the command body is documentation, not a
prompt. `{expand! …}` is also the verified replacement for procedural
`mul!`/`add!` inside an expression.

## Output format

Results print through a PrettyWriter that drops the `{2}`-style value
braces of the legacy writer wherever the pretty string parses back to the
same expression (validated per call via a group-stripped normal form);
anything ambiguous falls back to the verbose form. Atom-substituted
results additionally drop the parentheses the substitution machinery adds
wherever the expression stays unambiguous (`5(\sin x)` → `5 \sin x`,
`x(\sin x)` → `x \sin x`), print powered atoms in standard notation where
the position cannot rebind the argument (`\sin^{2}x`, but
`( \sin x)^{2} \cos^{2}x` keeps the parens), and keep them where they
bind (coefficient sums like `(\sin x + \cos x)x`); instantiated rewrite
templates go through the same pass (`(x+2)(x-2)`, not `(x+(2))(x-(2))`).
The pass is purely cosmetic and every emitted result is still
oracle-checked. Ledger `replay` tolerates formatting drift across
versions by falling back to a semantic `equal?` comparison when recorded
and replayed strings differ.
