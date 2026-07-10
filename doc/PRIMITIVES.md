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
| `engine/primitives.py` | The seven primitives + `equal_exprs` checker + numeric oracle |
| `engine/ledger.py` | Step ledger: JSON persistence, assumption accumulation, replay verification, chain-continuity detection |
| `toymath_cli.py` | Agent-facing CLI; one deterministic JSON object per call |
| `engine/agent_do.py` | The `do!` Jupyter endpoint: OpenRouter-backed agent (OpenAI Agents SDK) whose only way to do math is calling the primitives |
| `engine/plot_sandbox.py` | Sandboxed plotting backends for the do! agent (backend seam; v1: Pyodide under Deno) |
| `engine/pyodide_runner.mjs` | Vendored Deno script: runs agent plot code in the Pyodide WASM sandbox, captures matplotlib figures as base64 PNGs |
| `engine/unittests_primitives.py` | 150 tests |
| `engine/unittests_do.py` | do! endpoint tests (scripted fake model; live OpenRouter test behind `TOYMATH_LIVE_TESTS=1`) |

## The primitives

Every transforming primitive returns
`{ok, op, args, input, result, assumptions, check}` where `check` is the
independent numeric spot-check (`agree` / `disagree` / `skipped`).

1. **substitute(expr, var, value)** — notation-graph replacement, wraps
   values in parens so juxtaposition never corrupts (`2x`, x:=3 → `2(3)`,
   never `23`). Checked by evaluating input-with-binding vs output.
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
   grouping (`\sin(x)` ≡ `\sin x`), and function powers enter the
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
autonomous `integrate` (tactics the agent picks are the honest shape), and
general `factor` (specialized named factorings instead).

## Two independent trust legs

1. Each primitive is a narrow, tested implementation.
2. The numeric oracle shares nothing with the symbolic code: random rational
   sample points, assumption-aware (skips points where an assumed-nonzero
   divisor vanishes), fixed seed for reproducibility.

A bug in either leg is caught by the other. During development the oracle
caught three real bugs (function symbols treated as polynomial variables,
`\left(...\right)` vgroups unevaluated, wrong function-argument binding in
products) — the design works.

## The ledger

```
s1#dff1198 [ok] apply_both_sides: 2x + 3 = 7  ==>  {2}x+{3} - {3} = {7} - {3}
s2#43bdac6 [ok] expand: {2}x+{3} - {3} = {7} - {3}  ==>  {2}x = {4}
...
```

- steps carry id, content hash, op, args, result, assumptions, check;
- assumptions accumulate in the header — the derivation is honestly
  conditional;
- `continues` flags whether a step's input matches the previous result
  (semantic comparison), so branches are visible;
- `replay` re-runs every step and confirms recorded results — cheap
  verification of the whole derivation.

## Known limitations

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
- The agent designates the cell's final value with a validated
  `set_result` tool (falling back to the last step's result); that value
  is stored in the execution history, so later cells — plain math or
  another `do!` — can chain on it with `[[n]]`.
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
