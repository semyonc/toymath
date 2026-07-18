# Verified-derivation primitives

This document is the contract for ToyMath's agent-scoped mathematical layer.
It explains what makes a step admissible, how tactics are registered and
replayed, and how to extend the system without growing the model prompt.

For the product and interface tour, read [OVERVIEW.md](OVERVIEW.md). Exact
current tactic names and argument orders are generated from code:

```bash
python toymath_cli.py skills
python toymath_cli.py tactics [--skill NAME]
python toymath_cli.py describe TACTIC
```

## Contract

The agent decides strategy. Each primitive performs one narrow named move and
returns a JSON-able record:

```text
{ok, op, args, input, result, assumptions, check}
```

`check.status` is normally `agree`, `disagree`, `domain-differs`, or `skipped`;
pure structural rules may be `exact`. An unchecked transformation is a
regression.

The layer deliberately has no general `solve`, `simplify`, `factor`, or
autonomous `integrate`/`limit`. Smart behavior is decomposed into tactics the
agent must choose explicitly. Assumptions are recorded, not proved—for example,
division by a symbolic expression records that it is nonzero.

Never mutate an input notation graph. Transformations build fresh or cloned
graphs, then render a new LaTeX result.

## Modules and authority

| Module | Responsibility |
|---|---|
| `engine/primitives.py` | Shared parsing/writing, binder/substitution, result-record, and independent numeric-oracle infrastructure |
| `engine/tactics/*.py` | Static tactic implementations grouped by the same core/subject ownership used by the registry and skills |
| `engine/tactic_registry.py` | Single allowlist for public names, ledger operations, ordered arguments, skill ownership, CLI generation, agent invocation, replay, and source-provenance validation |
| `engine/tactic_skills.py` | Discovers committed domain skills and appends exact registry-generated tactic interfaces when loaded |
| `engine/ledger.py` | Persists steps/claims, checks chain continuity, renders, and delegates replay to the registry |
| `engine/agent_do.py` | Stable `load_skill`/`run_tactic` tool surface plus ledger controls; no per-tactic function tools |
| `toymath_cli.py` | Registry-generated positional tactic CLI plus explicit ledger/discovery commands |
| `engine/polyrat.py` | Canonical rational-function core used by algebraic primitives |

Execution authority comes only from `TacticSpec` entries. A skill can teach the
model when and how to request a tactic, but Markdown never loads Python by name
and cannot create a ledger operation.

## Progressive tactic skills

The core skill is always present and contains the trust rules plus common
algebra/checking tactics. Subject workflows are separate:

| Skill | Workflow file |
|---|---|
| core | [main ToyMath skill](../.claude/skills/toymath/SKILL.md) |
| differentiation | [differentiation](../.claude/skills/toymath/domains/differentiation/SKILL.md) |
| equations | [equations and roots](../.claude/skills/toymath/domains/equations/SKILL.md) |
| integration | [integration](../.claude/skills/toymath/domains/integration/SKILL.md) |
| limits | [limits](../.claude/skills/toymath/domains/limits/SKILL.md) |
| finite_operators | [finite sums/products](../.claude/skills/toymath/domains/finite_operators/SKILL.md) |

The always-on prompt contains only core guidance, core signatures, and one-line
domain descriptions. `load_skill` adds one domain's strategy and a tactic
interface rendered from the registry. `run_tactic(name, arguments)` accepts an
ordered string list, validates it against that registry entry, rejects tactics
whose subject skill is not loaded, and records successful transformations.

This is an SDK-independent simulation of progressive skills, which keeps the
OpenRouter chat-completions model path. The committed files can later be mapped
to a native sandbox skill loader without changing tactic authority.

## Two independent trust legs

Every transformation has two intentionally separate paths:

1. symbolic code constructs the proposed result;
2. the numeric oracle evaluates old and new expressions at reproducible sample
   points and compares their values.

The oracle must share no algebra implementation with the symbolic path. It has
its own graph evaluator, function binding, matrix arithmetic, finite
sum/product loops, derivative estimates, and limit extrapolation. Unsupported
numeric evaluation is honest ignorance, not evidence against a transformation.

Canonical rational comparison is exact. Outside the rational fragment,
`equal_exprs` may use shared opaque atoms for conclusive canonical equality and
independent numeric sampling for probabilistic evidence. Canonical inequality
between opaque atoms is never conclusive: distinct forms may still be related
by an identity the canonicalizer does not know.

Domain changes are surfaced. A value agreement on a common domain does not
erase the fact that one expression is defined where the other is not.

## Ledger and claims

A transforming record becomes a step only when it succeeds and its `op` is a
registry entry marked transforming. Steps carry a stable id, content hash,
goal, assumptions, check, and optional `sources`.

The ledger provides:

- semantic continuity flags between consecutive results;
- accumulated assumptions;
- comments that are visible but never usable as provenance;
- claims with open, established, or conditional verdicts;
- replay that re-invokes the registered primitive and compares the result;
- source validation for tactics that consume earlier ledger results.

`conclude` accepts only goal-owned `agree`/`exact` steps whose chain or explicit
source graph closes the claim. A relation-valued endpoint must itself be
mechanically true; an equivalent no-op rewrite cannot establish an arbitrary
relation.

### Provenance-aware composition

Linearity and squeeze tactics expose an important rule: checking a value is not
the same as checking where it came from. Integration/limit assembly therefore
retrieves ordered recorded pieces, recomputes the combined result, persists the
source ids, and revalidates them during replay. Squeeze likewise requires the
recorded lower- and upper-bound limit steps.

The same principle governs inline notebook commands:

```text
certificate(f(g(x))) = certificate(g)
                     + certificate(f)
                     + oracle-check(the splice glue)
```

The final glue check certifies only the expression assembled from already
recorded sub-results. Composition depth does not turn prose or retyped values
into evidence.

## Mathematical representation

The parser produces a notation DAG rather than a conventional syntax tree.
Important verified-layer behaviors include:

- `polyrat.py` canonicalizes the rational fragment using sparse polynomials
  and rational functions;
- comma-separated equations/inequalities form a relation collection rather
  than one comparison with a comma-valued right side; one-column `\cases`
  is also recognized as a system, and whole-system substitution/apply merge
  independently checked per-relation steps;
- maximal non-fragment subtrees become opaque atoms so rational algebra can
  still combine coefficients around `sin`, `ln`, integrals, and other forms;
- indexed big operators consume their scoped bodies as one atom; free-symbol
  discovery and substitution respect lexical binders;
- postfix factorial and `\binom{n}{k}` are first-class notation nodes rather
  than command-shaped symbols; the independent oracle evaluates closed cases
  only on their nonnegative-integer domain, while symbolic algebra keeps them
  as opaque atoms;
- matrix-valued factor runs become ordered noncommutative word atoms, while
  scalar factors remain commutative;
- rational powers of positive plain-variable bases have a limited Puiseux
  fold; composite bases remain opaque because identities such as
  `(x^2)^(1/2)=|x|` are domain-sensitive;
- ellipsis tokens have no semantics and are rejected except by the explicit
  finite-sum/product interpretation tactics, which record continuation as an
  assumption;
- big-operator bound variables are integer-valued by the semantics of
  `\sum`/`\prod`: the closed-form tactics check agent proposals by literal
  accumulation at several integer bounds and record the integer domain as an
  assumption, while general equality checking never silently integer-samples
  a free variable.

See [NOTATION.md](NOTATION.md) for graph shapes and `AGENTS.md` for parser,
writer, comparer, and replicator landmines.

## Known boundaries

- Partial multivariate common factors may remain uncancelled, although exact
  cross-multiplication keeps equality checking sound.
- Symbolic-sign multiplication/division of inequalities is refused; there is
  no case-split constraint store yet.
- Rewrite currently selects the first matching subterm unless a tactic adds a
  future position selector.
- Absolute value is a sound opaque atom, but its grammar/table coverage is
  intentionally narrow; floor and ceiling are not modeled.
- Literal matrices have oracle support and scalar-linear algebra, but no
  general matrix multiplication/determinant tactic.
- Relation systems parse and support whole-system substitution and uniform
  both-sides operations, but there is no autonomous elimination, row-combine,
  or general linear-system solver.
- Quadratic root finding returns complete rational roots; irrational and
  complex roots remain outside the executable equation tactics.
- Infinite sums/products enter only through the definitional
  partial-sums rewrite (value = limit of partial sums, if it exists);
  ellipsis and infinite bounds never become sampled ring variables.
- Numeric oracle `skipped` means the checker lacks evidence. It must never be
  presented as a counterexample or a successful verification.

## Adding or changing a tactic

The extension workflow is intentionally centralized:

1. Implement the narrow tactic and its independent check in the owning
   module under `engine/tactics/`. Put only genuinely cross-domain
   infrastructure in `engine/primitives.py`.
2. Add one `TacticSpec` in `engine/tactic_registry.py` with public name,
   stable ledger `op`, owning skill, arguments, summary, callable, and whether
   it transforms.
3. For a tactic that consumes recorded steps, add its session adapter and
   replay provenance validator to the same registry entry and persist source
   ids in the result.
4. Update only the owning domain `SKILL.md` with strategy, tactic ordering,
   examples, and pitfalls. Exact signatures are generated; do not copy a
   command table into prose.
5. Add focused primitive/oracle tests and registry/CLI/do!/replay tests. Keep
   the runtime-tool-count and prompt-size regression tests green.
6. Run the full offline suite. Grammar changes additionally require regenerated
   `engine/parsetab.py` and zero conflicts in `parser.out`.

Do not add a function tool to `agent_do.py`, a handwritten CLI branch, a
`TRANSFORMING_OPS` entry, or a replay lambda. Those are all derived from the
registry. Add a new skill only when tactics form a coherent subject workflow;
one skill per tactic defeats progressive disclosure.

The public descriptions stay intentionally layered:

- `OVERVIEW.md`: interfaces and system tour;
- this file: trust and extension contract;
- domain skills: strategy and examples;
- registry-generated CLI discovery: exact current tactic interface.
