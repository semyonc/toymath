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

The tool surface is also provider-independent. `make_tool_bindings(session)`
defines each model-visible tool once — name, description, JSON schema, and
handler — and a backend adapter converts that record into its own dialect
(Agents SDK function tools, or Codex app-server `dynamicTools`). Whichever
provider runs the agent, it sees the same tools, and only allowlisted registry
calls can append a step. A cancelled run closes the session under its lock:
steps committed before closure remain and still replay, while everything after
it is refused, so an interrupted cell can never be half-written.

## Two independent trust legs

Every transformation has two intentionally separate paths:

1. symbolic code constructs the proposed result;
2. the numeric oracle evaluates old and new expressions at reproducible sample
   points and compares their values.

The oracle must share no algebra implementation with the symbolic path. It has
its own graph evaluator, function binding, matrix arithmetic, finite
sum/product loops, derivative estimates, and limit extrapolation. Unsupported
numeric evaluation is honest ignorance, not evidence against a transformation.

That cuts both ways, so a `disagree` verdict must be backed by evidence the
oracle actually holds. `disagree` is a positive claim that two expressions
differ, and it bars the step from the ledger for good; a float evaluation whose
significant digits were consumed by cancellation cannot support it. Before
reporting a difference, the oracle measures its own numerical noise at that
sample point — how far the evaluation moves when one coordinate is nudged by a
single ULP, which the same intermediate magnitudes amplify as they amplify
round-off — and a gap that does not clear that noise leaves the point undecided
rather than accusing. Undecided points are counted (`unresolved_points`), and a
comparison where no point could resolve is `skipped`, never `disagree`.

Canonical rational comparison is exact. Outside the rational fragment,
`equal_exprs` may use shared opaque atoms for conclusive canonical equality and
independent numeric sampling for probabilistic evidence. Canonical inequality
between opaque atoms is never conclusive: distinct forms may still be related
by an identity the canonicalizer does not know.

Domain changes are surfaced. A value agreement on a common domain does not
erase the fact that one expression is defined where the other is not.

A recorded assumption may carry a constraint, and the oracle then samples only
inside it — strict hypotheses only, so the assumed region always has an
interior; an unsatisfiable one leaves the check `skipped` rather than agreed.
Relations get a second, independent leg: the input and output relation must be
true at exactly the same sampled points. Per-side value comparison cannot see a
wrong inequality direction, which is what that leg exists for.

## Ledger and claims

A transforming record becomes a step only when it succeeds and its `op` is a
registry entry marked transforming. Steps carry a stable id, content hash,
goal, assumptions, check, and optional `sources`.

The ledger provides:

- semantic continuity flags between consecutive results;
- accumulated assumptions;
- comments that are visible but never usable as provenance;
- structured exploration markers that name an earlier transforming step and
  explain why the agent resumed there, plus a presentation-only edge to the
  next transforming step in the same goal;
- replay-validated final-result selections, stored separately from steps so
  selecting an earlier result remains visible after a session is reopened;
- a replay-validated run-level open outcome (`set_open` in `do!`, `open` in
  the CLI) recording, with a capped unverified reason, that a session ended
  without a certified result; it suppresses any fallback result display and
  is deliberately vocabulary-guarded: it never asserts that no solution
  exists;
- claims with open, established, or conditional verdicts;
- replay that re-invokes the registered primitive and compares the result;
- source validation for tactics that consume earlier ledger results.

`conclude` accepts only goal-owned `agree`/`exact` steps whose chain or explicit
source graph closes the claim. A relation-valued endpoint must itself be
mechanically true; an equivalent no-op rewrite cannot establish an arbitrary
relation.

A claim that states what an unknown IS (`A = 1/2`) can never be decided on its
own — asking whether `A` equals the value is the claim itself. Such a claim
closes only when the chain's endpoint is the claim, and then it closes
CONDITIONAL on that chain's first input, which is recorded as the conclusion's
premise and displayed with the verdict. The implication is one-directional and
is all that is asserted; a chain starting from a mechanically false relation, or
from the claim itself, is refused rather than closed vacuously.

Re-recording a claim whose statement matches an existing same-parent claim —
open or concluded — focuses that claim instead of minting a duplicate id, and
a repeated `conclude` replaces the closing chain (for example with one that
carries fewer assumptions). In prove mode, `set_result` also accepts the root
claim's statement and records the concluded endpoint it closes to.

An exploration marker is recorded through `comment(text, from_step=...)` in
`do!` or the explicit CLI `branch FROM_STEP REASON` control. Replay validates
that its source is an earlier transforming step in the same goal. The next
transforming step for that goal must consume the source result — modulo the
wrapper/body convention the primitives already accept in goal gating, so an
integrand-consuming tactic can resume an `\int`-shaped result — or, to
abandon the source step itself (for example when the very first checked move
was the wrong route), it may restart from that step's recorded input; the
persisted edge then carries an explicit `input` anchor and presentation
classifies the source step with its own dead route. Either way the step
persists a hash-checked marker/source edge, derived from ledger order rather
than a hidden mutable cursor. Step-to-step chain continuity uses the same
structural convention, so a linear substitution workflow presents as one
spine. A marker at the end of a partial session remains
valid but visibly awaits its continuation. Legacy marker files without the
target half derive the same edge deterministically during replay.

`set_result` appends a separate selection record containing the chosen value
and its already-validated step or claim provenance. The selection is
presentation metadata, not a transforming step. `set_open` is the matching
terminal control for a run that certifies nothing: it appends an open
outcome (a selection that selects no value) whose only content is a capped
reason naming the missing move. It is sound because it claims nothing about
the mathematics — "this session exhibits no certified result" is decidable
from the ledger itself — and a later certified selection supersedes it for
display. An unresolved exploration marker at an open ending is presented as
left unresolved rather than awaiting continuation. A concluded claim or the
latest selection determines the displayed spine. Markdown and notebook output
collapse each marker-classified off-spine route behind its source and reason,
while retaining every checked step in an expandable body; assumptions from a
dead route do not condition the selected final result. Exploration topology
remains comment-grade annotation and is deliberately distinct from future
assumption-bearing mathematical case splits. The ledger stays append-only:
checked work on an abandoned path is never deleted.

Rich ledger views derive one additional presentation spelling without changing
the replayable record: an agent's explicit keyboard multiplication `*` renders
as `\cdot`. The candidate is accepted only when both strings parse to the same
notation shape; invalid or non-mathematical star contexts remain verbatim.
Hashes, tactic arguments, persisted inputs/results, replay, and terse text/JSON
views continue to use the exact recorded spelling.

### Provenance-aware composition

Linearity and squeeze tactics expose an important rule: checking a value is not
the same as checking where it came from. Integration/limit assembly therefore
retrieves ordered recorded pieces, recomputes the combined result, persists the
source ids, and revalidates them during replay. Squeeze likewise requires the
recorded lower- and upper-bound limit steps.

Complete answers that are collections of associated values follow the same
rule. A solution set of points is assembled by `points_assemble` from the
recorded solution step and one recorded value step per root: every
root-to-value association is gated symbolically and then re-derived by the
independent evaluator, which is what sees the pairing. A typed collection an
agent writes out by hand is a claim about where each number came from, so it
is not admissible as a result on its own.

An answer that is several statements at once — the values of A, B and C, the
assignment satisfying a system — is assembled the same way by `system_assemble`
from one recorded `unknown = value` step per unknown. It puts the recorded
values back into the stated target, gates the outcome symbolically, and has the
independent evaluator re-derive the whole assignment by binding each unknown to
its own value. The record says the assignment satisfies the target, never that
it is the only one that does; a target that already states one of the values
verifies nothing and is refused.

The same principle governs inline notebook commands:

```text
certificate(f(g(x))) = certificate(g)
                     + certificate(f)
                     + oracle-check(the splice glue)
```

The final glue check certifies only the expression assembled from already
recorded sub-results. Composition depth does not turn prose or retyped values
into evidence.

Notebook commands declare parser-owned `input`/`output` types in their
frontmatter. This separates two questions the older `expr: true` flag
conflated: whether a command returns a spliceable mathematical value, and
whether a whole-derivation command may consume an input containing such
values. A compatible nested input is resolved inside-out and gets the same
oracle-checked glue before the outer command runs; a top-level type mismatch
is refused. The special output type `derivation` means the checked chain is
the artifact and cannot be spliced as a value. These boundary types do not
invent new mathematical judgments—a classification such as maximum/minimum
still requires its own notation and mechanically checked tactic.

## Mathematical representation

The parser produces a notation DAG rather than a conventional syntax tree.
Important verified-layer behaviors include:

- `polyrat.py` canonicalizes the rational fragment using sparse polynomials
  and rational functions;
- comma-separated equations/inequalities form a relation collection rather
  than one comparison with a comma-valued right side; one-column `\cases`
  is also recognized as a system, and whole-system substitution/apply merge
  independently checked per-relation steps;
- finite escaped-brace collections and exactly two parenthesized items are
  first-class `COLLECTION` and ordered `PAIR` nodes, so complete values such
  as `\{(-1,2),(1,-2)\}` survive parse/write, graph replication, normal-form
  comparison, and notebook `[[n]]` history; they deliberately have no set
  algebra or scalar numeric-oracle meaning, while bare `x,y` remains the
  existing command/system `C_LIST`. Because the oracle cannot sample such a
  value, selecting one as the final result relies on structural identity with
  a recorded step result — the same comparison the ledger uses to validate the
  selection — so item order is significant and never rearranged;
- maximal non-fragment subtrees become opaque atoms so rational algebra can
  still combine coefficients around `sin`, `ln`, integrals, and other forms;
- indexed big operators consume their scoped bodies as one atom; free-symbol
  discovery and substitution respect lexical binders;
- postfix factorial and `\binom{n}{k}` are first-class notation nodes rather
  than command-shaped symbols; the independent oracle evaluates closed cases
  only on their nonnegative-integer domain, while symbolic algebra keeps them
  as opaque atoms;
- numeral-subscript names such as `C_{1}` are atomic variables in both trust
  legs, and their positive integer powers enter the same canonical atom power
  as repeated multiplication;
- standard `matrix`, `pmatrix`, `bmatrix`, `Bmatrix`, `vmatrix`, `Vmatrix`,
  and `smallmatrix` environments normalize at the shared parser boundary;
  the delimiter-bearing forms remain dedicated matrix nodes, so vertical-bar
  matrices are never confused with scalar absolute value;
- matrix-valued factor runs become ordered noncommutative word atoms, while
  scalar factors remain commutative;
- absolute value, floor, and ceiling are bracket *operators*, not grouping:
  `|x+1|`, `\lfloor x \rfloor`, and `\lceil x \rceil` (bare, `\left`-sized, or
  in the `\lvert`/`\rvert` spelling) stay opaque atoms for symbolic algebra
  while the independent oracle computes the real `|·|`, floor, and ceiling,
  so `\lfloor x+1 \rfloor = \lfloor x \rfloor + 1` is checkable and
  `\lfloor x \rfloor = x` is refused with a witness;
- rule-built derivatives pass through checked canonical algebra before being
  recorded; if removing zero-multiplied domain-bearing terms widens the
  written domain, the step remains visibly `domain-differs` rather than being
  presented as unconditional cleanup;
- rational powers of positive plain-variable bases have a limited Puiseux
  fold; composite bases remain opaque because identities such as
  `(x^2)^(1/2)=|x|` are domain-sensitive;
- a function head followed by a bracketed argument carrying a power —
  `\cos(x)^{2}` — reads as the power of the APPLICATION and normalizes
  to the powered-head spelling `\cos^{2}(x)` at the parse boundary, so
  the numeric and symbolic legs cannot read the spelling apart; write
  `\cos(x^{2})` when the power belongs inside the argument;
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
- Moving an inequality by a symbolic factor requires the agent to state the
  case as a strict hypothesis; the tactic records it, derives the direction
  from it, and refuses when it does not pin the factor's sign. Cases are
  separate steps, not one branching record: there is no case-split branch
  topology yet, so a claim whose chain mixes mutually exclusive hypotheses is
  refused and such hypotheses are displayed as alternatives, never as one
  condition.
- Rewrite defaults to the first matching subterm; the optional `at` argument
  (target subterm LaTeX or 1-based match index) selects among several
  matches, and a failed selection lists the available positions. Match
  positions count each visible subterm once and include the numeric
  perfect-power variants of the lemma pattern.
- Absolute value, floor, and ceiling are sound bracket operators with real
  oracle evaluation, but their grammar/table coverage is intentionally
  narrow.
- A definite integral closes only through `integrate_definite`, which
  consumes a recorded antiderivative step, substitutes both bounds itself,
  and is checked by independent quadrature over the integrand; continuity
  on the interval is recorded, not proved, and an interior domain break
  makes the quadrature leg refuse. A symbolic bound is sampled by that
  leg as a parameter. A chain rooted
  at the bare integrand — or the bare body of any value-bearing binder,
  such as a limit — is never accepted as the bounded expression's value.
- An integrand singular at one finite endpoint — or an integral with
  one infinite bound — closes through `integrate_improper`, the
  definitional door: the integral is read as the limit of its
  truncated integrals, consuming a recorded truncated
  `integrate_definite` step (the improper bound replaced by a fresh
  variable) and a recorded limit of that evaluation (one-sided from
  inside at a singular bound; at ±infinity for an infinite one). The
  definitional reading and half-open continuity are recorded, not
  proved; the independent leg re-integrates the integrand over a
  graded truncation ladder and extrapolates, refusing only past its
  own measured residual, with genuinely-huge growth saturating to
  infinity inside the quadrature so an exponentially decaying
  integrand stays evaluable at deep truncation points. A ladder that
  does not settle certifies nothing — a refusal is never evidence of
  divergence, and a divergent integral has no finite limit step to
  cite in the first place. Interior singularities and doubly-improper
  intervals still have no closing tactic.
- A parameterized reduction formula is stated by `integrate_reduction`:
  one integral family related to itself at an integer parameter shift,
  with the coefficient proposed by the agent. The structural fence —
  same bounds and variable verbatim, the right integrand mechanically
  equal to the left integrand at the shifted parameter — means the
  step can state a recurrence and nothing else; it is not an
  integral-equality oracle. Both sides are evaluated by truncation
  quadrature at parameter values sampled inside the stated `assuming`
  domain; agreement is evidence about the sampled region, not a proof,
  and the record says so. The continental hyperbolic
  spellings `\operatorname{ch}`, `\operatorname{sh}`,
  `\operatorname{th}`, `\operatorname{cth}` normalize to
  `\cosh`/`\sinh`/`\tanh`/`\coth` at the lexer; other `\operatorname`
  names are unchanged.
- A limit whose body contains a variable-bound definite integral is
  checkable: the limit tactics' approach oracle evaluates such an
  integral by graded quadrature (and l'Hôpital's form gate samples
  through it, with the FTC bound rule differentiating it). This
  evaluator belongs to the limits oracle leg only — general equality
  checking deliberately does not gain it, so a definite integral's own
  value still closes only through the integration tactics, never
  through a sampled proposal.
- In the other direction, `differentiate` closes the derivative of a
  variable-bound definite integral through the FTC bound rule
  (`f(b(x)) b'(x) - f(a(x)) a'(x)`), recording continuity of the integrand
  between the bounds; the check leg evaluates the integral as a function
  by quadrature and central-differences it. An integrand that itself
  depends on the differentiation variable is refused — differentiation
  under the integral sign is not in the rule set — as is an unevaluated
  indefinite integral, which the derivative rules never reach across. The
  Leibniz operator spelling `\frac{d}{d x} (...)` is a reading convention
  at the differentiate boundary (and diff! variable inference), not a
  notation node; elsewhere it remains an ordinary fraction.
- Matrix arithmetic is literal-only and cell-wise: the named tactics add,
  scale, multiply (ordered, exactly two factors), transpose, and take 2x2
  determinants of written-out matrices, delegating each cell to checked
  scalar algebra and verifying cell placement against an independent
  whole-matrix numeric comparison. Symbolic matrix declarations, inverses,
  row operations, and larger determinants have no tactic; multiplying both
  sides of a relation by a matrix-valued expression records invertibility
  (not the scalar `!= 0`), and dividing by one is refused.
- Alignment-bearing `array` and starred matrix environments are not accepted;
  supporting them requires notation and writer metadata for their alignment
  preambles rather than silently discarding presentation semantics.
- Relation systems parse and support whole-system substitution and uniform
  both-sides operations, but there is no autonomous elimination, row-combine,
  or general linear-system solver.
- Quadratic root finding returns complete rational roots; irrational and
  complex roots remain outside the executable equation tactics. Point
  assembly completes those roots into `\{(x_1,y_1),...\}` only from recorded
  steps: it associates values with roots and checks each association, but it
  finds no roots, evaluates nothing on its own, and has no collection
  algebra.
- Infinite sums/products enter only through the definitional
  partial-sums rewrite (value = limit of partial sums, if it exists);
  ellipsis and infinite bounds never become sampled ring variables.
- Convergence certification is comparison-only and absolute: the recorded
  step is a bound relation against a geometric or p-series tail, with
  domination recorded as a spot-checked assumption. There is no
  divergence or conditional-convergence test; a refusal is never
  evidence of divergence.
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
