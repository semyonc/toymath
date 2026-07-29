#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_do.py - the `do!` agent endpoint for the Jupyter kernel.

A natural-language instruction goes in; a verified derivation comes out.
The LLM (reached through OpenRouter via the OpenAI Agents SDK) is an
untrusted planner: it can only *call* the trusted primitives, and only
those tool executions append ledger steps. A hallucinated step cannot
enter the artifact - the rendered cell is the ledger, not the model's
prose.

The always-on instructions are small; subject workflows are committed
SKILL.md files loaded progressively through a stable tactic dispatcher.
"""
import json
import os
import re
import threading

import observability
import primitives
import tactic_registry
import tactic_skills
from model_config import DEFAULT_MODEL, MODEL_VAR
from tactics import core as core_tactics
from ledger import Ledger, TRANSFORMING_OPS

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover - dotenv ships with the kernel env
    pass

OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
API_KEY_VAR = 'OPEN_ROUTER'
DEFAULT_MAX_TURNS = 64

_SKILL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '.claude', 'skills', 'toymath', 'SKILL.md')


class DoAgentError(Exception):
    """Configuration/runtime error of the do! endpoint (not a math error)."""


# ---------------------------------------------------------------------------
# prompt: SKILL.md adapted for tool calling
# ---------------------------------------------------------------------------

_TOOL_PREAMBLE = """## Invocation inside do!

The compact core tactic skill is already present. For a subject workflow,
call `load_skill` before attempting one of its tactics. The loaded skill gives
the exact interface. Call `run_tactic` with a tactic name and an ordered list
of string arguments. Successful transformations are appended to the notebook
ledger and rendered immediately.
"""

_DO_RULES = """
## do! mode

- Work strictly step by step: plan the next move, call the tool, read the
  record, and feed its result verbatim into the next call.
- Load only the subject skills needed for this derivation. If the work crosses
  domains, load the next skill when the boundary is reached.
- When the goal is reached, call set_result once with an established ledger
  value, then stop calling tools and answer in one short sentence. A detached
  value after transforming steps is refused; query-only values are explicitly
  unverified.
- The notebook renders the verified chain, claims, assumptions, figures, and
  final value from the ledger itself. Never restate ledger steps, retype
  results, draw markdown tables, or write image links in your final answer -
  it is displayed as plain text commentary only.
- If a tactic refuses or its check disagrees, change strategy. Never present
  an unverified result as the answer.
- Confirm equation candidates with substitute plus evaluate when cheap.
- Use `claim` only for a relation you intend to establish as true, never for
  an equation whose solutions you are seeking. Ordinary calculations and
  root-finding should stay as tactic steps followed by `set_result`.
- Use comment only for short strategy annotations in plain text. When
  abandoning a recorded path and resuming from an earlier result, pass that
  exact step id as from_step, then feed that source result verbatim to the
  next tactic. To abandon a recorded step ITSELF (e.g. the very first move
  went the wrong way), pass that step id as from_step and restart the next
  tactic from that step's recorded input. This records an exploration edge,
  never mathematical case data or provenance. Pass from_step ONLY when the
  very next tactic resumes there; ordinary notes and summaries must omit
  it. Do not put scratch work in notes.
- If every applicable route is exhausted and no ledger value certifiably
  answers the instruction, call set_open once with a short reason naming
  the exact missing move and what you tried, with formulas in $...$, then
  stop. Open records that
  THIS session certified nothing - it never means no solution exists.
  Never dress the last step up as the answer instead.
"""

_PLOT_RULES = """
## Plotting (optional)

A `plot` tool runs self-contained Python in an isolated sandbox and shows
the figures inline, marked as unverified illustration. Use it only when a
picture genuinely helps (shape of a function, roots, area under a curve).
You translate LaTeX to Python yourself. Plots are never ledger steps - if
plotting fails, continue the derivation without it.

- matplotlib and seaborn render as static images. numpy, pandas, scipy and
  sympy are importable; import what you use, label axes, and add a legend
  for multiple curves.
- plotly renders as an interactive figure: build it and leave it in a
  variable (`fig = go.Figure(...)`). Do not call `fig.show()`, and do not
  call `fig.write_image()` - static export needs a binary that is not
  available here. An interactive figure needs a network connection when
  the notebook is reopened later, so prefer matplotlib when the figure
  must keep working offline.
- The code runs through `exec`, so top-level `await` is a syntax error.
  Never call `micropip.install` yourself: your imports are installed for
  you before the code runs.
"""

_TIKZ_RULES = """
## TikZ figures (optional)

A `tikz` tool renders a TeX/TikZ document to a figure, shown inline and
marked as unverified illustration. Prefer it over `plot` for anything
diagrammatic - commutative diagrams, geometry, number lines, labelled
constructions - and whenever the labels should be set in real math type:
you pass LaTeX straight through instead of translating it to Python.

Pass a whole document: any `\\usepackage{...}` lines, then
`\\begin{document}`, one `tikzpicture`, then `\\end{document}`. Available
packages: pgfplots, tikz-cd, circuitikz, chemfig, tikz-3dplot, amsmath,
amssymb, array. Reach for `plot` instead when the picture is driven by
computed data. TikZ figures are never ledger steps; a failed render hands
you the TeX log, so fix the source or continue without the figure.
"""

_PROVE_RULES = """
## prove! mode

The harness has already recorded the requested statement as root claim
`ROOT_CLAIM`.
Every transforming step and comment is tagged with that goal. A value is not
a proof: when the checked chain reaches the claim, call `conclude` with
`ROOT_CLAIM` and the ordered step ids that form the closing chain, then call
`set_result`.
If no available tactic closes the claim, leave it OPEN and give only a short
description of the missing move. Never bridge the gap with prose.
Re-stating the statement with `claim` re-focuses `ROOT_CLAIM` (it never
mints a duplicate), and a repeated `conclude` may replace the closing chain
with a better one (e.g. fewer assumptions). `set_result` takes the
concluded endpoint, or the claim statement itself which maps to it.
"""

_FALLBACK_SKILL = """# ToyMath verified derivations

You are the strategist; toymath is the mechanical checker. Never do
algebra in your head when a primitive can do it verified. Every tool
returns one JSON record; `check.status: "agree"` means an independent
numeric oracle confirmed the step. There is deliberately no `solve`,
`simplify`, or autonomous `integrate`: drive the derivation move by move,
feeding each step's `result` verbatim into the next call. Surface recorded
assumptions; results are
"mechanically checked", not "proved".
"""


def build_prompt(skill_path=_SKILL_PATH, plotting=False, prove_mode=False,
                 proof_claim_id='c1', tikz=False):
    """Build the small always-on core prompt and skill catalog."""
    try:
        if os.path.abspath(skill_path) == os.path.abspath(_SKILL_PATH):
            text = tactic_skills.render('core')
        else:
            with open(skill_path, 'r', encoding='utf-8') as fh:
                text = fh.read()
    except (OSError, ValueError):
        text = _FALLBACK_SKILL
        interface = tactic_skills.interface_markdown('core')
        if interface:
            text += '\n\n' + interface
    if text.startswith('---'):
        end = text.find('\n---', 3)
        if end != -1:
            text = text[end + len('\n---'):].lstrip('\n')
    replaced = re.sub(r'## Invocation.*?(?=\n## )', _TOOL_PREAMBLE, text,
                      count=1, flags=re.S)
    if replaced == text:
        replaced = text.rstrip() + '\n\n' + _TOOL_PREAMBLE
    text = (replaced.rstrip() + '\n\n'
            + tactic_skills.catalog_markdown() + '\n' + _DO_RULES)
    if plotting:
        text += _PLOT_RULES
    if tikz:
        text += _TIKZ_RULES
    if prove_mode:
        text += _PROVE_RULES.replace('ROOT_CLAIM', proof_claim_id)
    return text


# ---------------------------------------------------------------------------
# tool layer
# ---------------------------------------------------------------------------

class DoSession(object):
    """Shared state of one do! run: the ledger the steps land in, the
    streaming callbacks, the figure backends, and step-range bookkeeping."""

    def __init__(self, ledger=None, on_step=None, on_plot=None,
                 plot_backend=None, tikz_backend=None):
        self.ledger = ledger if ledger is not None else Ledger()
        self.on_step = on_step
        self.on_plot = on_plot
        self.plot_backend = plot_backend
        self.tikz_backend = tikz_backend
        self.start = len(self.ledger.steps)
        self.selection_start = len(self.ledger.selections)
        self.result_override = None
        self.result_provenance = None
        self.result_selection = None
        self.open_selection = None
        self.current_goal = None
        self.chain_goal = None
        self.proof_claim_id = None
        self.claim_start = len(self.ledger.claims)
        self.loaded_skills = {'core'}
        # the SDK executes sync tools on a thread pool, so parallel tool
        # calls hit the ledger concurrently - serialize the appends
        self._lock = threading.Lock()

    def record(self, result):
        """Ledger a successful transforming result; always return the
        (possibly step-annotated) record."""
        if result.get('ok') and result.get('op') in TRANSFORMING_OPS:
            with self._lock:
                try:
                    step = self.ledger.record(
                        result, goal=self.current_goal)
                except ValueError as exc:
                    refused = dict(result)
                    refused['ok'] = False
                    refused['error'] = str(exc)
                    return refused
                result = dict(result)
                result['step'] = {'id': step['id'], 'hash': step['hash']}
                if self.on_step is not None:
                    self.on_step(step)
        return result

    def new_steps(self):
        return self.ledger.steps[self.start:]

    def new_claims(self):
        return self.ledger.claims[self.claim_start:]

    def new_selections(self):
        return self.ledger.selections[self.selection_start:]

    def claim(self, statement, parent=None, root=False):
        """Record and focus a claim. Root claims govern prove-mode output."""
        with self._lock:
            claim = self.ledger.record_claim(statement, parent=parent)
            self.current_goal = claim['id']
            # Only the harness-created prove! claim governs final output.
            # A plain do! subclaim may remain open without converting the
            # whole calculation into prove mode.
            if root:
                self.proof_claim_id = claim['id']
        return claim

    def conclude(self, claim_id, step_ids):
        """Close a claim and return focus to its parent, if any."""
        with self._lock:
            claim = self.ledger.conclude(claim_id, step_ids)
            self.current_goal = claim.get('parent') or claim['id']
        return claim

    def comment(self, text, from_step=None):
        """Append a narrative note or structured branch marker and stream it."""
        with self._lock:
            if from_step:
                pending = self.ledger._pending_branch(self.current_goal)
                if pending is not None and from_step == pending['id']:
                    # the note annotates the pending marker itself — the
                    # agent meant a plain comment, not a second marker
                    # (markers do not stack); live agents hit this shape
                    step = self.ledger.record_comment(
                        text, goal=self.current_goal)
                else:
                    step = self.ledger.record_branch(
                        from_step, text, goal=self.current_goal)
            else:
                step = self.ledger.record_comment(
                    text, goal=self.current_goal)
            if self.on_step is not None:
                self.on_step(step)
        return step

    def set_open(self, reason):
        """Record a replay-validated run-level OPEN outcome.

        Sound because it claims nothing about the mathematics: "this
        session exhibits no certified result" is ledger-decidable.  The
        vocabulary is deliberately the claim layer's "open" — never a
        statement that no solution exists."""
        reason = (reason or '').strip()
        if not reason:
            raise ValueError(
                'set_open needs a short reason naming the exact missing '
                'move')
        if self.result_override is not None:
            raise ValueError(
                'this run already designated a certified result; the '
                'outcome is not open')
        if self.open_selection is not None:
            raise ValueError(
                f'open outcome already recorded '
                f'({self.open_selection["id"]}); stop, or continue only '
                'to certify a result')
        if self.proof_claim_id is not None:
            root = self.ledger.get_claim(self.proof_claim_id)
            if root is not None and root.get('verdict') != 'open':
                raise ValueError(
                    f'root claim {self.proof_claim_id} is concluded; call '
                    'set_result with its endpoint')
        with self._lock:
            selection = self.ledger.record_open(
                _cap_prove_summary(reason), goal=self.current_goal)
            self.open_selection = selection
        return selection

    def root_statement_endpoint(self, expr):
        """Map a retyped root-claim statement to its concluded endpoint.

        Agents naturally restate the proved claim when designating the
        final result; the selection machinery records endpoint values, so
        accept the statement spelling and answer with the endpoint its
        conclusion closes to. Returns None when the run has no concluded
        root claim or ``expr`` is not that claim's statement."""
        if self.proof_claim_id is None:
            return None
        claim = self.ledger.get_claim(self.proof_claim_id)
        if claim is None or claim.get('verdict') == 'open':
            return None
        endpoint = (claim.get('conclusion') or {}).get('endpoint')
        if not endpoint:
            return None
        statement = claim.get('statement') or ''
        if expr == statement:
            return endpoint
        try:
            if primitives.same_expression(statement, expr):
                return endpoint
        except Exception:
            pass
        return None

    def designate_result(self, expr):
        """Select a chainable value without letting the agent invent a
        detached conclusion.

        A ledger result is provenance even when it came from an earlier
        step or an earlier run sharing this ledger.  Harmless formatting
        and algebraic presentation changes are admitted only through the
        independent equality checker.  Query-only runs have no transforming
        result to select, so their value remains usable but is explicitly
        labelled unverified.
        """
        if self.proof_claim_id is not None:
            claim = self.ledger.get_claim(self.proof_claim_id)
            if claim is None or claim.get('verdict') == 'open':
                return None
            endpoint = (claim.get('conclusion') or {}).get('endpoint')
            if endpoint and (expr == endpoint or _equivalent(expr, endpoint)):
                return {
                    'status': claim['verdict'], 'source': 'claim',
                    'claim': claim['id'],
                    'steps': claim['conclusion']['steps'],
                    'method': claim['conclusion']['closure'],
                }
            return None
        with self._lock:
            steps = list(self.ledger.steps)
            # notes are not transforming steps: a comment-only run still
            # counts as query-only
            has_new_steps = any(s.get('result') is not None
                                for s in steps[self.start:])
        for step in reversed(steps):
            established = step.get('result')
            if not established:
                continue
            if expr == established:
                return {
                    'status': 'verified', 'source': 'ledger',
                    'step': step['id'], 'method': 'exact-result',
                }
            # Structural identity first, exactly as the ledger validates the
            # selection this is about to create. It is strictly stronger
            # evidence than numeric agreement, and it is the only reading
            # available for a typed result the scalar oracle refuses to
            # sample (a point collection, a pair).
            try:
                if primitives.same_expression(expr, established):
                    return {
                        'status': 'verified', 'source': 'ledger',
                        'step': step['id'], 'method': 'same-expression',
                    }
            except Exception:
                pass
            try:
                eq = core_tactics.equal_exprs(expr, established)
            except Exception:
                continue
            if eq.get('ok') and eq.get('verdict') == 'yes':
                return {
                    'status': 'verified', 'source': 'ledger',
                    'step': step['id'],
                    'method': eq.get('method', 'equal?'),
                }
        if not has_new_steps:
            return {
                'status': 'unverified', 'source': 'query-only',
                'reason': 'no transforming step was recorded in this run',
            }
        return None


def _equivalent(left, right):
    try:
        if primitives.same_expression(left, right):
            return True
    except Exception:
        pass
    try:
        rec = core_tactics.equal_exprs(left, right)
    except Exception:
        return False
    return rec.get('ok') and rec.get('verdict') == 'yes'


def make_api(session):
    """Return the stable runtime API plus generated direct test adapters.

    Only the stable control/dispatcher functions are exposed to the model by
    make_tools. The generated tactic-name adapters preserve the plain-Python
    test API without creating model-visible schemas.
    """

    def load_skill(skill: str) -> str:
        """Load one subject tactic skill before using its tactics.

        Args:
            skill: catalog subject, common subject alias, or exact tactic
                name whose owning subject should be loaded.
        """
        try:
            subject = tactic_skills.resolve(skill)
            content = tactic_skills.render(subject)
        except ValueError as exc:
            return f'Cannot load skill: {exc}'
        if subject in session.loaded_skills:
            return f'Skill {subject!r} is already loaded.'
        session.loaded_skills.add(subject)
        routed = ('' if subject == skill
                  else f' (resolved from {skill!r})')
        return f'Loaded skill {subject!r}{routed}.\n\n{content}'

    def run_tactic(tactic: str, arguments: list[str]) -> str:
        """Run one tactic from a loaded skill and return its checked record.

        Args:
            tactic: exact tactic name from a loaded skill.
            arguments: ordered string arguments shown by that skill.
        """
        result = tactic_registry.invoke_agent(tactic, arguments, session)
        return json.dumps(result, ensure_ascii=False, default=str)

    def comment(text: str, from_step: str = '') -> str:
        """Add a short unverified strategy note or exploration marker.

        Args:
            text: one or two short plain-text sentences; no scratch work.
            from_step: earlier transforming step id to resume from, or empty
                for an ordinary note. The next tactic must consume that
                source result — or, to abandon that step itself, its
                recorded input; this never supplies mathematical provenance.
        """
        try:
            step = session.comment(text, from_step=from_step or None)
        except ValueError as exc:
            return json.dumps({'ok': False,
                               'op': 'branch' if from_step else 'comment',
                               'error': str(exc)}, ensure_ascii=False)
        reply = {'ok': True, 'op': step['op'], 'id': step['id']}
        if step['op'] == 'branch':
            reply['from'] = step['args']['from']
        elif from_step:
            reply['hint'] = ('from_step named the pending marker; recorded '
                             'as a plain note — markers do not stack')
        if len(text) > 400:
            reply['hint'] = 'keep comments to one or two short sentences'
        return json.dumps(reply, ensure_ascii=False)

    def claim(statement: str, parent: str = '') -> str:
        """Record and focus a relation that will be established as true.

        Args:
            statement: LaTeX proposition, not an equation merely to solve.
            parent: parent claim id, or empty for none.
        """
        try:
            record = session.claim(statement, parent=parent or None)
        except ValueError as exc:
            return json.dumps({'ok': False, 'op': 'claim',
                               'error': str(exc)}, ensure_ascii=False)
        return json.dumps({'ok': True, 'op': 'claim', **record},
                          ensure_ascii=False)

    def conclude(claim_id: str, step_ids: list[str]) -> str:
        """Mechanically close a claim from an ordered checked chain.

        Args:
            claim_id: claim id such as c1.
            step_ids: ordered transforming step ids forming the closure.
        """
        try:
            record = session.conclude(claim_id, step_ids)
        except ValueError as exc:
            return json.dumps({'ok': False, 'op': 'conclude',
                               'claim': claim_id, 'error': str(exc)},
                              ensure_ascii=False)
        return json.dumps({'ok': True, 'op': 'conclude', 'claim': record},
                          ensure_ascii=False)

    def set_result(expr: str) -> str:
        """Designate the established value that later notebook cells use.

        Args:
            expr: LaTeX expression or relation already established in the
                shared ledger.
        """
        try:
            primitives.parse_latex(expr)
        except primitives.PrimitiveError as exc:
            return json.dumps({'ok': False, 'op': 'set_result',
                               'error': str(exc)}, ensure_ascii=False)
        mapped = session.root_statement_endpoint(expr)
        if mapped is not None:
            expr = mapped
        provenance = session.designate_result(expr)
        if (provenance is not None and session.chain_goal is not None
                and provenance.get('source') != 'claim'):
            # admission mirrors the composite closure gate: an inline
            # command (int!, diff!, ...) will refuse a final value whose
            # step is not connected to the requested expression by a
            # checked chain — refuse it HERE, while the agent can still
            # repair (live: a retyped final spelling severed the chain
            # and the run only learned after it had ended)
            from expr_commands import _chains_to_goal
            if not _chains_to_goal(session.new_steps(),
                                   provenance.get('step'),
                                   session.chain_goal):
                return json.dumps({'ok': False, 'op': 'set_result',
                                   'error': (
                    'value is established but its step is not connected '
                    'to the requested expression by a checked chain; '
                    'select the result of a step that continues the '
                    'derivation (feed recorded results forward verbatim '
                    'instead of retyping them)')}, ensure_ascii=False)
        if provenance is None:
            if session.proof_claim_id is not None:
                root = session.ledger.get_claim(session.proof_claim_id)
                if root is None or root.get('verdict') == 'open':
                    error = ('root claim is still open; call conclude with '
                             'a mechanically checked closing chain first')
                else:
                    endpoint = ((root.get('conclusion') or {})
                                .get('endpoint'))
                    error = ('value is not the endpoint of the concluded '
                             f'root claim {session.proof_claim_id}; call '
                             f'set_result with its endpoint {endpoint!r}')
            else:
                error = ('value is not mechanically equivalent to any '
                         'result in the shared ledger; use a tactic to '
                         'establish it or select an earlier ledger result')
            return json.dumps({'ok': False, 'op': 'set_result',
                               'error': error}, ensure_ascii=False)
        try:
            with session._lock:
                session.result_selection = session.ledger.record_selection(
                    expr, provenance, goal=session.current_goal)
        except ValueError as exc:
            return json.dumps({'ok': False, 'op': 'set_result',
                               'error': str(exc)}, ensure_ascii=False)
        session.result_override = expr
        session.result_provenance = provenance
        return json.dumps({'ok': True, 'op': 'set_result', 'result': expr,
                           'provenance': provenance,
                           'selection': session.result_selection['id']},
                          ensure_ascii=False)

    def set_open(reason: str) -> str:
        """Record that this run ends OPEN: no certified result.

        Only for exhausted routes — never a substitute for trying the
        applicable tactics, and never a claim that no solution exists.

        Args:
            reason: one or two short sentences naming the exact missing
                move and what was tried; wrap each formula in $...$ so
                the notebook renders it.
        """
        try:
            selection = session.set_open(reason)
        except ValueError as exc:
            return json.dumps({'ok': False, 'op': 'set_open',
                               'error': str(exc)}, ensure_ascii=False)
        return json.dumps({'ok': True, 'op': 'set_open', 'outcome': 'open',
                           'selection': selection['id']},
                          ensure_ascii=False)

    def plot(code: str, caption: str) -> str:
        """Render an unverified matplotlib/seaborn/plotly illustration.

        Args:
            code: self-contained Python that builds a figure.
            caption: one-line description displayed under the figure.
        """
        result = session.plot_backend.run_plot(code)
        figures = result.get('figures')
        if figures is None:  # a PNG-only backend still satisfies the seam
            figures = [{'kind': 'png', 'data': d}
                       for d in (result.get('images') or [])]
        if figures and session.on_plot is not None:
            session.on_plot(caption, figures)
        reply = {'ok': bool(result.get('ok')) and bool(figures),
                 'plots': len(figures),
                 'stdout': (result.get('stdout') or '')[-800:]}
        if not reply['ok']:
            reply['error'] = (result.get('error')
                              or 'the code produced no figure')[-1200:]
        return json.dumps(reply, ensure_ascii=False)

    def tikz(code: str, caption: str) -> str:
        """Render an unverified TikZ illustration.

        Args:
            code: a whole TeX document containing one tikzpicture.
            caption: one-line description displayed under the figure.
        """
        result = session.tikz_backend.render(code)
        svg = result.get('svg') if result.get('ok') else None
        if svg and session.on_plot is not None:
            session.on_plot(caption, [{'kind': 'svg', 'data': svg}])
        reply = {'ok': bool(svg), 'plots': 1 if svg else 0}
        if not reply['ok']:
            # keep the tail: that is where the TeX log's `!` lines are
            reply['error'] = (result.get('error')
                              or 'the source produced no figure')[-1200:]
        return json.dumps(reply, ensure_ascii=False)

    api = {
        'load_skill': load_skill,
        'run_tactic': run_tactic,
        'comment': comment,
        'claim': claim,
        'conclude': conclude,
        'set_result': set_result,
        'set_open': set_open,
    }
    if session.plot_backend is not None:
        api['plot'] = plot
    if session.tikz_backend is not None:
        api['tikz'] = tikz

    # Internal compatibility only: generated from the registry, never exposed
    # as model tools. This keeps focused tests and embedders off private
    # primitive functions while the public runtime has one stable dispatcher.
    for spec in tactic_registry.TACTICS:
        def legacy(*args, _spec=spec):
            argv = tactic_registry.legacy_agent_arguments(_spec, args)
            result = tactic_registry.invoke_agent(
                _spec.name, argv, session, require_loaded=False)
            return json.dumps(result, ensure_ascii=False, default=str)
        api[spec.name] = legacy
    return api


def make_tools(session):
    from agents import function_tool
    api = make_api(session)
    names = ['load_skill', 'run_tactic', 'comment', 'claim', 'conclude',
             'set_result', 'set_open']
    if session.plot_backend is not None:
        names.append('plot')
    if session.tikz_backend is not None:
        names.append('tikz')
    return [function_tool(api[name]) for name in names]


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def build_model(model_name=None):
    """OpenRouter-backed chat-completions model for the Agents SDK."""
    key = os.environ.get(API_KEY_VAR)
    if not key:
        raise DoAgentError(
            f'{API_KEY_VAR} is not set - put the OpenRouter key in .env')
    from openai import AsyncOpenAI
    from agents import OpenAIChatCompletionsModel, set_tracing_disabled
    # Leave the Agents SDK tracing ENABLED only when observability is active:
    # the OpenInference instrumentor rides that pipeline, so disabling it
    # would silently starve Langfuse. When inactive, disable it as before so
    # no traces are shipped to OpenAI's backend.
    set_tracing_disabled(not observability.active())
    client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
    return OpenAIChatCompletionsModel(
        model=model_name or os.environ.get(MODEL_VAR, DEFAULT_MODEL),
        openai_client=client)


def run_instruction(instruction, ledger=None, on_step=None, model=None,
                    max_turns=DEFAULT_MAX_TURNS, on_plot=None,
                    plot_backend=None, proof_goal=None, tikz_backend=None,
                    model_name=None, providers=(), chain_goal=None):
    """Run one do! instruction through the agent.

    Returns {ok, steps, assumptions, premises, final_result,
    final_provenance, branch_topology, abandoned_paths, summary[, error]}.
    `premises` are the inputs this run stated rather than derived — the
    boundary of what it checked.
    `steps` are the ledger steps this run added; `final_result` is the
    cell's chainable value. Figures reach
    `on_plot(caption, [{kind, data, height?}, ...])` where kind is
    png/html/svg; when a backend argument is None the configured one is
    auto-detected (TOYMATH_SANDBOX). The agent loop runs in a private
    thread with its own asyncio loop, so this is safe to call from the
    Jupyter kernel's event-loop thread.
    """
    from agents import Agent, ModelSettings, Runner, RunConfig
    from agents.exceptions import MaxTurnsExceeded
    # Set up tracing before the model is built: build_model consults
    # observability.active() to decide whether to leave Agents-SDK tracing on.
    observability.setup()
    if plot_backend is None:
        import plot_sandbox
        plot_backend = plot_sandbox.get_backend()
    if tikz_backend is None:
        import plot_sandbox
        tikz_backend = plot_sandbox.get_tikz_backend()
    session = DoSession(ledger=ledger, on_step=on_step, on_plot=on_plot,
                        plot_backend=plot_backend,
                        tikz_backend=tikz_backend)
    session.chain_goal = chain_goal
    root_claim = None
    if proof_goal is not None:
        try:
            root_claim = session.claim(proof_goal, root=True)
        except ValueError as e:
            raise DoAgentError(f'invalid proof claim: {e}')
    agent = Agent(name='toymath',
                  instructions=build_prompt(
                      plotting=session.plot_backend is not None,
                      tikz=session.tikz_backend is not None,
                      prove_mode=proof_goal is not None,
                      proof_claim_id=(root_claim['id']
                                      if root_claim is not None else 'c1')),
                  tools=make_tools(session),
                  model=(model if model is not None
                         else build_model(model_name=model_name)))
    def _tool_error_message(fargs):
        # A model naming a TACTIC as a tool must be steered back to
        # run_tactic in one turn, never abort the whole derivation
        # (live: a hallucinated `integrate_table` call killed a run
        # with three green steps).
        if getattr(fargs, 'kind', None) != 'tool_not_found':
            return fargs.default_message
        spec = tactic_registry.BY_NAME.get(fargs.tool_name)
        if spec is not None:
            return (f"'{fargs.tool_name}' is a tactic, not a tool. Call "
                    f"run_tactic with tactic='{fargs.tool_name}' and its "
                    f"ordered string arguments (load_skill "
                    f"'{spec.skill}' first if that skill is not loaded).")
        return (f"Tool '{fargs.tool_name}' does not exist. Use only the "
                "fixed tools: load_skill, run_tactic, comment, claim, "
                "conclude, set_result, set_open.")

    provider_order = tuple(providers or ())
    model_settings = None
    if provider_order:
        model_settings = ModelSettings(extra_body={
            'provider': {
                'order': list(provider_order),
                'allow_fallbacks': False,
            },
        })
    run_config = RunConfig(tool_not_found_behavior='return_error_to_model',
                           tool_error_formatter=_tool_error_message,
                           model_settings=model_settings)
    holder = {}
    trace_meta = {
        'mode': 'prove' if proof_goal is not None else 'do',
        'model': (model_name or os.environ.get(MODEL_VAR, DEFAULT_MODEL)
                  if model is None else type(model).__name__),
        'providers': list(provider_order),
        'max_turns': max_turns,
    }

    def worker():
        import asyncio
        try:
            # trace_run must be entered on this thread: the instrumentor's
            # root span inherits the active OTEL context here so the whole
            # agent trace nests under one Langfuse observation.
            with observability.trace_run(instruction,
                                         metadata=trace_meta) as span:
                res = asyncio.run(
                    Runner.run(agent, instruction, max_turns=max_turns,
                               run_config=run_config))
                observability.set_output(
                    span, str(res.final_output or '').strip() or None)
                holder['res'] = res
        except Exception as e:  # surfaced below, never swallowed
            holder['err'] = e

    t = threading.Thread(target=worker, name='toymath-do', daemon=True)
    t.start()
    t.join()
    observability.flush()

    steps = session.new_steps()
    last_transform = next((s for s in reversed(steps)
                           if s.get('result') is not None), None)
    root_claim = (session.ledger.get_claim(session.proof_claim_id)
                  if session.proof_claim_id is not None else None)
    if session.result_override is not None:
        final = session.result_override
        provenance = session.result_provenance
    elif root_claim is not None and root_claim.get('verdict') != 'open':
        final = root_claim['conclusion']['endpoint']
        provenance = {
            'status': root_claim['verdict'], 'source': 'claim',
            'claim': root_claim['id'],
            'steps': root_claim['conclusion']['steps'],
            'method': root_claim['conclusion']['closure'],
        }
    elif session.open_selection is not None:
        # the agent recorded a run-level open outcome: no certified
        # result exists, so no fallback value may masquerade as one
        final = None
        provenance = dict(session.open_selection.get('provenance') or {})
        provenance['selection'] = session.open_selection['id']
    elif root_claim is not None:
        final = None
        provenance = {
            'status': 'open', 'source': 'claim',
            'claim': root_claim['id'],
            'reason': 'no mechanically checked closing chain was recorded',
        }
    elif last_transform is not None:
        # fallback, not a designation: the step is verified, but nothing
        # says its result answers the instruction — consumers that need a
        # goal-covering value (inline expr commands) must check the chain
        final = last_transform['result']
        provenance = {
            'status': 'verified', 'source': 'ledger',
            'step': last_transform['id'], 'method': 'last-step',
        }
    else:
        final = None
        provenance = None
    marker_ids = [s['id'] for s in steps if s.get('op') == 'branch']
    topology = session.ledger.presentation_topology(
        final_provenance=(None if session.result_selection is not None
                          else provenance),
        marker_ids=marker_ids)
    if provenance and provenance.get('status') == 'unverified':
        final_assumptions = []
    elif topology['spine']:
        final_assumptions = topology['spine_assumptions']
    else:
        final_assumptions = list(session.ledger.assumptions)
    # premises this run STATED rather than derived: the boundary of what it
    # checked. Restricted to steps this run recorded, so a shared notebook
    # ledger does not re-attribute an earlier cell's givens to this one.
    run_ids = [s['id'] for s in steps if s.get('result') is not None]
    spine = [sid for sid in topology['spine'] if sid in set(run_ids)]
    final_premises = session.ledger.premises(spine or run_ids or None)
    out = {'ok': 'err' not in holder, 'steps': steps,
           'claims': session.new_claims(),
           'assumptions': final_assumptions,
           'premises': final_premises,
           'final_result': final, 'final_provenance': provenance,
           'branch_topology': topology,
           'abandoned_paths': topology['abandoned_paths'],
           'summary': None}
    if 'err' in holder:
        e = holder['err']
        if isinstance(e, MaxTurnsExceeded):
            out['turn_limit_reached'] = True
            # a run whose LAST turn committed a verified designated result
            # is finished, not failed: only the closing prose was lost. The
            # last-step fallback does not count - nothing there says the
            # value answers the instruction - and an open claim or open
            # outcome means the derivation really is unfinished.
            open_claims = any((c.get('verdict') or 'open') == 'open'
                              for c in out['claims'])
            if (session.result_override is not None
                    and (provenance or {}).get('status') == 'verified'
                    and not open_claims
                    and session.open_selection is None):
                out['ok'] = True
                out['summary'] = None
            else:
                out['error'] = (f'stopped after {max_turns} turns - partial '
                                f'derivation shown')
        else:
            out['error'] = f'{type(e).__name__}: {e}'
    else:
        summary = str(holder['res'].final_output or '').strip()
        open_claims = any((c.get('verdict') or 'open') == 'open'
                          for c in out['claims'])
        if (proof_goal is not None or open_claims
                or session.open_selection is not None):
            # an open claim must stay visibly unfinished: prose can never
            # substitute for the missing chain, in prove! or plain do!
            summary = _cap_prove_summary(summary)
            out['summary_unverified'] = True
        out['summary'] = summary
    return out


def _strip_dangling_math(text):
    """Drop an unterminated trailing math block a hard cap left behind."""
    while True:
        for opener, closer in (('\\[', '\\]'), ('\\(', '\\)')):
            idx = text.rfind(opener)
            if idx != -1 and text.find(closer, idx) == -1:
                text = text[:idx].rstrip()
                break
        else:
            if text.count('$') % 2:
                text = text[:text.rfind('$')].rstrip()
            return text


def _cap_prove_summary(text, max_lines=2, max_chars=280):
    """Keep prove-mode prose subordinate to the replayable artifact."""
    lines = [line.strip() for line in (text or '').splitlines()
             if line.strip()]
    clipped = ' '.join(lines[:max_lines])
    truncated = len(lines) > max_lines or len(clipped) > max_chars
    if len(clipped) > max_chars:
        cut = clipped[:max_chars]
        # prefer a sentence boundary so the cap never stops mid-formula
        dot = cut.rfind('. ')
        clipped = cut[:dot + 1] if dot >= 40 else cut.rstrip()
    stripped = _strip_dangling_math(clipped)
    truncated = truncated or stripped != clipped
    if truncated:
        stripped += ' … [narrative truncated; record claims and steps]'
    return stripped
