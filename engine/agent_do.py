#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_do.py - the `do!` agent endpoint for the Jupyter kernel.

A natural-language instruction goes in; a verified derivation comes out.
The LLM is an untrusted planner: it can only *call* the trusted primitives,
and only those tool executions append ledger steps. A hallucinated step
cannot enter the artifact - the rendered cell is the ledger, not the
model's prose.

This module owns the parts that are true of every provider: the session,
the prompt, the canonical tool bindings, cancellation, and the finalizer
that assembles a run's result. Which provider actually runs the loop lives
behind `agent_backends` (OpenRouter through the OpenAI Agents SDK today).

The always-on instructions are small; subject workflows are committed
SKILL.md files loaded progressively through a stable tactic dispatcher.
"""
import contextlib
import inspect
import json
import os
import re
import threading

import agent_config
import observability
import primitives
import tactic_registry
import tactic_skills
from agent_backends import base as agent_base
from agent_backends.base import (AgentBudget, AgentRequest, CancellationToken,
                                 DoAgentError, ToolBinding, ToolDispatcher,
                                 USER, wait_interruptibly)
from agent_backends.openrouter import DEFAULT_MAX_TURNS
from tactics import core as core_tactics
from ledger import Ledger, TRANSFORMING_OPS

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover - dotenv ships with the kernel env
    pass

_SKILL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '.claude', 'skills', 'toymath', 'SKILL.md')

# DoAgentError is defined with the backend seam (both layers raise it) and
# re-exported here: `except agent_do.DoAgentError` remains the caller's
# contract.
__all__ = ['DoAgentError', 'DoSession', 'build_prompt', 'make_api',
           'make_tool_bindings', 'make_tools', 'run_instruction',
           'finalize_session']


class SessionClosed(ValueError):
    """The run was cancelled: this session accepts no further mutations.

    A ValueError subclass so every existing tool-level `except ValueError`
    degrades into an ordinary refusal rather than an exception escaping
    into the agent loop.
    """


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
- Redrawing a figure you already drew? Pass its `figure` id as `replaces`,
  or the cell shows both the draft and the correction.
"""

_TIKZ_RULES = """
## TikZ figures (optional)

A `tikz` tool renders a TeX/TikZ document to a figure, shown inline and
marked as unverified illustration. Prefer it over `plot` for anything
diagrammatic - commutative diagrams, geometry, number lines, labelled
constructions - and whenever the labels should be set in real math type:
you pass LaTeX straight through instead of translating it to Python.

Pass preamble then body: any `\\usepackage{...}` lines, then
`\\begin{document}`, one `tikzpicture` or `tikzcd`, then `\\end{document}`.
Omit `\\documentclass` - the engine supplies its own, and a second one is a
hard TeX error. Available packages: pgfplots, tikz-cd, circuitikz, chemfig,
tikz-3dplot, amsmath, amssymb, array; use `tikz-cd` and a `tikzcd`
environment for commutative diagrams. Reach for `plot` instead when the
picture is driven by computed data. TikZ figures are never ledger steps; a
failed render hands you the TeX log, so fix the source or continue without
the figure.
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
    streaming callbacks, the figure backends, and step-range bookkeeping.

    The lock is also the cancellation boundary. `close(reason)` takes it and
    marks the session shut; a mutation that acquired it first commits
    atomically, and every later record, comment, claim, conclusion, result
    selection, and figure is refused. So an interrupted cell can never be
    half-written, and nothing the abandoned worker does afterwards lands in
    the notebook.
    """

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
        # Figures are cell output, but tool callbacks run on provider worker
        # threads. Keep an ordered, run-local copy so the notebook can render
        # them after the provider returns, on the kernel thread. They remain
        # illustrations only: none of this enters the ledger or replay.
        self._figure_events = []
        #: ids are issued per session, so a `replaces` can only ever name a
        #: figure from the run that is speaking
        self._figure_seq = 0
        self.closed = False
        self.close_reason = None
        # the SDK executes sync tools on a thread pool, so parallel tool
        # calls hit the ledger concurrently - serialize the appends
        self._lock = threading.RLock()

    # -- cancellation boundary --------------------------------------------
    def close(self, reason=USER):
        """Shut the session to further mutation. Idempotent; returns True
        only for the call that closed it."""
        with self._lock:
            if self.closed:
                return False
            self.closed = True
            self.close_reason = reason or USER
            return True

    def _refuse(self):
        raise SessionClosed(
            f'this run was stopped ({self.close_reason}); the session is '
            'closed and the steps already recorded are final')

    def deliver_figure(self, caption, figures, replaces=None):
        """Accept one successful figure, or refuse it because the run is over.

        A figure is cell output, so it needs the same boundary the ledger
        has. The durable-for-this-run copy is what the notebook renders on
        its kernel thread; `on_plot` remains an optional best-effort streaming
        seam for non-notebook callers.

        Returns `{'id', 'replaced'}` for an accepted figure, or False when
        the run is over.

        `replaces` supersedes an earlier figure IN PLACE, keeping its
        position, so a corrected render shows once where the first one
        appeared instead of the cell carrying both. Measured: a model whose
        own post-processing failed re-rendered the same number line and the
        notebook showed it twice. An id this run never issued is not
        something the figure should die of - the picture rendered - so it is
        appended and reported instead. Ids are per session: `replaces` can
        reach no other run's figures. Streaming cannot be unsent, so a
        replacement streams too; the buffer is the authoritative copy.
        """
        figures = [dict(figure) for figure in (figures or [])
                   if isinstance(figure, dict)]
        if not figures:
            return False
        with self._lock:
            if self.closed:
                return False
            self._figure_seq += 1
            event = {'status': 'ok', 'kind': 'figure',
                     'id': f'f{self._figure_seq}',
                     'caption': str(caption or ''), 'figures': figures}
            index = self._figure_position(replaces)
            replaced = None
            if index is None:
                self._figure_events.append(event)
            else:
                replaced = self._figure_events[index]['id']
                self._figure_events[index] = event
            if self.on_plot is not None:
                try:
                    self.on_plot(caption, [dict(figure)
                                           for figure in figures])
                except Exception:
                    # Streaming is optional UI only. The buffered copy still
                    # reaches the caller, so a display-handler fault may not
                    # turn a successfully rendered plot into a tool failure.
                    pass
            return {'id': event['id'], 'replaced': replaced}

    def _figure_position(self, figure_id):
        """Where a delivered figure of this run sits, or None. Caller holds
        the lock. Only successful figures have ids; a failure notice is not
        something a later render supersedes by id."""
        figure_id = (figure_id or '').strip()
        if not figure_id:
            return None
        return next((index for index, event
                     in enumerate(self._figure_events)
                     if event.get('id') == figure_id), None)

    def record_figure_failure(self, kind, caption, error):
        """Keep one failed render attempt for an end-of-run notebook notice.

        The model still receives the error in-band and may repair it. A later
        successful figure supersedes this notice; a final failed attempt is
        shown to the user instead of disappearing behind the tool protocol.
        """
        with self._lock:
            if self.closed:
                return False
            self._figure_events.append({
                'status': 'error', 'kind': str(kind or 'figure'),
                'caption': str(caption or ''), 'error': str(error or '')})
            return True

    def figure_events(self):
        """A detached snapshot of this run's local illustration events."""
        with self._lock:
            events = []
            for event in self._figure_events:
                copy = dict(event)
                if 'figures' in copy:
                    copy['figures'] = [dict(figure)
                                       for figure in copy['figures']]
                events.append(copy)
            return events

    @contextlib.contextmanager
    def _mutate(self):
        """Hold the ledger boundary for one mutation, or refuse it."""
        with self._lock:
            if self.closed:
                self._refuse()
            yield

    def record(self, result):
        """Ledger a successful transforming result; always return the
        (possibly step-annotated) record."""
        if result.get('ok') and result.get('op') in TRANSFORMING_OPS:
            with self._lock:
                if self.closed:
                    refused = dict(result)
                    refused['ok'] = False
                    refused['cancelled'] = True
                    refused['error'] = (
                        f'this run was stopped ({self.close_reason}); no '
                        'further step was recorded')
                    return refused
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
        with self._mutate():
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
        with self._mutate():
            claim = self.ledger.conclude(claim_id, step_ids)
            self.current_goal = claim.get('parent') or claim['id']
        return claim

    def comment(self, text, from_step=None):
        """Append a narrative note or structured branch marker and stream it."""
        with self._mutate():
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
        with self._mutate():
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


def _closed_reply(op, session):
    """A refusal for a non-ledger tool reached after the run was stopped."""
    return json.dumps({
        'ok': False, 'op': op, 'cancelled': True,
        'error': f'this run was stopped ({session.close_reason})',
    }, ensure_ascii=False)


def _with_figure_id(reply, delivery, replaces):
    """Tell the model which figure it just drew, and what it superseded.

    The id is what a later corrected render passes back as `replaces`, so
    the notebook shows one figure instead of a retry's duplicate. A
    `replaces` naming nothing in this run is reported rather than failed:
    the figure did render, and dropping it over a bookkeeping mistake would
    lose the picture the user asked for.
    """
    if not isinstance(delivery, dict):
        return reply
    reply['figure'] = delivery['id']
    if delivery['replaced']:
        reply['replaced'] = delivery['replaced']
    elif (replaces or '').strip():
        reply['note'] = (f"no figure {str(replaces).strip()!r} in this run; "
                         'shown as a new figure')
    return reply


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
            with session._mutate():
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

    def plot(code: str, caption: str, replaces: str = '') -> str:
        """Render an unverified matplotlib/seaborn/plotly illustration.

        The figure goes to the user's notebook, not back to you: this call
        answers {"ok", "plots", "figure"}, plus "error" when it failed.

        Args:
            code: self-contained Python that builds a figure.
            caption: one-line description displayed under the figure.
            replaces: the "figure" id of an earlier figure THIS RUN drew
                that this call corrects or redraws. The notebook then shows
                the corrected figure once, in the original's place, instead
                of both. Leave empty for a new figure.
        """
        if session.closed:
            return _closed_reply('plot', session)
        result = session.plot_backend.run_plot(code)
        figures = result.get('figures')
        if figures is None:  # a PNG-only backend still satisfies the seam
            figures = [{'kind': 'png', 'data': d}
                       for d in (result.get('images') or [])]
        ok = bool(result.get('ok')) and bool(figures)
        error = (result.get('error')
                 or 'the code produced no figure')[-1200:]
        accepted = (session.deliver_figure(caption, figures,
                                           replaces=replaces) if ok else
                    session.record_figure_failure('plot', caption, error))
        if not accepted and session.closed:
            return _closed_reply('plot', session)
        reply = {'ok': ok, 'plots': len(figures) if ok else 0,
                 'stdout': (result.get('stdout') or '')[-800:]}
        if not reply['ok']:
            reply['error'] = error
        return json.dumps(_with_figure_id(reply, accepted, replaces),
                          ensure_ascii=False)

    def tikz(code: str, caption: str, replaces: str = '') -> str:
        """Render an unverified TikZ illustration.

        The figure goes to the user's notebook, not back to you: this call
        answers {"ok", "plots", "figure"}, plus "error" when it failed.

        Args:
            code: preamble then body — any \\usepackage /
                \\usetikzlibrary lines, then \\begin{document}, one
                tikzpicture or tikzcd, then \\end{document}. Omit
                \\documentclass; the engine supplies it. tikz-cd is
                available, so use tikzcd for commutative diagrams.
            caption: one-line description displayed under the figure.
            replaces: the "figure" id of an earlier figure THIS RUN drew
                that this call corrects or redraws. The notebook then shows
                the corrected figure once, in the original's place, instead
                of both. Leave empty for a new figure.
        """
        if session.closed:
            return _closed_reply('tikz', session)
        result = session.tikz_backend.render(code)
        svg = result.get('svg') if result.get('ok') else None
        if svg:
            accepted = session.deliver_figure(
                caption, [{'kind': 'svg', 'data': svg}], replaces=replaces)
        else:
            error = (result.get('error')
                     or 'the source produced no figure')[-1200:]
            accepted = session.record_figure_failure('tikz', caption, error)
        if not accepted and session.closed:
            return _closed_reply('tikz', session)
        reply = {'ok': bool(svg), 'plots': 1 if svg else 0}
        if not reply['ok']:
            # keep the tail: that is where the TeX log's `!` lines are
            reply['error'] = error
        return json.dumps(_with_figure_id(reply, accepted, replaces),
                          ensure_ascii=False)

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


# ---------------------------------------------------------------------------
# one canonical tool surface
# ---------------------------------------------------------------------------

# the stable model-visible surface, in the order the model sees it. Tactic
# growth must never grow this list: subjects arrive through load_skill and
# run_tactic.
TOOL_NAMES = ('load_skill', 'run_tactic', 'comment', 'claim', 'conclude',
              'set_result', 'set_open')
FIGURE_TOOL_NAMES = ('plot', 'tikz')

_JSON_TYPES = {
    str: {'type': 'string'},
    list[str]: {'type': 'array', 'items': {'type': 'string'}},
}


def make_tool_bindings(session):
    """The canonical, backend-neutral tool surface for one run.

    Every provider adapter converts *these* records - name, description,
    schema, handler - and nothing else, so the two backends cannot drift
    into showing the model different tools. The bindings close over
    `make_api(session)`; the generated tactic-name adapters in that API stay
    internal compatibility helpers and never become model-visible.
    """
    api = make_api(session)
    names = list(TOOL_NAMES)
    if session.plot_backend is not None:
        names.append('plot')
    if session.tikz_backend is not None:
        names.append('tikz')
    return tuple(_binding(api[name]) for name in names)


def make_tools(session):
    """Agents SDK tools for `session` (compatibility for direct callers)."""
    from agent_backends.openrouter import function_tools
    return function_tools(ToolDispatcher(make_tool_bindings(session)))


def _binding(function):
    """Derive one binding from a handler's signature and docstring.

    The schema is generated once, here, rather than by each provider SDK:
    that is what lets a parity test assert both backends advertise the same
    tool. `unittests_do` pins it against the Agents SDK's own reading of
    these functions.
    """
    summary, argument_docs = _parse_docstring(
        function.__doc__ or '', tuple(inspect.signature(function).parameters))
    properties = {}
    for name, parameter in inspect.signature(function).parameters.items():
        spec = dict(_JSON_TYPES.get(parameter.annotation, {'type': 'string'}))
        if argument_docs.get(name):
            spec['description'] = argument_docs[name]
        spec['title'] = name.replace('_', ' ').title()
        if parameter.default is not inspect.Parameter.empty:
            spec['default'] = parameter.default
        properties[name] = spec
    schema = {
        'properties': properties,
        # strict schemas list every property as required; an argument with a
        # default is still supplied by the model, or restored by the shared
        # validator when it is omitted
        'required': list(properties),
        'title': f'{function.__name__}_args',
        'type': 'object',
        'additionalProperties': False,
    }
    return ToolBinding(name=function.__name__, description=summary,
                       input_schema=schema,
                       invoke=lambda payload: function(**payload))


_ARG_RE = re.compile(r'^(\w+):\s*(.*)$')


def _parse_docstring(text, parameters):
    """Split a Google-style docstring into (summary, {argument: text})."""
    lines = inspect.cleandoc(text).splitlines()
    start = next((i for i, line in enumerate(lines)
                  if line.strip() == 'Args:'), None)
    if start is None:
        return '\n'.join(lines).rstrip(), {}
    summary = '\n'.join(lines[:start]).rstrip()
    documented = {}
    pending = set(parameters)
    current = None
    for line in lines[start + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        match = _ARG_RE.match(stripped)
        if match is not None and match.group(1) in pending:
            current = match.group(1)
            pending.discard(current)
            documented[current] = [match.group(2)]
        elif current is not None:
            documented[current].append(stripped)
    return summary, {name: '\n'.join(parts).strip()
                     for name, parts in documented.items()}


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def prepare_session(ledger=None, on_step=None, on_plot=None,
                    plot_backend=None, tikz_backend=None, proof_goal=None,
                    chain_goal=None):
    """Build one run's session and, in prove! mode, its root claim."""
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
    if proof_goal is not None:
        try:
            session.claim(proof_goal, root=True)
        except ValueError as e:
            raise DoAgentError(f'invalid proof claim: {e}')
    return session


def resolve_backend(backend=None, model=None, model_name=None, providers=(),
                    max_turns=DEFAULT_MAX_TURNS):
    """Pick the backend for one run.

    `backend` is a ready backend object or a name. A run never fails over to
    the other provider once it has begun - a Codex limit must not silently
    create OpenRouter charges, and an OpenRouter outage must not silently
    consume a user's Codex allowance.
    """
    if backend is not None and not isinstance(backend, str):
        return backend
    if backend == agent_config.AUTO:
        # resolved once, here, before anything is spent
        backend = agent_config.resolve(backend).backend
    if backend == agent_config.CODEX:
        from agent_backends.codex import CodexBackend
        return CodexBackend(model=model_name)
    from agent_backends.openrouter import OpenRouterBackend
    return OpenRouterBackend(model=model, model_name=model_name,
                             providers=providers, max_turns=max_turns)


def _stated_premises(premises, chain_goal):
    """Premises worth naming beside the result.

    A premise that merely restates the requested expression — or a piece
    the goal itself wraps, like the integrand of the goal integral —
    names nothing the reader must grant beyond the cell they typed: the
    task is not a given (live: a one-step FTC cell reported "rests on 1
    stated premise" naming its own d/dx input). Typed coefficients and
    other genuine givens never cover the goal, so they always survive;
    with no goal to compare (plain do!), every premise stands."""
    if chain_goal is None:
        return premises
    return [p for p in premises
            if not primitives.covers_goal(p['input'], chain_goal)]


def run_instruction(instruction, ledger=None, on_step=None, model=None,
                    max_turns=DEFAULT_MAX_TURNS, on_plot=None,
                    plot_backend=None, proof_goal=None, tikz_backend=None,
                    model_name=None, providers=(), chain_goal=None,
                    budget=None, grace_period=None, backend=None, route=None):
    """Run one do! instruction through the agent.

    Returns {ok, status, steps, assumptions, premises, final_result,
    final_provenance, branch_topology, abandoned_paths, figures,
    figure_error, summary[, error]}.
    `premises` are the inputs this run stated rather than derived — the
    boundary of what it checked.
    `steps` are the ledger steps this run added; `final_result` is the
    cell's chainable value. Successful figures are returned as run-local
    illustration records and also reach the optional
    `on_plot(caption, [{kind, data, height?}, ...])` streaming callback, where
    kind is png/html/svg. A final failed figure attempt is returned in
    `figure_error`. When a backend argument is None the configured one is
    auto-detected (TOYMATH_SANDBOX). The agent loop runs in a private thread
    with its own asyncio loop, so notebook display must happen after this
    function returns to the Jupyter kernel's event-loop thread.

    Jupyter Stop (KeyboardInterrupt) cancels the provider run and closes
    the session within a bounded grace period. A cancelled run keeps every
    step it committed - those replay - but designates no chainable result:
    `status` is "interrupted" (or "budget_exhausted"), `final_result` is
    None, and a verified candidate appears only as a labelled
    `partial_result`.
    """
    if route is not None:
        # notebook-local routing: backend, model, and provider order travel
        # together so a run cannot mix one backend with another's model
        backend = backend if backend is not None else route.backend
        model_name = model_name if model_name is not None else route.model
        providers = providers or route.providers
    # Set up tracing before the backend builds its model: build_model
    # consults observability.active() to decide whether to leave Agents-SDK
    # tracing on.
    observability.setup()
    session = prepare_session(ledger=ledger, on_step=on_step,
                              on_plot=on_plot, plot_backend=plot_backend,
                              tikz_backend=tikz_backend,
                              proof_goal=proof_goal, chain_goal=chain_goal)
    budget = budget if budget is not None else AgentBudget()
    cancellation = CancellationToken()
    backend = resolve_backend(backend=backend, model=model,
                              model_name=model_name, providers=providers,
                              max_turns=max_turns)
    dispatcher = ToolDispatcher(
        make_tool_bindings(session), cancellation, budget=budget,
        serialize=getattr(backend, 'serialize_tools', False))
    request = AgentRequest(
        instruction=instruction,
        developer_instructions=build_prompt(
            plotting=session.plot_backend is not None,
            tikz=session.tikz_backend is not None,
            prove_mode=proof_goal is not None,
            proof_claim_id=(session.proof_claim_id or 'c1')),
        dispatcher=dispatcher,
        cancellation=cancellation,
        model=model_name,
        budget=budget,
        trace_metadata={'mode': 'prove' if proof_goal is not None else 'do'})
    handle = backend.start(request)
    # One place decides what cancellation means, whoever noticed first: the
    # kernel thread on Stop, or a tool worker that ran out of budget. The
    # session closes BEFORE the provider is asked to stop, so no tool call
    # racing the interrupt can still append a step.
    cancellation.add_listener(
        lambda reason: (session.close(reason), handle.cancel(reason)))
    try:
        outcome = wait_interruptibly(handle, cancellation,
                                     grace_period=grace_period,
                                     budget=budget)
    except KeyboardInterrupt:
        # Stop landed in the narrow window between starting the run and
        # entering the wait; the run is still ours to contain, and the
        # notebook must not be handed a live worker with an open session
        session.close(USER)
        handle.cancel(USER)
        outcome = wait_interruptibly(handle, cancellation,
                                     grace_period=grace_period,
                                     budget=budget)
    if outcome.cancelled:
        session.close(cancellation.reason or USER)
    observability.flush()
    return finalize_session(session, outcome, max_turns=max_turns)


# ---------------------------------------------------------------------------
# finalizer: one run's result, assembled from the ledger
# ---------------------------------------------------------------------------

def _designated_result(session, steps):
    """The run's value and its provenance, in precedence order."""
    root_claim = (session.ledger.get_claim(session.proof_claim_id)
                  if session.proof_claim_id is not None else None)
    if session.result_override is not None:
        return session.result_override, session.result_provenance
    if root_claim is not None and root_claim.get('verdict') != 'open':
        return root_claim['conclusion']['endpoint'], {
            'status': root_claim['verdict'], 'source': 'claim',
            'claim': root_claim['id'],
            'steps': root_claim['conclusion']['steps'],
            'method': root_claim['conclusion']['closure'],
        }
    if session.open_selection is not None:
        # the agent recorded a run-level open outcome: no certified result
        # exists, so no fallback value may masquerade as one
        provenance = dict(session.open_selection.get('provenance') or {})
        provenance['selection'] = session.open_selection['id']
        return None, provenance
    if root_claim is not None:
        return None, {
            'status': 'open', 'source': 'claim',
            'claim': root_claim['id'],
            'reason': 'no mechanically checked closing chain was recorded',
        }
    last_transform = next((s for s in reversed(steps)
                           if s.get('result') is not None), None)
    if last_transform is not None:
        # fallback, not a designation: the step is verified, but nothing
        # says its result answers the instruction — consumers that need a
        # goal-covering value (inline expr commands) must check the chain
        return last_transform['result'], {
            'status': 'verified', 'source': 'ledger',
            'step': last_transform['id'], 'method': 'last-step',
        }
    return None, None


def finalize_session(session, outcome, max_turns=None):
    """Assemble one run's result from the ledger and the provider outcome.

    Provider-neutral by construction: everything mathematical here is read
    back out of the session's own records, and `outcome.final_text` is
    unverified narrative that can never become a result.
    """
    steps = session.new_steps()
    final, provenance = _designated_result(session, steps)
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
    final_premises = _stated_premises(
        session.ledger.premises(spine or run_ids or None),
        session.chain_goal)
    figure_events = session.figure_events()
    figures = [{'id': event['id'], 'caption': event['caption'],
                'figures': event['figures']}
               for event in figure_events if event['status'] == 'ok']
    last_figure = figure_events[-1] if figure_events else None
    figure_error = (dict(last_figure)
                    if last_figure and last_figure['status'] == 'error'
                    else None)
    out = {'ok': outcome.status == agent_base.COMPLETED,
           'status': outcome.status,
           # provider-side facts about the run (backend, model, turn ids,
           # any native tool the model reached for) - never mathematics
           'outcome_metadata': dict(outcome.metadata or {}),
           'steps': steps,
           'claims': session.new_claims(),
           'assumptions': final_assumptions,
           'premises': final_premises,
           'final_result': final, 'final_provenance': provenance,
           'branch_topology': topology,
           'abandoned_paths': topology['abandoned_paths'],
           # Run-local UI artifacts only; never ledger evidence or replay.
           'figures': figures, 'figure_error': figure_error,
           'summary': None}
    if outcome.cancelled:
        return _finalize_cancelled(out, outcome, final, provenance)
    if outcome.status == agent_base.FAILED:
        _finalize_failure(out, outcome, session, provenance, max_turns)
    else:
        summary = outcome.final_text or ''
        open_claims = any((c.get('verdict') or 'open') == 'open'
                          for c in out['claims'])
        if (session.proof_claim_id is not None or open_claims
                or session.open_selection is not None):
            # an open claim must stay visibly unfinished: prose can never
            # substitute for the missing chain, in prove! or plain do!
            summary = _cap_prove_summary(summary)
            out['summary_unverified'] = True
        out['summary'] = summary
    return out


_CANCEL_MESSAGES = {
    agent_base.INTERRUPTED: 'stopped by the user',
    agent_base.BUDGET_EXHAUSTED: 'the run budget was exhausted',
    agent_base.CAPABILITY_VIOLATION: (
        'the run used a capability outside the ToyMath tool surface and was '
        'invalidated'),
}


def _finalize_cancelled(out, outcome, final, provenance):
    """A cancelled cell keeps its checked steps and chains nothing.

    Everything committed before the session closed stays in the notebook
    ledger and still replays. What the cell must not do is act finished:
    no chainable value, no execution-history backreference, and no closing
    narrative — the model never got to write one, and a candidate recorded
    moments before Stop was never confirmed as the answer.
    """
    verified = bool(final) and (provenance or {}).get('status') not in (
        None, 'unverified', 'open')
    message = _CANCEL_MESSAGES.get(outcome.status, 'stopped')
    if outcome.metadata.get('grace_exceeded'):
        message += '; the provider did not acknowledge cancellation in time'
    out.update(ok=False, cancelled=True, final_result=None,
               final_provenance=None, summary=None,
               partial_result=final if verified else None,
               partial_provenance=provenance if verified else None,
               error=outcome.error or message)
    return out


def _finalize_failure(out, outcome, session, provenance, max_turns):
    if not outcome.metadata.get('turn_limit'):
        out['error'] = outcome.error or 'the agent run failed'
        return
    out['turn_limit_reached'] = True
    # a run whose LAST turn committed a verified designated result is
    # finished, not failed: only the closing prose was lost. The last-step
    # fallback does not count - nothing there says the value answers the
    # instruction - and an open claim or open outcome means the derivation
    # really is unfinished.
    open_claims = any((c.get('verdict') or 'open') == 'open'
                      for c in out['claims'])
    if (session.result_override is not None
            and (provenance or {}).get('status') == 'verified'
            and not open_claims
            and session.open_selection is None):
        out['ok'] = True
        out['status'] = agent_base.COMPLETED
        out['summary'] = None
    else:
        limit = max_turns if max_turns is not None else '?'
        out['error'] = (f'stopped after {limit} turns - partial '
                        f'derivation shown')


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
