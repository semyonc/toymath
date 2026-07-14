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

import primitives
import tactic_registry
import tactic_skills
from tactics import core as core_tactics
from ledger import Ledger, TRANSFORMING_OPS

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover - dotenv ships with the kernel env
    pass

OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
API_KEY_VAR = 'OPEN_ROUTER'
MODEL_VAR = 'OPENROUTER_MODEL'
DEFAULT_MODEL = 'anthropic/claude-sonnet-5'
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
- If a tactic refuses or its check disagrees, change strategy. Never present
  an unverified result as the answer.
- Confirm equation candidates with substitute plus evaluate when cheap.
- Use comment only for short strategy/branch annotations in plain text. Notes
  are unverified prose, never steps or results; do not put scratch work there.
"""

_PLOT_RULES = """
## Plotting (optional)

A `plot` tool renders matplotlib/seaborn figures in an isolated sandbox;
they appear to the user inline, marked as unverified illustration. Use it
only when a picture genuinely helps (shape of a function, roots, area
under a curve). The code must be self-contained: import what you use,
build numpy grids, label axes, add a legend for multiple curves; one
figure per call. You translate LaTeX to Python yourself. Plots are never
ledger steps - if plotting fails, continue the derivation without it.
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
                 proof_claim_id='c1'):
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
    if prove_mode:
        text += _PROVE_RULES.replace('ROOT_CLAIM', proof_claim_id)
    return text


# ---------------------------------------------------------------------------
# tool layer
# ---------------------------------------------------------------------------

class DoSession(object):
    """Shared state of one do! run: the ledger the steps land in, the
    streaming callbacks, the plot backend, and step-range bookkeeping."""

    def __init__(self, ledger=None, on_step=None, on_plot=None,
                 plot_backend=None):
        self.ledger = ledger if ledger is not None else Ledger()
        self.on_step = on_step
        self.on_plot = on_plot
        self.plot_backend = plot_backend
        self.start = len(self.ledger.steps)
        self.result_override = None
        self.result_provenance = None
        self.current_goal = None
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
                step = self.ledger.record(result, goal=self.current_goal)
                result = dict(result)
                result['step'] = {'id': step['id'], 'hash': step['hash']}
                if self.on_step is not None:
                    self.on_step(step)
        return result

    def new_steps(self):
        return self.ledger.steps[self.start:]

    def new_claims(self):
        return self.ledger.claims[self.claim_start:]

    def claim(self, statement, parent=None, root=False):
        """Record and focus a claim. Root claims govern prove-mode output."""
        with self._lock:
            claim = self.ledger.record_claim(statement, parent=parent)
            self.current_goal = claim['id']
            if root or (parent is None and self.proof_claim_id is None):
                self.proof_claim_id = claim['id']
        return claim

    def conclude(self, claim_id, step_ids):
        """Close a claim and return focus to its parent, if any."""
        with self._lock:
            claim = self.ledger.conclude(claim_id, step_ids)
            self.current_goal = claim.get('parent') or claim['id']
        return claim

    def comment(self, text):
        """Append a narrative note to the ledger and stream it."""
        with self._lock:
            step = self.ledger.record_comment(text, goal=self.current_goal)
            if self.on_step is not None:
                self.on_step(step)
        return step

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
            skill: skill name from the available-skills catalog.
        """
        if skill in session.loaded_skills:
            return f'Skill {skill!r} is already loaded.'
        try:
            content = tactic_skills.render(skill)
        except ValueError as exc:
            return f'Cannot load skill: {exc}'
        session.loaded_skills.add(skill)
        return f'Loaded skill {skill!r}.\n\n{content}'

    def run_tactic(tactic: str, arguments: list[str]) -> str:
        """Run one tactic from a loaded skill and return its checked record.

        Args:
            tactic: exact tactic name from a loaded skill.
            arguments: ordered string arguments shown by that skill.
        """
        result = tactic_registry.invoke_agent(tactic, arguments, session)
        return json.dumps(result, ensure_ascii=False, default=str)

    def comment(text: str) -> str:
        """Add a short unverified strategy or branch note to the ledger.

        Args:
            text: one or two short plain-text sentences; no scratch work.
        """
        try:
            step = session.comment(text)
        except ValueError as exc:
            return json.dumps({'ok': False, 'op': 'comment',
                               'error': str(exc)}, ensure_ascii=False)
        reply = {'ok': True, 'op': 'comment', 'id': step['id']}
        if len(text) > 400:
            reply['hint'] = 'keep comments to one or two short sentences'
        return json.dumps(reply, ensure_ascii=False)

    def claim(statement: str, parent: str = '') -> str:
        """Record and focus a parseable mathematical subclaim.

        Args:
            statement: LaTeX statement of the subclaim.
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
        provenance = session.designate_result(expr)
        if provenance is None:
            if session.proof_claim_id is not None:
                root = session.ledger.get_claim(session.proof_claim_id)
                if root is None or root.get('verdict') == 'open':
                    error = ('root claim is still open; call conclude with '
                             'a mechanically checked closing chain first')
                else:
                    error = ('value is not the endpoint of the concluded '
                             f'root claim {session.proof_claim_id}')
            else:
                error = ('value is not mechanically equivalent to any '
                         'result in the shared ledger; use a tactic to '
                         'establish it or select an earlier ledger result')
            return json.dumps({'ok': False, 'op': 'set_result',
                               'error': error}, ensure_ascii=False)
        session.result_override = expr
        session.result_provenance = provenance
        return json.dumps({'ok': True, 'op': 'set_result', 'result': expr,
                           'provenance': provenance}, ensure_ascii=False)

    def plot(code: str, caption: str) -> str:
        """Render an unverified matplotlib/seaborn illustration.

        Args:
            code: self-contained Python that builds a figure.
            caption: one-line description displayed under the figure.
        """
        result = session.plot_backend.run_plot(code)
        images = result.get('images') or []
        if images and session.on_plot is not None:
            session.on_plot(caption, images)
        reply = {'ok': bool(result.get('ok')) and bool(images),
                 'plots': len(images),
                 'stdout': (result.get('stdout') or '')[-800:]}
        if not reply['ok']:
            reply['error'] = (result.get('error')
                              or 'the code produced no figure')[-1200:]
        return json.dumps(reply, ensure_ascii=False)

    api = {
        'load_skill': load_skill,
        'run_tactic': run_tactic,
        'comment': comment,
        'claim': claim,
        'conclude': conclude,
        'set_result': set_result,
    }
    if session.plot_backend is not None:
        api['plot'] = plot

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
             'set_result']
    if session.plot_backend is not None:
        names.append('plot')
    return [function_tool(api[name]) for name in names]


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def build_model():
    """OpenRouter-backed chat-completions model for the Agents SDK."""
    key = os.environ.get(API_KEY_VAR)
    if not key:
        raise DoAgentError(
            f'{API_KEY_VAR} is not set - put the OpenRouter key in .env')
    from openai import AsyncOpenAI
    from agents import OpenAIChatCompletionsModel, set_tracing_disabled
    set_tracing_disabled(True)
    client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
    return OpenAIChatCompletionsModel(
        model=os.environ.get(MODEL_VAR, DEFAULT_MODEL),
        openai_client=client)


def run_instruction(instruction, ledger=None, on_step=None, model=None,
                    max_turns=DEFAULT_MAX_TURNS, on_plot=None,
                    plot_backend=None, proof_goal=None):
    """Run one do! instruction through the agent.

    Returns {ok, steps, assumptions, final_result, final_provenance,
    summary[, error]}.
    `steps` are the ledger steps this run added; `final_result` is the
    cell's chainable value. Figures reach `on_plot(caption, images)`;
    when `plot_backend` is None the configured one is auto-detected
    (TOYMATH_SANDBOX). The agent loop runs in a private thread with its
    own asyncio loop, so this is safe to call from the Jupyter kernel's
    event-loop thread.
    """
    from agents import Agent, Runner
    from agents.exceptions import MaxTurnsExceeded
    if plot_backend is None:
        import plot_sandbox
        plot_backend = plot_sandbox.get_backend()
    session = DoSession(ledger=ledger, on_step=on_step, on_plot=on_plot,
                        plot_backend=plot_backend)
    root_claim = None
    if proof_goal is not None:
        try:
            root_claim = session.claim(proof_goal, root=True)
        except ValueError as e:
            raise DoAgentError(f'invalid proof claim: {e}')
    agent = Agent(name='toymath',
                  instructions=build_prompt(
                      plotting=session.plot_backend is not None,
                      prove_mode=proof_goal is not None,
                      proof_claim_id=(root_claim['id']
                                      if root_claim is not None else 'c1')),
                  tools=make_tools(session),
                  model=model if model is not None else build_model())
    holder = {}

    def worker():
        import asyncio
        try:
            holder['res'] = asyncio.run(
                Runner.run(agent, instruction, max_turns=max_turns))
        except Exception as e:  # surfaced below, never swallowed
            holder['err'] = e

    t = threading.Thread(target=worker, name='toymath-do', daemon=True)
    t.start()
    t.join()

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
    out = {'ok': 'err' not in holder, 'steps': steps,
           'claims': session.new_claims(),
           'assumptions': list(session.ledger.assumptions),
           'final_result': final, 'final_provenance': provenance,
           'summary': None}
    if 'err' in holder:
        e = holder['err']
        if isinstance(e, MaxTurnsExceeded):
            out['error'] = (f'stopped after {max_turns} turns - partial '
                            f'derivation shown')
        else:
            out['error'] = f'{type(e).__name__}: {e}'
    else:
        summary = str(holder['res'].final_output or '').strip()
        open_claims = any((c.get('verdict') or 'open') == 'open'
                          for c in out['claims'])
        if proof_goal is not None or open_claims:
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
