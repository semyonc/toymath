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

The agent instructions are generated from the committed skill file
(.claude/skills/toymath/SKILL.md) at load time, so the CLI skill and the
in-kernel agent cannot drift apart.
"""
import json
import os
import re
import threading

import primitives
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

_TOOL_PREAMBLE = """## Invocation

You are running inside a Jupyter notebook cell. Call one tool per
derivation step; every tool wraps exactly one toymath primitive and
returns its JSON record. Successful transforming steps are appended to
the notebook's session ledger automatically and rendered to the user as
they happen - you never need to repeat them.

Reading the command table and examples below: each command name is a
tool of the same name, positional arguments are the tool's parameters in
order, and `--session`/`show`/`replay` are handled by the notebook (there
are no such tools). `--direction backward` is the `direction` parameter
of `rewrite`.
"""

_DO_RULES = """
## do! mode

- Work strictly step by step: plan the next move, call the tool, read the
  record, feed its `result` verbatim into the next call.
- When the instruction's goal is reached, call `set_result` once with the
  established expression or equation (that becomes the cell's value,
  which later cells reference), then stop calling tools and reply with
  ONE short sentence (the final result and any assumptions it is
  conditional on). `set_result` accepts any value mechanically equivalent
  to a result already in the shared ledger, including an earlier result
  selected after later verification. It rejects a newly synthesized value
  after transforming steps. In a query-only run with no new transforming
  step it accepts the value but marks it explicitly unverified. Do not
  restate the steps; the ledger rendering does that.
- If a tool refuses or a check disagrees, change strategy; never present
  an unverified result as the answer.
- For equations, confirm candidate solutions with substitute + evaluate
  when that is cheap.
- In long derivations, use `comment` to annotate the ledger: the strategy
  you are starting, which piece of a split you are on, why you branch.
  Notes are unverified prose - never a step, never a result. Keep them to
  one or two short sentences in PLAIN TEXT: no LaTeX/MathML markup, no
  $ delimiters (comments are not rendered as math - write x^2/(1-x^2)^3
  style notation). Never write derivations or scratch work into
  comments; reason silently and record only the decision.
- For integrals beyond the direct tactics: propose an equivalent integrand
  (e.g. partial fractions) with `integrate_rewrite` (it is verified
  mechanically), split sums with `integrate_linearity`, solve each piece,
  then call `integrate_assemble` with the linearity step id and the ordered
  ledger step ids of the piece antiderivatives. Never type the final sum into
  `expand`: that checks only the typed expression, not piece provenance.
- For limits: use `limit_rewrite` for an equal body, `limit_substitute` only
  at a continuity point, `limit_table` for a named standard form, and
  `limit_lhopital` for one sampled 0/0 or infinity/infinity step. Split sums
  with `limit_linearity`, solve every returned piece, then call
  `limit_assemble` with the linearity step id and ordered value-step ids.
  Never type the final sum into `expand`; provenance must stay explicit.
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

_FALLBACK_SKILL = """# ToyMath verified derivations

You are the strategist; toymath is the mechanical checker. Never do
algebra in your head when a primitive can do it verified. Every tool
returns one JSON record; `check.status: "agree"` means an independent
numeric oracle confirmed the step. There is deliberately no `solve`,
`simplify`, or autonomous `integrate`: drive the derivation move by move
(apply / expand / collect / substitute / evaluate / diff / rewrite /
factor_* / integrate_* / limit_* tactics), feeding each step's `result` verbatim
into the next call. Surface recorded assumptions; results are
"mechanically checked", not "proved".
"""


def build_prompt(skill_path=_SKILL_PATH, plotting=False):
    """Agent instructions derived from SKILL.md: frontmatter stripped, the
    bash invocation section replaced by tool-calling guidance, do!-mode
    (and, when available, plotting) rules appended."""
    try:
        with open(skill_path, 'r', encoding='utf-8') as fh:
            text = fh.read()
    except OSError:
        text = _FALLBACK_SKILL
    if text.startswith('---'):
        end = text.find('\n---', 3)
        if end != -1:
            text = text[end + len('\n---'):].lstrip('\n')
    text = re.sub(r'## Invocation.*?(?=\n## )', _TOOL_PREAMBLE, text,
                  count=1, flags=re.S)
    text = text.rstrip() + '\n' + _DO_RULES
    if plotting:
        text += _PLOT_RULES
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
        # the SDK executes sync tools on a thread pool, so parallel tool
        # calls hit the ledger concurrently - serialize the appends
        self._lock = threading.Lock()

    def record(self, result):
        """Ledger a successful transforming result; always return the
        (possibly step-annotated) record."""
        if result.get('ok') and result.get('op') in TRANSFORMING_OPS:
            with self._lock:
                step = self.ledger.record(result)
                result = dict(result)
                result['step'] = {'id': step['id'], 'hash': step['hash']}
                if self.on_step is not None:
                    self.on_step(step)
        return result

    def new_steps(self):
        return self.ledger.steps[self.start:]

    def comment(self, text):
        """Append a narrative note to the ledger and stream it."""
        with self._lock:
            step = self.ledger.record_comment(text)
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
                eq = primitives.equal_exprs(expr, established)
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


def make_api(session):
    """The plain-python tool implementations, name -> callable returning
    the JSON record as a string. `make_tools` wraps these for the SDK;
    tests call them directly."""

    def _run(fn, *args):
        return json.dumps(session.record(fn(*args)),
                          ensure_ascii=False, default=str)

    def apply(equation: str, op: str, arg: str) -> str:
        """Apply an operation to both sides of an equation or inequality.

        Args:
            equation: LaTeX relation, e.g. "2x + 3 = 7" or "-2x \\lt 4".
            op: one of + - * / ^.
            arg: LaTeX operand to apply on both sides.
        """
        return _run(primitives.apply_both_sides, equation, op, arg)

    def expand(expr: str) -> str:
        """Distribute products/powers and canonicalize; simplifies each
        side of an equation; merges like terms over opaque atoms outside
        the rational fragment.

        Args:
            expr: LaTeX expression or relation.
        """
        return _run(primitives.expand, expr)

    def collect(expr: str, var: str) -> str:
        """Group an expression by powers of a variable (equations and
        rational functions supported, opaque atoms handled).

        Args:
            expr: LaTeX expression or relation.
            var: variable name to collect by, e.g. "x".
        """
        return _run(primitives.collect, expr, var)

    def substitute(expr: str, var: str, value: str) -> str:
        """Replace every free occurrence of a variable by a value.

        Args:
            expr: LaTeX expression or relation.
            var: variable name to replace.
            value: LaTeX replacement value.
        """
        return _run(primitives.substitute, expr, var, value)

    def evaluate(expr: str) -> str:
        """Exact arithmetic when no free variables remain; on an equation
        reports holds: true/false (solution checking).

        Args:
            expr: LaTeX expression or relation with numeric leaves.
        """
        return _run(primitives.evaluate, expr)

    def diff(expr: str, var: str) -> str:
        """Differentiate with respect to a variable (verified by central
        differences).

        Args:
            expr: LaTeX expression.
            var: differentiation variable.
        """
        return _run(primitives.differentiate, expr, var)

    def limit_rewrite(expr: str, new_body: str) -> str:
        """Replace a limit body by an expression YOU propose; equal?
        mechanically verifies the bodies before carrying the binder over.

        Args:
            expr: full LaTeX limit, e.g. "\\lim_{x \\to 1} ...".
            new_body: mechanically equivalent body in LaTeX.
        """
        return _run(primitives.limit_rewrite, expr, new_body)

    def limit_substitute(expr: str) -> str:
        """Evaluate a finite limit by continuity substitution. Records the
        continuity assumption and requires the approach oracle to converge.

        Args:
            expr: full LaTeX limit expression.
        """
        return _run(primitives.limit_substitute, expr)

    def limit_linearity(expr: str) -> str:
        """Split the limit of a top-level sum into ordered limit pieces.
        The result records existence assumptions; solve the pieces and close
        them with limit_assemble.

        Args:
            expr: full LaTeX limit whose body is a sum.
        """
        return _run(primitives.limit_linearity, expr)

    def limit_table(expr: str) -> str:
        """Apply a standard finite limit rule, constant rule, or finite
        rational leading-coefficient rule at infinity.

        Args:
            expr: full LaTeX limit expression.
        """
        return _run(primitives.limit_table, expr)

    def limit_lhopital(expr: str) -> str:
        """Apply one l'Hopital step after the oracle observes 0/0 or
        infinity/infinity; records the theorem's differentiability and
        existence premises.

        Args:
            expr: full quotient limit expression.
        """
        return _run(primitives.limit_lhopital, expr)

    def limit_assemble(linearity_step: str,
                       value_steps: list[str]) -> str:
        """Assemble a limit-linearity split from RECORDED piece values.

        Args:
            linearity_step: ledger id of the limit_linearity step.
            value_steps: ordered ledger ids whose results are piece limits.
        """
        with session._lock:
            steps = list(session.ledger.steps)
        by_id = {step['id']: step for step in steps}
        linearity = by_id.get(linearity_step)
        if linearity is None or linearity.get('op') != 'limit_linearity':
            return json.dumps({
                'ok': False, 'op': 'limit_assemble',
                'error': (f'{linearity_step!r} is not a recorded '
                          'limit_linearity step'),
            }, ensure_ascii=False)
        if not isinstance(value_steps, list):
            return json.dumps({
                'ok': False, 'op': 'limit_assemble',
                'error': 'value_steps must be an ordered list',
            }, ensure_ascii=False)
        values = []
        for source_id in value_steps:
            source = by_id.get(source_id)
            if source is None or source.get('result') is None:
                return json.dumps({
                    'ok': False, 'op': 'limit_assemble',
                    'error': f'unknown transforming step {source_id!r}',
                }, ensure_ascii=False)
            values.append(source['result'])
        result = primitives.limit_assemble(linearity['input'], values)
        if result.get('ok'):
            result['sources'] = {
                'linearity': linearity_step,
                'values': list(value_steps),
            }
        return json.dumps(session.record(result), ensure_ascii=False,
                          default=str)

    def rewrite(expr: str, lemma: str, direction: str) -> str:
        """Apply a registered equality lemma at the root or first matching
        subterm.

        Args:
            expr: LaTeX expression.
            lemma: lemma name from the lemmas tool, e.g. "diff_squares".
            direction: "forward" (usual) or "backward".
        """
        return _run(primitives.rewrite, expr, lemma, direction)

    def integrate_power_rule(expr: str, var: str) -> str:
        """Term-by-term power-rule antiderivative; accepts \\int wrappers;
        refuses the 1/x case (use integrate_table).

        Args:
            expr: LaTeX integrand or \\int ... dx expression.
            var: integration variable.
        """
        return _run(primitives.integrate_power_rule, expr, var)

    def integrate_table(expr: str, var: str) -> str:
        """Table antiderivatives (sin, cos, e^x, sinh, cosh, 1/x -> ln x);
        closed under sums and constant factors.

        Args:
            expr: LaTeX integrand or \\int ... dx expression.
            var: integration variable.
        """
        return _run(primitives.integrate_table, expr, var)

    def integrate_by_parts(expr: str, var: str, u: str, dv: str) -> str:
        """One application of integration by parts; verifies u*dv equals
        the integrand and returns u v - \\int v du plus a
        remaining_integral for the next tactic.

        Args:
            expr: LaTeX integrand or \\int ... dx expression.
            var: integration variable.
            u: the u part, LaTeX.
            dv: the dv part, LaTeX.
        """
        return _run(primitives.integrate_by_parts, expr, var, u, dv)

    def integrate_substitute(expr: str, var: str, u_expr: str, u_var: str,
                             new_integrand: str) -> str:
        """u-substitution: supply u and the integrand rewritten in the new
        variable; toymath verifies f(u(x))*u'(x) equals the integrand.

        Args:
            expr: LaTeX integrand or \\int ... dx expression.
            var: original integration variable.
            u_expr: substitution expression, e.g. "x^2".
            u_var: new variable name, e.g. "u".
            new_integrand: integrand rewritten in u_var, e.g. "\\cos(u)".
        """
        return _run(primitives.integrate_substitute, expr, var, u_expr,
                    u_var, new_integrand)

    def integrate_rewrite(expr: str, var: str, new_integrand: str) -> str:
        """Replace the integrand with an expression YOU propose (e.g. a
        partial-fraction decomposition); toymath verifies the two
        integrands are mechanically equal before rewriting the integral.

        Args:
            expr: LaTeX integrand or \\int ... dx expression.
            var: integration variable.
            new_integrand: equivalent integrand, LaTeX.
        """
        return _run(primitives.integrate_rewrite, expr, var, new_integrand)

    def integrate_linearity(expr: str, var: str) -> str:
        """Split the integral of a top-level sum into a signed sum of
        integrals, one per term (exact sum rule); attack each resulting
        integral separately, then use integrate_assemble.

        Args:
            expr: LaTeX integrand or \\int ... dx expression whose
                integrand is a sum.
            var: integration variable.
        """
        return _run(primitives.integrate_linearity, expr, var)

    def integrate_assemble(linearity_step: str,
                           antiderivative_steps: list[str]) -> str:
        """Assemble a linearity split from RECORDED piece results. Supply
        one ledger step id per term, in the exact order returned by
        integrate_linearity. ToyMath retrieves those results, verifies each
        derivative against its corresponding integrand, applies the signs,
        and adds one fresh constant.

        Args:
            linearity_step: ledger id of the integrate_linearity step,
                e.g. "s4".
            antiderivative_steps: ordered ledger ids whose results are the
                completed antiderivatives, e.g. ["s8", "s12", "s16"].
        """
        with session._lock:
            steps = list(session.ledger.steps)
        by_id = {step['id']: step for step in steps}
        linearity = by_id.get(linearity_step)
        if linearity is None:
            return json.dumps({
                'ok': False, 'op': 'integrate_assemble',
                'error': f'unknown linearity step {linearity_step!r}',
            }, ensure_ascii=False)
        if linearity.get('op') != 'integrate_linearity':
            return json.dumps({
                'ok': False, 'op': 'integrate_assemble',
                'error': (f'{linearity_step!r} is '
                          f'{linearity.get("op")!r}, not '
                          'integrate_linearity'),
            }, ensure_ascii=False)
        if not isinstance(antiderivative_steps, list):
            return json.dumps({
                'ok': False, 'op': 'integrate_assemble',
                'error': 'antiderivative_steps must be an ordered list',
            }, ensure_ascii=False)
        values = []
        for source_id in antiderivative_steps:
            source = by_id.get(source_id)
            if source is None or source.get('result') is None:
                return json.dumps({
                    'ok': False, 'op': 'integrate_assemble',
                    'error': f'unknown transforming step {source_id!r}',
                }, ensure_ascii=False)
            values.append(source['result'])
        result = primitives.integrate_assemble(
            linearity['input'], linearity['args']['var'], values)
        if result.get('ok'):
            result['sources'] = {
                'linearity': linearity_step,
                'antiderivatives': list(antiderivative_steps),
            }
        return json.dumps(session.record(result), ensure_ascii=False,
                          default=str)

    def comment(text: str) -> str:
        """Add a short narrative note to the ledger (strategy, which piece
        you are working on, why you branch). Notes are unverified prose:
        never a transforming step, never a final result.

        Args:
            text: one or two SHORT plain-text sentences. No LaTeX/MathML
                markup, no $ delimiters - notes are not rendered as math
                (write x^2/(1-x^2)^3 style notation). Never scratch work:
                reason silently, record only the decision.
        """
        try:
            step = session.comment(text)
        except ValueError as e:
            return json.dumps({'ok': False, 'op': 'comment',
                               'error': str(e)}, ensure_ascii=False)
        reply = {'ok': True, 'op': 'comment', 'id': step['id']}
        if len(text) > 400:
            reply['hint'] = ('note recorded, but comments should be one '
                             'or two short sentences - reason silently '
                             'instead of writing derivations here')
        return json.dumps(reply, ensure_ascii=False)

    def factor_gcd(expr: str) -> str:
        """Pull common factors from a sum or applicable relation sides,
        e.g. 6x^2+9x=3 -> 3x(2x+3)=3.

        Args:
            expr: LaTeX expression, equation, or inequality.
        """
        return _run(primitives.factor_gcd, expr)

    def factor_quadratic(expr: str, var: str) -> str:
        """Factor quadratics with rational roots in an expression or on
        applicable relation sides; reports roots and refuses
        irrational/complex cases.

        Args:
            expr: LaTeX quadratic expression, equation, or inequality.
            var: the variable, e.g. "x".
        """
        return _run(primitives.factor_quadratic, expr, var)

    def equal(expr1: str, expr2: str) -> str:
        """Check whether two expressions are equal: verdict yes/no/unknown
        (query only, not a ledger step).

        Args:
            expr1: LaTeX expression.
            expr2: LaTeX expression.
        """
        return _run(primitives.equal_exprs, expr1, expr2)

    def lemmas() -> str:
        """List the registered rewrite lemmas (query only)."""
        return _run(primitives.list_lemmas)

    def set_result(expr: str) -> str:
        """Designate the cell's final value (what later cells reference as
        [[n]]). Call once, right before your closing sentence, with the
        expression or equation the derivation established. The value must
        be mechanically equivalent to any result in the shared ledger; an
        earlier result may be selected after later verification. A run with
        no transforming steps may designate a query-only value, but it is
        explicitly marked unverified. Not a ledger step - the steps above
        are what justify it.

        Args:
            expr: LaTeX expression or relation, e.g. "x = 2".
        """
        try:
            primitives.parse_latex(expr)
        except primitives.PrimitiveError as e:
            return json.dumps({'ok': False, 'op': 'set_result',
                               'error': str(e)}, ensure_ascii=False)
        provenance = session.designate_result(expr)
        if provenance is None:
            return json.dumps({
                'ok': False, 'op': 'set_result',
                'error': ('value is not mechanically equivalent to any '
                          'result in the shared ledger; use a primitive to '
                          'establish it or select an earlier ledger result'),
            }, ensure_ascii=False)
        session.result_override = expr
        session.result_provenance = provenance
        return json.dumps({'ok': True, 'op': 'set_result', 'result': expr,
                           'provenance': provenance}, ensure_ascii=False)

    def plot(code: str, caption: str) -> str:
        """Render matplotlib/seaborn figures in an isolated sandbox; they
        are shown to the user inline as unverified illustration (never a
        ledger step).

        Args:
            code: self-contained Python; import everything you use
                (numpy/matplotlib/seaborn available), build the figure,
                no plt.show() needed.
            caption: one-line description displayed under the figure.
        """
        r = session.plot_backend.run_plot(code)
        images = r.get('images') or []
        if images and session.on_plot is not None:
            session.on_plot(caption, images)
        reply = {'ok': bool(r.get('ok')) and bool(images),
                 'plots': len(images),
                 'stdout': (r.get('stdout') or '')[-800:]}
        if not reply['ok']:
            reply['error'] = (r.get('error')
                              or 'the code produced no figure')[-1200:]
        return json.dumps(reply, ensure_ascii=False)

    fns = [apply, expand, collect, substitute, evaluate, diff, rewrite,
           limit_rewrite, limit_substitute, limit_linearity, limit_table,
           limit_lhopital, limit_assemble,
           integrate_power_rule, integrate_table, integrate_by_parts,
           integrate_substitute, integrate_rewrite, integrate_linearity,
           integrate_assemble,
           factor_gcd, factor_quadratic, equal, lemmas, comment,
           set_result]
    if session.plot_backend is not None:
        fns.append(plot)
    return {f.__name__: f for f in fns}


def make_tools(session):
    from agents import function_tool
    return [function_tool(f) for f in make_api(session).values()]


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
                    plot_backend=None):
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
    agent = Agent(name='toymath',
                  instructions=build_prompt(
                      plotting=session.plot_backend is not None),
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
    if session.result_override is not None:
        final = session.result_override
        provenance = session.result_provenance
    elif last_transform is not None:
        final = last_transform['result']
        provenance = {
            'status': 'verified', 'source': 'ledger',
            'step': last_transform['id'], 'method': 'exact-result',
        }
    else:
        final = None
        provenance = None
    out = {'ok': 'err' not in holder, 'steps': steps,
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
        out['summary'] = str(holder['res'].final_output or '').strip()
    return out
