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
DEFAULT_MAX_TURNS = 24

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
  conditional on). Do not restate the steps; the ledger rendering does
  that.
- If a tool refuses or a check disagrees, change strategy; never present
  an unverified result as the answer.
- For equations, confirm candidate solutions with substitute + evaluate
  when that is cheap.
"""

_FALLBACK_SKILL = """# ToyMath verified derivations

You are the strategist; toymath is the mechanical checker. Never do
algebra in your head when a primitive can do it verified. Every tool
returns one JSON record; `check.status: "agree"` means an independent
numeric oracle confirmed the step. There is deliberately no `solve`,
`simplify`, or autonomous `integrate`: drive the derivation move by move
(apply / expand / collect / substitute / evaluate / diff / rewrite /
factor_* / integrate_* tactics), feeding each step's `result` verbatim
into the next call. Surface recorded assumptions; results are
"mechanically checked", not "proved".
"""


def build_prompt(skill_path=_SKILL_PATH):
    """Agent instructions derived from SKILL.md: frontmatter stripped, the
    bash invocation section replaced by tool-calling guidance, do!-mode
    rules appended."""
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
    return text.rstrip() + '\n' + _DO_RULES


# ---------------------------------------------------------------------------
# tool layer
# ---------------------------------------------------------------------------

class DoSession(object):
    """Shared state of one do! run: the ledger the steps land in, the
    streaming callback, and the step-range bookkeeping."""

    def __init__(self, ledger=None, on_step=None):
        self.ledger = ledger if ledger is not None else Ledger()
        self.on_step = on_step
        self.start = len(self.ledger.steps)
        self.result_override = None

    def record(self, result):
        """Ledger a successful transforming result; always return the
        (possibly step-annotated) record."""
        if result.get('ok') and result.get('op') in TRANSFORMING_OPS:
            step = self.ledger.record(result)
            result = dict(result)
            result['step'] = {'id': step['id'], 'hash': step['hash']}
            if self.on_step is not None:
                self.on_step(step)
        return result

    def new_steps(self):
        return self.ledger.steps[self.start:]


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

    def factor_gcd(expr: str) -> str:
        """Pull the common factor out of a sum, e.g. 6x^2+9x -> 3x(2x+3).

        Args:
            expr: LaTeX expression.
        """
        return _run(primitives.factor_gcd, expr)

    def factor_quadratic(expr: str, var: str) -> str:
        """Factor a quadratic with rational roots; reports the roots;
        refuses irrational/complex cases.

        Args:
            expr: LaTeX quadratic expression.
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
        expression or equation the derivation established. Not a ledger
        step - the steps above are what justify it.

        Args:
            expr: LaTeX expression or relation, e.g. "x = 2".
        """
        try:
            primitives.parse_latex(expr)
        except primitives.PrimitiveError as e:
            return json.dumps({'ok': False, 'op': 'set_result',
                               'error': str(e)}, ensure_ascii=False)
        session.result_override = expr
        return json.dumps({'ok': True, 'op': 'set_result', 'result': expr},
                          ensure_ascii=False)

    fns = (apply, expand, collect, substitute, evaluate, diff, rewrite,
           integrate_power_rule, integrate_table, integrate_by_parts,
           integrate_substitute, factor_gcd, factor_quadratic, equal,
           lemmas, set_result)
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
                    max_turns=DEFAULT_MAX_TURNS):
    """Run one do! instruction through the agent.

    Returns {ok, steps, assumptions, final_result, summary[, error]}.
    `steps` are the ledger steps this run added; `final_result` is the
    last transforming step's result (the cell's chainable value). The
    agent loop runs in a private thread with its own asyncio loop, so
    this is safe to call from the Jupyter kernel's event-loop thread.
    """
    from agents import Agent, Runner
    from agents.exceptions import MaxTurnsExceeded
    session = DoSession(ledger=ledger, on_step=on_step)
    agent = Agent(name='toymath', instructions=build_prompt(),
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
    final = session.result_override or (steps[-1]['result'] if steps
                                        else None)
    out = {'ok': 'err' not in holder, 'steps': steps,
           'assumptions': list(session.ledger.assumptions),
           'final_result': final, 'summary': None}
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
