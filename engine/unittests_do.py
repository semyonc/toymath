#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the do! agent endpoint (agent_do.py + MathShell integration).

Everything here runs offline: the agent loop is exercised with a scripted
fake Model. Set TOYMATH_LIVE_TESTS=1 to also run real OpenRouter
round-trips (the OPEN_ROUTER credential is the one thing they still take
from the environment/.env; backend, model, tracing, and figure sandboxes
are pinned by the live classes themselves - see LIVE_MODEL).
"""
import _thread
import base64
import dataclasses
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

# A developer may enable production tracing in the repository's .env.  The
# scripted test suite must still make neither OpenRouter nor Langfuse calls;
# the dedicated observability test below opts back in with an in-memory
# exporter and an HTTP-mocked model transport.
os.environ['TOYMATH_OBSERVABILITY'] = 'off'
os.environ['OPENAI_AGENTS_DISABLE_TRACING'] = 'true'
# The context-preparation stage is ON by default in production, and it makes
# one model call of its own before the executor. A scripted turn list is a
# transcript of the EXECUTOR, so leaving the stage on would feed each script's
# first turn to the preparation call instead. Tests that exercise the stage
# pass `prewarm=True` explicitly and script that turn themselves.
os.environ['TOYMATH_PREWARM'] = 'off'

import agent_config
import agent_do
import mathShell
import observability
import plot_sandbox
import tactic_registry
from tactics import core as core_tactics
from tactics import integration as integration_tactics
from agent_backends import (base as agent_base, codex as codex_backend,
                            codex_transport,
                            openrouter as openrouter_backend)
from agent_do import DoSession, make_api, run_instruction
from ledger import Ledger

from agents.models.interface import Model
from agents.items import ModelResponse
from agents.usage import Usage
from openai.types.responses import (ResponseFunctionToolCall,
                                    ResponseOutputMessage,
                                    ResponseOutputText)


_NO_BROWSER = None
_SANDBOX_HOME = None
_CLEARED_ENV = None
#: Whatever `TOYMATH_CODEX_HOME` said before the throwaway home replaced it.
#: The live Codex class is the one place that must see the developer's own
#: signed-in home again; everything else keeps the sandbox.
_REAL_CODEX_HOME = None

#: Notebook-default routing this module must decide for itself. ToyMath loads
#: `.env`, so leaving these set makes the suite assert against whatever the
#: developer happens to have configured - a `TOYMATH_CODEX_MODEL` in a working
#: `.env` failed three routing tests that are about ToyMath's own defaults.
#: Tests that care about a value set it explicitly with `mock.patch.dict`.
_ENV_DEFAULTS = ('TOYMATH_CODEX_MODEL', 'OPENROUTER_MODEL',
                 'TOYMATH_AGENT_BACKEND')


def setUpModule():
    """No test may reach a real browser, the real Codex home, or the
    developer's own agent configuration.

    Browser: `login!` hands its one-time sign-in URL to the OS browser
    instead of printing it, so any test touching that path would otherwise
    open a real window — pointed at the fake app-server's
    `auth.openai.test`, which is not a real page. Individual tests still
    patch `mathShell._open_browser` when they assert on the outcome; this is
    the backstop that makes forgetting harmless.

    Home: backend auto-resolution probes for a signed-in Codex account, and
    that probe starts a real app-server against `~/.toymath/codex-home` —
    the user's own authenticated home — which then lives for the rest of the
    process. An offline suite must not spawn that, read that account, or
    write to that directory, so the whole module runs against a throwaway
    home instead.

    Environment: `.env` is loaded for real runs, so the routing defaults are
    cleared here and reinstated afterwards.
    """
    global _NO_BROWSER, _SANDBOX_HOME, _CLEARED_ENV, _REAL_CODEX_HOME
    _NO_BROWSER = mock.patch('webbrowser.open', return_value=False)
    _NO_BROWSER.start()
    _SANDBOX_HOME = tempfile.mkdtemp(prefix='toymath-test-codex-home-')
    _REAL_CODEX_HOME = os.environ.get(codex_backend.HOME_VAR)
    os.environ[codex_backend.HOME_VAR] = _SANDBOX_HOME
    # `load_dotenv()` runs at import of each module that needs it, and
    # `toymathkernel` is imported lazily inside tests - late enough to put a
    # cleared variable straight back. Force the remaining loads to happen
    # first, so clearing them below actually sticks.
    import toymathkernel                                       # noqa: F401
    _CLEARED_ENV = {name: os.environ.pop(name)
                    for name in _ENV_DEFAULTS if name in os.environ}


def _set_codex_home(home):
    """Point `TOYMATH_CODEX_HOME` at `home`, or unset it when None.

    A plain assignment cannot express "unset", and the difference matters:
    an unset variable means the default `~/.toymath/codex-home`, which is
    exactly the authenticated home the offline suite must stay out of.
    """
    if home is None:
        os.environ.pop(codex_backend.HOME_VAR, None)
    else:
        os.environ[codex_backend.HOME_VAR] = home


def tearDownModule():
    if _NO_BROWSER is not None:
        _NO_BROWSER.stop()
    codex_backend.close_runtime()
    if _SANDBOX_HOME is not None:
        _set_codex_home(_REAL_CODEX_HOME)
        shutil.rmtree(_SANDBOX_HOME, True)
    os.environ.update(_CLEARED_ENV or {})


class ScriptedModel(Model):
    """Deterministic Model: replays a fixed list of turns."""

    def __init__(self, turns):
        self.turns = list(turns)

    async def get_response(self, system_instructions, input, model_settings,
                           tools, output_schema, handoffs, tracing, *,
                           previous_response_id=None, conversation_id=None,
                           prompt=None):
        if not self.turns:
            raise AssertionError('scripted model ran out of turns')
        return ModelResponse(output=self.turns.pop(0), usage=Usage(),
                             response_id=None)

    def stream_response(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def tool_call(name, args, cid):
    if name in tactic_registry.BY_NAME:
        spec = tactic_registry.BY_NAME[name]
        ordered = []
        for arg in spec.agent_args:
            value = args.get(arg.name, arg.default)
            if value is tactic_registry._MISSING:
                raise AssertionError(f'missing scripted argument {arg.name}')
            if value is None:
                # the model-visible schema is a list of strings: an omitted
                # optional argument is a shorter list, never a null
                continue
            if arg.nargs in ('+', '*'):
                ordered.extend(value)
            else:
                ordered.append(value)
        name = 'run_tactic'
        args = {'tactic': spec.name, 'arguments': ordered}
    return ResponseFunctionToolCall(type='function_call', name=name,
                                    arguments=json.dumps(args),
                                    call_id=cid, id=cid)


def message(text):
    return ResponseOutputMessage(
        id='m1', type='message', role='assistant', status='completed',
        content=[ResponseOutputText(type='output_text', text=text,
                                    annotations=[])])


SOLVE_SCRIPT = [
    [tool_call('apply', {'equation': '2x + 3 = 7', 'op': '-', 'arg': '3'},
               'c1')],
    [tool_call('expand', {'expr': '2x+3 - 3 = 7 - 3'}, 'c2')],
    [message('Subtracted 3 from both sides.')],
]


class TestPromptBuilder(unittest.TestCase):
    def test_skill_derived_prompt(self):
        p = agent_do.build_prompt()
        # frontmatter stripped, bash invocation replaced, rules appended
        self.assertFalse(p.startswith('---'))
        self.assertNotIn('python toymath_cli.py <command>', p)
        self.assertIn('Invocation inside do!', p)
        self.assertIn('## do! mode', p)
        # core signatures and the small domain catalog are present; detailed
        # subject tactics are progressively loaded
        self.assertIn('factor_quadratic', p)
        self.assertIn('`integration`', p)
        self.assertIn('`equations`', p)
        self.assertNotIn('integrate_by_parts EXPR', p)
        self.assertIn('never for\n  an equation whose solutions', p)
        self.assertIn('mechanically checked', p)

    def test_missing_skill_falls_back(self):
        p = agent_do.build_prompt(skill_path='/nonexistent/SKILL.md')
        self.assertIn('mechanical checker', p)
        self.assertIn('## do! mode', p)

    def test_prompt_forbids_restating_the_ledger(self):
        # live defect: an agent hand-retyped the step chain as a markdown
        # table (raw pipes in the em line, broken attachment: image link).
        # The notebook renders the verified chain from the ledger; the
        # prompt must keep final answers to plain commentary.
        p = agent_do.build_prompt()
        self.assertIn('Never restate ledger steps', p)
        self.assertIn('write image links', p)

    def test_prompt_names_structured_exploration_markers(self):
        p = agent_do.build_prompt()
        self.assertIn('exact step id as from_step', p)
        self.assertIn('source result verbatim to the\n  next tactic', p)
        self.assertIn('never mathematical case data', p)
        # the abandon-the-step-itself form (root-anchor gap) is steered too
        self.assertIn('abandon a recorded step ITSELF', p)
        self.assertIn("from that step's recorded input", p)
        # from_step is only for resume intent; notes must omit it
        self.assertIn('Pass from_step ONLY when the', p)

    def test_prompt_steers_open_outcome_with_naming_guard(self):
        p = agent_do.build_prompt()
        self.assertIn('call set_open once', p)
        self.assertIn('it never means no solution exists', p)
        self.assertIn('Never dress the last step up as the answer', p)
        # math in the reason is $-delimited so the notebook banner
        # typesets it instead of showing raw backslash commands
        self.assertIn('with formulas in $...$', p)

    def test_prove_prompt_names_actual_shared_ledger_claim(self):
        p = agent_do.build_prompt(prove_mode=True, proof_claim_id='c7')
        self.assertIn('root claim\n`c7`', p)
        self.assertIn('`c7` and the ordered step ids', p)
        self.assertNotIn('ROOT_CLAIM', p)


class TestDoSessionApi(unittest.TestCase):
    def test_transforming_call_recorded_and_streamed(self):
        seen = []
        session = DoSession(on_step=seen.append)
        api = make_api(session)
        rec = json.loads(api['apply']('2x + 3 = 7', '-', '3'))
        self.assertTrue(rec['ok'])
        self.assertEqual(rec['step']['id'], 's1')
        self.assertEqual(len(session.new_steps()), 1)
        self.assertEqual(seen[0]['id'], 's1')

    def test_query_calls_not_recorded(self):
        session = DoSession()
        api = make_api(session)
        rec = json.loads(api['equal']('(x+1)^2', 'x^2+2x+1'))
        self.assertEqual(rec['verdict'], 'yes')
        self.assertEqual(json.loads(api['lemmas']())['ok'], True)
        self.assertEqual(session.new_steps(), [])

    def test_failed_call_not_recorded(self):
        session = DoSession()
        api = make_api(session)
        rec = json.loads(api['apply']('2x + 3 = 7', '/', '0'))
        self.assertFalse(rec['ok'])
        self.assertEqual(session.new_steps(), [])

    def test_set_result_validates_query_only_as_unverified(self):
        session = DoSession()
        api = make_api(session)
        rec = json.loads(api['set_result']('x = 2'))
        self.assertTrue(rec['ok'])
        self.assertEqual(rec['provenance']['status'], 'unverified')
        self.assertEqual(rec['selection'], 'r1')
        self.assertEqual(session.ledger.selections[0]['result'], 'x = 2')
        self.assertEqual(session.result_override, 'x = 2')
        self.assertFalse(json.loads(api['set_result']('\\frac{'))['ok'])
        self.assertEqual(session.result_override, 'x = 2')  # kept
        self.assertEqual(session.new_steps(), [])  # never a ledger step

    def test_comment_streamed_but_not_transforming(self):
        seen = []
        session = DoSession(on_step=seen.append)
        api = make_api(session)
        rec = json.loads(api['comment']('strategy: partial fractions'))
        self.assertTrue(rec['ok'])
        self.assertEqual(rec['id'], 's1')
        self.assertEqual(seen[0]['op'], 'comment')
        # a note is not provenance: the run is still query-only
        res = json.loads(api['set_result']('x = 2'))
        self.assertEqual(res['provenance']['status'], 'unverified')
        self.assertFalse(json.loads(api['comment']('  '))['ok'])

    def test_comment_from_step_records_a_structured_branch_marker(self):
        seen = []
        session = DoSession(on_step=seen.append)
        api = make_api(session)
        source = json.loads(api['expand']('(x+1)^2'))['step']['id']
        rec = json.loads(api['comment'](
            'the substitution route stalled', source))
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['op'], 'branch')
        self.assertEqual(rec['from'], 's1')
        marker = session.new_steps()[-1]
        self.assertEqual(marker['args']['reason'],
                         'the substitution route stalled')
        self.assertIsNone(marker['result'])
        self.assertEqual(seen[-1]['op'], 'branch')
        self.assertEqual(session.ledger.replay()['status'], 'verified')
        bad = json.loads(api['comment']('resume', 's999'))
        self.assertFalse(bad['ok'])
        self.assertEqual(bad['op'], 'branch')

    def test_branch_marker_attaches_to_next_transform(self):
        session = DoSession()
        api = make_api(session)
        first = json.loads(api['expand']('(x+1)^2'))
        json.loads(api['substitute'](first['result'], 'x', '1'))
        marker = json.loads(api['comment'](
            'the numeric detour is not the symbolic answer', 's1'))
        refused = json.loads(api['expand']('(y+1)^2'))
        self.assertFalse(refused['ok'])
        self.assertIn('does not resume branch marker', refused['error'])
        self.assertEqual(len(session.new_steps()), 3)
        resumed = json.loads(api['factor_quadratic'](
            first['result'], 'x'))
        self.assertEqual(resumed['step']['id'], 's4')
        target = session.ledger.steps[-1]
        self.assertEqual(target['exploration']['marker'], marker['id'])
        self.assertEqual(target['exploration']['from'], 's1')
        selected = json.loads(api['set_result'](resumed['result']))
        self.assertEqual(selected['selection'], 'r1')
        topology = session.ledger.presentation_topology()
        self.assertEqual(topology['spine'], ['s1', 's4'])
        self.assertEqual(topology['abandoned_paths'][0]['steps'], ['s2'])
        self.assertEqual(session.ledger.replay()['status'], 'verified')

    def test_wedged_marker_refusal_teaches_the_fresh_start(self):
        """Live int! trace shape: the agent marked from_step=s1 meaning
        'the whole ansatz was wrong', then had every fresh start refused
        and went open without finding either escape. The refusal must name
        the unwedging move, and the move must actually work."""
        session = DoSession()
        api = make_api(session)
        json.loads(api['expand']('(x+1)^2'))
        marker = json.loads(api['comment']('the ansatz itself was wrong',
                                           's1'))
        self.assertEqual(marker['op'], 'branch')
        refused = json.loads(api['expand']('(z+5)^2'))
        self.assertFalse(refused['ok'])
        self.assertIn('first resolve the marker', refused['error'])
        # the taught escape: one tactic on the marked step's recorded
        # input resolves the marker...
        resolved = json.loads(api['expand']('(x+1)^2'))
        self.assertTrue(resolved['ok'], resolved.get('error'))
        # ...and the fresh line is then admitted
        fresh = json.loads(api['expand']('(z+5)^2'))
        self.assertTrue(fresh['ok'], fresh.get('error'))
        self.assertEqual(session.ledger.replay()['status'], 'verified')

    def test_pending_marker_note_downgrades_and_refusal_steers(self):
        # live conv! trace shape: 5 of 8 comment calls died against the
        # pending-marker gate because the refusal never named the repair
        session = DoSession()
        api = make_api(session)
        first = json.loads(api['expand']('(x+1)^2'))
        json.loads(api['substitute'](first['result'], 'x', '1'))
        marker = json.loads(api['comment']('detour went nowhere', 's1'))
        self.assertEqual(marker['op'], 'branch')
        # a note whose from_step names the pending marker ITSELF is the
        # observed misuse: record the plain note the agent meant
        note = json.loads(api['comment']('status: routes exhausted',
                                         marker['id']))
        self.assertTrue(note['ok'])
        self.assertEqual(note['op'], 'comment')
        self.assertIn('markers do not stack', note['hint'])
        # any other from_step while a marker is pending names both repairs
        refused = json.loads(api['comment']('analysis note', 's1'))
        self.assertFalse(refused['ok'])
        self.assertIn('plain comment (no from_step)', refused['error'])
        self.assertEqual(session.ledger.replay()['status'], 'verified')

    def test_set_open_records_open_outcome(self):
        session = DoSession()
        api = make_api(session)
        json.loads(api['expand']('(x+1)^2'))
        self.assertFalse(json.loads(api['set_open'](' '))['ok'])
        rec = json.loads(api['set_open']('missing: a checked lower bound'))
        self.assertTrue(rec['ok'])
        self.assertEqual(rec['outcome'], 'open')
        self.assertEqual(rec['selection'], 'r1')
        outcome = session.ledger.selections[-1]
        self.assertIsNone(outcome['result'])
        self.assertEqual(outcome['provenance']['status'], 'open')
        dup = json.loads(api['set_open']('again'))
        self.assertFalse(dup['ok'])
        self.assertIn('already recorded', dup['error'])
        self.assertEqual(session.new_steps()[-1]['op'], 'expand')
        self.assertEqual(session.ledger.replay()['status'], 'verified')

    def test_set_open_refused_after_designation_and_recovery_after_open(
            self):
        session = DoSession()
        api = make_api(session)
        first = json.loads(api['expand']('(x+2)^2'))
        json.loads(api['set_result'](first['result']))
        refused = json.loads(api['set_open']('stalled'))
        self.assertFalse(refused['ok'])
        self.assertIn('already designated', refused['error'])
        # and the reverse order recovers: set_result supersedes set_open
        other = DoSession()
        api2 = make_api(other)
        step = json.loads(api2['expand']('(x+3)^2'))
        self.assertTrue(json.loads(api2['set_open']('thought stuck'))['ok'])
        recovered = json.loads(api2['set_result'](step['result']))
        self.assertTrue(recovered['ok'], recovered.get('error'))
        self.assertEqual([s['id'] for s in other.ledger.selections],
                         ['r1', 'r2'])
        self.assertEqual(other.ledger.replay()['status'], 'verified')

    def test_set_result_rejects_detached_conclusion(self):
        session = DoSession()
        api = make_api(session)
        api['factor_quadratic']('x^2 - 5x + 6', 'x')
        api['substitute']('x^2 - 5x + 6 = 0', 'x', '3')
        rec = json.loads(api['set_result']('(x-3)(x-2)=0'))
        self.assertFalse(rec['ok'])
        self.assertIn('shared ledger', rec['error'])
        self.assertIsNone(session.result_override)

    def test_set_result_can_select_earlier_semantic_result(self):
        session = DoSession()
        api = make_api(session)
        api['expand']('(x+1)^2')
        api['substitute']('x^2+2x+1=4', 'x', '1')
        rec = json.loads(api['set_result']('(x+1)^2'))
        self.assertTrue(rec['ok'])
        self.assertEqual(rec['provenance']['status'], 'verified')
        self.assertEqual(rec['provenance']['step'], 's1')
        self.assertNotEqual(rec['provenance']['method'], 'exact-result')
        self.assertEqual(session.ledger.selections[-1]['provenance']['step'],
                         's1')

    def test_set_result_can_select_shared_ledger_result(self):
        ledger = Ledger()
        first = DoSession(ledger=ledger)
        first_api = make_api(first)
        established = json.loads(first_api['expand']('(x+1)^2'))['result']
        session = DoSession(ledger=ledger)
        rec = json.loads(make_api(session)['set_result'](established))
        self.assertTrue(rec['ok'])
        self.assertEqual(rec['provenance']['step'], 's1')
        self.assertEqual(session.new_steps(), [])

    def test_plain_do_claim_does_not_become_governing_proof(self):
        session = DoSession()
        claim = session.claim('x^2-1=0')
        self.assertIsNone(session.proof_claim_id)
        step = json.loads(make_api(session)['expand']('x^2-1'))
        selected = json.loads(make_api(session)['set_result'](
            step['result']))
        self.assertTrue(selected['ok'], selected.get('error'))
        self.assertEqual(selected['provenance']['step'], 's1')
        self.assertEqual(session.ledger.get_claim(claim['id'])['verdict'],
                         'open')

    def test_plain_do_reuses_equivalent_open_claim(self):
        session = DoSession()
        first = session.claim('3x^{2}-3=0')
        again = session.claim('3x^2-3=0')
        self.assertEqual(again['id'], first['id'])
        self.assertEqual(len(session.ledger.claims), 1)

    def test_open_proof_rejects_unrelated_verified_value(self):
        session = DoSession()
        root = session.claim(
            r'\lim_{n \to \infty} \frac{n}{2^n} = 0', root=True)
        api = make_api(session)
        step = json.loads(api['limit_table'](
            r'\lim_{n \to \infty} \frac{1}{n}'))
        self.assertEqual(session.new_steps()[0]['goal'], root['id'])
        close = json.loads(api['conclude'](root['id'],
                                           [step['step']['id']]))
        self.assertFalse(close['ok'])
        self.assertIn('does not close claim', close['error'])
        selected = json.loads(api['set_result']('0'))
        self.assertFalse(selected['ok'])
        self.assertIn('still open', selected['error'])

    def test_concluded_proof_result_has_claim_provenance(self):
        session = DoSession()
        root = session.claim(r'\lim_{n \to \infty} \frac{1}{n} = 0',
                             root=True)
        api = make_api(session)
        step = json.loads(api['limit_table'](
            r'\lim_{n \to \infty} \frac{1}{n}'))
        close = json.loads(api['conclude'](root['id'],
                                           [step['step']['id']]))
        self.assertTrue(close['ok'], close.get('error'))
        selected = json.loads(api['set_result']('0'))
        self.assertTrue(selected['ok'])
        self.assertEqual(selected['provenance']['source'], 'claim')
        self.assertEqual(selected['provenance']['claim'], 'c1')

    def test_integrate_assemble_uses_ordered_ledger_results(self):
        session = DoSession()
        api = make_api(session)
        linearity = json.loads(api['integrate_linearity'](
            '\\int (x+x^2) \\, d x', 'x'))
        first = json.loads(api['integrate_power_rule'](
            linearity['integrals'][0], 'x'))
        second = json.loads(api['integrate_power_rule'](
            linearity['integrals'][1], 'x'))
        rec = json.loads(api['integrate_assemble'](
            linearity['step']['id'],
            [first['step']['id'], second['step']['id']]))
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['sources'], {
            'linearity': 's1', 'antiderivatives': ['s2', 's3']})
        self.assertEqual(rec['step']['id'], 's4')
        selected = json.loads(api['set_result'](rec['result']))
        self.assertTrue(selected['ok'])
        self.assertEqual(selected['provenance']['step'], 's4')
        self.assertEqual(session.ledger.replay()['status'], 'verified')

    def test_limit_assemble_uses_ordered_ledger_results(self):
        session = DoSession()
        api = make_api(session)
        linearity = json.loads(api['limit_linearity'](
            '\\lim_{x \\to 0} (\\frac{\\sin x}{x}+x^2)'))
        first = json.loads(api['limit_table'](
            linearity['limits'][0]))
        second = json.loads(api['limit_substitute'](
            linearity['limits'][1]))
        rec = json.loads(api['limit_assemble'](
            linearity['step']['id'],
            [first['step']['id'], second['step']['id']]))
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], '1')
        self.assertEqual(rec['sources'], {
            'linearity': 's1', 'values': ['s2', 's3']})
        self.assertEqual(session.ledger.replay()['status'], 'verified')

    def test_limit_assemble_rejects_wrong_source_order(self):
        session = DoSession()
        api = make_api(session)
        linearity = json.loads(api['limit_linearity'](
            '\\lim_{x \\to 0} (\\frac{\\sin x}{x}+x^2)'))
        first = json.loads(api['limit_table'](linearity['limits'][0]))
        second = json.loads(api['limit_substitute'](linearity['limits'][1]))
        rec = json.loads(api['limit_assemble'](
            's1', [second['step']['id'], first['step']['id']]))
        self.assertFalse(rec['ok'])
        self.assertIn('piece 1', rec['error'])

    def test_replay_rejects_tampered_limit_assembly_provenance(self):
        session = DoSession()
        api = make_api(session)
        linearity = json.loads(api['limit_linearity'](
            '\\lim_{x \\to 0} (\\frac{\\sin x}{x}+x^2)'))
        first = json.loads(api['limit_table'](linearity['limits'][0]))
        second = json.loads(api['limit_substitute'](linearity['limits'][1]))
        json.loads(api['limit_assemble'](
            's1', [first['step']['id'], second['step']['id']]))
        session.ledger.steps[-1]['sources']['values'][0] = 's3'
        replay = session.ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('provenance mismatch', replay['reason'])

    WALLIS_EXPLICIT = '\\lim_{n \\to \\infty} \\prod_{k=1}^{n} ' \
                      '\\frac{2k-1}{2k}'
    WALLIS_UPPER = '\\frac{1}{\\sqrt{2n+1}}'

    def _squeeze_setup(self, api):
        upper = json.loads(api['limit_table'](
            '\\lim_{n \\to \\infty} ' + self.WALLIS_UPPER))
        lower = json.loads(api['limit_table']('\\lim_{n \\to \\infty} 0'))
        return lower['step']['id'], upper['step']['id']

    def test_limit_squeeze_uses_recorded_bound_limits(self):
        session = DoSession()
        api = make_api(session)
        lower_id, upper_id = self._squeeze_setup(api)
        rec = json.loads(api['limit_squeeze'](
            self.WALLIS_EXPLICIT, '0', self.WALLIS_UPPER,
            lower_id, upper_id))
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], '0')
        self.assertEqual(rec['sources'],
                         {'lower': lower_id, 'upper': upper_id})
        self.assertEqual(session.ledger.replay()['status'], 'verified')

    def test_limit_squeeze_rejects_unknown_bound_step(self):
        session = DoSession()
        api = make_api(session)
        rec = json.loads(api['limit_squeeze'](
            self.WALLIS_EXPLICIT, '0', self.WALLIS_UPPER, 's7', 's8'))
        self.assertFalse(rec['ok'])
        self.assertIn('unknown transforming step', rec['error'])

    def test_limit_squeeze_rejects_mismatched_bound_step(self):
        session = DoSession()
        api = make_api(session)
        other = json.loads(api['limit_table'](
            '\\lim_{n \\to \\infty} \\frac{1}{n}'))
        lower = json.loads(api['limit_table']('\\lim_{n \\to \\infty} 0'))
        rec = json.loads(api['limit_squeeze'](
            self.WALLIS_EXPLICIT, '0', self.WALLIS_UPPER,
            lower['step']['id'], other['step']['id']))
        self.assertFalse(rec['ok'])
        self.assertIn('does not record', rec['error'])

    def test_replay_rejects_tampered_squeeze_provenance(self):
        session = DoSession()
        api = make_api(session)
        lower_id, upper_id = self._squeeze_setup(api)
        json.loads(api['limit_squeeze'](
            self.WALLIS_EXPLICIT, '0', self.WALLIS_UPPER,
            lower_id, upper_id))
        session.ledger.steps[-1]['sources']['upper'] = lower_id
        replay = session.ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('provenance', replay['reason'])

    OSC_ROOT = '\\lim_{x \\to 0} x \\sqrt{\\cos\\frac{1}{x}}'

    def _one_sided_setup(self, api):
        right = json.loads(api['limit_substitute']('\\lim_{x \\to 0^+} x'))
        left = json.loads(api['limit_substitute']('\\lim_{x \\to 0^-} x'))
        return left['step']['id'], right['step']['id']

    def test_limit_from_sides_uses_recorded_one_sided_limits(self):
        session = DoSession()
        api = make_api(session)
        left_id, right_id = self._one_sided_setup(api)
        rec = json.loads(api['limit_from_sides'](
            '\\lim_{x \\to 0} x', left_id, right_id))
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], '0')
        self.assertEqual(rec['sources'],
                         {'left': left_id, 'right': right_id})
        self.assertEqual(session.ledger.replay()['status'], 'verified')

    def test_limit_from_sides_closes_oscillating_root(self):
        # the live-failure workflow: direction-dependent squeezes close
        # each side, then the sides combine into the two-sided limit
        session = DoSession()
        api = make_api(session)
        r_low = json.loads(api['limit_substitute'](
            '\\lim_{x \\to 0^+} (-x)'))
        r_up = json.loads(api['limit_substitute']('\\lim_{x \\to 0^+} x'))
        right = json.loads(api['limit_squeeze'](
            '\\lim_{x \\to 0^+} x \\sqrt{\\cos\\frac{1}{x}}',
            '(-x)', 'x', r_low['step']['id'], r_up['step']['id']))
        self.assertTrue(right['ok'], right.get('error'))
        l_low = json.loads(api['limit_substitute']('\\lim_{x \\to 0^-} x'))
        l_up = json.loads(api['limit_substitute'](
            '\\lim_{x \\to 0^-} (-x)'))
        left = json.loads(api['limit_squeeze'](
            '\\lim_{x \\to 0^-} x \\sqrt{\\cos\\frac{1}{x}}',
            'x', '(-x)', l_low['step']['id'], l_up['step']['id']))
        self.assertTrue(left['ok'], left.get('error'))
        rec = json.loads(api['limit_from_sides'](
            self.OSC_ROOT, left['step']['id'], right['step']['id']))
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], '0')
        self.assertEqual(session.ledger.replay()['status'], 'verified')

    def test_limit_from_sides_rejects_mismatched_side_step(self):
        session = DoSession()
        api = make_api(session)
        left_id, _ = self._one_sided_setup(api)
        other = json.loads(api['limit_substitute'](
            '\\lim_{x \\to 0^+} (-x)'))
        rec = json.loads(api['limit_from_sides'](
            '\\lim_{x \\to 0} x', left_id, other['step']['id']))
        self.assertFalse(rec['ok'])
        self.assertIn('does not record', rec['error'])

    def test_limit_from_sides_rejects_unequal_side_values(self):
        session = DoSession()
        api = make_api(session)
        left_id, right_id = self._one_sided_setup(api)
        for step in session.ledger.steps:
            if step['id'] == left_id:
                step['result'] = '1'
        rec = json.loads(api['limit_from_sides'](
            '\\lim_{x \\to 0} x', left_id, right_id))
        self.assertFalse(rec['ok'])
        self.assertIn('not the same value', rec['error'])

    def test_replay_rejects_tampered_from_sides_provenance(self):
        session = DoSession()
        api = make_api(session)
        left_id, right_id = self._one_sided_setup(api)
        json.loads(api['limit_from_sides'](
            '\\lim_{x \\to 0} x', left_id, right_id))
        session.ledger.steps[-1]['sources']['left'] = right_id
        replay = session.ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('provenance', replay['reason'])

    def test_replay_rejects_sourceless_from_sides_step(self):
        session = DoSession()
        api = make_api(session)
        left_id, right_id = self._one_sided_setup(api)
        json.loads(api['limit_from_sides'](
            '\\lim_{x \\to 0} x', left_id, right_id))
        del session.ledger.steps[-1]['sources']
        replay = session.ledger.replay()
        self.assertEqual(replay['status'], 'failed')

    def test_integrate_assemble_rejects_wrong_source_order(self):
        session = DoSession()
        api = make_api(session)
        linearity = json.loads(api['integrate_linearity'](
            '\\int (x+x^2) \\, d x', 'x'))
        first = json.loads(api['integrate_power_rule'](
            linearity['integrals'][0], 'x'))
        second = json.loads(api['integrate_power_rule'](
            linearity['integrals'][1], 'x'))
        rec = json.loads(api['integrate_assemble'](
            's1', [second['step']['id'], first['step']['id']]))
        self.assertFalse(rec['ok'])
        self.assertIn('piece 1', rec['error'])
        self.assertEqual(len(session.ledger.steps), 3)

    def test_integrate_assemble_requires_linearity_step(self):
        session = DoSession()
        api = make_api(session)
        step = json.loads(api['expand']('(x+1)^2'))['step']['id']
        rec = json.loads(api['integrate_assemble'](step, [step]))
        self.assertFalse(rec['ok'])
        self.assertIn('not integrate_linearity', rec['error'])

    def _stationary_session(self):
        session = DoSession()
        api = make_api(session)
        json.loads(api['diff']('x^3-3x', 'x'))
        roots = json.loads(api['quadratic_roots']('3x^{2}-3', 'x'))
        values = []
        for root, substituted in (('-1', '(-1)^{3}-3(-1)'),
                                  ('1', '(1)^{3}-3(1)')):
            json.loads(api['substitute']('x^3-3x', 'x', root))
            values.append(json.loads(
                api['evaluate'](substituted))['step']['id'])
        return session, api, roots['step']['id'], values

    def test_points_assemble_completes_the_stationary_point_answer(self):
        session, api, roots_id, values = self._stationary_session()
        rec = json.loads(api['points_assemble'](
            'x^3-3x', 'x', roots_id, values))
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], r'\{(-1,2),(1,-2)\}')
        self.assertEqual(rec['sources'],
                         {'roots': 's2', 'values': ['s4', 's6']})
        self.assertEqual(rec['check']['status'], 'agree')
        selected = json.loads(api['set_result'](rec['result']))
        self.assertTrue(selected['ok'], selected.get('error'))
        self.assertEqual(selected['provenance']['step'], rec['step']['id'])
        self.assertEqual(session.ledger.replay()['status'], 'verified')
        # every checked step stays on the presented spine, and both
        # renderings name the function the values were paired from
        topology = session.ledger.presentation_topology()
        self.assertEqual(topology['spine'],
                         [s['id'] for s in session.ledger.steps])
        self.assertIn('values of x^3-3x from s4, s6',
                      session.ledger.render())
        self.assertIn('values of $x^3-3x$ from `s2` → `s4, s6`',
                      session.ledger.render_markdown())

    ANSATZ = r'\frac{1}{x^2-1} = \frac{A}{x-1}+\frac{B}{x+1}'

    def _coefficient_session(self):
        session = DoSession()
        api = make_api(session)
        api['load_skill']('equations')
        values = []
        for equation, divisor in (('2A = 1', '2'), ('-2B = 1', '-2')):
            applied = json.loads(api['apply'](equation, '/', divisor))
            values.append(json.loads(
                api['expand'](applied['result']))['step']['id'])
        return session, api, values

    def test_system_assemble_states_the_several_part_answer(self):
        session, api, values = self._coefficient_session()
        rec = json.loads(api['run_tactic'](
            'system_assemble', [self.ANSATZ] + values))
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertTrue(agent_do.primitives.same_expression(
            rec['result'], r'A=\frac{1}{2},B=-\frac{1}{2}'))
        self.assertEqual(rec['sources'], {'assignments': values})
        self.assertEqual(rec['check']['status'], 'agree')
        selected = json.loads(api['set_result'](rec['result']))
        self.assertTrue(selected['ok'], selected.get('error'))
        self.assertEqual(selected['provenance']['step'], rec['step']['id'])
        self.assertEqual(session.ledger.replay()['status'], 'verified')
        self.assertIn('values for A, B from s2, s4', session.ledger.render())
        self.assertIn('values for $A, B$ from `s2, s4`',
                      session.ledger.render_markdown())

    def test_system_assemble_refuses_a_value_that_fails_the_target(self):
        # every cited step is itself checked; what this tactic adds is that
        # the values TOGETHER satisfy the stated target
        session, api, values = self._coefficient_session()
        applied = json.loads(api['apply']('2B = 1', '/', '2'))
        wrong = json.loads(api['expand'](applied['result']))['step']['id']
        rec = json.loads(api['run_tactic'](
            'system_assemble', [self.ANSATZ, values[0], wrong]))
        self.assertFalse(rec['ok'])
        self.assertIn('do not satisfy', rec['error'])
        self.assertEqual(len(session.ledger.steps), 6)

    def test_assignment_order_never_stands_in_for_the_association(self):
        # unlike a point list, each assignment names its own unknown, so
        # citing the steps in the other order is a reordering, not a swap
        session, api, values = self._coefficient_session()
        rec = json.loads(api['run_tactic'](
            'system_assemble', [self.ANSATZ] + list(reversed(values))))
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertTrue(agent_do.primitives.same_expression(
            rec['result'], r'B=-\frac{1}{2},A=\frac{1}{2}'))
        self.assertEqual(rec['check']['status'], 'agree')

    def test_system_assemble_needs_recorded_value_steps(self):
        session, api, values = self._coefficient_session()
        rec = json.loads(api['run_tactic'](
            'system_assemble', [self.ANSATZ, values[0], 's99']))
        self.assertFalse(rec['ok'])
        self.assertIn('unknown transforming step', rec['error'])

    def test_replay_rejects_tampered_assignment_provenance(self):
        session, api, values = self._coefficient_session()
        json.loads(api['run_tactic'](
            'system_assemble', [self.ANSATZ] + values))
        session.ledger.steps[-1]['sources']['assignments'][1] = values[0]
        replay = session.ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('provenance mismatch', replay['reason'])

    def test_replay_rejects_a_retyped_assignment_association(self):
        session, api, values = self._coefficient_session()
        json.loads(api['run_tactic'](
            'system_assemble', [self.ANSATZ] + values))
        session.ledger.steps[-1]['unknowns'][0]['value'] = '0'
        replay = session.ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('association mismatch', replay['reason'])

    def test_an_unknowns_claim_closes_conditional_on_its_premise(self):
        session = DoSession()
        api = make_api(session)
        claim = json.loads(api['claim'](r'A = \frac{1}{2}'))
        applied = json.loads(api['apply']('2A = 1', '/', '2'))
        expanded = json.loads(api['expand'](applied['result']))
        closed = json.loads(api['conclude'](
            claim['id'], [applied['step']['id'], expanded['step']['id']]))
        self.assertTrue(closed['ok'], closed.get('error'))
        self.assertEqual(closed['claim']['verdict'], 'conditional')
        self.assertEqual(closed['claim']['conclusion']['premise'], '2A = 1')
        self.assertEqual(closed['claim']['conclusion']['closure'],
                         'derived-from-premise')
        selected = json.loads(api['set_result'](expanded['result']))
        self.assertTrue(selected['ok'], selected.get('error'))
        self.assertEqual(session.ledger.replay()['status'], 'verified')
        self.assertIn('given 2A = 1', session.ledger.render())

    INEQUALITY = r'\frac{1}{x} \lt 2'
    UNION = r'x \gt \frac{1}{2} \lor x \lt 0'

    def _case_session(self):
        session = DoSession()
        api = make_api(session)
        api['load_skill']('equations')
        endpoints = []
        for hypothesis in (r'x \gt 0', r'x \lt 0'):
            applied = json.loads(api['apply'](
                self.INEQUALITY, '*', 'x', hypothesis))
            cleared = json.loads(api['expand'](applied['result']))
            halved = json.loads(api['apply'](cleared['result'], '/', '2'))
            endpoint = json.loads(api['expand'](halved['result']))
            endpoints.append(endpoint['step']['id'])
        return session, api, endpoints

    def test_cases_assemble_states_the_union_of_cases(self):
        session, api, endpoints = self._case_session()
        rec = json.loads(api['run_tactic'](
            'cases_assemble', [self.INEQUALITY, self.UNION] + endpoints))
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['sources'], {'cases': endpoints})
        self.assertEqual(rec['check']['status'], 'agree')
        selected = json.loads(api['set_result'](rec['result']))
        self.assertTrue(selected['ok'], selected.get('error'))
        self.assertEqual(selected['provenance']['step'], rec['step']['id'])
        self.assertEqual(session.ledger.replay()['status'], 'verified')

    def test_cases_assemble_closes_the_between_workflow(self):
        # Realistic sign-case path: the endpoint x<1 remains conditioned on
        # x>-1 until the assembly combines both into one bounded answer.
        target = r'x^2 \lt 1'
        session = DoSession()
        api = make_api(session)
        api['load_skill']('equations')
        moved = json.loads(api['apply'](target, '-', '1'))
        expanded = json.loads(api['expand'](moved['result']))
        factored = json.loads(api['run_tactic'](
            'factor_quadratic', [expanded['result'], 'x']))
        divided = json.loads(api['apply'](
            factored['result'], '/', 'x+1', r'x \gt -1'))
        cancelled = json.loads(api['expand'](divided['result']))
        shifted = json.loads(api['apply'](cancelled['result'], '+', '1'))
        endpoint = json.loads(api['expand'](shifted['result']))

        rec = json.loads(api['run_tactic'](
            'cases_assemble',
            [target, r'-1 \lt x \lt 1', endpoint['step']['id']]))
        self.assertTrue(rec['ok'], rec.get('error'))
        self.assertEqual(rec['result'], r'-1 \lt x \land x \lt 1')
        self.assertEqual(rec['sources'],
                         {'cases': [endpoint['step']['id']]})
        selected = json.loads(api['set_result'](rec['result']))
        self.assertTrue(selected['ok'], selected.get('error'))
        self.assertEqual(
            session.ledger.presentation_topology()['spine_assumptions'], [])
        self.assertEqual(session.ledger.replay()['status'], 'verified')

    def test_cases_assemble_discharges_the_case_hypotheses(self):
        # the union was checked unconditionally across all cases, so the
        # mutually exclusive hypotheses stop conditioning the endpoint;
        # a run that stops INSIDE one case still shows that case's own
        session, api, endpoints = self._case_session()
        rec = json.loads(api['run_tactic'](
            'cases_assemble', [self.INEQUALITY, self.UNION] + endpoints))
        json.loads(api['set_result'](rec['result']))
        topology = session.ledger.presentation_topology()
        self.assertEqual(topology['spine_assumptions'], [])
        inside = session.ledger.presentation_topology(
            final_provenance={'status': 'verified', 'source': 'ledger',
                              'step': endpoints[0],
                              'method': 'exact-result'})
        self.assertEqual([a.get('constraint')
                          for a in inside['spine_assumptions']],
                         [r'x \gt 0'])

    def test_cases_assemble_needs_a_recorded_hypothesis(self):
        session, api, endpoints = self._case_session()
        moved = json.loads(api['apply'](self.INEQUALITY, '-', '2'))
        bare = json.loads(api['expand'](moved['result']))
        rec = json.loads(api['run_tactic'](
            'cases_assemble', [self.INEQUALITY, self.UNION,
                               endpoints[0], bare['step']['id']]))
        self.assertFalse(rec['ok'])
        self.assertIn('no case hypothesis', rec['error'])

    def test_cases_assemble_requires_the_equations_skill(self):
        session = DoSession()
        api = make_api(session)
        rec = json.loads(api['run_tactic'](
            'cases_assemble', [self.INEQUALITY, self.UNION, 's1', 's2']))
        self.assertFalse(rec['ok'])
        self.assertIn("unloaded skill 'equations'", rec['error'])

    def test_replay_rejects_tampered_case_provenance(self):
        session, api, endpoints = self._case_session()
        json.loads(api['run_tactic'](
            'cases_assemble', [self.INEQUALITY, self.UNION] + endpoints))
        session.ledger.steps[-1]['sources']['cases'][1] = endpoints[0]
        replay = session.ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('provenance mismatch', replay['reason'])

    def test_replay_rejects_a_forged_case_hypothesis(self):
        session, api, endpoints = self._case_session()
        json.loads(api['run_tactic'](
            'cases_assemble', [self.INEQUALITY, self.UNION] + endpoints))
        session.ledger.steps[-1]['args']['hypotheses'][0] = r'x \gt 5'
        replay = session.ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('hypothesis provenance mismatch', replay['reason'])

    def test_points_assemble_refuses_swapped_value_steps(self):
        session, api, roots_id, values = self._stationary_session()
        rec = json.loads(api['points_assemble'](
            'x^3-3x', 'x', roots_id, list(reversed(values))))
        self.assertFalse(rec['ok'])
        self.assertIn('is not the value of', rec['error'])
        self.assertEqual(len(session.ledger.steps), 6)

    def test_points_assemble_needs_recorded_value_steps(self):
        session, api, roots_id, values = self._stationary_session()
        rec = json.loads(api['points_assemble'](
            'x^3-3x', 'x', roots_id, ['s4', 's99']))
        self.assertFalse(rec['ok'])
        self.assertIn('unknown transforming step', rec['error'])

    def test_replay_rejects_tampered_point_provenance(self):
        session, api, roots_id, values = self._stationary_session()
        json.loads(api['points_assemble']('x^3-3x', 'x', roots_id, values))
        session.ledger.steps[-1]['sources']['values'][1] = 's4'
        replay = session.ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('provenance mismatch', replay['reason'])

    def test_typed_result_is_selectable_after_harmless_respelling(self):
        session, api, roots_id, values = self._stationary_session()
        json.loads(api['points_assemble']('x^3-3x', 'x', roots_id, values))
        selected = json.loads(api['set_result'](r'\{ (-1, 2), (1, -2) \}'))
        self.assertTrue(selected['ok'], selected.get('error'))
        self.assertEqual(selected['provenance']['method'], 'same-expression')
        reordered = json.loads(api['set_result'](r'\{(1,-2),(-1,2)\}'))
        self.assertFalse(reordered['ok'])

    def test_replay_rejects_tampered_assembly_provenance(self):
        session = DoSession()
        api = make_api(session)
        linearity = json.loads(api['integrate_linearity'](
            '\\int (x+x^2) \\, d x', 'x'))
        first = json.loads(api['integrate_power_rule'](
            linearity['integrals'][0], 'x'))
        second = json.loads(api['integrate_power_rule'](
            linearity['integrals'][1], 'x'))
        json.loads(api['integrate_assemble']('s1', [
            first['step']['id'], second['step']['id']]))
        session.ledger.steps[-1]['sources']['antiderivatives'][0] = 's3'
        replay = session.ledger.replay()
        self.assertEqual(replay['status'], 'failed')
        self.assertIn('provenance mismatch', replay['reason'])

    def test_concurrent_recording_is_thread_safe(self):
        # the Agents SDK runs sync tools on a thread pool; parallel tool
        # calls must not produce duplicate step ids
        import threading
        session = DoSession()
        api = make_api(session)

        def work():
            for _ in range(10):
                api['expand']('x + x')

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ids = [s['id'] for s in session.new_steps()]
        self.assertEqual(len(ids), 40)
        self.assertEqual(len(set(ids)), 40)

    def test_existing_ledger_range(self):
        ledger = Ledger()
        ledger.record({'ok': True, 'op': 'expand', 'args': {'expr': 'x+x'},
                       'input': 'x+x', 'result': '2x'})
        session = DoSession(ledger=ledger)
        api = make_api(session)
        api['expand']('3x + 3x')
        self.assertEqual(len(ledger.steps), 2)
        self.assertEqual(len(session.new_steps()), 1)


LIM_SUM_EXPR = ('\\lim _{n \\rightarrow \\infty}\\left['
                '\\frac{1}{1 \\cdot 2}+\\frac{1}{2 \\cdot 3}'
                '+\\ldots+\\frac{1}{n(n+1)}\\right]')

LIM_SUM_SCRIPT = [
    [tool_call('load_skill', {'skill': 'finite_operators'}, 'sk1')],
    [tool_call('sum_from_ellipsis',
               {'expr': LIM_SUM_EXPR,
                'sum_form': '\\sum_{k=1}^{n} \\frac{1}{k(k+1)}'}, 'c1')],
    [tool_call('sum_telescope',
               {'expr': ('\\lim_{n \\to \\infty} \\sum_{k=1}^{n} '
                         '\\frac{1}{k(k+1)}'),
                'term': '\\frac{1}{k}'}, 'c2')],
    [tool_call('load_skill', {'skill': 'limits'}, 'sk2')],
    [tool_call('limit_table',
               {'expr': '\\lim_{n \\to \\infty} \\frac{n}{n+1}'}, 'c3')],
    [message('Interpreted the ellipsis, telescoped, closed at 1.')],
]


class TestScriptedAgent(unittest.TestCase):
    def test_derivative_roots_and_plot_finishes_with_solution_result(self):
        shown = []
        script = [
            [tool_call('load_skill', {'skill': 'differentiation'}, 'sk1')],
            [tool_call('diff', {'expr': 'x^3-3x', 'var': 'x'}, 'd1')],
            [tool_call('load_skill', {'skill': 'roots'}, 'sk2')],
            [tool_call('quadratic_roots', {
                'expr': '3x^{2}-3', 'var': 'x'}, 'r1')],
            [tool_call('substitute', {
                'expr': 'x^3-3x', 'var': 'x', 'value': '-1'}, 'p1')],
            [tool_call('evaluate', {
                'expr': '(-1)^3-3(-1)'}, 'p2')],
            [tool_call('substitute', {
                'expr': 'x^3-3x', 'var': 'x', 'value': '1'}, 'p3')],
            [tool_call('evaluate', {'expr': '(1)^3-3(1)'}, 'p4')],
            [tool_call('plot', {
                'code': 'import matplotlib.pyplot as plt; plt.plot([0])',
                'caption': 'stationary points'}, 'plot1')],
            [tool_call('set_result', {
                'expr': r'x=-1 \lor x=1'}, 'done')],
            [message('Mechanically checked the derivative and roots.')],
        ]
        ledger = Ledger()
        res = run_instruction(
            'differentiate x^3-3x, find where the derivative is zero, '
            'plot with those points marked',
            model=ScriptedModel(script), ledger=ledger,
            plot_backend=FakePlotBackend(),
            on_plot=lambda caption, images: shown.append(caption))
        self.assertTrue(res['ok'], res.get('error'))
        self.assertEqual(res['final_result'], r'x=-1 \lor x=1')
        self.assertEqual(res['final_provenance']['step'], 's2')
        self.assertEqual([step['op'] for step in res['steps']], [
            'differentiate', 'quadratic_roots', 'substitute', 'evaluate',
            'substitute', 'evaluate'])
        self.assertTrue(res['steps'][1]['continues'])
        self.assertEqual(shown, ['stationary points'])
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_stationary_points_finish_as_one_assembled_collection(self):
        script = [
            [tool_call('load_skill', {'skill': 'differentiation'}, 'sk1')],
            [tool_call('diff', {'expr': 'x^3-3x', 'var': 'x'}, 'd1')],
            [tool_call('load_skill', {'skill': 'equations'}, 'sk2')],
            [tool_call('quadratic_roots', {
                'expr': '3x^{2}-3', 'var': 'x'}, 'r1')],
            [tool_call('substitute', {
                'expr': 'x^3-3x', 'var': 'x', 'value': '-1'}, 'p1')],
            [tool_call('evaluate', {'expr': '(-1)^{3}-3(-1)'}, 'p2')],
            [tool_call('substitute', {
                'expr': 'x^3-3x', 'var': 'x', 'value': '1'}, 'p3')],
            [tool_call('evaluate', {'expr': '(1)^{3}-3(1)'}, 'p4')],
            [tool_call('points_assemble', {
                'expr': 'x^3-3x', 'var': 'x', 'roots_step': 's2',
                'value_steps': ['s4', 's6']}, 'a1')],
            [tool_call('set_result', {
                'expr': r'\{(-1,2),(1,-2)\}'}, 'done')],
            [message('Assembled the checked stationary points.')],
        ]
        ledger = Ledger()
        res = run_instruction(
            'find the stationary points of x^3-3x',
            model=ScriptedModel(script), ledger=ledger)
        self.assertTrue(res['ok'], res.get('error'))
        self.assertEqual(res['final_result'], r'\{(-1,2),(1,-2)\}')
        self.assertEqual(res['final_provenance']['status'], 'verified')
        self.assertEqual(res['final_provenance']['step'], 's7')
        self.assertEqual(res['steps'][-1]['op'], 'points_assemble')
        self.assertEqual(res['steps'][-1]['check']['status'], 'agree')
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_points_assemble_needs_its_subject_skill(self):
        session = DoSession()
        api = make_api(session)
        refusal = json.loads(api['run_tactic'](
            'points_assemble', ['x^3-3x', 'x', 's1', 's2']))
        self.assertFalse(refusal['ok'])
        self.assertIn("load_skill('equations')", refusal['error'])

    def test_ellipsis_series_limit_closes_via_sum_tactics(self):
        res = run_instruction('evaluate ' + LIM_SUM_EXPR,
                              model=ScriptedModel(LIM_SUM_SCRIPT))
        self.assertTrue(res['ok'])
        self.assertEqual([s['op'] for s in res['steps']],
                         ['sum_from_ellipsis', 'sum_telescope',
                          'limit_table'])
        self.assertEqual(res['final_result'], '1')
        self.assertEqual(res['final_provenance']['status'], 'verified')
        # the pattern continuation is honestly conditional
        self.assertTrue(any('\\ldots' in a['text']
                            for a in res['assumptions']))

    def test_full_run(self):
        seen = []
        res = run_instruction('solve 2x + 3 = 7 for x',
                              model=ScriptedModel(SOLVE_SCRIPT),
                              on_step=lambda s: seen.append(s['id']))
        self.assertTrue(res['ok'])
        self.assertEqual([s['op'] for s in res['steps']],
                         ['apply_both_sides', 'expand'])
        self.assertEqual(res['final_result'], '2x = 4')
        self.assertEqual(res['final_provenance']['step'], 's2')
        self.assertEqual(res['summary'], 'Subtracted 3 from both sides.')
        self.assertEqual(seen, ['s1', 's2'])

    def test_set_result_overrides_final(self):
        script = [
            [tool_call('factor_quadratic', {'expr': 'x^2 - 5x + 6',
                                            'var': 'x'}, 'c1')],
            [tool_call('substitute', {'expr': 'x^2 - 5x + 6 = 0',
                                      'var': 'x', 'value': '3'}, 'c2')],
            [tool_call('set_result', {'expr': '(x - 3)(x - 2)'}, 'c3'),
             ],
            [message('factored; root 3 checked')],
        ]
        res = run_instruction('factor and verify',
                              model=ScriptedModel(script))
        self.assertTrue(res['ok'])
        # the designated value wins over the last step's result
        self.assertEqual(res['final_result'], '(x - 3)(x - 2)')
        self.assertEqual(res['final_provenance']['step'], 's1')
        self.assertEqual(len(res['steps']), 2)

    def test_a_run_reports_the_premises_it_stated(self):
        # the live int1 failure: typed coefficients laundered into green
        # steps by multiplying by 1. Every step is honestly checked; what
        # was missing is that the answer rests on nothing but assertions.
        script = [
            [tool_call('apply', {'equation': 'A = \\frac{1}{2}',
                                 'op': '*', 'arg': '1'}, 'c1')],
            [tool_call('expand', {
                'expr': 'A \\cdot \\left(1\\right) = '
                        '\\frac {1} {2} \\cdot \\left(1\\right)'}, 'c2')],
            [message('A determined')],
        ]
        res = run_instruction('find A', model=ScriptedModel(script))
        self.assertTrue(res['ok'], res.get('error'))
        self.assertEqual([p['input'] for p in res['premises']],
                         ['A = \\frac{1}{2}'])

    ISOLATION_TARGET = '2A x^2 + 2B x + 2C = 4x^2 + 10'

    def _isolation_run(self, row):
        """The shipped route's shape: match the system, isolate one unknown.

        `expand` refuses a C_LIST and `apply` divides EVERY row by the same
        thing, so isolating one unknown forces the agent to RETYPE a row the
        previous step just produced and the oracle just checked.
        """
        script = [
            [tool_call('load_skill', {'skill': 'equations'}, 'sk1')],
            [tool_call('match_coefficients',
                       {'expr': self.ISOLATION_TARGET, 'var': 'x'}, 'c1')],
            [tool_call('apply', {'equation': row, 'op': '/', 'arg': '2'},
                       'c2')],
            [message('isolated')],
        ]
        res = run_instruction('solve for A', model=ScriptedModel(script))
        self.assertTrue(res['ok'], res.get('error'))
        return res

    def test_a_retyped_system_row_is_labelled_not_counted(self):
        # the run payload must carry the distinction, or the reader counts
        # the engine's own bookkeeping as an assertion
        res = self._isolation_run('2A = 4')
        self.assertEqual(
            [(p['input'], p.get('derived_from')) for p in res['premises']],
            [('2A = 4', 's1')])

    def test_a_fabricated_row_is_still_reported_as_stated(self):
        # THE DISCRIMINATOR, same shape as above: a fabricated row records
        # IDENTICALLY to a correct one - every step 'agree' - but is not a
        # member of any checked result, so it must keep counting against the
        # boundary. Over-discharging here would hide the one thing the
        # premise boundary exists to expose.
        res = self._isolation_run('2A = 5')
        self.assertEqual(
            [(p['input'], p.get('derived_from')) for p in res['premises']],
            [('2A = 5', None)])

    def test_premises_are_scoped_to_the_run_not_the_shared_ledger(self):
        ledger = Ledger()
        run_instruction('expand it', ledger=ledger, model=ScriptedModel(
            [[tool_call('expand', {'expr': '(x+1)^2'}, 'c1')],
             [message('done')]]))
        second = run_instruction('expand another', ledger=ledger,
                                 model=ScriptedModel(
            [[tool_call('expand', {'expr': '(y+1)^2'}, 'c2')],
             [message('done')]]))
        # the earlier cell's given belongs to that cell
        self.assertEqual([p['input'] for p in second['premises']],
                         ['(y+1)^2'])

    def test_set_result_admission_mirrors_chain_goal(self):
        # admission mirrors the composite closure gate (live: the agent
        # retyped hand-simplified algebra as its final expand input, the
        # chain severed silently, and int! refused only after the run had
        # ended). With a chain_goal, set_result must refuse a verified but
        # DISCONNECTED value while the agent can still repair.
        script = [
            [tool_call('expand', {'expr': '(y+1)^2'}, 'c1')],
            [tool_call('set_result', {'expr': 'y^{2}+2y+1'}, 'c2')],
            [message('done')],
        ]
        res = run_instruction('expand it', model=ScriptedModel(list(script)),
                              chain_goal='(x+1)^2')
        # The designation was refused, and the same disconnected value may
        # not re-enter as the undesignated last-step fallback.
        self.assertIsNone(res['final_result'])
        self.assertEqual(res['final_provenance']['method'],
                         'last-step-disconnected')
        # control 1: the same selection with a matching goal is admitted
        res = run_instruction('expand it', model=ScriptedModel(list(script)),
                              chain_goal='(y+1)^2')
        self.assertEqual(res['final_result'], 'y^{2}+2y+1')
        self.assertNotEqual(res['final_provenance']['method'], 'last-step')
        # control 2: plain do! runs carry no chain goal and stay permissive
        res = run_instruction('expand it', model=ScriptedModel(list(script)))
        self.assertNotEqual(res['final_provenance']['method'], 'last-step')

    def test_connected_last_step_fallback_still_closes_a_command_goal(self):
        res = run_instruction(
            'expand it',
            model=ScriptedModel([
                [tool_call('expand', {'expr': '(x+1)^2'}, 'c1')],
                [message('done')],
            ]),
            chain_goal='(x+1)^2')
        self.assertEqual(res['final_result'], 'x^{2}+2x+1')
        self.assertEqual(res['final_provenance']['method'], 'last-step')

    def test_set_open_suppresses_last_step_fallback(self):
        # the conv! pseudo-answer: a run that certified nothing used to
        # hand the cell its last checked step as a "verified" result
        script = [
            [tool_call('expand', {'expr': '(x+1)^2'}, 'c1')],
            [tool_call('set_open', {
                'reason': 'no tactic covers the goal shape'}, 'c2')],
            [message('Left open: the checked split answers nothing yet.')],
        ]
        ledger = Ledger()
        res = run_instruction('decide the undecidable',
                              model=ScriptedModel(script), ledger=ledger)
        self.assertTrue(res['ok'], res.get('error'))
        self.assertIsNone(res['final_result'])
        self.assertEqual(res['final_provenance']['status'], 'open')
        self.assertEqual(res['final_provenance']['source'], 'open')
        self.assertEqual(res['final_provenance']['reason'],
                         'no tactic covers the goal shape')
        self.assertEqual(res['final_provenance']['selection'], 'r1')
        # prose stays subordinate to the recorded outcome
        self.assertTrue(res.get('summary_unverified'))
        self.assertEqual(res['branch_topology']['spine'], [])
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_set_open_closes_out_unresolved_marker_visibly(self):
        script = [
            [tool_call('expand', {'expr': '(x+1)^2'}, 'c1')],
            [tool_call('comment', {'text': 'wrong route',
                                   'from_step': 's1'}, 'c2')],
            [tool_call('set_open', {
                'reason': 'the resume needs a tactic this run lacks'},
                'c3')],
            [message('open')],
        ]
        ledger = Ledger()
        res = run_instruction('explore', model=ScriptedModel(script),
                              ledger=ledger)
        self.assertTrue(res['ok'], res.get('error'))
        self.assertIsNone(res['final_result'])
        self.assertEqual(res['branch_topology']['unresolved_markers'],
                         ['s2'])
        md = ledger.render_markdown()
        self.assertIn('left unresolved; outcome recorded open', md)
        self.assertNotIn('awaiting a continuing step', md)
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_an_abandoned_marker_does_not_leak_into_later_runs(self):
        # live: one cell ended open with its branch marker still pending,
        # and every later cell in the notebook was refused for not
        # resuming a step it never saw. Markers are run-scoped: a new run
        # starts unconstrained, may branch on its own steps, and the stale
        # marker stays honestly unresolved instead of being presented as
        # resumed by unrelated work.
        ledger = Ledger()
        first = run_instruction('explore', ledger=ledger,
                                model=ScriptedModel([
            [tool_call('expand', {'expr': '(x+1)^2'}, 'c1')],
            [tool_call('comment', {'text': 'wrong route',
                                   'from_step': 's1'}, 'c2')],
            [tool_call('set_open', {
                'reason': 'the resume needs a tactic this run lacks'},
                'c3')],
            [message('open')],
        ]))
        self.assertEqual(first['branch_topology']['unresolved_markers'],
                         ['s2'])
        second = run_instruction('factor another', ledger=ledger,
                                 model=ScriptedModel([
            [tool_call('expand', {'expr': '(y+1)^2'}, 'c4')],
            [tool_call('comment', {'text': 'factor instead',
                                   'from_step': 's3'}, 'c5')],
            [tool_call('factor_quadratic', {'expr': 'y^{2}+2y+1',
                                            'var': 'y'}, 'c6')],
            [tool_call('set_result', {'expr': '(y+1)^{2}'}, 'c7')],
            [message('done')],
        ]))
        self.assertTrue(second['ok'], second.get('error'))
        self.assertEqual(second['final_result'], '(y+1)^{2}')
        # the new run recorded its steps and its own (resumed) marker
        self.assertEqual([(s['id'], s['op']) for s in second['steps']],
                         [('s3', 'expand'), ('s4', 'branch'),
                          ('s5', 'factor_quadratic')])
        # the earlier run's marker is not presented as resumed by this run
        stale = next(e for e in ledger.branch_edges()
                     if e['marker'] == 's2')
        self.assertIsNone(stale['to'])
        resumed = next(e for e in ledger.branch_edges()
                       if e['marker'] == 's4')
        self.assertEqual(resumed['to'], 's5')
        # admission still mirrors replay across the run boundary
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_a_run_with_no_recorded_result_states_no_premises(self):
        # live: a run whose every tactic call was refused — and the
        # committed figure-only cell — reported earlier cells' givens as
        # its own stated premises. A run that recorded no result-bearing
        # step stated nothing.
        ledger = Ledger()
        run_instruction('expand it', ledger=ledger, model=ScriptedModel([
            [tool_call('expand', {'expr': '(x+1)^2'}, 'c1')],
            [message('done')],
        ]))
        empty = run_instruction('just narrate', ledger=ledger,
                                model=ScriptedModel([
            [tool_call('comment', {'text': 'nothing to record'}, 'c2')],
            [message('no derivation')],
        ]))
        self.assertEqual(empty['premises'], [])

    def test_branch_run_returns_spine_and_abandoned_path_summary(self):
        script = [
            [tool_call('expand', {'expr': '(x+1)^2'}, 'c1')],
            [tool_call('substitute', {
                'expr': 'x^{2}+2x+1', 'var': 'x', 'value': '1'}, 'c2')],
            [tool_call('comment', {
                'text': 'the numeric detour does not answer the symbolic goal',
                'from_step': 's1'}, 'c3')],
            [tool_call('factor_quadratic', {
                'expr': 'x^{2}+2x+1', 'var': 'x'}, 'c4')],
            [tool_call('set_result', {'expr': '(x+1)^{2}'}, 'c5')],
            [message('Used the factored symbolic route.')],
        ]
        ledger = Ledger()
        res = run_instruction('explore then factor',
                              model=ScriptedModel(script), ledger=ledger)
        self.assertTrue(res['ok'], res.get('error'))
        self.assertEqual(res['final_provenance']['step'], 's4')
        self.assertEqual(res['branch_topology']['spine'], ['s1', 's4'])
        self.assertEqual(res['abandoned_paths'], [{
            'marker': 's3', 'source': 's1', 'continues_at': 's4',
            'reason': 'the numeric detour does not answer the symbolic goal',
            'steps': ['s2'],
        }])
        self.assertEqual(ledger.selections[-1]['id'], 'r1')
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_dead_branch_assumption_is_not_a_final_assumption(self):
        script = [
            [tool_call('expand', {'expr': 'xy=1'}, 'c1')],
            [tool_call('apply', {
                'equation': 'xy=1', 'op': '/', 'arg': 'y'}, 'c2')],
            [tool_call('comment', {
                'text': 'division is unnecessary', 'from_step': 's1'}, 'c3')],
            [tool_call('apply', {
                'equation': 'xy=1', 'op': '-', 'arg': '1'}, 'c4')],
            [tool_call('set_result', {'expr': 'xy-1=0'}, 'c5')],
            [message('Kept the assumption-free route.')],
        ]
        ledger = Ledger()
        res = run_instruction('rearrange without division',
                              model=ScriptedModel(script), ledger=ledger)
        self.assertTrue(res['ok'], res.get('error'))
        self.assertTrue(ledger.assumptions)
        self.assertEqual(res['assumptions'], [])
        self.assertEqual(res['abandoned_paths'][0]['steps'], ['s2'])

    def test_set_result_cannot_override_with_detached_value(self):
        script = [
            [tool_call('expand', {'expr': '(x+1)^2'}, 'c1')],
            [tool_call('set_result', {'expr': 'x=999'}, 'c2')],
            [message('done')],
        ]
        res = run_instruction('try to invent a result',
                              model=ScriptedModel(script))
        self.assertEqual(res['final_result'], 'x^{2}+2x+1')
        self.assertEqual(res['final_provenance']['step'], 's1')

    def test_trailing_comment_does_not_eat_the_final(self):
        script = [
            [tool_call('expand', {'expr': '(x+1)^2'}, 'c1')],
            [tool_call('comment', {'text': 'done expanding'}, 'c2')],
            [message('Expanded the square.')],
        ]
        res = run_instruction('expand and annotate',
                              model=ScriptedModel(script))
        self.assertTrue(res['ok'])
        # the note is streamed as a step but the chainable value is the
        # last transforming result
        self.assertEqual([s['op'] for s in res['steps']],
                         ['expand', 'comment'])
        self.assertEqual(res['final_result'], 'x^{2}+2x+1')
        self.assertEqual(res['final_provenance']['step'], 's1')

    def test_max_turns_returns_partial(self):
        endless = [[tool_call('expand', {'expr': 'x + x'}, f'c{i}')]
                   for i in range(10)]
        res = run_instruction('loop forever', model=ScriptedModel(endless),
                              max_turns=2)
        self.assertFalse(res['ok'])
        self.assertIn('turns', res['error'])
        self.assertTrue(res['turn_limit_reached'])
        self.assertEqual(len(res['steps']), 2)  # partial work is kept

    def test_empty_model_response_is_retried_not_accepted(self):
        # gen 61: a live int1 run died at turn 5 when the provider returned a
        # zero-token empty message (trace 4c93d48b01599905e10933c7d0e55d3d).
        # The SDK reads that as the final answer, so the run ended holding an
        # unfinished integral. Ask again instead.
        empty = ResponseOutputMessage(
            id='m', role='assistant', status='completed', type='message',
            content=[])
        calls = {'n': 0}

        class FlakyModel(ScriptedModel):
            async def get_response(self, *args, **kwargs):
                calls['n'] += 1
                if calls['n'] == 2:      # one empty blip mid-derivation
                    return ModelResponse(output=[empty], usage=Usage(),
                                         response_id=None)
                return await ScriptedModel.get_response(self, *args, **kwargs)

        script = [
            [tool_call('expand', {'expr': '(x+1)^2'}, 'c1')],
            [tool_call('set_result', {'expr': 'x^{2}+2x+1'}, 'c2')],
            [message('done')],
        ]
        model = openrouter_backend._retrying_model(FlakyModel)(script)
        res = run_instruction('expand it', model=model)
        self.assertTrue(res['ok'])
        self.assertEqual(res['final_result'], 'x^{2}+2x+1')
        self.assertEqual(calls['n'], 4)  # 3 scripted turns + 1 retried blip

    def test_persistently_empty_model_still_terminates(self):
        empty = ResponseOutputMessage(
            id='m', role='assistant', status='completed', type='message',
            content=[])

        class MuteModel(ScriptedModel):
            def __init__(self):
                ScriptedModel.__init__(self, [])
                self.calls = 0

            async def get_response(self, *args, **kwargs):
                self.calls += 1
                return ModelResponse(output=[empty], usage=Usage(),
                                     response_id=None)

        model = openrouter_backend._retrying_model(MuteModel)()
        res = run_instruction('anything', model=model)
        self.assertTrue(res['ok'])          # ends, does not hang or raise
        self.assertEqual(model.calls, openrouter_backend.EMPTY_RESPONSE_RETRIES + 1)

    def test_a_real_response_is_never_retried(self):
        calls = {'n': 0}

        class CountingModel(ScriptedModel):
            async def get_response(self, *args, **kwargs):
                calls['n'] += 1
                return await ScriptedModel.get_response(self, *args, **kwargs)

        script = [[tool_call('expand', {'expr': '(x+1)^2'}, 'c1')],
                  [message('done')]]
        model = openrouter_backend._retrying_model(CountingModel)(script)
        res = run_instruction('expand it', model=model)
        self.assertTrue(res['ok'])
        self.assertEqual(calls['n'], 2)

    def test_result_committed_on_the_last_turn_is_not_a_failure(self):
        # gen 61: a live int! run committed a verified result with its 64th
        # of 64 turns and was reported as "Max turns exceeded", discarding a
        # correct answer. Only the closing prose was actually lost.
        script = [
            [tool_call('expand', {'expr': '(x+1)^2'}, 'c1')],
            [tool_call('set_result', {'expr': 'x^{2}+2x+1'}, 'c2')],
            [message('done')],
        ]
        res = run_instruction('expand it', model=ScriptedModel(script),
                              max_turns=2)
        self.assertTrue(res['ok'])
        self.assertIsNone(res.get('error'))
        self.assertTrue(res['turn_limit_reached'])
        self.assertEqual(res['final_result'], 'x^{2}+2x+1')
        self.assertEqual(res['final_provenance']['status'], 'verified')

    def test_unverified_designation_at_the_limit_still_fails(self):
        # the last-step fallback is not a designation; nothing says that
        # value answers the instruction, so exhaustion stays a failure
        script = [[tool_call('expand', {'expr': f'x + {i}x'}, f'c{i}')]
                  for i in range(10)]
        res = run_instruction('loop', model=ScriptedModel(script),
                              max_turns=3)
        self.assertFalse(res['ok'])
        self.assertIn('turns', res['error'])

    def test_prove_mode_closes_root_claim(self):
        script = [
            [tool_call('load_skill', {'skill': 'limits'}, 'sk1')],
            [tool_call('limit_table', {
                'expr': r'\lim_{n \to \infty} \frac{1}{n}'}, 'c1')],
            [tool_call('conclude', {'claim_id': 'c1',
                                    'step_ids': ['s1']}, 'c2')],
            [tool_call('set_result', {'expr': '0'}, 'c3')],
            [message('Mechanically checked.\nThe table closes it.\n'
                     'This extra prose must not dominate.')],
        ]
        res = run_instruction(
            'prove the limit', model=ScriptedModel(script),
            proof_goal=r'\lim_{n \to \infty} \frac{1}{n} = 0')
        self.assertTrue(res['ok'])
        self.assertEqual(res['claims'][0]['verdict'], 'established')
        self.assertEqual(res['steps'][0]['goal'], 'c1')
        self.assertEqual(res['final_result'], '0')
        self.assertEqual(res['final_provenance']['source'], 'claim')
        self.assertTrue(res['summary_unverified'])
        self.assertIn('narrative truncated', res['summary'])

    def test_prove_mode_closes_ellipsis_product_claim(self):
        # the reported cell: prove! \lim (1/2 * 3/4 ... (2n-1)/2n) = 0.
        # The ellipsis claim records, the product door interprets it, the
        # bound limits close by table rules, and the squeeze concludes.
        goal = ('\\lim _{n \\rightarrow \\infty}\\left(\\frac{1}{2} \\cdot '
                '\\frac{3}{4} \\ldots \\frac{2 n-1}{2 n}\\right) = 0')
        expr = ('\\lim _{n \\rightarrow \\infty}\\left(\\frac{1}{2} \\cdot '
                '\\frac{3}{4} \\ldots \\frac{2 n-1}{2 n}\\right)')
        explicit = '\\lim_{n \\to \\infty} \\prod_{k=1}^{n} \\frac{2k-1}{2k}'
        upper = '\\frac{1}{\\sqrt{2n+1}}'
        script = [
            [tool_call('load_skill', {'skill': 'finite_operators'}, 'sk1')],
            [tool_call('prod_from_ellipsis', {
                'expr': expr,
                'prod_form': '\\prod_{k=1}^{n} \\frac{2k-1}{2k}'}, 'c1')],
            [tool_call('load_skill', {'skill': 'limits'}, 'sk2')],
            [tool_call('limit_table', {
                'expr': '\\lim_{n \\to \\infty} 0'}, 'c2')],
            [tool_call('limit_table', {
                'expr': '\\lim_{n \\to \\infty} ' + upper}, 'c3')],
            [tool_call('limit_squeeze', {
                'expr': explicit, 'lower': '0', 'upper': upper,
                'lower_step': 's2', 'upper_step': 's3'}, 'c4')],
            [tool_call('conclude', {'claim_id': 'c1',
                                    'step_ids': ['s1', 's4']}, 'c5')],
            [tool_call('set_result', {'expr': '0'}, 'c6')],
            [message('Interpreted the product, squeezed it to 0.')],
        ]
        ledger = Ledger()
        res = run_instruction('prove the product limit', ledger=ledger,
                              model=ScriptedModel(script), proof_goal=goal)
        self.assertTrue(res['ok'], res.get('error'))
        self.assertEqual([s['op'] for s in res['steps']],
                         ['prod_from_ellipsis', 'limit_table',
                          'limit_table', 'limit_squeeze'])
        # conditional: the ellipsis reading and the ordering are assumptions
        self.assertEqual(res['claims'][0]['verdict'], 'conditional')
        self.assertEqual(res['final_result'], '0')
        self.assertEqual(res['final_provenance']['source'], 'claim')
        # the full session, ellipsis root claim included, must replay
        rep = ledger.replay()
        self.assertEqual(rep['status'], 'verified', rep.get('reason'))

    def test_prove_mode_reclaim_refocuses_root_and_statement_sets_result(
            self):
        # The live n/2^n trap: after a first conditional closure the agent
        # re-stated the root claim (minting c2, which stole the goal focus
        # of every later step, so conclude(c1) could never succeed again)
        # and then passed the claim STATEMENT to set_result (refused; the
        # selection wants the endpoint). Both moves must now work: the
        # re-claim focuses c1, a repeated conclude replaces the chain, and
        # the statement maps to the concluded endpoint.
        goal = r'\lim_{n \to \infty} \frac{n}{2^n} = 0'
        start = r'\lim_{n \to \infty} \frac{n}{2^n}'
        transformed = (r'\lim_{n \to \infty} \frac{1}'
                       r'{({2}^n) \ln\left (2 \right )}')
        script = [
            [tool_call('load_skill', {'skill': 'limits'}, 'sk1')],
            [tool_call('limit_lhopital', {'expr': start}, 'c1')],
            [tool_call('limit_table', {'expr': transformed}, 'c2')],
            [tool_call('conclude', {'claim_id': 'c1',
                                    'step_ids': ['s1', 's2']}, 'c3')],
            # the agent re-states the claim hunting for a better chain
            [tool_call('claim', {'statement': goal}, 'c4')],
            [tool_call('limit_lhopital', {'expr': start}, 'c5')],
            [tool_call('limit_table', {'expr': transformed}, 'c6')],
            [tool_call('conclude', {'claim_id': 'c1',
                                    'step_ids': ['s3', 's4']}, 'c7')],
            [tool_call('set_result', {'expr': goal}, 'c8')],
            [message('Mechanically checked via l\'Hopital.')],
        ]
        ledger = Ledger()
        res = run_instruction('prove the limit', ledger=ledger,
                              model=ScriptedModel(script), proof_goal=goal)
        self.assertTrue(res['ok'], res.get('error'))
        # the re-claim reused c1: one claim, and the late steps kept its goal
        self.assertEqual(len(res['claims']), 1)
        self.assertEqual([s.get('goal') for s in res['steps']
                          if s.get('result') is not None],
                         ['c1', 'c1', 'c1', 'c1'])
        self.assertEqual(res['claims'][0]['verdict'], 'conditional')
        self.assertEqual(res['claims'][0]['conclusion']['steps'],
                         ['s3', 's4'])
        # the statement spelling designated the concluded endpoint
        self.assertEqual(res['final_result'], '0')
        self.assertEqual(res['final_provenance']['source'], 'claim')
        rep = ledger.replay()
        self.assertEqual(rep['status'], 'verified', rep.get('reason'))

    def test_prove_mode_leaves_failed_target_open(self):
        script = [
            [tool_call('load_skill', {'skill': 'limits'}, 'sk1')],
            [tool_call('limit_table', {
                'expr': r'\lim_{n \to \infty} \frac{1}{n}'}, 'c1')],
            [tool_call('set_result', {'expr': '0'}, 'c2')],
            [message('A prose squeeze argument would go here.')],
        ]
        res = run_instruction(
            'try the proof', model=ScriptedModel(script),
            proof_goal=r'\lim_{n \to \infty} \frac{n}{2^n} = 0')
        self.assertTrue(res['ok'])
        self.assertEqual(res['claims'][0]['verdict'], 'open')
        self.assertIsNone(res['final_result'])
        self.assertEqual(res['final_provenance']['status'], 'open')

    def test_prove_mode_set_open_refines_reason_and_respects_conclusion(
            self):
        # with the root still open, set_open records the agent's specific
        # missing move instead of the generic no-closing-chain reason
        script = [
            [tool_call('load_skill', {'skill': 'limits'}, 'sk1')],
            [tool_call('limit_table', {
                'expr': r'\lim_{n \to \infty} \frac{1}{n}'}, 'c1')],
            [tool_call('set_open', {
                'reason': 'no tactic relates the table value to the '
                          'claimed sequence'}, 'c2')],
            [message('Left open.')],
        ]
        res = run_instruction(
            'try the proof', model=ScriptedModel(script),
            proof_goal=r'\lim_{n \to \infty} \frac{n}{2^n} = 0')
        self.assertTrue(res['ok'], res.get('error'))
        self.assertEqual(res['claims'][0]['verdict'], 'open')
        self.assertIsNone(res['final_result'])
        self.assertEqual(res['final_provenance']['source'], 'open')
        self.assertIn('no tactic relates', res['final_provenance']['reason'])
        # a concluded root refuses the open outcome and names the repair
        session = DoSession()
        api = make_api(session)
        session.claim(r'\lim_{n \to \infty} \frac{1}{n} = 0', root=True)
        json.loads(api['limit_table'](r'\lim_{n \to \infty} \frac{1}{n}'))
        json.loads(api['conclude']('c1', ['s1']))
        refused = json.loads(api['set_open']('stalled'))
        self.assertFalse(refused['ok'])
        self.assertIn('concluded', refused['error'])
        self.assertIn('set_result', refused['error'])

    def test_open_claim_caps_plain_do_summary(self):
        # even outside prove!, prose must not dominate an open claim
        script = [
            [tool_call('claim', {
                'statement': r'\lim_{n \to \infty} \frac{n}{2^n} = 0'},
                'c1')],
            [message('An informal argument follows.\n'
                     '\\[\n2^n \\ge n^2 \\text{ for } n \\ge 4\n\\]\n'
                     'Therefore the sequence is squeezed to zero, which '
                     'this prose must not be allowed to assert.')],
        ]
        res = run_instruction('prove informally',
                              model=ScriptedModel(script))
        self.assertTrue(res['ok'])
        self.assertEqual(res['claims'][0]['verdict'], 'open')
        self.assertTrue(res['summary_unverified'])
        self.assertIn('narrative truncated', res['summary'])
        self.assertNotIn('\\[', res['summary'])

    def test_cap_prefers_sentence_boundary_and_strips_math(self):
        from agent_do import _cap_prove_summary
        capped = _cap_prove_summary(
            'The claim remains open. Mechanically checked, the chain '
            'reached\n\\[\n\\lim_{n \\to \\infty} e^{-n}\n\\]')
        self.assertIn('narrative truncated', capped)
        self.assertNotIn('\\[', capped)
        long_prose = ('This sentence is padded out to be long enough to '
                      'trigger the character cap on its own. ' * 5)
        capped = _cap_prove_summary(long_prose)
        self.assertIn('narrative truncated', capped)
        self.assertTrue(capped.split(' … ')[0].endswith('.'))

    def test_notebook_ledger_accumulates(self):
        ledger = Ledger()
        run_instruction('solve', model=ScriptedModel(SOLVE_SCRIPT),
                        ledger=ledger)
        res = run_instruction(
            'again',
            model=ScriptedModel([
                [tool_call('expand', {'expr': '2x = 4'}, 'c9')],
                [message('done')]]),
            ledger=ledger)
        self.assertEqual(len(ledger.steps), 3)
        self.assertEqual(len(res['steps']), 1)  # only this run's slice


class TestCanonicalToolBindings(unittest.TestCase):
    """One tool surface, defined once, converted per provider."""

    def test_binding_schema_matches_the_agents_sdk_derivation(self):
        # the canonical schema is generated by ToyMath, not by a provider
        # SDK - that is what lets two backends advertise the same tool. Pin
        # it against the Agents SDK's own reading of the same handlers so a
        # docstring edit can never silently change only one of them.
        from agents import function_tool
        session = DoSession(plot_backend=FakePlotBackend(),
                            tikz_backend=FakeTikzBackend())
        api = make_api(session)
        bindings = agent_do.make_tool_bindings(session)
        self.assertEqual([b.name for b in bindings], [
            'load_skill', 'run_tactic', 'comment', 'claim', 'conclude',
            'set_result', 'set_open', 'plot', 'tikz'])
        for binding in bindings:
            derived = function_tool(api[binding.name])
            self.assertEqual(binding.description, derived.description,
                             binding.name)
            self.assertEqual(binding.input_schema,
                             derived.params_json_schema, binding.name)

    def test_a_figure_binding_states_what_it_returns(self):
        # LANDMINE (measured on the Codex path): a figure tool whose
        # description only says "render an illustration" leaves the model
        # to guess the reply. One run drove `tikz` from the runtime's
        # scripting tool, iterated the reply as `r.content` (an MCP shape
        # it never had), and then passed it to the script helper
        # `image(...)`; the appended `image_url` was
        # `{"ok": true, "plots": 1}`, and the provider refused the next
        # request outright - `Invalid 'input[0].output[1].image_url'` -
        # killing a run whose figures had already rendered.
        #
        # The scripts are authored BEFORE any result exists, so the field
        # names are the part the model cannot observe; where the figure
        # went, it can never observe. Keep those two, here, where they ship
        # only with the tool - not the prohibition, which is one backend's
        # scripting layer and belongs in that backend's policy.
        bindings = {b.name: b for b in agent_do.make_tool_bindings(
            DoSession(plot_backend=FakePlotBackend(),
                      tikz_backend=FakeTikzBackend()))}
        for name in ('plot', 'tikz'):
            # the docstring wraps, so read it the way the model does
            description = ' '.join(bindings[name].description.split())
            self.assertIn('not back to you', description, name)
            self.assertIn('{"ok", "plots", "figure"}', description, name)
            # backend-neutral: no Codex scripting helper in the canonical
            # surface, which OpenRouter advertises verbatim
            self.assertNotIn('image(', description, name)
            self.assertNotIn('generatedImage', description, name)
        # the Args block still parses: the contract is in the summary, and
        # `_parse_docstring` folds anything after `Args:` into an argument
        self.assertNotIn('not back to you',
                         bindings['tikz'].input_schema
                         ['properties']['caption']['description'])

    def test_figure_bindings_appear_only_with_their_backends(self):
        names = [b.name for b in agent_do.make_tool_bindings(DoSession())]
        self.assertNotIn('plot', names)
        self.assertNotIn('tikz', names)
        only_tikz = agent_do.make_tool_bindings(
            DoSession(tikz_backend=FakeTikzBackend()))
        self.assertEqual([b.name for b in only_tikz][-1], 'tikz')

    def test_tactic_adapters_are_never_model_visible(self):
        session = DoSession()
        names = {b.name for b in agent_do.make_tool_bindings(session)}
        self.assertIn('expand', make_api(session))     # internal test API
        self.assertNotIn('expand', names)              # not a model tool
        self.assertFalse(names & set(tactic_registry.BY_NAME))

    def test_arguments_are_validated_against_the_canonical_schema(self):
        schema = {b.name: b.input_schema
                  for b in agent_do.make_tool_bindings(DoSession())}
        validate = agent_base.validate_arguments
        # an omitted optional argument falls back to its schema default,
        # and an explicit null reads as omission (no field is nullable)
        self.assertEqual(validate(schema['comment'], {'text': 'hi'}),
                         {'text': 'hi', 'from_step': ''})
        self.assertEqual(
            validate(schema['comment'], {'text': 'hi', 'from_step': None}),
            {'text': 'hi', 'from_step': ''})
        with self.assertRaises(agent_base.ToolArgumentError):
            validate(schema['comment'], {'from_step': 's1'})
        with self.assertRaises(agent_base.ToolArgumentError):
            validate(schema['run_tactic'],
                     {'tactic': 'expand', 'arguments': 'x+1'})
        with self.assertRaises(agent_base.ToolArgumentError):
            validate(schema['run_tactic'],
                     {'tactic': 'expand', 'arguments': [7]})


class TestBackendToolParity(unittest.TestCase):
    """Two backends, one tool surface. Adapters convert; they never define."""

    def _dispatcher(self, **kwargs):
        session = DoSession(**kwargs)
        return session, agent_base.ToolDispatcher(
            agent_do.make_tool_bindings(session),
            agent_base.CancellationToken())

    def test_both_adapters_advertise_the_same_tools(self):
        _, dispatcher = self._dispatcher(plot_backend=FakePlotBackend(),
                                         tikz_backend=FakeTikzBackend())
        sdk = openrouter_backend.function_tools(dispatcher)
        codex = codex_backend.dynamic_tools(dispatcher)
        self.assertEqual([tool.name for tool in sdk],
                         [tool['name'] for tool in codex])
        for tool, record in zip(sdk, codex):
            self.assertEqual(record['type'], 'function')
            self.assertEqual(tool.description, record['description'])
            self.assertEqual(tool.params_json_schema, record['inputSchema'])
            self.assertEqual(sorted(record['inputSchema']['required']),
                             sorted(record['inputSchema']['properties']))
            self.assertFalse(record['inputSchema']['additionalProperties'])

    def test_optional_figure_tools_track_their_backends_in_both(self):
        _, plain = self._dispatcher()
        names = [tool['name'] for tool in codex_backend.dynamic_tools(plain)]
        self.assertNotIn('plot', names)
        self.assertNotIn('tikz', names)
        _, with_plot = self._dispatcher(plot_backend=FakePlotBackend())
        self.assertIn('plot', [t['name'] for t
                               in codex_backend.dynamic_tools(with_plot)])

    def test_the_version_contract_lists_the_reviewed_residual_tools(self):
        # the exact model-visible set: canonical ToyMath tools plus the
        # three residual native names the project accepted for 0.144.4.
        # Anything else must fail closed before a live model runs.
        _, dispatcher = self._dispatcher()
        self.assertEqual(codex_backend.expected_model_tools(dispatcher), (
            'load_skill', 'run_tactic', 'comment', 'claim', 'conclude',
            'set_result', 'set_open',
            'update_plan', 'request_user_input', 'view_image'))

    def test_tactic_names_never_reach_the_codex_surface(self):
        _, dispatcher = self._dispatcher()
        names = {tool['name']
                 for tool in codex_backend.dynamic_tools(dispatcher)}
        self.assertFalse(names & set(tactic_registry.BY_NAME))


class TestCodexRuntimeSpawn(unittest.TestCase):
    def test_runtime_starts_in_its_own_process_group(self):
        """Jupyter's Interrupt sends SIGINT to the kernel's WHOLE process
        group (jupyter_client LocalProvisioner uses killpg): a runtime left
        in that group dies on the first Stop press, the interrupted cell
        errors with 'the Codex runtime closed' instead of a clean
        interrupted outcome, and every later run fails the same way. The
        spawn must detach the runtime into its own group."""
        transport = codex_transport.AppServerTransport(binary='/fake/codex')
        with mock.patch.object(codex_transport.subprocess, 'Popen') as popen, \
                mock.patch.object(codex_transport.AppServerTransport,
                                  'initialize'), \
                mock.patch.object(codex_transport.threading, 'Thread',
                                  return_value=mock.Mock()):
            transport.start()
        self.assertTrue(popen.call_args.kwargs.get('start_new_session'))


class TestCodexToolCallBridge(unittest.TestCase):
    """`item/tool/call` semantics: what ran versus what failed to run."""

    def setUp(self):
        self.ledger = Ledger()
        self.session = DoSession(ledger=self.ledger)
        self.cancellation = agent_base.CancellationToken()
        self.dispatcher = agent_base.ToolDispatcher(
            agent_do.make_tool_bindings(self.session), self.cancellation,
            serialize=True)

    def _call(self, tool, arguments=None, **extra):
        params = {'tool': tool, 'arguments': arguments or {},
                  'callId': 'c1', 'threadId': 't1', 'turnId': 'r1'}
        params.update(extra)
        return codex_backend.tool_call_result(self.dispatcher, params)

    def _text(self, reply):
        return reply['contentItems'][0]['text']

    def test_a_checked_step_comes_back_as_one_input_text_item(self):
        reply = self._call('run_tactic',
                           {'tactic': 'expand', 'arguments': ['(x+1)^2']})
        self.assertTrue(reply['success'])
        self.assertEqual([item['type'] for item in reply['contentItems']],
                         ['inputText'])
        record = json.loads(self._text(reply))
        self.assertTrue(record['ok'])
        self.assertEqual(record['result'], 'x^{2}+2x+1')
        self.assertEqual(record['step']['id'], 's1')
        self.assertEqual(record['check']['status'], 'agree')

    def test_a_refused_tactic_is_a_successful_call_the_model_can_repair(self):
        # ToyMath answering {"ok": false} is a tool that RAN and said no;
        # calling that a transport failure would hide the reason
        reply = self._call('run_tactic',
                           {'tactic': 'factor_quadratic',
                            'arguments': ['x^{2}+2x+3', 'x']})
        self.assertTrue(reply['success'])
        record = json.loads(self._text(reply))
        self.assertFalse(record['ok'])
        self.assertTrue(record['error'])
        self.assertEqual(len(self.ledger.steps), 0)

    def test_unknown_tools_malformed_calls_and_namespaces_are_distinct(self):
        unknown = self._call('shell', {'command': 'ls'})
        self.assertFalse(unknown['success'])
        self.assertIn("'shell' is not a ToyMath tool", self._text(unknown))
        self.assertIn('run_tactic', self._text(unknown))

        malformed = self._call('run_tactic', {'tactic': 'expand'})
        self.assertFalse(malformed['success'])
        self.assertIn('missing required argument', self._text(malformed))

        namespaced = self._call('run_tactic', {}, namespace='mcp__helper')
        self.assertFalse(namespaced['success'])
        self.assertIn('unknown tool namespace', self._text(namespaced))

        shapeless = codex_backend.tool_call_result(self.dispatcher, 'nope')
        self.assertFalse(shapeless['success'])

    def test_raw_json_arguments_are_accepted_and_validated(self):
        reply = self._call('run_tactic',
                           json.dumps({'tactic': 'expand',
                                       'arguments': ['(x+1)^2']}))
        self.assertTrue(reply['success'])
        broken = self._call('run_tactic', '{"tactic": ')
        self.assertFalse(broken['success'])
        self.assertIn('not valid JSON', self._text(broken))

    def test_dispatch_is_serial_so_step_ids_stay_deterministic(self):
        # provenance depends on step order; the Codex bridge serializes
        # rather than inheriting the SDK's tool-call concurrency
        order = []
        original = self.ledger.record

        def watched(result, **kwargs):
            order.append(result['op'])
            return original(result, **kwargs)

        self.ledger.record = watched
        workers = [threading.Thread(
            target=self._call, args=('run_tactic',
                                     {'tactic': 'expand',
                                      'arguments': [f'(x+{i})^2']}))
            for i in range(1, 5)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(10)
        self.assertEqual(len(order), 4)
        self.assertEqual([s['id'] for s in self.ledger.steps],
                         ['s1', 's2', 's3', 's4'])
        self.assertEqual(self.ledger.replay()['status'], 'verified')

    def test_a_stopped_run_refuses_further_calls_at_the_bridge(self):
        self.assertTrue(self._call(
            'run_tactic',
            {'tactic': 'expand', 'arguments': ['(x+1)^2']})['success'])
        self.cancellation.cancel('user')
        self.session.close('user')
        reply = self._call('run_tactic',
                           {'tactic': 'expand', 'arguments': ['(x+2)^2']})
        self.assertTrue(reply['success'])   # the transport worked
        record = json.loads(self._text(reply))
        self.assertFalse(record['ok'])      # the run did not
        self.assertTrue(record['cancelled'])
        self.assertEqual(len(self.ledger.steps), 1)


class TestCodexTranscriptDriver(unittest.TestCase):
    """A whole derivation over the fake app-server, no account, no network."""

    def _run(self, transcript, ledger=None):
        session = DoSession(ledger=ledger)
        cancellation = agent_base.CancellationToken()
        dispatcher = agent_base.ToolDispatcher(
            agent_do.make_tool_bindings(session), cancellation,
            serialize=True)
        transport = codex_transport.TranscriptTransport(transcript)
        request = codex_transport.CodexThreadRequest(
            instruction='solve it', developer_instructions='rules',
            dynamic_tools=tuple(codex_backend.dynamic_tools(dispatcher)))
        outcome = transport.run_thread(
            request,
            lambda params: codex_backend.tool_call_result(dispatcher,
                                                          params))
        return session, transport, outcome

    def test_a_scripted_solve_produces_the_same_ledger_as_openrouter(self):
        ledger = Ledger()
        session, _, outcome = self._run([
            {'tool': 'run_tactic',
             'arguments': {'tactic': 'apply',
                           'arguments': ['2x + 3 = 7', '-', '3']}},
            {'tool': 'run_tactic',
             'arguments': {'tactic': 'expand',
                           'arguments': ['2x+3 - 3 = 7 - 3']}},
            {'tool': 'set_result', 'arguments': {'expr': '2x = 4'}},
            {'message': 'Subtracted 3 from both sides.'},
        ], ledger=ledger)
        self.assertEqual(outcome.status, 'completed')
        self.assertEqual([s['op'] for s in ledger.steps],
                         ['apply_both_sides', 'expand'])
        self.assertEqual(session.result_override, '2x = 4')
        self.assertEqual(ledger.replay()['status'], 'verified')

        # the OpenRouter path over the same moves lands on the same ledger
        other = Ledger()
        res = run_instruction('solve 2x + 3 = 7 for x', ledger=other,
                              model=ScriptedModel(
                                  [list(t) for t in SOLVE_SCRIPT]))
        self.assertEqual([s['op'] for s in other.steps],
                         [s['op'] for s in ledger.steps])
        self.assertEqual([s['result'] for s in other.steps],
                         [s['result'] for s in ledger.steps])
        self.assertTrue(res['ok'])

    def test_codex_prose_alone_cannot_designate_a_result(self):
        session, _, outcome = self._run([
            {'tool': 'run_tactic',
             'arguments': {'tactic': 'expand', 'arguments': ['(x+1)^2']}},
            {'message': 'And therefore x = 2, trust me.'},
        ])
        self.assertEqual(outcome.final_text, 'And therefore x = 2, trust me.')
        self.assertIsNone(session.result_override)
        # the value exists only in the narrative: no ledger step establishes
        # it, so it cannot be designated
        self.assertIsNone(session.designate_result('x = 2'))
        self.assertEqual(session.designate_result('x^{2}+2x+1')['status'],
                         'verified')

    def test_an_interrupt_stops_the_transcript_where_it_stands(self):
        session = DoSession()
        cancellation = agent_base.CancellationToken()
        dispatcher = agent_base.ToolDispatcher(
            agent_do.make_tool_bindings(session), cancellation)
        transport = codex_transport.TranscriptTransport([
            {'tool': 'run_tactic',
             'arguments': {'tactic': 'expand', 'arguments': ['(x+1)^2']}},
            {'block': True},
            {'tool': 'run_tactic',
             'arguments': {'tactic': 'expand', 'arguments': ['(x+9)^2']}},
            {'message': 'never reached'},
        ])
        outcome = {}
        worker = threading.Thread(target=lambda: outcome.update(
            value=transport.run_thread(
                request=codex_transport.CodexThreadRequest(
                    instruction='x', developer_instructions='y'),
                on_tool_call=lambda params: codex_backend.tool_call_result(
                    dispatcher, params))))
        worker.start()
        transport._released.wait(5)
        transport.interrupt_turn('t1', 'r1')
        worker.join(10)
        self.assertEqual(outcome['value'].status, 'interrupted')
        self.assertEqual(outcome['value'].tool_calls, ('run_tactic',))
        self.assertEqual(transport.interrupts, [('t1', 'r1')])
        self.assertEqual(len(session.ledger.steps), 1)


SOLVE_TRANSCRIPT = [
    {'tool': 'run_tactic',
     'arguments': {'tactic': 'apply', 'arguments': ['2x + 3 = 7', '-', '3']}},
    {'tool': 'run_tactic',
     'arguments': {'tactic': 'expand', 'arguments': ['2x+3 - 3 = 7 - 3']}},
    {'tool': 'set_result', 'arguments': {'expr': '2x = 4'}},
    {'message': 'Subtracted 3 from both sides.'},
]


class TestCodexExecution(unittest.TestCase):
    """A whole do! run over the Codex path, offline."""

    def _backend(self, transcript, transport=None, **kwargs):
        transport = transport or codex_transport.TranscriptTransport(
            transcript)
        return codex_backend.CodexBackend(transport=transport, **kwargs)

    def test_a_derivation_lands_the_same_ledger_as_openrouter(self):
        ledger = Ledger()
        res = run_instruction('solve 2x + 3 = 7 for x', ledger=ledger,
                              backend=self._backend(SOLVE_TRANSCRIPT))
        self.assertTrue(res['ok'], res.get('error'))
        self.assertEqual(res['status'], 'completed')
        self.assertEqual(res['final_result'], '2x = 4')
        self.assertEqual(res['final_provenance']['status'], 'verified')
        self.assertEqual(res['summary'], 'Subtracted 3 from both sides.')
        self.assertEqual(ledger.replay()['status'], 'verified')

        other = Ledger()
        run_instruction('solve 2x + 3 = 7 for x', ledger=other,
                        model=ScriptedModel([list(t) for t in SOLVE_SCRIPT]))
        self.assertEqual([(s['op'], s['input'], s['result'], s['hash'])
                          for s in other.steps],
                         [(s['op'], s['input'], s['result'], s['hash'])
                          for s in ledger.steps])

    def test_a_signed_out_machine_is_told_to_log_in(self):
        transport = codex_transport.TranscriptTransport(
            SOLVE_TRANSCRIPT,
            account=codex_transport.CodexAccountStatus())
        with self.assertRaises(agent_do.DoAgentError) as caught:
            run_instruction('solve it', backend=self._backend(
                None, transport=transport))
        self.assertIn('login!', str(caught.exception))
        self.assertEqual(transport.requests, [])   # no tokens spent

    def test_the_thread_carries_the_generated_policy_and_the_tools(self):
        transport = codex_transport.TranscriptTransport(SOLVE_TRANSCRIPT)
        run_instruction('solve 2x + 3 = 7 for x',
                        backend=self._backend(None, transport=transport))
        thread = transport.requests[0]
        names = [tool['name'] for tool in thread.dynamic_tools]
        self.assertEqual(names[:7], list(agent_do.TOOL_NAMES))
        # the per-thread instructions enumerate, then carry the do! rules
        self.assertTrue(thread.developer_instructions.startswith(
            codex_backend.POLICY_HEADER))
        for name in names:
            self.assertIn(name, thread.developer_instructions)
        self.assertIn('## do! mode', thread.developer_instructions)

    def test_the_thread_policy_lists_every_tool_the_session_offers(self):
        # the regression: the allowlist the model is given must be the
        # session's real surface, figure tools included. A policy naming
        # fewer tools than are offered makes the model refuse the rest.
        transport = codex_transport.TranscriptTransport(SOLVE_TRANSCRIPT)
        run_instruction('solve 2x + 3 = 7 for x',
                        backend=self._backend(None, transport=transport),
                        plot_backend=FakePlotBackend(),
                        tikz_backend=FakeTikzBackend())
        thread = transport.requests[0]
        offered = [tool['name'] for tool in thread.dynamic_tools]
        self.assertIn('plot', offered)
        self.assertIn('tikz', offered)
        allowlist = thread.developer_instructions.split('\n\n')[1]
        for name in offered:
            self.assertIn(name, allowlist,
                          f'{name} is offered but missing from the policy')

    def test_the_thread_is_ephemeral_read_only_and_outside_the_repository(
            self):
        home = tempfile.mkdtemp(prefix='toymath-codex-home-')
        self.addCleanup(shutil.rmtree, home, True)
        _, workdir = codex_backend.ensure_home(home=home)
        transport = codex_transport.AppServerTransport(
            binary='/nonexistent', home=home, cwd=workdir)
        params = transport.thread_params(
            codex_transport.CodexThreadRequest(
                instruction='x', developer_instructions='y'))
        self.assertTrue(params['ephemeral'])
        self.assertEqual(params['sandbox'], 'read-only')
        self.assertEqual(params['approvalPolicy'], 'never')
        self.assertEqual(params['cwd'], workdir)
        repository = os.path.dirname(os.path.dirname(
            os.path.abspath(agent_do.__file__)))
        self.assertFalse(os.path.abspath(params['cwd']).startswith(
            repository))
        self.assertNotIn('model', params)      # the account's own default

    def test_a_refused_request_is_explained_rather_than_dumped(self):
        # Measured (Into.ipynb, `do! plot interval [[2]]`): the model drove
        # `tikz` from the runtime's scripting tool and passed the reply to
        # the script helper `image(...)`. The appended `image_url` was our
        # status JSON, so the provider refused the NEXT request and the
        # cell showed the raw 400 - a message about `input[0].output[1]`
        # that tells a mathematician nothing about what to do.
        refusal = json.dumps({'type': 'error', 'error': {
            'type': 'invalid_request_error', 'code': 'invalid_value',
            'message': ("Invalid 'input[0].output[1].image_url'. Expected a "
                        'valid URL, but got a value with an invalid '
                        'format.'),
            'param': 'input[0].output[1].image_url'}, 'status': 400})
        res = run_instruction(
            'draw the interval', tikz_backend=FakeTikzBackend(),
            backend=self._backend([
                {'tool': 'tikz', 'arguments': {'code': '\\draw (0,0);',
                                               'caption': 'interval'}},
                {'fail': refusal}]))
        self.assertEqual(res['status'], 'failed')
        self.assertIn('attached a ToyMath tool result', res['error'])
        self.assertIn('never returned to the model', res['error'])
        self.assertIn('re-run the cell', res['error'])
        # the provider's own words survive: this is a translation, not a
        # replacement, and the raw cause is what a bug report needs
        self.assertIn('image_url', res['error'])

    def test_an_unrelated_refusal_keeps_its_own_words(self):
        # a wrong guess about why a run died is worse than the raw message
        for error in ('the Codex turn failed',
                      json.dumps({'error': {'message': 'context too long',
                                            'code': 'context_length'}}),
                      '', None):
            self.assertEqual(codex_backend.explain_turn_failure(error), error)

    def test_tool_dispatch_is_serialized_on_the_codex_path(self):
        seen = {}
        original = agent_base.ToolDispatcher.__init__

        def record(self, bindings, *args, **kwargs):
            seen['serialize'] = kwargs.get('serialize')
            return original(self, bindings, *args, **kwargs)

        with mock.patch.object(agent_base.ToolDispatcher, '__init__', record):
            run_instruction('solve it',
                            backend=self._backend(SOLVE_TRANSCRIPT))
        self.assertTrue(seen['serialize'])

    def test_every_run_gets_a_fresh_thread(self):
        # ephemeral threads: no hidden conversational state carries from
        # one do! cell to the next
        transport = codex_transport.TranscriptTransport(SOLVE_TRANSCRIPT)
        backend = self._backend(None, transport=transport)
        run_instruction('solve it', backend=backend)
        run_instruction('solve it again', backend=backend)
        self.assertEqual(len(transport.requests), 2)
        self.assertEqual([r.instruction for r in transport.requests],
                         ['solve it', 'solve it again'])


class TestCodexCancellation(unittest.TestCase):
    def _blocking(self, transport_class=None):
        transport_class = (transport_class
                           or codex_transport.TranscriptTransport)
        return transport_class([
            {'tool': 'run_tactic',
             'arguments': {'tactic': 'expand', 'arguments': ['(x+1)^2']}},
            {'block': True},
            {'tool': 'run_tactic',
             'arguments': {'tactic': 'expand', 'arguments': ['(x+9)^2']}},
            {'message': 'never reached'},
        ])

    def _stop_when_blocked(self, transport):
        """Press Stop once the transcript has parked."""
        def stopper():
            transport._released.wait(5)
            time.sleep(0.05)
            _thread.interrupt_main()
        threading.Thread(target=stopper, daemon=True).start()

    def test_stop_interrupts_the_exact_turn_and_keeps_the_steps(self):
        ledger = Ledger()
        transport = self._blocking()
        self._stop_when_blocked(transport)
        res = run_instruction('expand it', ledger=ledger, grace_period=1.0,
                              backend=codex_backend.CodexBackend(
                                  transport=transport))
        self.assertEqual(res['status'], 'interrupted')
        self.assertTrue(res['cancelled'])
        self.assertIsNone(res['final_result'])
        self.assertEqual(transport.interrupts, [('t1', 'r1')])
        self.assertEqual([s['op'] for s in res['steps']], ['expand'])
        self.assertEqual(res['partial_result'], 'x^{2}+2x+1')
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_a_runtime_that_ignores_the_interrupt_is_poisoned(self):
        transport = self._blocking(codex_transport.StubbornTransport)
        codex_backend.set_runtime(transport)
        self.addCleanup(codex_backend.set_runtime, None)
        self.addCleanup(transport._released.set)
        self._stop_when_blocked(transport)
        res = run_instruction('expand it', grace_period=0.2,
                              backend=codex_backend.CodexBackend(
                                  transport=transport))
        self.assertEqual(res['status'], 'interrupted')
        self.assertTrue(res['error'])
        self.assertTrue(transport.poisoned)
        self.assertEqual(len(res['steps']), 1)

    def test_a_residual_native_tool_invalidates_the_run(self):
        ledger = Ledger()
        transport = codex_transport.TranscriptTransport([
            {'tool': 'run_tactic',
             'arguments': {'tactic': 'expand', 'arguments': ['(x+1)^2']}},
            {'tool': 'set_result', 'arguments': {'expr': 'x^{2}+2x+1'}},
            {'native': 'update_plan'},
            {'message': 'the answer is above'},
        ])
        res = run_instruction('expand it', ledger=ledger, grace_period=1.0,
                              backend=codex_backend.CodexBackend(
                                  transport=transport))
        self.assertEqual(res['status'], 'capability_violation')
        self.assertTrue(res['cancelled'])
        self.assertIsNone(res['final_result'])   # nothing chainable
        self.assertIsNone(res['summary'])        # the prose is discarded
        self.assertIn('update_plan', res['error'])
        # the mechanically checked step still stands and still replays
        self.assertEqual([s['op'] for s in res['steps']], ['expand'])
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_the_budget_stops_a_codex_run_with_its_own_status(self):
        res = run_instruction(
            'expand forever', grace_period=1.0,
            budget=agent_base.AgentBudget(max_tool_calls=1),
            backend=codex_backend.CodexBackend(
                transport=codex_transport.TranscriptTransport(
                    SOLVE_TRANSCRIPT)))
        self.assertEqual(res['status'], 'budget_exhausted')
        self.assertIsNone(res['final_result'])
        self.assertEqual(len(res['steps']), 1)

    def test_unknown_thread_items_fail_closed(self):
        # a native capability ToyMath has never seen must not be waved
        # through just because it is unrecognised
        self.assertIsNone(codex_transport.native_capability('agentMessage'))
        self.assertIsNone(
            codex_transport.native_capability('dynamicToolCall'))
        self.assertEqual(codex_transport.native_capability('plan'),
                         'update_plan')
        self.assertEqual(codex_transport.native_capability('commandExecution'),
                         'shell')
        self.assertEqual(codex_transport.native_capability('brandNewTool'),
                         'brandNewTool')


class TestCodexServerRequests(unittest.TestCase):
    """Server-initiated requests: only a ToyMath tool call is answered.

    The methods come from the pinned runtime's own `ServerRequest` schema.
    Each refusal has to be that method's own shape - a generic `{}` is not
    a valid response for most of them - and each has to invalidate the run.
    """

    def setUp(self):
        self.seen = []

    def _serve(self, method, params=None):
        return codex_transport.serve_request(
            method, params or {},
            on_tool_call=lambda p: {'success': True, 'contentItems': []},
            on_native_tool=self.seen.append)

    def test_a_toymath_tool_call_is_the_only_thing_answered_normally(self):
        reply = self._serve('item/tool/call', {'tool': 'comment'})
        self.assertTrue(reply['success'])
        self.assertEqual(self.seen, [])

    def test_each_approval_gets_its_own_schemas_refusal(self):
        cases = {
            'item/commandExecution/requestApproval': {'decision': 'decline'},
            'item/fileChange/requestApproval': {'decision': 'decline'},
            'mcpServer/elicitation/request': {'action': 'decline'},
            'item/permissions/requestApproval': {'permissions': {}},
            'applyPatchApproval': {'decision': 'denied'},
            'execCommandApproval': {'decision': 'denied'},
        }
        for method, expected in cases.items():
            with self.subTest(method=method):
                self.assertEqual(self._serve(method), expected)
        self.assertEqual(self.seen, ['shell', 'apply_patch', 'mcp',
                                     'permissions', 'apply_patch', 'shell'])

    def test_a_token_refresh_is_refused_rather_than_answered(self):
        # the only valid response carries an accessToken; ToyMath never
        # holds one, so this must be an error, not a shaped reply
        reply = self._serve('account/chatgptAuthTokens/refresh',
                            {'reason': 'unauthorized'})
        self.assertIsInstance(reply, codex_transport.ServerRefusal)
        self.assertEqual(self.seen, ['chatgpt_auth_tokens'])

    def test_asking_the_user_a_question_invalidates_the_run(self):
        reply = self._serve('item/tool/requestUserInput', {'questions': []})
        self.assertIsInstance(reply, codex_transport.ServerRefusal)
        self.assertEqual(self.seen, ['request_user_input'])

    def test_an_unknown_server_request_fails_closed(self):
        reply = self._serve('capability/inventedLater', {})
        self.assertIsInstance(reply, codex_transport.ServerRefusal)
        self.assertEqual(self.seen, ['capability/inventedLater'])

    def test_every_schema_method_but_the_tool_call_invalidates_the_run(self):
        # the full ServerRequest set from the pinned runtime's schema
        for method in codex_transport.SERVER_REQUEST_CAPABILITY:
            with self.subTest(method=method):
                self.seen.clear()
                self._serve(method)
                self.assertEqual(len(self.seen), 1)
        self.assertNotIn('item/tool/call',
                         codex_transport.SERVER_REQUEST_CAPABILITY)

    def test_a_refused_request_reaches_the_wire_as_an_error(self):
        written = []
        transport = codex_transport.AppServerTransport.__new__(
            codex_transport.AppServerTransport)
        transport._write = written.append
        transport._answer({'id': 'r1', 'method': 'attestation/generate'},
                          lambda method, params: codex_transport.ServerRefusal(
                              'nope'))
        self.assertEqual(written[0]['id'], 'r1')
        self.assertNotIn('result', written[0])
        self.assertEqual(written[0]['error']['message'], 'nope')
        self.assertEqual(written[0]['error']['code'],
                         codex_transport.METHOD_REFUSED)


class TestCodexHomeAndPolicy(unittest.TestCase):
    """The dedicated home, and the one generated role/tool policy."""

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix='toymath-codex-home-')
        self.addCleanup(shutil.rmtree, self.home, True)
        self.names = [b.name for b
                      in agent_do.make_tool_bindings(DoSession())]

    def test_the_policy_allowlist_is_generated_from_the_bindings(self):
        # no second handwritten tactic list in Markdown: the allowed names
        # come from the canonical binding map
        policy = codex_backend.role_policy(self.names)
        for name in self.names:
            self.assertIn(name, policy)
        for residual in codex_backend.RESIDUAL_NATIVE_TOOLS:
            self.assertIn(residual, policy)
        self.assertIn('Use only the client-provided ToyMath dynamic tools',
                      policy)
        self.assertIn('Model prose is', policy)
        # "use only these tools", never "never use tools" - the derivation
        # agent must call the verified surface
        self.assertNotIn('Never use tools', policy)

    def test_the_durable_home_policy_names_no_tools(self):
        # It outlives the session that would define the list. `plot` and
        # `tikz` exist only when the figure sandboxes resolved, so a home
        # file written by `login!` once listed seven tools while the thread
        # offered nine - and the model obeyed the stricter one, refusing to
        # draw a diagram it had a working `tikz` for.
        home, workdir = codex_backend.ensure_home(home=self.home)
        with open(os.path.join(home, 'AGENTS.md'), encoding='utf-8') as fh:
            written = fh.read()
        self.assertEqual(written, codex_backend.role_policy())
        for name in self.names:
            self.assertNotIn(name, written)
        self.assertIn('supplied with each thread', written)
        # the prohibitions do not vary, so they stay in the durable form
        for residual in codex_backend.RESIDUAL_NATIVE_TOOLS:
            self.assertIn(residual, written)
        # the runtime cwd is empty and inside the dedicated home, so no
        # project AGENTS.md can join the instruction chain
        self.assertEqual(os.listdir(workdir), [])
        self.assertTrue(workdir.startswith(home))

    def test_both_policy_forms_state_what_attaching_a_result_costs(self):
        # The Codex path does not hand the model our tools directly: it
        # reaches them through the runtime's scripting tool, whose helpers
        # can append images to the conversation. A model that reads `tikz`
        # as "returns a picture" writes `image(r)`, and the provider then
        # refuses the whole turn. This text is always-on, so it carries
        # only what the model cannot see for itself: the reply's shape it
        # observes on the first call, but the refused turn never reaches
        # it, and neither does the fact that the user already has the
        # figure. Neither varies with the session, so both forms carry it.
        for policy in (codex_backend.role_policy(),
                       codex_backend.role_policy(self.names)):
            self.assertIn("reach the user's notebook", policy)
            self.assertIn('image, generatedImage', policy)
            self.assertIn('refuses the whole turn', policy)
        # ... and naming those helpers must not smuggle the session's own
        # figure tools into the durable form, which outlives the session
        durable = codex_backend.role_policy()
        self.assertNotIn('plot', durable)
        self.assertNotIn('tikz', durable)

    def test_the_durable_policy_defers_to_the_threads_list(self):
        # the two reach the model together, so the durable one must not
        # read as a competing allowlist
        durable = codex_backend.role_policy()
        self.assertIn('treat that\nlist as authoritative', durable)

    def test_the_home_is_never_the_general_codex_home(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(codex_backend.HOME_VAR, None)
            default = codex_backend.home_path()
        self.assertNotEqual(os.path.normpath(default),
                            os.path.normpath(os.path.expanduser('~/.codex')))
        self.assertIn('toymath', default)
        with mock.patch.dict(os.environ,
                             {codex_backend.HOME_VAR: self.home}):
            self.assertEqual(codex_backend.home_path(), self.home)

    def test_rewriting_the_policy_is_idempotent(self):
        codex_backend.ensure_home(home=self.home)
        path = os.path.join(self.home, 'AGENTS.md')
        stamp = os.stat(path).st_mtime_ns
        codex_backend.ensure_home(home=self.home)
        self.assertEqual(os.stat(path).st_mtime_ns, stamp)

    def test_a_missing_extra_names_the_install_command(self):
        with mock.patch.object(codex_backend, 'runtime_binary',
                               return_value=None):
            self.assertFalse(codex_backend.available())
            with self.assertRaises(agent_do.DoAgentError) as caught:
                codex_backend.require_available()
        # the hint must name a command that actually resolves: ToyMath is
        # installed from a clone, so `toymath[codex]` is not on PyPI
        self.assertIn('.[codex]', str(caught.exception))

    # -- ownership: ensure_home rewrites AGENTS.md, so it must own the dir --
    def test_pointing_the_home_at_the_general_one_is_refused(self):
        general = os.path.join(self.home, 'dot-codex')
        os.makedirs(general)
        with mock.patch.dict(os.environ, {codex_backend.HOME_VAR: general,
                                          'CODEX_HOME': general}):
            with self.assertRaises(agent_do.DoAgentError) as caught:
                codex_backend.home_path()
        self.assertIn('general Codex home', str(caught.exception))

    def test_an_unrelated_populated_directory_is_never_adopted(self):
        # a mistyped TOYMATH_CODEX_HOME must not silently overwrite the
        # AGENTS.md of whatever it happened to name
        project = os.path.join(self.home, 'someones-project')
        os.makedirs(project)
        policy = os.path.join(project, 'AGENTS.md')
        with open(policy, 'w', encoding='utf-8') as handle:
            handle.write('# Their project instructions\n')
        with self.assertRaises(agent_do.DoAgentError) as caught:
            codex_backend.ensure_home(home=project)
        self.assertIn('not created by ToyMath', str(caught.exception))
        with open(policy, encoding='utf-8') as handle:
            self.assertEqual(handle.read(), '# Their project instructions\n')

    def test_a_home_toymath_created_earlier_is_adopted_and_marked(self):
        # the layout ToyMath wrote before the ownership marker existed
        os.makedirs(os.path.join(self.home, codex_backend.WORKDIR))
        with open(os.path.join(self.home, 'AGENTS.md'), 'w',
                  encoding='utf-8') as handle:
            handle.write(codex_backend.role_policy())
        codex_backend.ensure_home(home=self.home)
        self.assertTrue(os.path.exists(
            os.path.join(self.home, codex_backend.OWNER_MARKER)))

    def test_a_fresh_home_is_marked_as_toymaths_own(self):
        codex_backend.ensure_home(home=self.home)
        codex_backend.ensure_home(home=self.home)   # idempotent
        self.assertTrue(os.path.exists(
            os.path.join(self.home, codex_backend.OWNER_MARKER)))

    # -- MCP containment ---------------------------------------------------
    def _write_config(self, text):
        with open(os.path.join(self.home, codex_backend.CONFIG_FILE), 'w',
                  encoding='utf-8') as handle:
            handle.write(text)

    def test_no_config_means_no_mcp_overrides(self):
        self.assertEqual(codex_backend.mcp_overrides(self.home), ())

    def test_every_configured_mcp_server_is_disabled_by_name(self):
        # measured against the real runtime: a whole-table override merges
        # instead of replacing, so each server must be named individually
        self._write_config('[mcp_servers.alpha]\ncommand = "a"\n\n'
                           '[mcp_servers.beta]\ncommand = "b"\n')
        self.assertEqual(
            codex_backend.mcp_overrides(self.home),
            ('mcp_servers.alpha.enabled=false',
             'mcp_servers.beta.enabled=false'))

    def test_a_server_name_that_cannot_be_disabled_is_a_hard_error(self):
        # a dotted/quoted key cannot be addressed by a dotted override, and
        # leaving one enabled is not an option
        self._write_config('[mcp_servers."a.b"]\ncommand = "x"\n')
        with self.assertRaises(agent_do.DoAgentError) as caught:
            codex_backend.mcp_overrides(self.home)
        self.assertIn('cannot reliably disable', str(caught.exception))

    def test_an_unreadable_config_fails_closed(self):
        self._write_config('this is not = = toml\n')
        with self.assertRaises(agent_do.DoAgentError) as caught:
            codex_backend.mcp_overrides(self.home)
        self.assertIn('could not be read', str(caught.exception))

    # -- telemetry containment ---------------------------------------------
    def test_every_self_starting_exporter_is_pinned_off(self):
        # `metrics_exporter` is the one that defaults to a live destination
        # (statsig -> a baked-in ab.chatgpt.com endpoint in a release
        # build). The gate that spares it today is `app-server`'s own
        # default_analytics_enabled=false, which is the binary's argument
        # and not ToyMath's decision.
        for key in ('otel.exporter', 'otel.metrics_exporter'):
            self.assertIn(f'{key}="none"',
                          codex_backend.CONTAINMENT_OVERRIDES)

    def test_the_trace_exporter_is_left_to_the_home_config(self):
        # Not an omission: a --config override is SessionFlags, which
        # outranks $CODEX_HOME/config.toml, so pinning the trace exporter
        # would silently disable a trace exporter configured in the home's
        # own config.toml (which is how a credential-bearing exporter stays
        # out of argv) while buying no containment - it already defaults to
        # none. Measured: pinned on the CLI, the runtime exports zero spans
        # and reports nothing.
        self.assertFalse(
            [override for override in codex_backend.CONTAINMENT_OVERRIDES
             if override.startswith(('otel.trace_exporter',
                                     'otel.log_user_prompt'))])

    # -- the runtime's own trace export ------------------------------------
    #: the module runs with TOYMATH_OBSERVABILITY=off, so "tracing off" is
    #: the ambient state and an explicit target is how a test says "on"
    TARGET = ('http://localhost:3100/api/public/otel/v1/traces',
              {'Authorization': 'Basic ZmFrZTpmYWtl'})

    def _config_text(self):
        with open(os.path.join(self.home, codex_backend.CONFIG_FILE),
                  encoding='utf-8') as handle:
            return handle.read()

    def _parsed_config(self):
        import tomllib
        return tomllib.loads(self._config_text())

    def test_the_trace_block_is_written_where_mcp_can_still_read_it(self):
        self.assertTrue(codex_backend.ensure_trace_export(
            self.home, target=self.TARGET))
        exporter = self._parsed_config()['otel']['trace_exporter']['otlp-http']
        self.assertEqual(exporter['endpoint'], self.TARGET[0])
        self.assertEqual(exporter['protocol'], 'json')
        self.assertEqual(exporter['headers']['Authorization'],
                         self.TARGET[1]['Authorization'])
        # the file the containment reader now has to parse on every run
        self.assertEqual(codex_backend.mcp_overrides(self.home), ())

    def test_tracing_off_removes_the_block_and_the_credential(self):
        codex_backend.ensure_trace_export(self.home, target=self.TARGET)
        self.assertFalse(codex_backend.ensure_trace_export(self.home))
        text = self._config_text()
        self.assertNotIn('otlp-http', text)
        self.assertNotIn('Basic', text)

    def test_tracing_off_creates_no_file_at_all(self):
        self.assertFalse(codex_backend.ensure_trace_export(self.home))
        self.assertFalse(os.path.exists(
            os.path.join(self.home, codex_backend.CONFIG_FILE)))

    def test_foreign_configuration_survives_both_directions(self):
        self._write_config('[mcp_servers.alpha]\ncommand = "a"\n')
        codex_backend.ensure_trace_export(self.home, target=self.TARGET)
        self.assertEqual(codex_backend.mcp_overrides(self.home),
                         ('mcp_servers.alpha.enabled=false',))
        codex_backend.ensure_trace_export(self.home)          # off again
        self.assertIn('mcp_servers.alpha', self._config_text())

    def test_writing_the_same_block_twice_changes_nothing(self):
        codex_backend.ensure_trace_export(self.home, target=self.TARGET)
        first = self._config_text()
        self.assertTrue(codex_backend.ensure_trace_export(
            self.home, target=self.TARGET))
        self.assertEqual(self._config_text(), first)

    def test_a_config_that_would_not_parse_is_left_alone(self):
        # tracing is observability only: it may not make the backend worse
        self._write_config('this is not = = toml\n')
        with self.assertLogs('toymath.agent.codex', level='WARNING'):
            self.assertFalse(codex_backend.ensure_trace_export(
                self.home, target=self.TARGET))
        self.assertEqual(self._config_text(), 'this is not = = toml\n')

    def test_the_sampler_is_pinned_only_when_the_runtime_traces(self):
        # without it the runtime exports every span it raises, at any
        # level; there is no verbosity knob on that pipeline
        seen = {}

        class Recorder(object):
            def __init__(self, **kwargs):
                seen.clear()
                seen.update(kwargs)

            def start(self):
                return self

        with mock.patch.object(codex_backend, 'AppServerTransport', Recorder), \
                mock.patch.object(codex_backend, 'runtime_binary',
                                  lambda: '/nonexistent/codex'):
            codex_backend.open_transport(home=self.home)
            self.assertFalse(seen.get('env'))          # tracing off
            with mock.patch.object(codex_backend, 'ensure_trace_export',
                                   lambda home: True):
                codex_backend.open_transport(home=self.home)
            self.assertEqual(seen['env'], codex_backend.TRACE_SAMPLER_ENV)

    def test_an_interrupted_write_is_repaired_not_compounded(self):
        # an opened block with no terminator is our own half-written file
        self._write_config('model = "x"\n' + codex_backend.TRACE_BLOCK_BEGIN
                           + '\n[otel.trace_exporter.otlp-h')
        self.assertTrue(codex_backend.ensure_trace_export(
            self.home, target=self.TARGET))
        text = self._config_text()
        self.assertEqual(text.count(codex_backend.TRACE_BLOCK_BEGIN), 1)
        self.assertIn('model = "x"', text)
        self._parsed_config()                       # and it parses


class TestCodexRequestTracing(unittest.TestCase):
    """The runtime parents a request's spans on the `trace` field of the
    JSON-RPC envelope, so ToyMath's run span can adopt them."""

    TRACEPARENT = '00-' + 'a' * 32 + '-' + 'b' * 16 + '-01'

    def _sent(self, carrier):
        transport = codex_transport.AppServerTransport(trace_carrier=carrier)
        sent = []
        transport._write = sent.append
        with self.assertRaises(codex_transport.CodexUnavailable):
            transport.request('turn/start', {'threadId': 't'}, timeout=0.01)
        return sent[0]

    def test_a_request_carries_the_caller_trace(self):
        message = self._sent(lambda: {'traceparent': self.TRACEPARENT})
        # a sibling of params, never inside it
        self.assertEqual(message['trace'], {'traceparent': self.TRACEPARENT})
        self.assertEqual(message['params'], {'threadId': 't'})

    def test_no_carrier_leaves_the_envelope_untouched(self):
        self.assertNotIn('trace', self._sent(None))

    def test_an_untraced_run_adds_no_field(self):
        # observability.traceparent returns None when tracing is inactive
        self.assertNotIn('trace', self._sent(lambda: None))

    def test_a_broken_carrier_costs_the_trace_not_the_run(self):
        def carrier():
            raise RuntimeError('no tracer')
        self.assertNotIn('trace', self._sent(carrier))


class TestObservabilityExportTarget(unittest.TestCase):
    """Where a subprocess with its own OTel pipeline is told to ship."""

    CREDS = {'LANGFUSE_PUBLIC_KEY': 'pk-lf-1', 'LANGFUSE_SECRET_KEY': 'sk-lf-2'}

    def test_no_target_while_tracing_is_off(self):
        with mock.patch.dict(os.environ, dict(self.CREDS, **{
                observability.ENABLE_VAR: 'off'})):
            self.assertIsNone(observability.otlp_trace_target())

    def test_no_target_without_credentials(self):
        env = {observability.ENABLE_VAR: 'on'}
        with mock.patch.dict(os.environ, env, clear=False):
            for var in observability._CRED_VARS:
                os.environ.pop(var, None)
            self.assertIsNone(observability.otlp_trace_target())

    def test_the_endpoint_and_basic_header(self):
        with mock.patch.dict(os.environ, dict(self.CREDS, **{
                observability.ENABLE_VAR: 'on',
                'LANGFUSE_BASE_URL': 'http://localhost:3100/'})):
            endpoint, headers = observability.otlp_trace_target()
        self.assertEqual(endpoint,
                         'http://localhost:3100/api/public/otel/v1/traces')
        self.assertEqual(
            headers['Authorization'],
            'Basic ' + base64.b64encode(b'pk-lf-1:sk-lf-2').decode())

    def test_the_carrier_keeps_only_flags_codex_accepts(self):
        # measured against 0.144.4: flags 03 (sampled + the level-2
        # random-id hint) makes it reject the whole carrier, silently, and
        # the runtime's spans start their own trace instead of joining ours
        stem = '00-' + 'a' * 32 + '-' + 'b' * 16
        self.assertEqual(observability._plain_flags(stem + '-03'),
                         stem + '-01')
        self.assertEqual(observability._plain_flags(stem + '-01'),
                         stem + '-01')

    def test_an_unsampled_parent_is_not_promoted(self):
        stem = '00-' + 'a' * 32 + '-' + 'b' * 16
        self.assertEqual(observability._plain_flags(stem + '-02'),
                         stem + '-00')

    def test_an_unfamiliar_carrier_is_left_alone(self):
        for value in ('nonsense', '01-a-b-c-d', ''):
            self.assertEqual(observability._plain_flags(value) or '', value)

    def test_a_bare_host_gets_a_scheme(self):
        with mock.patch.dict(os.environ, dict(self.CREDS, **{
                observability.ENABLE_VAR: 'on',
                'LANGFUSE_BASE_URL': 'cloud.langfuse.com'})):
            endpoint, _ = observability.otlp_trace_target()
        self.assertTrue(endpoint.startswith('https://cloud.langfuse.com/'))


class TestCodexAuthentication(unittest.TestCase):
    """Managed ChatGPT sign-in: a URL and a status, never a token."""

    def setUp(self):
        self.transport = codex_transport.TranscriptTransport(
            account=codex_transport.CodexAccountStatus())
        codex_backend.set_runtime(self.transport)
        self.addCleanup(codex_backend.set_runtime, None)

    def test_browser_login_shows_a_url_and_reports_the_account(self):
        seen = []
        status = codex_backend.login('chatgpt', on_challenge=seen.append)
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].auth_url.startswith('https://'))
        self.assertIsNone(seen[0].user_code)
        self.assertTrue(status.logged_in)
        self.assertEqual(status.auth_mode, 'chatgpt')
        self.assertEqual(status.plan_type, 'plus')

    def test_device_login_shows_a_verification_url_and_code(self):
        seen = []
        codex_backend.login('chatgptDeviceCode', on_challenge=seen.append)
        self.assertEqual(seen[0].kind, 'chatgptDeviceCode')
        self.assertTrue(seen[0].verification_uri)
        self.assertEqual(seen[0].user_code, 'ABCD-EFGH')

    def test_only_managed_modes_are_accepted(self):
        # no externally managed chatgptAuthTokens flow, and no way to hand
        # ToyMath a raw token
        for mode in ('apiKey', 'chatgptAuthTokens', 'token'):
            with self.assertRaises(ValueError):
                self.transport.login_start(mode)

    def test_only_a_managed_chatgpt_account_may_run_a_derivation(self):
        # `logged_in` is also true for the runtime's apiKey and
        # amazonBedrock account types, which bill a different - possibly
        # organizational - credential. Running on one by accident is the
        # kind of mistake the user finds out about on an invoice.
        session = DoSession()
        dispatcher = agent_base.ToolDispatcher(
            agent_do.make_tool_bindings(session),
            agent_base.CancellationToken())
        request = agent_base.AgentRequest(
            instruction='x', developer_instructions='y',
            dispatcher=dispatcher, cancellation=dispatcher.cancellation)
        for mode in ('apiKey', 'amazonBedrock'):
            with self.subTest(mode=mode):
                transport = codex_transport.TranscriptTransport(
                    account=codex_transport.CodexAccountStatus(
                        logged_in=True, auth_mode=mode))
                backend = codex_backend.CodexBackend(transport=transport)
                with self.assertRaises(agent_do.DoAgentError) as caught:
                    backend.start(request)
                self.assertIn('managed ChatGPT', str(caught.exception))
                self.assertIn(mode, str(caught.exception))

    def test_a_signed_out_account_still_says_to_run_login(self):
        transport = codex_transport.TranscriptTransport(
            account=codex_transport.CodexAccountStatus())
        with self.assertRaises(agent_do.DoAgentError) as caught:
            codex_backend.require_managed_account(transport.account_read())
        self.assertIn('login!', str(caught.exception))

    def test_usable_is_stricter_than_logged_in(self):
        managed = codex_transport.CodexAccountStatus(logged_in=True,
                                                     auth_mode='chatgpt')
        self.assertTrue(managed.usable)
        self.assertTrue(codex_backend.require_managed_account(managed))
        for mode in ('apiKey', 'amazonBedrock', None):
            status = codex_transport.CodexAccountStatus(logged_in=True,
                                                        auth_mode=mode)
            self.assertTrue(status.logged_in)
            self.assertFalse(status.usable)

    def test_an_interrupted_login_is_cancelled_on_the_app_server(self):
        self.transport.pending_login = threading.Event()
        cancellation = agent_base.CancellationToken()
        worker = threading.Thread(
            target=lambda: self._swallow(cancellation))
        worker.start()
        for _ in range(200):
            if self.transport.logins:
                break
            time.sleep(0.01)
        cancellation.cancel('user')
        self.transport.pending_login.set()
        worker.join(10)
        self.assertEqual(self.transport.cancelled_login,
                         self.transport.logins[0])

    def _swallow(self, cancellation):
        try:
            codex_backend.login('chatgpt', cancellation=cancellation,
                                timeout=5)
        except agent_do.DoAgentError:
            pass

    def test_login_cancellation_has_its_own_short_deadline(self):
        # this runs on the Stop path: the transport's ordinary 60s request
        # timeout would hold the kernel long after the user gave up
        self.transport.pending_login = threading.Event()
        cancellation = agent_base.CancellationToken()
        worker = threading.Thread(target=lambda: self._swallow(cancellation))
        worker.start()
        for _ in range(200):
            if self.transport.logins:
                break
            time.sleep(0.01)
        cancellation.cancel('user')
        self.transport.pending_login.set()
        worker.join(10)
        self.assertEqual(self.transport.cancel_timeout,
                         codex_backend.LOGIN_CANCEL_TIMEOUT)
        self.assertLess(codex_backend.LOGIN_CANCEL_TIMEOUT,
                        codex_transport.AppServerTransport.REQUEST_TIMEOUT)

    def test_a_runtime_that_will_not_cancel_a_login_is_replaced(self):
        # an unacknowledged cancellation means the challenge is still open;
        # that runtime is not trusted again
        transport = codex_transport.TranscriptTransport()

        def refuse(login_id, timeout=None):
            raise codex_transport.CodexUnavailable('no answer')

        transport.login_cancel = refuse
        codex_backend.set_runtime(transport)
        codex_backend._cancel_login(transport, 'login-1')
        self.assertTrue(transport.poisoned)
        # dropped from the kernel runtime slot, so the next command builds
        # a fresh one rather than reusing it
        self.assertIsNone(codex_backend._runtime['transport'])

    def test_status_and_logout_go_through_the_managed_endpoints(self):
        codex_backend.login('chatgpt')
        self.assertTrue(codex_backend.account_status().logged_in)
        self.assertFalse(codex_backend.logout().logged_in)

    def test_the_account_record_carries_no_identifier(self):
        # app-server reports an email; ToyMath must not keep it, so no log
        # line, exception, ledger record, or trace can leak one
        transport = codex_transport.AppServerTransport(binary='/nonexistent')
        payload = {'requiresOpenaiAuth': False,
                   'account': {'type': 'chatgpt', 'planType': 'pro',
                               'email': 'someone@example.com'}}
        with mock.patch.object(transport, 'request', return_value=payload):
            status = transport.account_read()
        self.assertEqual(status.plan_type, 'pro')
        self.assertNotIn('email', dataclasses.asdict(status))
        self.assertNotIn('example.com', repr(status))


class TestBackendResolution(unittest.TestCase):
    """Auto-selection, and the rule that a run never fails over."""

    def setUp(self):
        # the observed-account cache is process-wide; no test may inherit it
        agent_config.forget_codex_account()
        self.addCleanup(agent_config.forget_codex_account)

    def _env(self, **overrides):
        env = {key: value for key, value in os.environ.items()
               if key not in (agent_config.BACKEND_VAR,
                              agent_config.OPENROUTER_KEY_VAR,
                              openrouter_backend.BASE_URL_VAR)}
        env.update(overrides)
        return mock.patch.dict(os.environ, env, clear=True)

    def test_an_explicit_notebook_choice_wins(self):
        route = agent_config.AgentRoute(backend='codex')
        with self._env(**{agent_config.BACKEND_VAR: 'openrouter',
                          agent_config.OPENROUTER_KEY_VAR: 'sk-test'}):
            resolution = agent_config.resolve(route)
        self.assertEqual(resolution.backend, 'codex')
        self.assertIn('notebook', resolution.reason)

    def test_the_environment_setting_comes_next(self):
        with self._env(**{agent_config.BACKEND_VAR: 'codex',
                          agent_config.OPENROUTER_KEY_VAR: 'sk-test'}):
            resolution = agent_config.resolve(agent_config.AgentRoute())
        self.assertEqual(resolution.backend, 'codex')
        self.assertIn(agent_config.BACKEND_VAR, resolution.reason)

    def test_an_existing_openrouter_install_is_unchanged(self):
        # the precedence that matters for everyone already using ToyMath
        with self._env(**{agent_config.OPENROUTER_KEY_VAR: 'sk-test'}):
            with mock.patch.object(agent_config, '_codex_usable',
                                   side_effect=AssertionError('probed')):
                resolution = agent_config.resolve(agent_config.AgentRoute())
        self.assertEqual(resolution.backend, 'openrouter')

    def test_a_redirected_endpoint_selects_the_openai_compatible_backend(self):
        # no OpenRouter key, no Codex login: pointing the base URL at a local
        # server is itself a complete configuration
        with self._env(**{openrouter_backend.BASE_URL_VAR:
                          'http://localhost:11434/v1'}):
            with mock.patch.object(agent_config, '_codex_usable',
                                   side_effect=AssertionError('probed')):
                resolution = agent_config.resolve(agent_config.AgentRoute())
        self.assertEqual(resolution.backend, 'openrouter')
        self.assertIn('localhost:11434', resolution.reason)

    def test_a_redirected_endpoint_outranks_a_leftover_openrouter_key(self):
        # the reason must name where the run actually goes, not a key that
        # sits unused in .env
        with self._env(**{openrouter_backend.BASE_URL_VAR:
                          'http://localhost:11434/v1',
                          agent_config.OPENROUTER_KEY_VAR: 'sk-secret'}):
            resolution = agent_config.resolve(agent_config.AgentRoute())
            routing = agent_config.describe(agent_config.AgentRoute())
        self.assertEqual(resolution.backend, 'openrouter')
        self.assertNotIn(agent_config.OPENROUTER_KEY_VAR, resolution.reason)
        self.assertEqual(routing['endpoint'], 'http://localhost:11434/v1')

    def test_plain_openrouter_reports_no_redirected_endpoint(self):
        with self._env(**{agent_config.OPENROUTER_KEY_VAR: 'sk-test'}):
            routing = agent_config.describe(agent_config.AgentRoute())
        self.assertIsNone(routing['endpoint'])

    def test_codex_is_chosen_only_when_signed_in(self):
        with self._env():
            with mock.patch.object(agent_config, '_codex_usable',
                                   return_value=True):
                self.assertEqual(
                    agent_config.resolve(agent_config.AgentRoute()).backend,
                    'codex')
            with mock.patch.object(agent_config, '_codex_usable',
                                   return_value=False):
                with self.assertRaises(agent_do.DoAgentError) as caught:
                    agent_config.resolve(agent_config.AgentRoute())
        message = str(caught.exception)
        self.assertIn('login!', message)
        self.assertIn(agent_config.OPENROUTER_KEY_VAR, message)

    def test_an_api_key_account_is_never_auto_selected(self):
        # logged in, but on a credential this backend refuses to spend
        status = codex_transport.CodexAccountStatus(logged_in=True,
                                                    auth_mode='apiKey')
        with self._env():
            with mock.patch('agent_backends.codex.available',
                            return_value=True):
                with mock.patch('agent_backends.codex.account_status',
                                return_value=status):
                    with self.assertRaises(agent_do.DoAgentError):
                        agent_config.resolve(agent_config.AgentRoute())

    def test_a_failing_account_probe_never_escapes(self):
        with self._env():
            with mock.patch('agent_backends.codex.account_status',
                            side_effect=RuntimeError('runtime is down')):
                self.assertFalse(agent_config._codex_usable())

    def test_preview_never_starts_a_codex_runtime(self):
        with self._env():
            with mock.patch('agent_backends.codex.account_status',
                            side_effect=AssertionError('probed')):
                resolution = agent_config.preview(agent_config.AgentRoute())
        self.assertEqual(resolution.backend, 'auto')

    def test_preview_agrees_with_routing_once_the_account_is_known(self):
        # a signed-in user must not be told the notebook is unconfigured
        # while do! would in fact route to Codex
        status = codex_transport.CodexAccountStatus(logged_in=True,
                                                    auth_mode='chatgpt')
        with self._env():
            self.assertEqual(
                agent_config.preview(agent_config.AgentRoute()).backend,
                'auto')                       # nothing has looked yet
            agent_config.note_codex_account(status)
            with mock.patch('agent_backends.codex.available',
                            return_value=True):
                with mock.patch('agent_backends.codex.account_status',
                                side_effect=AssertionError('probed')):
                    resolution = agent_config.preview(
                        agent_config.AgentRoute())
        self.assertEqual(resolution.backend, 'codex')

    def test_signing_out_stops_advertising_codex(self):
        with self._env():
            with mock.patch('agent_backends.codex.available',
                            return_value=True):
                agent_config.note_codex_account(
                    codex_transport.CodexAccountStatus(logged_in=True,
                                                       auth_mode='chatgpt'))
                self.assertEqual(
                    agent_config.preview(agent_config.AgentRoute()).backend,
                    'codex')
                agent_config.note_codex_account(
                    codex_transport.CodexAccountStatus())
                self.assertEqual(
                    agent_config.preview(agent_config.AgentRoute()).backend,
                    'auto')

    def test_switching_backend_resets_the_model_to_that_backend_default(self):
        route = agent_config.AgentRoute(backend='openrouter',
                                        model='z-ai/glm-5.2',
                                        providers=('Cerebras',))
        moved = route.with_backend('codex')
        self.assertEqual(moved.backend, 'codex')
        self.assertEqual(moved.providers, ())
        self.assertIsNone(moved.model)      # the account's own default
        self.assertTrue(moved.experimental)

    def test_a_run_never_falls_back_to_the_other_backend(self):
        # a Codex failure must not silently create OpenRouter charges
        transport = codex_transport.TranscriptTransport(
            [], account=codex_transport.CodexAccountStatus())
        with mock.patch.dict(os.environ,
                             {agent_config.OPENROUTER_KEY_VAR: 'sk-test'}):
            with self.assertRaises(agent_do.DoAgentError):
                run_instruction('solve it',
                                backend=codex_backend.CodexBackend(
                                    transport=transport))

    def test_backend_completion_offers_the_three_names(self):
        code = 'backend! co'
        reply = agent_config.complete_backend_command(code, len(code))
        self.assertEqual(reply['matches'], ['codex'])
        self.assertEqual(code[reply['cursor_start']:reply['cursor_end']],
                         'co')
        self.assertIn('experimental',
                      reply['metadata']['_jupyter_types_experimental'][0]
                      ['type'])
        self.assertEqual(
            agent_config.complete_backend_command('backend! ', 9)['matches'],
            ['auto', 'openrouter', 'codex'])
        self.assertIsNone(
            agent_config.complete_backend_command('solve! x', 8))

    def test_login_completion_offers_its_options(self):
        code = 'login! st'
        reply = agent_config.complete_login_command(code, len(code))
        self.assertEqual(reply['matches'], ['status'])
        self.assertEqual(code[reply['cursor_start']:reply['cursor_end']],
                         'st')
        self.assertEqual(reply['status'], 'ok')
        self.assertEqual(
            agent_config.complete_login_command('login! ', 7)['matches'],
            ['device', 'status', 'logout'])
        self.assertIsNone(agent_config.complete_login_command('solve! x', 8))

    def test_every_completed_login_option_reaches_its_own_handler(self):
        # sharing one list makes the popup and the validation agree by
        # construction; what it cannot prevent is an option that passes
        # validation and then falls through to the wrong branch - a new
        # entry with no dispatch of its own would silently become a
        # browser sign-in. Assert where each offered option actually lands.
        import engine
        import mathShell
        import IPython.display
        engine.setHandler(lambda *objs, **kw: None)
        self.addCleanup(engine.setHandler, IPython.display.display)
        shell = mathShell.MathShell()
        expected = {'device': ('login', 'chatgptDeviceCode'),
                    'status': ('account_status', None),
                    'logout': ('logout', None),
                    '': ('login', 'chatgpt')}
        offered = agent_config.complete_login_command('login! ', 7)['matches']
        self.assertTrue(offered)
        for option in list(offered) + ['']:      # '' is the bare `login!`
            with self.subTest(option=option):
                self.assertIn(option, expected,
                              'a login! option with no expected handler')
                called = []
                patches = {name: mock.DEFAULT for name in
                           ('login', 'logout', 'account_status')}
                with mock.patch.object(mathShell.MathShell, '_render_account'):
                    with mock.patch.multiple('agent_backends.codex',
                                             **patches) as mocks:
                        shell.exec_login(option)
                        called = [name for name, m in mocks.items()
                                  if m.called]
                        handler, mode = expected[option]
                        self.assertEqual(called, [handler])
                        if mode is not None:
                            self.assertEqual(mocks['login'].call_args[0][0],
                                             mode)

    def test_completion_only_fires_at_an_argument_position(self):
        # the frontend opens the popup unprompted, so a line that merely
        # mentions a command must not produce completions
        for code in ('backend!', 'login!', 'x + backend! ', '$login! '):
            with self.subTest(code=code):
                self.assertIsNone(
                    agent_config.complete_backend_command(code, len(code)))
                self.assertIsNone(
                    agent_config.complete_login_command(code, len(code)))

    def test_completion_reads_the_line_holding_the_cursor(self):
        code = 'x^2\nlogin! dev'
        reply = agent_config.complete_login_command(code, len(code))
        self.assertEqual(reply['matches'], ['device'])
        self.assertEqual(code[reply['cursor_start']:reply['cursor_end']],
                         'dev')


class TestBackendCommand(unittest.TestCase):
    def setUp(self):
        import engine
        self.displays = []
        engine.setHandler(lambda *objs, **kw: self.displays.extend(objs))
        from mathShell import MathShell
        self.shell = MathShell()

    def tearDown(self):
        import engine
        import IPython.display
        engine.setHandler(IPython.display.display)

    def _html(self):
        return ''.join(getattr(d, 'data', str(d)) for d in self.displays)

    def test_backend_is_a_reserved_dispatcher_command(self):
        import prompt_commands
        self.assertIn('backend', prompt_commands.RESERVED)
        with self.assertRaises(ValueError):
            prompt_commands.parse_command(
                '---\nname: backend\ndescription: d\n---\n$ARGUMENTS',
                'backend')

    def test_selection_is_notebook_local_and_explicit(self):
        self.shell.exec('backend! codex', 1)
        self.assertEqual(self.shell.route.backend, 'codex')
        self.assertIn('experimental', self._html())
        from mathShell import MathShell
        self.assertEqual(MathShell().route.backend, 'auto')

    def test_a_fresh_notebook_carries_no_backend_specific_model(self):
        # an OpenRouter model id means nothing to Codex: until the notebook
        # chooses one, each backend supplies its own default, so
        # auto-resolution can never hand one backend the other's model
        self.assertIsNone(self.shell.route.model)
        self.shell.exec('backend! codex', 1)
        self.assertIsNone(self.shell.route.model)
        self.assertIn('codex default', self._html())

    def test_an_unknown_backend_is_refused(self):
        self.shell.exec('backend! anthropic', 1)
        self.assertIn('backend! error', self._html())
        self.assertEqual(self.shell.route.backend, 'auto')

    def test_bare_backend_shows_the_effective_routing_and_why(self):
        with mock.patch.dict(os.environ,
                             {agent_config.OPENROUTER_KEY_VAR: 'sk-test'}):
            self.shell.exec('backend!', 1)
        out = self._html()
        self.assertIn('openrouter', out)
        self.assertIn(agent_config.OPENROUTER_KEY_VAR, out)
        self.assertIn('agent model', out)

    def test_switching_backend_resets_the_model_and_notifies_once(self):
        seen = []
        self.shell.model_change_handler = seen.append
        self.shell.exec('model! z-ai/glm-5.2', 1)
        self.shell.exec('backend! codex', 2)
        self.assertEqual([route.backend for route in seen],
                         ['auto', 'codex'])
        self.assertIsNone(self.shell.route.model)
        self.assertEqual(self.shell.route.providers, ())

    def test_a_legacy_two_argument_handler_still_works(self):
        seen = []
        self.shell.model_change_handler = (
            lambda model, providers: seen.append((model, providers)))
        self.shell.exec('model! z-ai/glm-5.2', 1)
        self.assertEqual(seen, [('z-ai/glm-5.2', ('Cerebras', 'Fireworks'))])

    def test_codex_model_selection_uses_the_account_catalog(self):
        transport = codex_transport.TranscriptTransport([])
        codex_backend.set_runtime(transport)
        self.addCleanup(codex_backend.set_runtime, None)
        self.shell.exec('backend! codex', 1)
        self.displays.clear()
        self.shell.exec('model!', 2)
        self.assertIn('gpt-5.6-luna', self._html())
        self.displays.clear()
        self.shell.exec('model! gpt-5.6-luna', 3)
        self.assertEqual(self.shell.route.model, 'gpt-5.6-luna')
        self.displays.clear()
        # a model this account does not offer, and provider routing, are
        # both refused rather than silently sent
        self.shell.exec('model! z-ai/glm-5.2', 4)
        self.assertIn('model! error', self._html())
        self.displays.clear()
        self.shell.exec('model! gpt-5.6-luna, Cerebras', 5)
        self.assertIn('no provider routing', self._html())
        self.assertEqual(self.shell.route.model, 'gpt-5.6-luna')

    def test_the_kernel_payload_carries_backend_and_model(self):
        from toymathkernel import MathKernel
        kernel = mock.Mock(spec=['mathShell'])
        kernel.mathShell = self.shell
        with mock.patch.dict(os.environ,
                             {agent_config.OPENROUTER_KEY_VAR: 'sk-test'}):
            payload = MathKernel._model_payload(kernel)
        self.assertEqual(payload['backend'], 'openrouter')
        self.assertFalse(payload['experimental'])
        self.assertTrue(payload['model'])
        self.shell.exec('backend! codex', 1)
        payload = MathKernel._model_payload(kernel)
        self.assertEqual(payload['backend'], 'codex')
        self.assertTrue(payload['experimental'])
        self.assertEqual(payload['model'], 'codex default')


class TestLoginCommand(unittest.TestCase):
    def setUp(self):
        import engine
        self.displays = []
        engine.setHandler(lambda *objs, **kw: self.displays.extend(objs))
        from mathShell import MathShell
        self.shell = MathShell()
        self.transport = codex_transport.TranscriptTransport(
            account=codex_transport.CodexAccountStatus())
        codex_backend.set_runtime(self.transport)
        self.addCleanup(codex_backend.set_runtime, None)
        # every test in this class exercises login!; none of them may open a
        # browser, so the default is "no browser available" and the tests
        # that care about the opened path say so explicitly
        import mathShell
        patch = mock.patch.object(mathShell, '_open_browser',
                                  return_value=False)
        self.opened = patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        import engine
        import IPython.display
        engine.setHandler(IPython.display.display)

    def _html(self):
        return ''.join(getattr(d, 'data', str(d)) for d in self.displays)

    def test_login_is_a_reserved_dispatcher_command(self):
        import prompt_commands
        self.assertIn('login', prompt_commands.RESERVED)
        with self.assertRaises(ValueError):
            prompt_commands.parse_command(
                '---\nname: login\ndescription: d\n---\n$ARGUMENTS', 'login')

    def test_browser_flow_keeps_the_one_time_link_out_of_the_notebook(self):
        # Jupyter persists cell output into the .ipynb, so the sign-in URL
        # goes to the OS browser instead of into the saved file
        import mathShell
        with mock.patch.object(mathShell, '_open_browser',
                               return_value=True) as opened:
            self.shell.exec('login!', 1)
        opened.assert_called_once()
        self.assertTrue(opened.call_args[0][0].startswith('https://'))
        out = self._html()
        self.assertNotIn('auth.openai.test', out)
        self.assertIn('not stored in this notebook', out)
        self.assertIn('signed in', out)
        self.assertIn('plus plan', out)

    def test_without_a_browser_the_link_is_shown_and_labelled(self):
        self.shell.exec('login!', 1)               # setUp: no browser
        out = self._html()
        self.assertIn('auth.openai.test', out)     # the only way in
        self.assertIn('No browser could be opened', out)
        self.opened.assert_called_once()

    def test_no_test_can_reach_a_real_browser(self):
        # the module-level backstop behind the per-class patch: a forgotten
        # patch must not open a window at auth.openai.test
        import webbrowser
        self.assertFalse(webbrowser.open('https://auth.openai.test/activate'))
        self.assertIsInstance(webbrowser.open, mock.MagicMock)

    def test_no_test_can_reach_the_real_codex_home(self):
        # backend auto-resolution probes for a signed-in account, which
        # starts an app-server against the user's own authenticated home
        # and keeps it for the rest of the process. Not from a test suite.
        home = codex_backend.home_path()
        self.assertEqual(home, _SANDBOX_HOME)
        self.assertNotEqual(
            os.path.realpath(home),
            os.path.realpath(os.path.expanduser('~/.toymath/codex-home')))

    def test_device_flow_renders_the_code(self):
        self.shell.exec('login! device', 1)
        out = self._html()
        self.assertIn('ABCD-EFGH', out)
        self.assertIn('auth.openai.test/device', out)

    def test_a_spent_challenge_is_cleared_from_the_cell(self):
        # the device code has to be readable while the user types it, and
        # must not survive into the saved notebook afterwards
        import engine
        seen = []
        engine.setHandler(lambda *objs, **kw: seen.append((objs, kw)))
        self.shell.exec('login! device', 1)
        challenge, account = seen[0], seen[-1]
        self.assertIn('ABCD-EFGH', challenge[0][0].data)
        self.assertFalse(challenge[1].get('clear_output'))
        self.assertIn('signed in', account[0][0].data)
        self.assertTrue(account[1].get('clear_output'))

    def test_an_interrupted_login_also_clears_the_challenge(self):
        import engine
        seen = []
        engine.setHandler(lambda *objs, **kw: seen.append((objs, kw)))
        with mock.patch.object(codex_backend, 'login',
                               side_effect=KeyboardInterrupt):
            self.shell.exec('login! device', 1)
        self.assertIn('login cancelled', seen[-1][0][0].data)
        self.assertTrue(seen[-1][1].get('clear_output'))

    def test_status_and_logout(self):
        self.shell.exec('login! status', 1)
        self.assertIn('signed out', self._html())
        self.displays.clear()
        self.shell.exec('login!', 2)
        self.displays.clear()
        self.shell.exec('login! logout', 3)
        self.assertIn('signed out', self._html())

    def test_unknown_option_is_refused_with_the_usage(self):
        self.shell.exec('login! sk-secret-token-value', 1)
        out = self._html()
        self.assertIn('login! error', out)
        self.assertIn('login! device', out)
        self.assertEqual(self.transport.logins, [])
        self.assertNotIn('sk-secret-token-value', repr(self.transport))

    def test_signing_in_never_changes_the_selected_model_routing(self):
        before = (self.shell.model_name, self.shell.model_providers)
        self.shell.exec('login!', 1)
        self.assertEqual((self.shell.model_name, self.shell.model_providers),
                         before)
        self.assertEqual(len(self.shell.ledger.steps), 0)

    def test_signing_in_republishes_the_effective_backend(self):
        # on `backend! auto` the effective backend is a function of the
        # login state, so it can change without the route changing at all.
        # The toolbar would otherwise keep advertising the pre-login answer.
        agent_config.forget_codex_account()
        self.addCleanup(agent_config.forget_codex_account)
        published = []
        self.shell.model_change_handler = published.append
        env = {key: value for key, value in os.environ.items()
               if key not in (agent_config.BACKEND_VAR,
                              agent_config.OPENROUTER_KEY_VAR)}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch('agent_backends.codex.available',
                            return_value=True):
                self.assertEqual(self.shell.backend_name, 'auto')
                self.shell.exec('login!', 1)
                self.assertEqual(self.shell.backend_name, 'codex')
                self.assertEqual(len(published), 1)
                self.displays.clear()
                self.shell.exec('login! logout', 2)
                self.assertEqual(self.shell.backend_name, 'auto')
                self.assertEqual(len(published), 2)

    def test_an_unusable_account_is_reported_as_such(self):
        self.transport.account = codex_transport.CodexAccountStatus(
            logged_in=True, auth_mode='apiKey')
        self.shell.exec('login! status', 1)
        out = self._html()
        self.assertIn('apiKey', out)
        self.assertIn('ToyMath does not use', out)
        self.assertNotIn('<strong>signed in</strong>', out)


class TestToolDispatcher(unittest.TestCase):
    def _dispatcher(self, session=None, **kwargs):
        session = session if session is not None else DoSession()
        cancellation = kwargs.pop('cancellation', None)
        return session, agent_base.ToolDispatcher(
            agent_do.make_tool_bindings(session),
            cancellation or agent_base.CancellationToken(), **kwargs)

    def test_a_refusing_tactic_is_still_a_successful_dispatch(self):
        # a ToyMath record answering {"ok": false} ran and rejected the
        # request; only transport-level failures are dispatch errors, so
        # the model can still repair
        _, dispatcher = self._dispatcher()
        record = json.loads(dispatcher.dispatch(
            'run_tactic', {'tactic': 'expand', 'arguments': ['\\frac{1}{']}))
        self.assertFalse(record['ok'])
        self.assertTrue(record.get('error'))

    def test_a_queued_serialized_call_does_not_start_after_cancellation(self):
        # the checks before the queue happened while the run was alive; a
        # call that waited behind an executing handler must re-check, or it
        # begins work on a session that was stopped meanwhile
        cancellation = agent_base.CancellationToken()
        session, dispatcher = self._dispatcher(cancellation=cancellation,
                                               serialize=True)
        holding = threading.Event()
        release = threading.Event()
        started = []

        def slow(payload):
            started.append('first')
            holding.set()
            release.wait(5)
            return '{"ok": true}'

        def second(payload):
            started.append('second')
            return '{"ok": true}'

        binding = dispatcher.bindings['comment']
        dispatcher.bindings['first'] = agent_base.ToolBinding(
            'first', 'd', binding.input_schema, slow)
        dispatcher.bindings['second'] = agent_base.ToolBinding(
            'second', 'd', binding.input_schema, second)

        replies = {}
        args = {'text': 'x'}
        threading.Thread(
            target=lambda: replies.setdefault(
                'first', dispatcher.dispatch('first', args)),
            daemon=True).start()
        self.assertTrue(holding.wait(5))
        queued = threading.Thread(
            target=lambda: replies.setdefault(
                'second', dispatcher.dispatch('second', args)),
            daemon=True)
        queued.start()
        time.sleep(0.1)                      # 'second' is now behind the lock
        cancellation.cancel(agent_base.USER)
        release.set()
        queued.join(5)

        self.assertEqual(started, ['first'])
        self.assertTrue(json.loads(replies['second'])['cancelled'])

    def test_unknown_tool_and_malformed_arguments_are_distinguished(self):
        _, dispatcher = self._dispatcher()
        with self.assertRaises(agent_base.UnknownToolError):
            dispatcher.dispatch('frobnicate', {})
        with self.assertRaises(agent_base.ToolArgumentError):
            dispatcher.dispatch('run_tactic', '{not json')
        with self.assertRaises(agent_base.ToolArgumentError):
            dispatcher.dispatch('run_tactic', {'tactic': 'expand'})

    def test_no_new_tool_call_starts_after_cancellation(self):
        session, dispatcher = self._dispatcher()
        dispatcher.dispatch('run_tactic',
                            {'tactic': 'expand', 'arguments': ['(x+1)^2']})
        dispatcher.cancellation.cancel('user')
        session.close('user')
        reply = json.loads(dispatcher.dispatch(
            'run_tactic', {'tactic': 'expand', 'arguments': ['(x+2)^2']}))
        self.assertFalse(reply['ok'])
        self.assertTrue(reply['cancelled'])
        self.assertEqual(len(session.ledger.steps), 1)

    def test_tool_budget_stops_the_run_and_keeps_earlier_steps(self):
        session, dispatcher = self._dispatcher(
            budget=agent_base.AgentBudget(max_tool_calls=1))
        first = json.loads(dispatcher.dispatch(
            'run_tactic', {'tactic': 'expand', 'arguments': ['(x+1)^2']}))
        self.assertTrue(first['ok'])
        second = json.loads(dispatcher.dispatch(
            'run_tactic', {'tactic': 'expand', 'arguments': ['(x+2)^2']}))
        self.assertFalse(second['ok'])
        self.assertEqual(dispatcher.cancellation.reason, agent_base.BUDGET)
        self.assertEqual(len(session.ledger.steps), 1)


class TestCancellationToken(unittest.TestCase):
    def test_first_reason_wins_and_cancel_is_idempotent(self):
        token = agent_base.CancellationToken()
        self.assertFalse(token.cancelled)
        self.assertTrue(token.cancel('user'))
        self.assertFalse(token.cancel('budget'))
        self.assertTrue(token.cancelled)
        self.assertEqual(token.reason, 'user')

    def test_listeners_run_once_and_never_lose_a_race(self):
        token = agent_base.CancellationToken()
        seen = []
        token.add_listener(seen.append)
        token.cancel('budget')
        token.cancel('user')
        self.assertEqual(seen, ['budget'])
        late = []
        token.add_listener(late.append)   # registered after the fact
        self.assertEqual(late, ['budget'])

    def test_a_failing_listener_cannot_break_cancellation(self):
        token = agent_base.CancellationToken()
        seen = []

        def boom(reason):
            raise RuntimeError('listener bug')
        token.add_listener(boom)
        token.add_listener(seen.append)
        self.assertTrue(token.cancel('user'))
        self.assertEqual(seen, ['user'])

    def test_reason_maps_to_the_outcome_status(self):
        self.assertEqual(agent_base.status_for_reason('user'),
                         agent_base.INTERRUPTED)
        self.assertEqual(agent_base.status_for_reason('budget'),
                         agent_base.BUDGET_EXHAUSTED)
        self.assertEqual(agent_base.status_for_reason('capability'),
                         agent_base.CAPABILITY_VIOLATION)


class TestSessionClosure(unittest.TestCase):
    """Closing is the ledger mutation boundary of a cancelled run."""

    def setUp(self):
        self.ledger = Ledger()
        self.session = DoSession(ledger=self.ledger,
                                 plot_backend=FakePlotBackend(),
                                 tikz_backend=FakeTikzBackend())
        self.api = make_api(self.session)

    def test_close_is_idempotent_and_records_the_first_reason(self):
        self.assertTrue(self.session.close('user'))
        self.assertFalse(self.session.close('budget'))
        self.assertEqual(self.session.close_reason, 'user')

    def test_committed_steps_survive_closure_and_still_replay(self):
        self.assertTrue(json.loads(self.api['expand']('(x+1)^2'))['ok'])
        self.session.close('user')
        refused = json.loads(self.api['expand']('(x+2)^2'))
        self.assertFalse(refused['ok'])
        self.assertTrue(refused['cancelled'])
        self.assertEqual(len(self.ledger.steps), 1)
        self.assertEqual(self.ledger.replay()['status'], 'verified')

    def test_every_late_callback_is_rejected(self):
        self.assertTrue(json.loads(self.api['expand']('(x+1)^2'))['ok'])
        self.session.close('user')
        for reply in (self.api['comment']('a late note'),
                      self.api['claim']('x = x'),
                      self.api['conclude']('c1', ['s1']),
                      self.api['set_result']('x^{2}+2x+1'),
                      self.api['set_open']('ran out of moves'),
                      self.api['plot']('import matplotlib', 'late'),
                      self.api['tikz']('\\begin{document}', 'late')):
            record = json.loads(reply)
            self.assertFalse(record['ok'], reply)
        self.assertIsNone(self.session.result_override)
        self.assertIsNone(self.session.open_selection)
        self.assertEqual(len(self.ledger.steps), 1)
        self.assertEqual(len(self.ledger.claims), 0)
        self.assertEqual(len(self.ledger.selections), 0)

    def test_a_figure_reached_after_closure_is_never_rendered(self):
        shown = []
        session = DoSession(on_plot=lambda *a: shown.append(a),
                            plot_backend=FakePlotBackend())
        session.close('user')
        make_api(session)['plot']('import matplotlib', 'late')
        self.assertEqual(shown, [])

    def test_figure_delivery_is_serialised_with_session_closure(self):
        # a figure is cell output, so it needs the ledger's own boundary.
        # Reading `closed` and then calling back leaves a window for a
        # cancellation to land in between and paint into a cell that has
        # already reported itself stopped; holding the lock closes it.
        shown = []
        session = DoSession(on_plot=lambda *a: shown.append(a),
                            plot_backend=FakePlotBackend())
        delivered = []
        entered = threading.Event()

        def deliver():
            entered.set()
            delivered.append(session.deliver_figure(
                'racy', [{'kind': 'png', 'data': 'x'}]))

        with session._lock:                  # a cancellation, mid-flight
            worker = threading.Thread(target=deliver)
            worker.start()
            self.assertTrue(entered.wait(5))
            time.sleep(0.1)
            self.assertEqual(delivered, [])  # waiting on the boundary
            session.close('user')
        worker.join(5)
        self.assertEqual(delivered, [False])
        self.assertEqual(shown, [])

    def test_a_record_that_wins_the_lock_commits_completely(self):
        # invariant: cancellation never rolls back a mechanically checked
        # step, and never leaves half of one behind
        holding = threading.Event()
        release = threading.Event()
        original = self.ledger.record

        def slow_record(*args, **kwargs):
            holding.set()
            release.wait(5)
            return original(*args, **kwargs)

        self.ledger.record = slow_record
        worker = threading.Thread(
            target=lambda: self.api['expand']('(x+1)^2'))
        worker.start()
        self.assertTrue(holding.wait(5))
        closer = threading.Thread(target=self.session.close, args=('user',))
        closer.start()               # blocks on the session lock
        release.set()
        worker.join(5)
        closer.join(5)
        self.assertEqual(len(self.ledger.steps), 1)
        self.assertTrue(self.session.closed)
        self.assertEqual(self.ledger.replay()['status'], 'verified')


class FakeRunHandle(agent_base.ThreadedRunHandle):
    """A run whose completion, cancellation, and stubbornness are scripted."""

    def __init__(self, request, outcome=None, acknowledge=True, delay=0.0):
        super(FakeRunHandle, self).__init__(request)
        self.outcome = outcome or agent_base.AgentOutcome(final_text='done')
        self.acknowledge = acknowledge
        self.delay = delay
        self.finish = threading.Event()
        self.cancels = []
        self.contained = []

    def run(self):
        self.finish.wait(10)
        return self.outcome

    def request_cancel(self, reason):
        self.cancels.append(reason)
        if self.acknowledge:
            self.outcome = agent_base.AgentOutcome(
                status=agent_base.INTERRUPTED)
            self.finish.set()

    def abandon(self, reason):
        self.contained.append(reason)


class TestWaitInterruptibly(unittest.TestCase):
    """The Jupyter interruption bridge, without the kernel."""

    def _request(self, cancellation):
        session = DoSession()
        return agent_base.AgentRequest(
            instruction='x', developer_instructions='y',
            dispatcher=agent_base.ToolDispatcher(
                agent_do.make_tool_bindings(session), cancellation),
            cancellation=cancellation)

    def _run(self, handle, cancellation, interrupts=(), **kwargs):
        """Deliver a KeyboardInterrupt on the given poll iterations."""
        polls = {'n': 0}
        real_wait = handle.wait

        def wait(timeout=None):
            if timeout:   # only the polling waits; wait(0) is the race check
                polls['n'] += 1
                if polls['n'] in interrupts:
                    raise KeyboardInterrupt
            return real_wait(min(timeout or 0, 0.01))

        handle.wait = wait
        return agent_base.wait_interruptibly(handle, cancellation,
                                             poll=0.01, **kwargs)

    def test_interrupt_cancels_exactly_this_run_once(self):
        cancellation = agent_base.CancellationToken()
        handle = FakeRunHandle(self._request(cancellation)).start()
        outcome = self._run(handle, cancellation, interrupts=(1, 2))
        self.assertEqual(outcome.status, agent_base.INTERRUPTED)
        self.assertEqual(handle.cancels, ['user'])   # idempotent
        self.assertEqual(cancellation.reason, 'user')

    def test_a_provider_that_won_the_race_stays_completed(self):
        cancellation = agent_base.CancellationToken()
        handle = FakeRunHandle(self._request(cancellation))
        handle.finish.set()
        handle.start()
        handle._done.wait(5)
        outcome = self._run(handle, cancellation, interrupts=(1,))
        self.assertEqual(outcome.status, agent_base.COMPLETED)
        self.assertEqual(handle.cancels, [])
        self.assertFalse(cancellation.cancelled)

    def test_a_late_completion_cannot_undo_cancellation(self):
        cancellation = agent_base.CancellationToken()
        handle = FakeRunHandle(self._request(cancellation),
                               acknowledge=False)

        def late_finish(reason):
            handle.outcome = agent_base.AgentOutcome(
                status=agent_base.COMPLETED, final_text='the answer is 4')
            handle.finish.set()
        handle.request_cancel = late_finish
        handle.start()
        outcome = self._run(handle, cancellation, interrupts=(1,))
        self.assertEqual(outcome.status, agent_base.INTERRUPTED)
        self.assertEqual(outcome.final_text, '')

    def test_a_run_that_ignores_cancellation_is_contained(self):
        cancellation = agent_base.CancellationToken()
        handle = FakeRunHandle(self._request(cancellation),
                               acknowledge=False).start()
        outcome = self._run(handle, cancellation, interrupts=(1,),
                            grace_period=0.05)
        self.assertEqual(outcome.status, agent_base.INTERRUPTED)
        self.assertTrue(outcome.metadata['grace_exceeded'])
        self.assertEqual(handle.contained, ['user'])
        handle.finish.set()

    def test_wall_clock_budget_is_exhaustion_not_interruption(self):
        cancellation = agent_base.CancellationToken()
        handle = FakeRunHandle(self._request(cancellation)).start()
        outcome = self._run(
            handle, cancellation,
            budget=agent_base.AgentBudget(max_seconds=0.01))
        self.assertEqual(outcome.status, agent_base.BUDGET_EXHAUSTED)
        self.assertEqual(handle.cancels, ['budget'])

    def test_a_stubborn_run_past_its_wall_clock_is_still_contained(self):
        # the grace deadline must be armed once. Renewing it on every poll
        # after the run deadline passed would keep a provider that ignores
        # cancellation here forever - the cell would never come back.
        cancellation = agent_base.CancellationToken()
        handle = FakeRunHandle(self._request(cancellation),
                               acknowledge=False).start()
        outcome = self._run(
            handle, cancellation, grace_period=0.05,
            budget=agent_base.AgentBudget(max_seconds=0.01))
        self.assertEqual(outcome.status, agent_base.BUDGET_EXHAUSTED)
        self.assertTrue(outcome.metadata['grace_exceeded'])
        self.assertEqual(handle.contained, ['budget'])
        handle.finish.set()

    def test_pressing_stop_again_does_not_extend_the_grace_period(self):
        # every poll delivers another Stop, as a user leaning on the button
        # would. The deadline is armed once, so containment still arrives;
        # re-arming it per interrupt would postpone it indefinitely.
        cancellation = agent_base.CancellationToken()
        handle = FakeRunHandle(self._request(cancellation),
                               acknowledge=False).start()
        ticks = {'n': 0}

        def clock():
            ticks['n'] += 1
            if ticks['n'] > 200:
                raise AssertionError('the grace deadline was never reached')
            return ticks['n'] * 0.01

        real_wait = handle.wait

        def wait(timeout=None):
            if timeout:
                raise KeyboardInterrupt
            return real_wait(0)

        handle.wait = wait
        outcome = agent_base.wait_interruptibly(
            handle, cancellation, poll=0.01, grace_period=0.05, clock=clock)
        self.assertEqual(outcome.status, agent_base.INTERRUPTED)
        self.assertEqual(handle.contained, ['user'])
        self.assertEqual(handle.cancels, ['user'])       # idempotent
        handle.finish.set()


class TestRunCancellation(unittest.TestCase):
    """run_instruction end to end: what a stopped cell keeps and refuses."""

    def _backend(self, run):
        """A backend whose worker body is `run(request, handle)`."""
        class Handle(agent_base.ThreadedRunHandle):
            def __init__(self, request):
                super(Handle, self).__init__(request)
                self.stopped = threading.Event()

            def run(self):
                return run(self.request, self)

            def request_cancel(self, reason):
                self.stopped.set()

        class Backend(object):
            name = 'fake'

            def start(self, request):
                self.handle = Handle(request)
                return self.handle.start()

        return Backend()

    def _tactic(self, request, expr):
        return json.loads(request.dispatcher.dispatch(
            'run_tactic', {'tactic': 'expand', 'arguments': [expr]}))

    def test_interrupted_run_keeps_its_steps_and_chains_nothing(self):
        ledger = Ledger()

        def run(request, handle):
            self._tactic(request, '(x+1)^2')
            _thread.interrupt_main()     # exactly what Jupyter Stop does
            handle.stopped.wait(5)
            # a late tool call after the user pressed Stop
            self._tactic(request, '(x+9)^2')
            return agent_base.AgentOutcome(final_text='never shown')

        backend = self._backend(run)
        with mock.patch.object(agent_do, 'resolve_backend',
                               lambda **kwargs: backend):
            try:
                res = run_instruction('expand it', ledger=ledger,
                                      grace_period=0.5)
            except KeyboardInterrupt:  # pragma: no cover - bridge failure
                self.fail('the interrupt escaped run_instruction')
        self.assertFalse(res['ok'])
        self.assertEqual(res['status'], 'interrupted')
        self.assertTrue(res['cancelled'])
        self.assertIsNone(res['final_result'])
        self.assertIsNone(res['summary'])
        # the step committed before the stop is kept, replays, and is
        # offered only as labelled partial work
        self.assertEqual([s['op'] for s in res['steps']], ['expand'])
        self.assertEqual(res['partial_result'], 'x^{2}+2x+1')
        self.assertEqual(res['partial_provenance']['method'], 'last-step')
        self.assertEqual(len(ledger.steps), 1)
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_a_designated_result_before_the_stop_is_not_chainable(self):
        def run(request, handle):
            self._tactic(request, '(x+1)^2')
            request.dispatcher.dispatch('set_result',
                                        {'expr': 'x^{2}+2x+1'})
            _thread.interrupt_main()
            handle.stopped.wait(5)
            return agent_base.AgentOutcome(final_text='ignored')

        backend = self._backend(run)
        with mock.patch.object(agent_do, 'resolve_backend',
                               lambda **kwargs: backend):
            res = run_instruction('expand it', grace_period=0.5)
        self.assertIsNone(res['final_result'])
        self.assertEqual(res['partial_result'], 'x^{2}+2x+1')
        self.assertEqual(res['partial_provenance']['status'], 'verified')

    def test_budget_exhaustion_is_its_own_status(self):
        def run(request, handle):
            self._tactic(request, '(x+1)^2')
            self._tactic(request, '(x+2)^2')      # over the tool budget
            handle.stopped.wait(5)
            return agent_base.AgentOutcome(final_text='ignored')

        backend = self._backend(run)
        with mock.patch.object(agent_do, 'resolve_backend',
                               lambda **kwargs: backend):
            res = run_instruction(
                'expand forever', grace_period=0.5,
                budget=agent_base.AgentBudget(max_tool_calls=1))
        self.assertEqual(res['status'], 'budget_exhausted')
        self.assertTrue(res['cancelled'])
        self.assertIsNone(res['final_result'])
        self.assertEqual(len(res['steps']), 1)

    def test_an_ordinary_run_still_reports_completed(self):
        res = run_instruction('expand it', model=ScriptedModel([
            [tool_call('expand', {'expr': '(x+1)^2'}, 'c1')],
            [message('done')]]))
        self.assertTrue(res['ok'])
        self.assertEqual(res['status'], 'completed')
        self.assertNotIn('cancelled', res)
        self.assertEqual(res['summary'], 'done')


class TestOpenRouterBackendCancellation(unittest.TestCase):
    def test_cancel_stops_the_running_agent_task(self):
        started = threading.Event()

        class BlockingModel(ScriptedModel):
            def __init__(self):
                ScriptedModel.__init__(self, [])

            async def get_response(self, *args, **kwargs):
                import asyncio
                started.set()
                await asyncio.sleep(30)   # pragma: no cover - cancelled
                raise AssertionError('the provider call was not cancelled')

        session = DoSession()
        cancellation = agent_base.CancellationToken()
        request = agent_base.AgentRequest(
            instruction='hang', developer_instructions='rules',
            dispatcher=agent_base.ToolDispatcher(
                agent_do.make_tool_bindings(session), cancellation),
            cancellation=cancellation)
        handle = openrouter_backend.OpenRouterBackend(
            model=BlockingModel()).start(request)
        self.assertTrue(started.wait(5))
        handle.cancel('user')
        outcome = handle.wait(5)
        self.assertIsNotNone(outcome, 'the run never acknowledged the stop')
        self.assertEqual(outcome.status, agent_base.INTERRUPTED)


class TestOpenAICompatibleEndpoint(unittest.TestCase):
    """The transport is chat-completions; only the endpoint is OpenRouter's.

    `TOYMATH_OPENAI_BASE_URL` points the same backend at Ollama, vLLM, or any
    other OpenAI-compatible server. Nothing here contacts a network: building
    the client is enough to see where it would go, and with which credential.
    """

    def _env(self, **overrides):
        env = {key: value for key, value in os.environ.items()
               if key not in (openrouter_backend.API_KEY_VAR,
                              openrouter_backend.BASE_URL_VAR,
                              openrouter_backend.ENDPOINT_KEY_VAR)}
        env.update(overrides)
        return mock.patch.dict(os.environ, env, clear=True)

    def _build(self, model_name=None):
        # the tracing global is process-wide; this test is about the endpoint
        with mock.patch('agents.set_tracing_disabled'):
            return openrouter_backend.build_model(model_name)

    def test_the_default_endpoint_is_openrouter_and_needs_its_key(self):
        with self._env(**{openrouter_backend.API_KEY_VAR: 'sk-or-test'}):
            built = self._build()
        client = built.toymath_client
        self.assertEqual(str(client.base_url).rstrip('/'),
                         openrouter_backend.OPENROUTER_BASE_URL)
        self.assertEqual(client.api_key, 'sk-or-test')
        with self._env():
            with self.assertRaises(agent_do.DoAgentError) as caught:
                openrouter_backend.build_model()
        self.assertIn(openrouter_backend.API_KEY_VAR, str(caught.exception))
        self.assertIn(openrouter_backend.BASE_URL_VAR, str(caught.exception))

    def test_a_redirected_endpoint_builds_a_client_there_without_a_key(self):
        with self._env(**{openrouter_backend.BASE_URL_VAR:
                          'http://localhost:11434/v1'}):
            built = self._build('qwen3:14b')
        self.assertEqual(built.model, 'qwen3:14b')
        self.assertEqual(str(built.toymath_client.base_url).rstrip('/'),
                         'http://localhost:11434/v1')

    def test_the_openrouter_key_never_reaches_a_redirected_endpoint(self):
        # THE landmine: an .env holding OPEN_ROUTER is the normal case, so a
        # generic key fallback would post a paid credential to whatever host
        # the base URL names. OPENAI_API_KEY must not leak in either - the
        # client would read it from the environment if we passed None.
        with self._env(**{openrouter_backend.BASE_URL_VAR:
                          'http://localhost:11434/v1',
                          openrouter_backend.API_KEY_VAR: 'sk-or-secret',
                          'OPENAI_API_KEY': 'sk-openai-secret'}):
            built = self._build()
        self.assertEqual(built.toymath_client.api_key,
                         openrouter_backend.PLACEHOLDER_KEY)

    def test_a_redirected_endpoint_may_carry_its_own_credential(self):
        with self._env(**{openrouter_backend.BASE_URL_VAR:
                          'https://gateway.example/v1',
                          openrouter_backend.ENDPOINT_KEY_VAR: 'sk-gateway'}):
            built = self._build()
        self.assertEqual(built.toymath_client.api_key, 'sk-gateway')

    def test_a_trailing_slash_still_reads_as_the_same_endpoint(self):
        with self._env(**{openrouter_backend.BASE_URL_VAR:
                          openrouter_backend.OPENROUTER_BASE_URL + '/'}):
            self.assertIsNone(openrouter_backend.custom_base_url())

    def test_provider_order_is_not_sent_to_a_redirected_endpoint(self):
        # `provider` is an OpenRouter routing extension, not chat-completions
        settings = []

        class RecordingModel(ScriptedModel):
            async def get_response(self, system_instructions, input,
                                   model_settings, *args, **kwargs):
                settings.append(model_settings)
                return await super().get_response(
                    system_instructions, input, model_settings,
                    *args, **kwargs)

        with self._env(**{openrouter_backend.BASE_URL_VAR:
                          'http://localhost:11434/v1'}):
            res = run_instruction(
                'say hello', model=RecordingModel([[message('hello')]]),
                providers=('Cerebras', 'Fireworks'))
        self.assertTrue(res['ok'])
        self.assertIsNone(getattr(settings[0], 'extra_body', None))


class TestObservability(unittest.TestCase):
    """Langfuse tracing is opt-in, non-fatal, and never touches the ledger.

    Runs fully offline: a Langfuse client backed by an in-memory OTEL
    exporter captures spans without any network or credentials.
    """

    def test_env_toggle_parsing(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(observability.ENABLE_VAR, None)
            self.assertFalse(observability.is_enabled())
        for on in ('1', 'true', 'on', 'yes', 'ON', 'Enabled'):
            with mock.patch.dict(os.environ,
                                 {observability.ENABLE_VAR: on}):
                self.assertTrue(observability.is_enabled(), on)
        for off in ('0', 'false', 'off', 'no', '', 'maybe'):
            with mock.patch.dict(os.environ,
                                 {observability.ENABLE_VAR: off}):
                self.assertFalse(observability.is_enabled(), off)

    def test_disabled_is_a_noop_and_run_is_unaffected(self):
        observability._reset_for_tests()
        self.addCleanup(observability._reset_for_tests)
        with mock.patch.dict(os.environ, {observability.ENABLE_VAR: 'off'}):
            self.assertFalse(observability.setup())
        self.assertFalse(observability.active())
        with observability.trace_run('anything') as span:
            self.assertIsNone(span)  # a no-op context manager
        observability.set_output(None, 'ignored')  # must not raise
        observability.flush()  # must not raise
        res = run_instruction(
            'solve 2x + 3 = 7 for x',
            model=ScriptedModel([list(t) for t in SOLVE_SCRIPT]))
        self.assertTrue(res['ok'])
        self.assertEqual(res['final_result'], '2x = 4')

    def test_build_model_tracing_toggle_follows_the_instrumentor(self):
        # THE landmine: the OpenInference instrumentor rides the Agents-SDK
        # tracing pipeline, so while it is attached that pipeline must stay
        # ENABLED. The predicate is `instrumented`, not `active`: a Langfuse
        # client without the instrumentor (a Codex-only install) would
        # otherwise leave the pipeline exporting to OpenAI's backend.
        for attached, expected_disabled in [(True, False), (False, True)]:
            calls = []
            with mock.patch.dict(os.environ, {'OPEN_ROUTER': 'sk-test'}), \
                 mock.patch('agents.set_tracing_disabled',
                            side_effect=calls.append), \
                 mock.patch.object(observability, 'instrumented',
                                   return_value=attached):
                openrouter_backend.build_model()
            self.assertEqual(calls, [expected_disabled], attached)

    def test_tracing_survives_a_missing_agents_instrumentor(self):
        # a Codex-only installation has no Agents SDK to instrument; that
        # must cost the nested spans, not the whole trace
        observability._reset_for_tests()
        self.addCleanup(observability._reset_for_tests)
        client = mock.Mock()
        with mock.patch.object(observability, '_instrument',
                               side_effect=ImportError('no openai-agents')):
            self.assertTrue(observability.setup(client=client))
        self.assertTrue(observability.active())
        self.assertFalse(observability.instrumented())

    def test_build_model_accepts_notebook_override(self):
        with mock.patch.dict(os.environ, {'OPEN_ROUTER': 'sk-test'}), \
             mock.patch.object(observability, 'active', return_value=False):
            built = openrouter_backend.build_model('z-ai/glm-5.2')
        self.assertEqual(built.model, 'z-ai/glm-5.2')

    def test_provider_order_reaches_model_settings_extra_body(self):
        settings = []

        class RecordingModel(ScriptedModel):
            async def get_response(self, system_instructions, input,
                                   model_settings, *args, **kwargs):
                settings.append(model_settings)
                return await super().get_response(
                    system_instructions, input, model_settings,
                    *args, **kwargs)

        res = run_instruction(
            'say hello', model=RecordingModel([[message('hello')]]),
            providers=('Cerebras', 'Fireworks'))
        self.assertTrue(res['ok'])
        self.assertEqual(settings[0].extra_body, {
            'provider': {
                'order': ['Cerebras', 'Fireworks'],
                'allow_fallbacks': False,
            },
        })

    def test_active_run_emits_one_nested_langfuse_trace(self):
        import asyncio

        import httpx
        from agents import (OpenAIChatCompletionsModel,
                            set_tracing_disabled)
        from langfuse import Langfuse
        from openai import AsyncOpenAI
        from openinference.instrumentation.openai_agents import (
            OpenAIAgentsInstrumentor)
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter)

        scripted_responses = []
        for index, turn in enumerate(SOLVE_SCRIPT):
            item = turn[0]
            if isinstance(item, ResponseFunctionToolCall):
                message_data = {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [{
                        'id': item.call_id,
                        'type': 'function',
                        'function': {
                            'name': item.name,
                            'arguments': item.arguments,
                        },
                    }],
                }
                finish_reason = 'tool_calls'
            else:
                message_data = {
                    'role': 'assistant',
                    'content': item.content[0].text,
                }
                finish_reason = 'stop'
            scripted_responses.append({
                'id': f'chatcmpl-test-{index}',
                'object': 'chat.completion',
                'created': 0,
                'model': 'test/provider-model',
                'choices': [{
                    'index': 0,
                    'message': message_data,
                    'finish_reason': finish_reason,
                }],
                'usage': {
                    'prompt_tokens': 7,
                    'completion_tokens': 3,
                    'total_tokens': 10,
                },
            })

        requests = []

        def handle_request(request):
            requests.append(request)
            if not scripted_responses:
                raise AssertionError('unexpected extra model request')
            return httpx.Response(200, json=scripted_responses.pop(0))

        exporter = InMemorySpanExporter()
        lf = Langfuse(public_key='pk-lf-test', secret_key='sk-lf-test',
                      tracing_enabled=True, span_exporter=exporter)
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request))
        openai_client = AsyncOpenAI(
            api_key='sk-test', base_url='https://mock.openrouter.test/v1',
            http_client=http_client)
        model = OpenAIChatCompletionsModel(
            model='test/provider-model', openai_client=openai_client)
        set_tracing_disabled(False)
        observability._reset_for_tests()

        def cleanup():
            OpenAIAgentsInstrumentor().uninstrument()
            observability._reset_for_tests()
            set_tracing_disabled(True)  # leave the process quiet again
            asyncio.run(openai_client.close())
            lf.shutdown()
        self.addCleanup(cleanup)

        self.assertTrue(observability.setup(client=lf))
        self.assertTrue(observability.active())

        ledger = Ledger()
        res = run_instruction(
            'solve 2x + 3 = 7 for x',
            model=model, ledger=ledger)
        self.assertTrue(res['ok'], res.get('error'))
        self.assertEqual(len(requests), 3)
        self.assertFalse(scripted_responses)
        lf.flush()

        spans = exporter.get_finished_spans()
        self.assertTrue(spans, 'no spans were exported')
        # one trace for the whole run, rooted at our wrapping observation
        self.assertEqual(len({s.context.trace_id for s in spans}), 1)
        root = next(s for s in spans if s.parent is None)
        self.assertEqual(root.name, 'do!')
        attrs = dict(root.attributes)
        self.assertEqual(attrs.get('langfuse.observation.input'),
                         'solve 2x + 3 = 7 for x')
        self.assertEqual(attrs.get('langfuse.observation.metadata.mode'),
                         'do')
        self.assertEqual(attrs.get('langfuse.observation.type'), 'agent')
        # the trusted-primitive calls appear as nested tool spans
        kinds = {s.attributes.get('openinference.span.kind') for s in spans}
        self.assertIn('TOOL', kinds)
        llm_spans = [
            s for s in spans
            if s.attributes.get('openinference.span.kind') == 'LLM'
        ]
        self.assertEqual(len(llm_spans), 3)
        for llm_span in llm_spans:
            llm_attrs = dict(llm_span.attributes)
            self.assertEqual(llm_attrs.get('llm.model_name'),
                             'test/provider-model')
            self.assertTrue(llm_attrs.get('input.value'))
            self.assertTrue(llm_attrs.get('output.value'))
            self.assertEqual(llm_attrs.get('llm.token_count.prompt'), 7)
            self.assertEqual(llm_attrs.get('llm.token_count.completion'), 3)
        self.assertIn('solve 2x + 3 = 7 for x',
                      llm_spans[0].attributes['input.value'])
        self.assertIn('run_tactic',
                      llm_spans[0].attributes['output.value'])
        self.assertIn('Subtracted 3 from both sides.',
                      llm_spans[-1].attributes['output.value'])
        # (the Codex path has no instrumentor and traces through the
        # neutral seam instead - see test_codex_run_emits_one_nested_trace)
        # tracing is observability only: the ledger is exactly what it would
        # be without it
        self.assertEqual([s['op'] for s in res['steps']],
                         ['apply_both_sides', 'expand'])
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_codex_run_emits_one_nested_trace_without_the_instrumentor(self):
        # The OpenInference instrumentor only sees Agents-SDK runs, so the
        # Codex path traces through the provider-neutral seam. Its tool
        # callbacks arrive on transport worker threads, which is exactly
        # where a naive child span would start a second, orphaned trace.
        from langfuse import Langfuse
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter)

        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        exporter = InMemorySpanExporter()
        lf = Langfuse(public_key='pk-lf-test', secret_key='sk-lf-test',
                      tracing_enabled=True, span_exporter=exporter)
        # OTel allows one global TracerProvider per process, so whether this
        # client's provider wins depends on test order; subscribe to the
        # effective one as well and de-duplicate below
        provider = otel_trace.get_tracer_provider()
        if hasattr(provider, 'add_span_processor'):
            provider.add_span_processor(SimpleSpanProcessor(exporter))
        observability._reset_for_tests()
        self.addCleanup(observability._reset_for_tests)
        self.addCleanup(lf.shutdown)
        # no instrumentor: exactly the Codex-only installation shape
        with mock.patch.object(observability, '_instrument',
                               side_effect=ImportError('no openai-agents')):
            self.assertTrue(observability.setup(client=lf))
        self.assertFalse(observability.instrumented())

        ledger = Ledger()
        res = run_instruction(
            'solve 2x + 3 = 7 for x', ledger=ledger,
            backend=codex_backend.CodexBackend(
                transport=codex_transport.TranscriptTransport(
                    SOLVE_TRANSCRIPT)))
        self.assertTrue(res['ok'], res.get('error'))
        lf.flush()

        spans = list({span.context.span_id: span
                      for span in exporter.get_finished_spans()}.values())
        self.assertTrue(spans, 'no spans were exported')
        # one trace for the whole run, despite the thread hop
        self.assertEqual(len({s.context.trace_id for s in spans}), 1)
        root = next(s for s in spans if s.parent is None)
        self.assertEqual(root.name, 'do!')
        attrs = dict(root.attributes)
        self.assertEqual(attrs.get('langfuse.observation.metadata.backend'),
                         'codex')
        self.assertEqual(attrs.get('langfuse.observation.output'),
                         'Subtracted 3 from both sides.')
        tools = sorted(s.name for s in spans if s.name.startswith('tool:'))
        self.assertEqual(tools, ['tool:run_tactic', 'tool:run_tactic',
                                 'tool:set_result'])
        for span in spans:
            if span.name.startswith('tool:'):
                self.assertIsNotNone(span.parent)   # nested, not orphaned
        # observability only: the ledger is what it would be untraced
        self.assertEqual([s['op'] for s in res['steps']],
                         ['apply_both_sides', 'expand'])
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_a_tracing_failure_never_breaks_a_codex_tool_call(self):
        observability._reset_for_tests()
        self.addCleanup(observability._reset_for_tests)
        with mock.patch.object(observability, '_instrument'):
            observability.setup(client=mock.Mock(
                start_as_current_observation=mock.Mock(
                    side_effect=RuntimeError('langfuse is down'))))
        ledger = Ledger()
        res = run_instruction(
            'solve 2x + 3 = 7 for x', ledger=ledger,
            backend=codex_backend.CodexBackend(
                transport=codex_transport.TranscriptTransport(
                    SOLVE_TRANSCRIPT)))
        self.assertTrue(res['ok'], res.get('error'))
        self.assertEqual(res['final_result'], '2x = 4')
        self.assertEqual(ledger.replay()['status'], 'verified')


FAKE_PNG_B64 ='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAAAABJRU5ErkJggg=='


FAKE_SVG = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>'


class FakePlotBackend(object):
    name = 'fake'

    def __init__(self, result=None):
        self.calls = []
        # the default deliberately speaks the older images-only dialect,
        # so the typed-figure fallback stays covered
        self.result = result or {'ok': True, 'stdout': 'drew it\n',
                                 'stderr': '', 'images': [FAKE_PNG_B64]}

    def run_plot(self, code, timeout=None):
        self.calls.append(code)
        return dict(self.result)


class FakeTikzBackend(object):
    name = 'fake-tikz'

    def __init__(self, result=None):
        self.calls = []
        self.result = result or {'ok': True, 'svg': FAKE_SVG}

    def render(self, code, timeout=None):
        self.calls.append(code)
        return dict(self.result)


class TestPlotTool(unittest.TestCase):
    def test_registered_only_with_backend(self):
        self.assertNotIn('plot', make_api(DoSession()))
        session = DoSession(plot_backend=FakePlotBackend())
        self.assertIn('plot', make_api(session))

    def test_prompt_mentions_plotting_only_when_available(self):
        self.assertNotIn('## Plotting', agent_do.build_prompt())
        self.assertIn('## Plotting', agent_do.build_prompt(plotting=True))

    def test_plot_streams_images_not_tokens(self):
        shown = []
        backend = FakePlotBackend()
        session = DoSession(plot_backend=backend,
                            on_plot=lambda cap, figs: shown.append(
                                (cap, figs)))
        api = make_api(session)
        reply = json.loads(api['plot']('plt.plot([1])', 'a parabola'))
        # the model sees counts and stdout, never image bytes
        self.assertTrue(reply['ok'])
        self.assertEqual(reply['plots'], 1)
        self.assertNotIn(FAKE_PNG_B64[:24], json.dumps(reply))
        # the user sees the figure with its caption; an images-only
        # backend is adapted up into the typed shape
        self.assertEqual(shown[0][0], 'a parabola')
        self.assertEqual(shown[0][1], [{'kind': 'png', 'data': FAKE_PNG_B64}])
        self.assertEqual(session.figure_events(), [{
            'status': 'ok', 'kind': 'figure', 'id': 'f1',
            'caption': 'a parabola',
            'figures': [{'kind': 'png', 'data': FAKE_PNG_B64}],
        }])
        # the model is told which figure it drew, so a corrected render can
        # supersede it instead of adding a second one
        self.assertEqual(reply['figure'], 'f1')
        self.assertNotIn('replaced', reply)
        # never a ledger step
        self.assertEqual(session.new_steps(), [])

    def test_plot_streams_typed_figures(self):
        shown = []
        backend = FakePlotBackend({
            'ok': True, 'stdout': '', 'stderr': '',
            'figures': [{'kind': 'html', 'data': '<html>fig</html>',
                         'height': 400}]})
        session = DoSession(plot_backend=backend,
                            on_plot=lambda cap, figs: shown.append(figs))
        reply = json.loads(make_api(session)['plot']('fig = go.Figure()',
                                                     'interactive'))
        self.assertTrue(reply['ok'])
        self.assertEqual(reply['plots'], 1)
        self.assertEqual(shown[0][0]['kind'], 'html')
        self.assertEqual(shown[0][0]['height'], 400)
        # figure bytes never reach the model
        self.assertNotIn('<html>', json.dumps(reply))

    def test_plot_failure_reported_in_band(self):
        backend = FakePlotBackend({'ok': False, 'images': [],
                                   'error': 'NameError: nope'})
        session = DoSession(plot_backend=backend)
        reply = json.loads(make_api(session)['plot']('bad', 'cap'))
        self.assertFalse(reply['ok'])
        self.assertIn('NameError', reply['error'])
        self.assertEqual(session.figure_events()[-1]['status'], 'error')
        self.assertIn('NameError', session.figure_events()[-1]['error'])

    def test_failed_python_does_not_publish_its_partial_figure(self):
        shown = []
        backend = FakePlotBackend({
            'ok': False, 'figures': [
                {'kind': 'png', 'data': FAKE_PNG_B64}],
            'error': 'RuntimeError: failed after drawing'})
        session = DoSession(plot_backend=backend,
                            on_plot=lambda *args: shown.append(args))
        reply = json.loads(make_api(session)['plot']('bad', 'partial'))
        self.assertFalse(reply['ok'])
        self.assertEqual(reply['plots'], 0)
        self.assertEqual(shown, [])
        self.assertEqual(session.figure_events(), [{
            'status': 'error', 'kind': 'plot', 'caption': 'partial',
            'error': 'RuntimeError: failed after drawing',
        }])

    def test_a_corrected_figure_supersedes_the_one_it_redraws(self):
        # Measured (Into.ipynb, `do! plot interval [[2]]`): the model's own
        # post-processing failed after a successful render, so it re-ran the
        # same tikz and the cell showed the number line twice. The redraw
        # now names the figure it replaces and lands in its place.
        session = DoSession(plot_backend=FakePlotBackend(),
                            tikz_backend=FakeTikzBackend())
        api = make_api(session)
        first = json.loads(api['plot']('plt.plot([1])', 'draft'))
        other = json.loads(api['tikz']('\\draw (0,0);', 'a diagram'))
        fixed = json.loads(api['plot']('plt.plot([2])', 'corrected',
                                       first['figure']))
        self.assertEqual(fixed['replaced'], first['figure'])
        self.assertNotIn('note', fixed)
        events = session.figure_events()
        # replaced in place: two figures, the correction where the draft was
        self.assertEqual([e['caption'] for e in events],
                         ['corrected', 'a diagram'])
        self.assertEqual([e['id'] for e in events],
                         [fixed['figure'], other['figure']])
        self.assertNotEqual(fixed['figure'], first['figure'])
        # omitting it is the normal case: the canonical validator restores
        # the default rather than the model having to send an empty string
        schema = {b.name: b.input_schema for b in
                  agent_do.make_tool_bindings(session)}['plot']
        self.assertEqual(
            agent_base.validate_arguments(
                schema, {'code': 'plt.plot([1])', 'caption': 'c'}),
            {'code': 'plt.plot([1])', 'caption': 'c', 'replaces': ''})

    def test_replacing_a_figure_this_run_never_drew_keeps_the_picture(self):
        # a bookkeeping mistake must not cost the user the figure that
        # rendered, and ids are per session: no run can reach another's
        session = DoSession(plot_backend=FakePlotBackend())
        api = make_api(session)
        drawn = json.loads(api['plot']('plt.plot([1])', 'first'))
        for stale in ('f9', drawn['figure'] + 'x', '   '):
            session_before = len(session.figure_events())
            reply = json.loads(api['plot']('plt.plot([2])', 'second', stale))
            self.assertTrue(reply['ok'])
            self.assertNotIn('replaced', reply)
            self.assertEqual(len(session.figure_events()), session_before + 1)
        self.assertIn('f9', json.loads(
            api['plot']('plt.plot([3])', 'third', 'f9'))['note'])

    def test_a_replacement_after_the_run_is_stopped_is_still_refused(self):
        session = DoSession(plot_backend=FakePlotBackend())
        api = make_api(session)
        first = json.loads(api['plot']('plt.plot([1])', 'kept'))
        session.close('user')
        reply = json.loads(api['plot']('plt.plot([2])', 'late',
                                       first['figure']))
        self.assertFalse(reply['ok'])
        self.assertTrue(reply['cancelled'])
        # the committed figure is untouched: a stopped run rewrites nothing
        self.assertEqual([e['caption'] for e in session.figure_events()],
                         ['kept'])

    def test_streaming_callback_failure_does_not_lose_buffered_figure(self):
        def broken_callback(*_args):
            raise RuntimeError('display socket unavailable')

        session = DoSession(plot_backend=FakePlotBackend(),
                            on_plot=broken_callback)
        reply = json.loads(make_api(session)['plot']('plt.plot([1])', 'safe'))
        self.assertTrue(reply['ok'])
        self.assertEqual(session.figure_events()[0]['status'], 'ok')

    def test_no_figure_is_an_error(self):
        backend = FakePlotBackend({'ok': True, 'stdout': '', 'stderr': '',
                                   'images': []})
        session = DoSession(plot_backend=backend)
        reply = json.loads(make_api(session)['plot']('print(1)', 'cap'))
        self.assertFalse(reply['ok'])
        self.assertIn('no figure', reply['error'])

    def test_scripted_agent_with_plot(self):
        shown = []
        script = [
            [tool_call('expand', {'expr': '(x-2)(x-3)'}, 'c1')],
            [tool_call('plot', {'code': 'plt.plot([1])',
                                'caption': 'the parabola'}, 'c2')],
            [message('done')],
        ]
        res = run_instruction('expand and plot',
                              model=ScriptedModel(script),
                              plot_backend=FakePlotBackend(),
                              on_plot=lambda cap, imgs: shown.append(cap))
        self.assertTrue(res['ok'])
        self.assertEqual(len(res['steps']), 1)  # plot is not a step
        self.assertEqual(shown, ['the parabola'])
        self.assertEqual(res['figures'][0]['caption'], 'the parabola')
        self.assertIsNone(res['figure_error'])

    def test_get_backend_off(self):
        with mock.patch.dict(os.environ, {'TOYMATH_SANDBOX': 'off'}):
            self.assertIsNone(plot_sandbox.get_backend())
            self.assertIsNone(plot_sandbox.get_tikz_backend())

    def test_parse_runner_output(self):
        good = plot_sandbox._parse_runner_output(
            'noise\n{"ok": true, "images": []}\n', '')
        self.assertTrue(good['ok'])
        bad = plot_sandbox._parse_runner_output('garbage only', 'boom')
        self.assertFalse(bad['ok'])
        self.assertIn('boom', bad['stderr'])

    def test_child_env_scrubs_secrets(self):
        with mock.patch.dict(os.environ, {'OPEN_ROUTER': 'sk-secret',
                                          'AWS_SECRET_ACCESS_KEY': 'nope',
                                          'PATH': '/usr/bin'}):
            env = plot_sandbox._child_env()
        self.assertNotIn('OPEN_ROUTER', env)
        self.assertNotIn('AWS_SECRET_ACCESS_KEY', env)
        self.assertEqual(env['PATH'], '/usr/bin')


class TestTikzSourceNormalization(unittest.TestCase):
    """node-tikzjax injects `\\documentclass` but no document environment.

    Both obvious submissions therefore failed, with TeX errors naming
    neither cause: a bare picture died on `Missing \\begin{document}` and a
    complete document on `Two \\documentclass commands`. Offline coverage of
    the reshaping; TestLiveTikzSandbox proves it against the real engine.
    """

    def _norm(self, code):
        return plot_sandbox.normalize_tikz_source(code)

    PICTURE = '\\begin{tikzpicture}\n\\draw (0,0) -- (1,1);\n' \
              '\\end{tikzpicture}'

    def test_a_bare_picture_gains_a_document(self):
        out = self._norm(self.PICTURE)
        self.assertIn('\\begin{document}', out)
        self.assertIn('\\end{document}', out)
        self.assertLess(out.index('\\begin{document}'),
                        out.index('\\begin{tikzpicture}'))

    def test_a_preamble_stays_outside_the_document(self):
        out = self._norm('\\usepackage{tikz-cd}\n' + self.PICTURE)
        self.assertLess(out.index('\\usepackage'),
                        out.index('\\begin{document}'))

    def test_a_documentclass_is_dropped(self):
        for line in ('\\documentclass{standalone}',
                     '\\documentclass[tikz,border=2pt]{standalone}',
                     '\\documentstyle{article}'):
            with self.subTest(line=line):
                out = self._norm(f'{line}\n\\begin{{document}}\n'
                                 f'{self.PICTURE}\n\\end{{document}}')
                self.assertNotIn('\\documentclass', out)
                self.assertNotIn('\\documentstyle', out)
                self.assertIn('\\begin{tikzpicture}', out)

    def test_an_existing_document_is_left_alone(self):
        code = ('\\usepackage{tikz}\n\\begin{document}\n' + self.PICTURE
                + '\n\\end{document}')
        self.assertEqual(self._norm(code), code)

    def test_a_tikzcd_body_is_wrapped_too(self):
        out = self._norm('\\usepackage{tikz-cd}\n'
                         '\\begin{tikzcd}\nA \\arrow[r] & B\n\\end{tikzcd}')
        self.assertLess(out.index('\\begin{document}'),
                        out.index('\\begin{tikzcd}'))
        self.assertIn('\\end{document}', out)

    def test_unrecognisable_source_is_passed_through(self):
        # no picture to wrap: let the TeX log explain rather than guessing
        self.assertEqual(self._norm('\\relax'), '\\relax')
        self.assertEqual(self._norm(''), '')

    def test_the_tool_description_states_the_real_contract(self):
        # the model reads this; it used to ask for a whole TeX document,
        # which is the one shape the engine rejects outright
        binding = {b.name: b for b
                   in agent_do.make_tool_bindings(
                       DoSession(tikz_backend=FakeTikzBackend()))}['tikz']
        text = json.dumps(binding.input_schema) + binding.description
        self.assertIn('begin{document}', text)
        self.assertIn('Omit', text)
        self.assertIn('tikz-cd', text)


class TestTikzTool(unittest.TestCase):
    def test_registered_only_with_backend(self):
        self.assertNotIn('tikz', make_api(DoSession()))
        session = DoSession(tikz_backend=FakeTikzBackend())
        self.assertIn('tikz', make_api(session))

    def test_prompt_mentions_tikz_only_when_available(self):
        self.assertNotIn('## TikZ', agent_do.build_prompt())
        self.assertIn('## TikZ', agent_do.build_prompt(tikz=True))

    def test_tikz_streams_svg_not_tokens(self):
        shown = []
        session = DoSession(tikz_backend=FakeTikzBackend(),
                            on_plot=lambda cap, figs: shown.append(
                                (cap, figs)))
        reply = json.loads(make_api(session)['tikz'](
            r'\begin{document}\end{document}', 'a diagram'))
        self.assertTrue(reply['ok'])
        self.assertEqual(reply['plots'], 1)
        self.assertNotIn('<svg', json.dumps(reply))
        self.assertEqual(shown[0][0], 'a diagram')
        self.assertEqual(shown[0][1], [{'kind': 'svg', 'data': FAKE_SVG}])
        # illustrations are never evidence
        self.assertEqual(session.new_steps(), [])

    def test_tikz_failure_returns_tex_log(self):
        backend = FakeTikzBackend({
            'ok': False,
            'error': 'TikZ render failed: TeX engine render failed.\n'
                     'TeX log:\n! Undefined control sequence.\n'
                     'l.4 \\draw (0,0) circle (\\nosuchmacro)'})
        session = DoSession(tikz_backend=backend)
        reply = json.loads(make_api(session)['tikz']('bad', 'cap'))
        self.assertFalse(reply['ok'])
        # the agent can only fix its source if the TeX error survives
        self.assertIn('Undefined control sequence', reply['error'])


@unittest.skipUnless(os.environ.get('TOYMATH_PLOT_TESTS') == '1',
                     'set TOYMATH_PLOT_TESTS=1 for a live deno/pyodide '
                     'sandbox test')
class TestLivePlotSandbox(unittest.TestCase):
    def test_matplotlib_figure_roundtrip(self):
        import base64
        backend = plot_sandbox.get_backend()
        self.assertIsNotNone(backend, 'deno not available')
        r = backend.run_plot(
            'import numpy as np\nimport matplotlib.pyplot as plt\n'
            'x = np.linspace(0, 1, 50)\nplt.plot(x, x*x)\n')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual(len(r['images']), 1)
        png = base64.b64decode(r['images'][0])
        self.assertEqual(png[:8], b'\x89PNG\r\n\x1a\n')

    def test_escape_probes_denied(self):
        backend = plot_sandbox.get_backend()
        for probe in ("import js\njs.Deno.env.get('OPEN_ROUTER')",
                      "import js\njs.Deno.readTextFileSync('/etc/hosts')"):
            r = backend.run_plot(probe)
            self.assertFalse(r['ok'])
            self.assertTrue('NotCapable' in (r.get('error') or '')
                            or 'PermissionDenied' in (r.get('error') or ''))

    def test_submodule_import_of_uninstalled_package(self):
        """find_imports reports 'seaborn.objects' alongside 'seaborn', and
        find_spec raises on the dotted name while the parent is absent.
        That used to kill the sandbox before the user code ever ran."""
        backend = plot_sandbox.get_backend()
        r = backend.run_plot(
            'import seaborn.objects as so\nimport matplotlib.pyplot as plt\n'
            'plt.plot([1, 2])\n')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual([f['kind'] for f in r['figures']], ['png'])

    def test_bundled_submodule_still_resolves(self):
        backend = plot_sandbox.get_backend()
        r = backend.run_plot(
            'from scipy.integrate import quad\n'
            'import matplotlib.pyplot as plt\n'
            'v, _ = quad(lambda t: t**2, 0, 3)\nprint(round(v, 6))\n'
            'plt.plot([0, v])\n')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertIn('9.0', r['stdout'])

    def test_plotly_figure_becomes_html(self):
        backend = plot_sandbox.get_backend()
        r = backend.run_plot(
            'import plotly.graph_objects as go\n'
            'fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[1, 4])])\n'
            'fig.update_layout(height=400)\n')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertEqual([f['kind'] for f in r['figures']], ['html'])
        self.assertEqual(r['images'], [])  # plotly cannot rasterise here
        self.assertEqual(r['figures'][0]['height'], 440)


@unittest.skipUnless(os.environ.get('TOYMATH_PLOT_TESTS') == '1',
                     'set TOYMATH_PLOT_TESTS=1 for a live deno/tikzjax '
                     'test')
class TestLiveTikzSandbox(unittest.TestCase):
    def test_tikz_renders_self_contained_svg(self):
        backend = plot_sandbox.get_tikz_backend()
        self.assertIsNotNone(backend, 'deno not available')
        r = backend.render(
            r'\usepackage{pgfplots}'
            '\n\\begin{document}\n\\begin{tikzpicture}\n'
            r'\begin{axis}[width=6cm]\addplot[domain=-2:2]{x^2};\end{axis}'
            '\n\\end{tikzpicture}\n\\end{document}')
        self.assertTrue(r['ok'], r.get('error'))
        svg = r['svg']
        self.assertTrue(svg.lstrip().startswith('<svg'))
        # fonts are inlined, so the figure still renders with no network
        self.assertIn('@font-face', svg)
        self.assertIn('data:font/ttf;base64', svg)
        self.assertNotIn('<script', svg.lower())

    def test_tikz_error_carries_the_tex_log(self):
        backend = plot_sandbox.get_tikz_backend()
        r = backend.render(r'\begin{document}\begin{tikzpicture}'
                           r'\draw (0,0) circle (\nosuchmacro);'
                           r'\end{tikzpicture}\end{document}')
        self.assertFalse(r['ok'])
        self.assertIn('Undefined control sequence', r['error'])

    def test_every_reasonable_document_shape_renders(self):
        # the engine injects \documentclass but no document environment, so
        # a bare picture and a complete document both used to fail with TeX
        # errors that named neither cause. One traced run burned four calls
        # guessing at the wrapper.
        backend = plot_sandbox.get_tikz_backend()
        picture = ('\\begin{tikzpicture}\n\\draw (0,0) -- (2,1);\n'
                   '\\end{tikzpicture}')
        shapes = {
            'bare picture': picture,
            'preamble + picture': '\\usepackage{tikz}\n' + picture,
            'whole document': ('\\documentclass{standalone}\n'
                               '\\usepackage{tikz}\n\\begin{document}\n'
                               + picture + '\n\\end{document}'),
            'documentclass with options': ('\\documentclass[tikz]{standalone}'
                                           '\n\\begin{document}\n' + picture
                                           + '\n\\end{document}'),
        }
        for name, code in shapes.items():
            with self.subTest(shape=name):
                r = backend.render(code)
                self.assertTrue(r['ok'], f'{name}: {r.get("error")}')

    def test_commutative_diagrams_render_with_tikz_cd(self):
        # what the failing run was actually asked for
        backend = plot_sandbox.get_tikz_backend()
        r = backend.render(
            '\\usepackage{tikz-cd}\n\\begin{document}\n\\begin{tikzcd}\n'
            'A \\arrow[r, "f"] \\arrow[d, "g"\'] & B \\arrow[d, "h"] \\\\\n'
            'C \\arrow[r, "k"\'] & D\n\\end{tikzcd}\n\\end{document}')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertTrue(r['svg'].lstrip().startswith('<svg'))


class TestFigureHtml(unittest.TestCase):
    def setUp(self):
        from mathShell import MathShell
        self.render = MathShell._figure_html

    def test_png_inlined_as_data_uri(self):
        html = self.render({'kind': 'png', 'data': FAKE_PNG_B64})
        self.assertIn(f'src="data:image/png;base64,{FAKE_PNG_B64}"', html)

    def test_svg_delivered_as_image_not_inline_markup(self):
        # inline <svg> does not survive an untrusted notebook: JupyterLab
        # sanitizes text/html outputs and strips the tag, leaving the glyph
        # text as debris. A data-URI <img> is allowlisted, so the figure
        # renders with no trust signature, and image context cannot script.
        import base64
        html = self.render({'kind': 'svg', 'data': FAKE_SVG})
        payload = base64.b64encode(FAKE_SVG.encode()).decode()
        self.assertIn(f'src="data:image/svg+xml;base64,{payload}"', html)
        self.assertNotIn('<svg', html)

    def test_html_iframed_and_escaped(self):
        # JupyterLab strips <script> from cell output, so plotly only runs
        # inside an iframe - and srcdoc must be attribute-escaped or the
        # figure's own quotes break out of it
        html = self.render({'kind': 'html', 'height': 400,
                            'data': '<p class="x">hi</p>'})
        self.assertIn('<iframe', html)
        self.assertIn('height:400px', html)
        self.assertIn('sandbox="allow-scripts"', html)
        self.assertIn('&lt;p class=&quot;x&quot;&gt;', html)
        self.assertNotIn('<p class="x">', html)

    def test_unknown_kind_falls_back_to_png(self):
        self.assertIn('<img', self.render({'data': FAKE_PNG_B64}))


class TestMathShellDo(unittest.TestCase):
    def setUp(self):
        import engine
        self.displays = []
        engine.setHandler(lambda *objs, **kw: self.displays.extend(objs))
        from mathShell import MathShell
        self.shell = MathShell()

    def tearDown(self):
        import engine
        import IPython.display
        engine.setHandler(IPython.display.display)

    def _html(self):
        return ''.join(getattr(d, 'data', str(d)) for d in self.displays)

    def test_assumption_html_prose_and_math(self):
        mixed = self.shell._assumption_html(
            {'text': 'x \\ne 0 in that neighborhood',
             'display': '$x \\ne 0$ in that <neighborhood>'})
        self.assertEqual(mixed,
                         '$x \\ne 0$ in that &lt;neighborhood&gt;')
        bare = self.shell._assumption_html({'text': 'x > 0'})
        self.assertEqual(bare, '$x > 0$')

    def test_resolve_backrefs(self):
        self.shell.exec('2 + 3', 1, add_to_history=True)
        text = self.shell.resolve_backrefs('check [[1]] please')
        self.assertIn('5', text)
        with self.assertRaises(ValueError):
            self.shell.resolve_backrefs('[[99]]')

    def test_backref_survives_later_parses(self):
        # parser.parse() clears parsedNotation in place; the stored history
        # snapshot must not be affected (regression: [[n]] printed _nNN)
        sym = self.shell.parser.parse('x^{2}+1')
        self.shell.output(sym, self.shell.parsedNotation, 7, True)
        self.shell.parser.parse('y')
        text = self.shell.resolve_backrefs('[[7]]')
        self.assertNotIn('textit', text)
        self.assertIn('x^', text)

    def test_backref_of_fractional_power_reparses(self):
        # \int dx/(x^{1/2}+x^{1/3}) stored, then referenced from a
        # composite cell: the re-rendered LaTeX must parse (regression:
        # the writer emitted x^\frac{1}{2}, syntax error on int! [[n]])
        import primitives
        self.shell.exec('\\int \\frac {dx} {(x^{\\frac 1 2} '
                        '+ x^{\\frac 1 3})}', 3, add_to_history=True)
        resolved = self.shell.resolve_backrefs('[[3]]')
        primitives.parse_latex(resolved)

    def test_typed_collection_survives_history_snapshot(self):
        import primitives
        from notation import Notation

        latex = '\\{(-1,2),(1,-2)\\}'
        sym = self.shell.parser.parse(latex)
        rendered = self.shell.output(
            sym, self.shell.parsedNotation, 48, True)
        self.assertEqual(rendered, latex)

        # A later parse clears parsedNotation in place. The private history
        # snapshot must retain both the collection and its pair children.
        self.shell.parser.parse('z')
        resolved = self.shell.resolve_backrefs('[[48]]')
        chained, notation = primitives.parse_latex(resolved)
        collection = notation.getf(chained, Notation.COLLECTION)
        self.assertIsNotNone(collection)
        self.assertTrue(all(notation.getf(item, Notation.PAIR) is not None
                            for item in collection.args))

    def test_do_cell_streams_and_chains(self):
        with mock.patch.object(openrouter_backend, 'build_model',
                               lambda model_name=None: ScriptedModel(
                                   SOLVE_SCRIPT)):
            self.shell.exec('do! solve 2x + 3 = 7 for x', 2,
                            add_to_history=True)
        out = self._html()
        self.assertIn('s1#', out)
        self.assertIn('s2#', out)
        self.assertIn('Subtracted 3', out)
        # the harness renders the end-of-run chain table from the ledger
        self.assertIn('verified chain', out)
        # the final result is chainable from later cells
        self.assertIn('2', self.shell.resolve_backrefs('[[2]]'))
        self.assertEqual(len(self.shell.ledger.steps), 2)

    def test_codex_plot_is_rendered_once_on_the_kernel_thread(self):
        import engine

        main_thread = threading.get_ident()
        displayed = []

        def capture(*objects, **_kwargs):
            for obj in objects:
                self.displays.append(obj)
                displayed.append((threading.get_ident(),
                                  getattr(obj, 'data', str(obj))))

        engine.setHandler(capture)
        plot_threads = []

        class WorkerPlotBackend(FakePlotBackend):
            def run_plot(inner_self, code, timeout=None):
                plot_threads.append(threading.get_ident())
                return super(WorkerPlotBackend, inner_self).run_plot(
                    code, timeout=timeout)

        transport = codex_transport.TranscriptTransport([
            {'tool': 'plot',
             'arguments': {'code': 'plt.plot([1])',
                           'caption': 'worker-thread figure'}},
            {'message': 'Rendered the requested illustration.'},
        ])
        self.shell.route = agent_config.AgentRoute(
            backend=agent_config.CODEX, model='gpt-5.6-terra')
        with mock.patch.object(codex_backend, 'runtime',
                               return_value=transport), \
                mock.patch.object(plot_sandbox, 'get_backend',
                                  return_value=WorkerPlotBackend()), \
                mock.patch.object(plot_sandbox, 'get_tikz_backend',
                                  return_value=None):
            self.shell.exec('do! draw it', 22, add_to_history=True)

        self.assertEqual(len(plot_threads), 1)
        self.assertNotEqual(plot_threads[0], main_thread)
        figure_threads = [thread for thread, html in displayed
                          if '<img' in html]
        self.assertEqual(figure_threads, [main_thread])
        self.assertEqual(self._html().count('<img'), 1)
        self.assertIn('worker-thread figure', self._html())

    def test_failed_codex_plot_is_visible_without_partial_image(self):
        backend = FakePlotBackend({
            'ok': False,
            'figures': [{'kind': 'png', 'data': FAKE_PNG_B64}],
            'error': 'NameError: missing_name',
        })
        transport = codex_transport.TranscriptTransport([
            {'tool': 'plot',
             'arguments': {'code': 'missing_name()',
                           'caption': 'broken worker plot'}},
            {'message': 'The illustration could not be produced.'},
        ])
        self.shell.route = agent_config.AgentRoute(
            backend=agent_config.CODEX, model='gpt-5.6-terra')
        with mock.patch.object(codex_backend, 'runtime',
                               return_value=transport), \
                mock.patch.object(plot_sandbox, 'get_backend',
                                  return_value=backend), \
                mock.patch.object(plot_sandbox, 'get_tikz_backend',
                                  return_value=None):
            self.shell.exec('do! draw it', 23, add_to_history=True)

        out = self._html()
        self.assertIn('<strong>plot failed:</strong>', out)
        self.assertIn('broken worker plot', out)
        self.assertIn('NameError: missing_name', out)
        self.assertNotIn('<img', out)
        self.assertIn('mechanically checked mathematics is unaffected', out)

    def test_do_open_outcome_banner_typesets_dollar_math(self):
        # the reason's $-delimited math must reach the banner intact and
        # the banner div must stay MathJax-eligible (unlike note prose,
        # which is deliberately tex2jax_ignore'd)
        script = [
            [tool_call('expand', {'expr': '(x+1)^2'}, 'c1')],
            [tool_call('set_open', {
                'reason': 'the missing move is a checked bound for '
                          '$\\sum_{n=1}^{m} \\frac{1}{n}$'}, 'c2')],
            [message('Left open.')],
        ]
        with mock.patch.object(openrouter_backend, 'build_model',
                               lambda model_name=None: ScriptedModel(
                                   script)):
            self.shell.exec('do! decide something out of reach', 5,
                            add_to_history=True)
        out = self._html()
        self.assertIn('outcome: open', out)
        self.assertIn('$\\sum_{n=1}^{m} \\frac{1}{n}$', out)
        self.assertIn('(unverified reason)', out)
        banner = next(getattr(d, 'data', '') for d in self.displays
                      if 'outcome: open' in getattr(d, 'data', ''))
        self.assertNotIn('tex2jax_ignore', banner)

    def test_cancelled_cell_is_amber_and_never_chainable(self):
        # a stopped cell keeps what was checked and produces no result:
        # no [[n]] backreference, no red agent-error block
        cancelled = {
            'ok': False, 'status': 'interrupted', 'cancelled': True,
            'steps': [self._chain_step('s1', 'expand', 'x^{2}+2x+1')],
            'claims': [], 'assumptions': [], 'premises': [],
            'final_result': None, 'final_provenance': None,
            'partial_result': 'x^{2}+2x+1',
            'partial_provenance': {'status': 'verified', 'step': 's1',
                                   'method': 'last-step'},
            'branch_topology': {'spine': [], 'abandoned_paths': [],
                                'parents': {}},
            'abandoned_paths': [], 'summary': None,
            'error': 'stopped by the user',
        }
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: cancelled):
            self.shell.exec('do! solve something long', 9,
                            add_to_history=True)
        out = self._html()
        self.assertIn('cancelled', out)
        self.assertIn('1 mechanically checked step preserved', out)
        self.assertIn('partial result from', out)
        self.assertNotIn('do! error', out)
        with self.assertRaises(ValueError):
            self.shell.resolve_backrefs('[[9]]')

    def test_budget_exhausted_cell_says_so(self):
        exhausted = {
            'ok': False, 'status': 'budget_exhausted', 'cancelled': True,
            'steps': [], 'claims': [], 'assumptions': [], 'premises': [],
            'final_result': None, 'final_provenance': None,
            'partial_result': None, 'partial_provenance': None,
            'branch_topology': {'spine': [], 'abandoned_paths': [],
                                'parents': {}},
            'abandoned_paths': [], 'summary': None,
        }
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: exhausted):
            self.shell.exec('do! loop forever', 10, add_to_history=True)
        out = self._html()
        self.assertIn('budget exhausted', out)
        self.assertIn('nothing was recorded', out)
        with self.assertRaises(ValueError):
            self.shell.resolve_backrefs('[[10]]')

    def test_int_composite_closes_across_substitution_chain(self):
        # the reported live cell (gemini, trace-replayed): u = x^{1/6}
        # substitution, polynomial division, linearity, assemble, back-
        # substitution, expand — every step green, set_result accepted,
        # and the goal-chain gate then refused the splice because its
        # step hops used strict spelling identity across \int boundaries
        def rt(tactic, arguments, cid):
            return ResponseFunctionToolCall(
                type='function_call', name='run_tactic',
                arguments=json.dumps({'tactic': tactic,
                                      'arguments': arguments}),
                call_id=cid, id=cid)

        calls = [
            tool_call('load_skill', {'skill': 'integration'}, 'c0'),
            rt('integrate_substitute',
               ['\\frac{1}{x^{\\frac{1}{2}}+x^{\\frac{1}{3}}}', 'x',
                'x^{\\frac{1}{6}}', 'u', '\\frac{6u^5}{u^3+u^2}'], 'c1'),
            rt('integrate_rewrite',
               ['\\frac{6u^5}{u^3+u^2}', 'u', '\\frac{6u^3}{u+1}'], 'c2'),
            rt('integrate_rewrite',
               ['\\frac{6u^3}{u+1}', 'u', '6u^2-6u+6-\\frac{6}{u+1}'],
               'c3'),
            rt('integrate_linearity',
               ['6u^2-6u+6-\\frac{6}{u+1}', 'u'], 'c4'),
            rt('integrate_power_rule', ['6u^2', 'u'], 'c5'),
            rt('substitute', ['2u^{3} + C', 'C', '0'], 'c6'),
            rt('integrate_power_rule', ['6u', 'u'], 'c7'),
            rt('substitute', ['3u^{2} + C', 'C', '0'], 'c8'),
            rt('integrate_power_rule', ['6', 'u'], 'c9'),
            rt('substitute', ['6u + C', 'C', '0'], 'c10'),
            rt('integrate_substitute',
               ['\\frac{6}{u+1}', 'u', 'u+1', 'v', '\\frac{6}{v}'], 'c11'),
            rt('integrate_table', ['\\frac{6}{v}', 'v'], 'c12'),
            rt('substitute',
               ['6 \\ln\\left(v\\right) + C', 'v', 'u+1'], 'c13'),
            rt('substitute',
               ['6 \\ln\\left ((u+1) \\right )+C', 'C', '0'], 'c14'),
            rt('integrate_assemble',
               ['s4', 's6', 's8', 's10', 's14'], 'c15'),
            rt('substitute',
               ['\\left(2u^{3}\\right) - \\left(3u^{2}\\right) + '
                '\\left(6u\\right) - '
                '\\left(6 \\ln\\left ((u+1) \\right )\\right) + C',
                'u', 'x^{\\frac{1}{6}}'], 'c16'),
            rt('expand',
               ['\\left (2(x^{\\frac {1} {6}})^{3}\\right )- '
                '\\left (3(x^{\\frac {1} {6}})^{2}\\right )+ '
                '\\left (6(x^{\\frac {1} {6}}) \\right )- '
                '\\left (6 \\ln\\left (((x^{\\frac {1} {6}})+1) \\right )'
                ' \\right )+C'], 'c17'),
            tool_call('set_result', {'expr':
                '2x^{\\frac {1} {2}}-3x^{\\frac {1} {3}}+C'
                '+6x^{\\frac {1} {6}}'
                '-6 \\ln\\left ((x^{\\frac {1} {6}}+1) \\right )'}, 'c18'),
        ]
        turns = [[c] for c in calls]
        turns.append([message('The indefinite integral is set.')])
        with mock.patch.object(openrouter_backend, 'build_model',
                               lambda model_name=None: ScriptedModel(
                                   turns)):
            self.shell.exec(
                'int! \\int \\frac {dx} (x^{\\frac 1 2} + x^{\\frac 1 3})',
                6, add_to_history=True)
        out = self._html()
        self.assertNotIn('did not close', out)
        self.assertNotIn('do! error', out)
        # 17 replayed steps; a whole-cell single command keeps its
        # designated result — no composite glue step to respell it
        self.assertEqual(len(self.shell.ledger.steps), 17)
        self.assertEqual(self.shell.ledger.replay()['status'], 'verified')
        chained = self.shell.resolve_backrefs('[[6]]')
        self.assertIn('\\ln', chained)

    def test_do_missing_backref_fails_fast(self):
        called = []
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: called.append(1)):
            self.shell.exec('do! solve [[42]]', 3, add_to_history=True)
        self.assertEqual(called, [])  # no agent run, no tokens
        self.assertIn('do! error', self._html())

    def test_do_missing_key_reports_cleanly(self):
        env = {k: v for k, v in os.environ.items()
               if k not in (openrouter_backend.API_KEY_VAR,
                            openrouter_backend.BASE_URL_VAR)}
        with mock.patch.dict(os.environ, env, clear=True):
            self.shell.exec('do! solve 2x = 4', 4, add_to_history=True)
        self.assertIn(openrouter_backend.API_KEY_VAR, self._html())

    def _chain_step(self, sid, op, result, check='agree', continues=True,
                    assumptions=()):
        return {'id': sid, 'hash': 'h' + sid, 'op': op,
                'args': {}, 'input': 'x', 'result': result,
                'check': {'status': check}, 'continues': continues,
                'assumptions': list(assumptions)}

    def test_chain_table_renders_from_step_records(self):
        html = self.shell.render_do_chain([
            self._chain_step('s1', 'expand', 'x^{2}+2x+1'),
            self._chain_step('s2', 'factor_quadratic', '(x+1)^{2}',
                             assumptions=[{'text': 'x \\ne 0'}]),
            self._chain_step('s3', 'sum_telescope', '\\frac{n}{n+1}',
                             check='skipped', continues=False),
        ])
        self.assertIn('verified chain', html)
        for frag in ('<code>s1</code>', '<code>expand</code>',
                     '$x^{2}+2x+1$', 'new chain; no marker',
                     '+1 assum.'):
            self.assertIn(frag, html)
        self.assertNotIn('(branch)', html)
        # check-status colors: agree green, skipped grey
        self.assertIn('#176b2c', html)
        self.assertIn('#888', html)

    def test_rich_ledger_views_prettify_formula_stars(self):
        step = self._chain_step('s1', 'expand', 'x*y')
        step['input'] = '2*x'
        streamed = self.shell.render_do_step(step)
        chain = self.shell.render_do_chain([
            step, self._chain_step('s2', 'expand', '2*x')])
        claim = self.shell.render_do_claim({
            'id': 'c1', 'statement': 'x*y=2*x', 'verdict': 'open'})
        assumption = self.shell._assumption_html({'text': 'x*y \\ne 0'})
        for rendered in (streamed, chain, claim, assumption):
            self.assertIn('\\cdot', rendered)
            self.assertNotIn('*', rendered)
        mixed = self.shell._assumption_html({
            'text': 'unused',
            'display': 'literal * prose, then $x*y$'})
        self.assertIn('literal * prose', mixed)
        self.assertIn('$x \\cdot y$', mixed)

    def test_chain_table_skips_runs_with_nothing_checked(self):
        comment = {'id': 's1', 'hash': 'h1', 'op': 'comment',
                   'args': {'text': 'strategy note'}, 'input': None,
                   'result': None, 'check': {'status': 'note'},
                   'assumptions': []}
        branch = {'id': 's2', 'hash': 'h2', 'op': 'branch',
                  'args': {'from': 's1', 'reason': 'try another route'},
                  'input': None, 'result': None,
                  'check': {'status': 'note'}, 'assumptions': []}
        self.assertIsNone(self.shell.render_do_chain([]))
        self.assertIsNone(self.shell.render_do_chain([comment, branch]))

    def test_single_step_run_renders_its_receipt(self):
        # a one-step derivation still shows the kernel-rendered check
        # verdict: without its row, the only on-screen "verified" is
        # agent prose — exactly what the table exists to replace (live:
        # the one-step FTC cell showed its result with no receipt)
        html = self.shell.render_do_chain(
            [self._chain_step('s1', 'differentiate',
                              '2x \\sqrt{x^{4}+1}')])
        self.assertIn('verified chain', html)
        self.assertIn('<code>s1</code>', html)
        self.assertIn('<code>differentiate</code>', html)
        self.assertIn('agree', html)

    def test_goal_restatement_is_not_a_stated_premise(self):
        # the task is not a given: a premise naming the requested
        # expression (or a piece the goal wraps) says nothing beyond the
        # cell the reader typed; genuine givens never cover the goal
        premises = [
            {'step': 's1',
             'input': '\\frac{d}{dx}\\int_{0}^{x^{2}}'
                      '\\sqrt{1+t^{2}}\\,dt'},
            {'step': 's2', 'input': 'A = 3'},
        ]
        goal = '\\frac  {d} {dx}\\int_{0}^{x^{2}}\\sqrt{{1}+t^{2}}dt'
        kept = agent_do._stated_premises(premises, goal)
        self.assertEqual([p['step'] for p in kept], ['s2'])
        # with no goal to compare (plain do!), every premise stands
        self.assertEqual(agent_do._stated_premises(premises, None),
                         premises)

    def test_branch_marker_stream_render_is_plain_annotation(self):
        html = self.shell.render_do_step({
            'id': 's3', 'hash': 'h3', 'op': 'branch',
            'args': {'from': 's1',
                     'reason': 'substitution < increased complexity'},
            'input': None, 'result': None, 'continues': None,
            'check': {'status': 'note'}, 'assumptions': [],
        })
        self.assertIn('tex2jax_ignore', html)
        self.assertIn('branch from s1', html)
        self.assertIn('substitution &lt; increased complexity', html)
        self.assertNotIn('$', html)

    def test_chain_table_folds_abandoned_path_with_marker_reason(self):
        first = self._chain_step('s1', 'expand', 'x^{2}+2x+1',
                                 continues=None)
        dead = self._chain_step('s2', 'substitute', '4')
        resumed = self._chain_step('s4', 'factor_quadratic', '(x+1)^{2}',
                                   continues=False)
        resumed['exploration'] = {
            'marker': 's3', 'from': 's1',
            'reason': 'numeric < detour', 'hash': 'edge',
        }
        topology = {'abandoned_paths': [{
            'marker': 's3', 'source': 's1', 'continues_at': 's4',
            'reason': 'numeric < detour', 'steps': ['s2'],
        }]}
        html = self.shell.render_do_chain([first, dead, resumed], topology)
        self.assertIn('<details', html)
        self.assertIn('abandoned path from s1', html)
        self.assertIn('numeric &lt; detour', html)
        self.assertIn('<code>s2</code>', html)  # expandable body retained
        self.assertIn('resumed from s1 via s3', html)
        self.assertNotIn('(branch)', html)

    def test_chain_table_can_expand_prior_run_steps_named_by_new_marker(self):
        first = self._chain_step('s1', 'expand', 'x^{2}+2x+1',
                                 continues=None)
        dead = self._chain_step('s2', 'substitute', '4')
        resumed = self._chain_step('s4', 'factor_quadratic', '(x+1)^{2}',
                                   continues=False)
        resumed['exploration'] = {
            'marker': 's3', 'from': 's1', 'reason': 'old detour',
            'hash': 'edge',
        }
        topology = {'abandoned_paths': [{
            'marker': 's3', 'source': 's1', 'continues_at': 's4',
            'reason': 'old detour', 'steps': ['s2'],
        }]}
        html = self.shell.render_do_chain(
            [resumed], topology, all_steps=[first, dead, resumed])
        self.assertIn('old detour', html)
        self.assertIn('<code>s2</code>', html)
        self.assertIn('<code>s4</code>', html)

    def test_query_only_final_is_rendered_unverified(self):
        result = {'ok': True, 'steps': [], 'assumptions': [],
                  'final_result': 'x=999', 'summary': None,
                  'final_provenance': {
                      'status': 'unverified', 'source': 'query-only',
                      'reason': 'no transforming step was recorded'}}
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: result):
            self.shell.exec('do! query only', 5, add_to_history=True)
        self.assertIn('unverified final value', self._html())
        self.assertIn('999', self.shell.resolve_backrefs('[[5]]'))


class TestPromptCommandModel(unittest.TestCase):
    """The discovery/parse model in prompt_commands.py (no shell, no agent)."""

    def test_repo_commands_load(self):
        import prompt_commands as pc
        reg = pc.load_commands()
        for name in ('int', 'diff', 'solve', 'expand', 'prove', 'conv'):
            self.assertIn(name, reg)
        self.assertIsNone(reg['int'].direct)      # LLM tactic tier
        self.assertEqual(reg['diff'].direct, 'differentiate')
        self.assertEqual(reg['expand'].direct, 'expand')
        self.assertTrue(reg['diff'].expr)         # direct implies expr
        self.assertEqual(reg['prove'].mode, 'prove')
        # conv! is a plain agent command: the verdict must live in the
        # recorded series_converges step, never in an expr splice
        self.assertFalse(reg['conv'].expr)
        self.assertIsNone(reg['conv'].direct)

    def test_repo_model_endpoint_config(self):
        import model_config
        endpoints = model_config.load_model_config()
        glm = model_config.find_model(endpoints, 'z-ai/glm-5.2')
        self.assertIsNotNone(glm)
        self.assertEqual(glm.providers, ('Cerebras', 'Fireworks'))

    def test_model_endpoint_config_validation(self):
        import model_config
        with self.assertRaises(model_config.ModelConfigError):
            model_config.parse_model_config('models: []')
        with self.assertRaises(model_config.ModelConfigError):
            model_config.parse_model_config(
                'models:\n  - model: a\n    providers: Cerebras')
        with self.assertRaises(model_config.ModelConfigError):
            model_config.parse_model_config(
                'models:\n  - model: a\n  - model: a')

    def test_model_command_completion_replaces_only_current_token(self):
        import model_config
        endpoints = (
            model_config.ModelEndpoint('alpha/one', ()),
            model_config.ModelEndpoint('beta/two', ('Fast', 'Safe')),
        )
        code = 'model! be'
        reply = model_config.complete_model_command(
            code, len(code), endpoints=endpoints)
        self.assertEqual(reply['matches'], ['alpha/one', 'beta/two'])
        self.assertEqual(code[reply['cursor_start']:reply['cursor_end']],
                         'be')
        self.assertEqual(
            [item['type'] for item in
             reply['metadata']['_jupyter_types_experimental']],
            ['model', 'model'])

    def test_provider_completion_uses_selected_model_and_skips_used(self):
        import model_config
        endpoints = (
            model_config.ModelEndpoint('beta/two', ('Fast', 'Safe')),
        )
        code = 'model! beta/two, Fast, S'
        reply = model_config.complete_model_command(
            code, len(code), endpoints=endpoints)
        self.assertEqual(reply['matches'], ['Safe'])
        self.assertEqual(code[reply['cursor_start']:reply['cursor_end']],
                         'S')
        self.assertEqual(
            reply['metadata']['_jupyter_types_experimental'][0]['type'],
            'provider')

    def test_non_model_command_has_no_model_completions(self):
        import model_config
        self.assertIsNone(model_config.complete_model_command(
            'solve! x = 2', len('solve! x = 2'), endpoints=()))

    def test_kernel_do_complete_routes_to_model_configuration(self):
        import asyncio
        from toymathkernel import MathKernel
        code = 'model! z-ai'
        reply = asyncio.run(MathKernel.do_complete(None, code, len(code)))
        self.assertIn('z-ai/glm-5.2', reply['matches'])
        self.assertEqual(reply['status'], 'ok')

    def test_kernel_do_complete_routes_backend_and_login(self):
        import asyncio
        from toymathkernel import MathKernel
        for code, expected in (('backend! co', ['codex']),
                               ('login! log', ['logout'])):
            with self.subTest(code=code):
                reply = asyncio.run(
                    MathKernel.do_complete(None, code, len(code)))
                self.assertEqual(reply['matches'], expected)
                self.assertEqual(reply['status'], 'ok')

    def test_static_lexer_command_table_matches_committed_registries(self):
        import prompt_commands as pc
        from lexer import MathLexer
        from processor import register_actions
        expected = (set(register_actions()) | set(pc.load_commands())
                    | set(pc.RESERVED))
        self.assertEqual(MathLexer.KNOWN_COMMANDS, expected)

    def test_render_substitutes_arguments(self):
        import prompt_commands as pc
        cmd = pc.parse_command(
            '---\nname: int\ndescription: d\n---\nIntegrate $ARGUMENTS now',
            'int')
        self.assertEqual(pc.render(cmd, 'x^2'), 'Integrate x^2 now')

    def test_fallback_name_from_stem(self):
        import prompt_commands as pc
        cmd = pc.parse_command('---\ndescription: d\n---\n$ARGUMENTS', 'foo')
        self.assertEqual(cmd.name, 'foo')

    def test_expr_flag_parsed(self):
        import prompt_commands as pc
        plain = pc.parse_command('---\nname: a\ndescription: d\n---\n$ARGUMENTS',
                                 'a')
        self.assertFalse(plain.expr)
        inline = pc.parse_command(
            '---\nname: b\ndescription: d\nexpr: true\n---\n$ARGUMENTS', 'b')
        self.assertTrue(inline.expr)

    def test_command_input_output_types_parsed(self):
        import prompt_commands as pc
        cmd = pc.parse_command(
            '---\nname: typed\ndescription: d\n'
            'input: [expression, relation]\noutput: derivation\n'
            '---\n$ARGUMENTS', 'typed')
        self.assertEqual(cmd.input_types, ('expression', 'relation'))
        self.assertEqual(cmd.output_types, ('derivation',))
        self.assertEqual(pc.signature(cmd),
                         'expression|relation -> derivation')

    def test_unknown_command_type_rejected(self):
        import prompt_commands as pc
        with self.assertRaisesRegex(ValueError, 'unknown command input type'):
            pc.parse_command(
                '---\nname: typed\ndescription: d\ninput: theorem\n'
                '---\n$ARGUMENTS', 'typed')

    def test_inline_command_cannot_return_derivation(self):
        import prompt_commands as pc
        with self.assertRaisesRegex(ValueError, 'cannot return a derivation'):
            pc.parse_command(
                '---\nname: typed\ndescription: d\nexpr: true\n'
                'output: derivation\n---\n$ARGUMENTS', 'typed')

    def test_committed_commands_declare_both_type_boundaries(self):
        import prompt_commands as pc
        commands = pc.load_commands()
        self.assertTrue(commands)
        for command in commands.values():
            with self.subTest(command=command.name):
                self.assertTrue(command.input_types)
                self.assertTrue(command.output_types)

    def test_factor_command_boundary(self):
        # factor! is a whole-cell derivation command like simplify!: the
        # agent chooses the factors, so it must never be inline-spliceable,
        # and its typed boundary refuses relation input before any agent run.
        import prompt_commands as pc
        cmd = pc.load_commands().get('factor')
        self.assertIsNotNone(cmd)
        self.assertFalse(cmd.expr)
        self.assertIsNone(cmd.direct)
        self.assertEqual(cmd.input_types, ('expression', 'relation'))
        self.assertEqual(cmd.output_types, ('derivation',))
        self.assertEqual(pc.signature(cmd),
                         'expression|relation -> derivation')

    def test_prove_mode_parsed_and_cannot_compose(self):
        import prompt_commands as pc
        cmd = pc.parse_command(
            '---\nname: p\ndescription: d\nmode: prove\n---\n$ARGUMENTS',
            'p')
        self.assertEqual(cmd.mode, 'prove')
        with self.assertRaises(ValueError):
            pc.parse_command(
                '---\nname: p\ndescription: d\nmode: prove\nexpr: true\n'
                '---\n$ARGUMENTS', 'p')

    def test_fresh_field_parsed(self):
        import prompt_commands as pc
        plain = pc.parse_command('---\nname: a\ndescription: d\n---\n$ARGUMENTS',
                                 'a')
        self.assertEqual(plain.fresh, ())
        one = pc.parse_command(
            '---\nname: b\ndescription: d\nfresh: C\n---\n$ARGUMENTS', 'b')
        self.assertEqual(one.fresh, ('C',))
        many = pc.parse_command(
            '---\nname: c\ndescription: d\nfresh: [C, K]\n---\n$ARGUMENTS', 'c')
        self.assertEqual(many.fresh, ('C', 'K'))
        with self.assertRaises(ValueError):
            pc.parse_command(
                '---\nname: e\ndescription: d\nfresh: "\\\\bad"\n---\n$ARGUMENTS',
                'e')

    def test_rejects_missing_frontmatter(self):
        import prompt_commands as pc
        with self.assertRaises(ValueError):
            pc.parse_command('bare body $ARGUMENTS', 'x')

    def test_rejects_missing_description(self):
        import prompt_commands as pc
        with self.assertRaises(ValueError):
            pc.parse_command('---\nname: x\n---\n$ARGUMENTS', 'x')

    def test_rejects_missing_placeholder(self):
        import prompt_commands as pc
        with self.assertRaises(ValueError):
            pc.parse_command('---\nname: x\ndescription: d\n---\nno slot', 'x')

    def test_rejects_reserved_name(self):
        import prompt_commands as pc
        with self.assertRaises(ValueError):
            pc.parse_command('---\nname: do\ndescription: d\n---\n$ARGUMENTS',
                             'do')

    def test_bad_file_is_skipped_not_fatal(self):
        import tempfile
        import prompt_commands as pc
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'good.md'), 'w') as fh:
                fh.write('---\nname: good\ndescription: g\n---\n$ARGUMENTS')
            with open(os.path.join(d, 'bad.md'), 'w') as fh:
                fh.write('no frontmatter, no placeholder')
            reg = pc.load_commands(d)
        self.assertIn('good', reg)
        self.assertNotIn('bad', reg)


class TestPromptCommandDispatch(unittest.TestCase):
    """Kernel dispatch of `name!` cells (mocked agent — offline)."""

    def setUp(self):
        import engine
        self.displays = []
        engine.setHandler(lambda *objs, **kw: self.displays.extend(objs))
        from mathShell import MathShell
        self.shell = MathShell()

    def tearDown(self):
        import engine
        import IPython.display
        engine.setHandler(IPython.display.display)

    def _html(self):
        return ''.join(getattr(d, 'data', str(d)) for d in self.displays)

    def _capture_instruction(self):
        box = {}

        def fake_run(instruction, **kw):
            box['instruction'] = instruction
            box['kwargs'] = kw
            return {'ok': True, 'steps': [], 'assumptions': [],
                    'claims': [], 'final_result': None, 'summary': None}
        return box, fake_run

    def test_command_prefix_renders_template(self):
        # solve! is a plain (non-expr) command: whole-cell prefix -> exec_do
        box, fake = self._capture_instruction()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('solve! 2x = 4', 1, add_to_history=True)
        self.assertIn('Solve', box['instruction'])
        self.assertIn('2x = 4', box['instruction'])
        self.assertEqual(box['kwargs']['chain_goal'], '2x = 4')

    def test_simplify_command_passes_requested_expression_as_chain_goal(self):
        box, fake = self._capture_instruction()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('simplify! (x+1)^2', 1, add_to_history=True)
        self.assertEqual(box['kwargs']['chain_goal'], '(x+1)^2')

    def test_disconnected_simplify_result_never_enters_cell_history(self):
        # The committed-notebook failure shape: a saved command's agent
        # checks a different expression, its designation is refused, then it
        # stops.  The disconnected last step must not become the cell value.
        script = [
            [tool_call('expand', {'expr': '(y+1)^2'}, 'c1')],
            [tool_call('set_result', {'expr': 'y^{2}+2y+1'}, 'c2')],
            [message('done')],
        ]
        with mock.patch.object(openrouter_backend, 'build_model',
                               lambda model_name=None: ScriptedModel(script)):
            self.shell.exec('simplify! (x+1)^2', 17,
                            add_to_history=True)
        with self.assertRaisesRegex(ValueError, 'does not reference'):
            self.shell.resolve_backrefs('[[17]]')
        self.assertEqual(self.shell.ledger.selections, [])
        self.assertIn('no goal-connected result', self._html())

    def test_prove_command_passes_raw_claim_to_harness(self):
        box, fake = self._capture_instruction()
        claim = r'\lim_{n \to \infty} \frac{1}{n} = 0'
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('prove! ' + claim, 1, add_to_history=True)
        self.assertEqual(box['kwargs']['proof_goal'], claim)
        self.assertEqual(box['kwargs']['chain_goal'], claim)
        self.assertIn('Prove ' + claim, box['instruction'])

    def test_plain_do_has_no_chain_goal(self):
        box, fake = self._capture_instruction()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('do! explore anything', 1, add_to_history=True)
        self.assertIsNone(box['kwargs']['chain_goal'])

    def test_prove_command_resolves_backref_in_harness_claim(self):
        self.shell.exec('2 = 2', 1, add_to_history=True)
        box, fake = self._capture_instruction()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('prove! [[1]]', 2, add_to_history=True)
        self.assertNotIn('[[1]]', box['kwargs']['proof_goal'])
        self.assertIn('=', box['kwargs']['proof_goal'])

    def test_backref_resolves_inside_command_args(self):
        self.shell.exec('2 + 3', 1, add_to_history=True)  # result 5
        box, fake = self._capture_instruction()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('solve! [[1]]', 2, add_to_history=True)
        self.assertIn('5', box['instruction'])
        self.assertNotIn('[[1]]', box['instruction'])
        self.assertEqual(box['kwargs']['chain_goal'], '5')

    def test_plain_command_resolves_typed_inline_input_before_agent(self):
        box, fake = self._capture_instruction()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec(
                'solve! {expand! (x+1)^2}=4', 1, add_to_history=True)
        self.assertNotIn('expand!', box['instruction'])
        self.assertIn('x^{2}+2x+1 = 4', box['instruction'])
        # One checked step produces the inner value; another checks its
        # placement on the relation side before solve! sees the input.
        self.assertEqual([s['op'] for s in self.shell.ledger.steps],
                         ['expand', 'expand'])
        self.assertEqual(self.shell.ledger.steps[-1]['result'],
                         'x^{2}+2x+1 = 4')
        self.assertEqual(self.shell.ledger.replay()['status'], 'verified')

    def test_prove_goal_uses_the_resolved_typed_input(self):
        box, fake = self._capture_instruction()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec(
                'prove! {expand! (x-1)^2}=x^2-2x+1', 1,
                add_to_history=True)
        self.assertNotIn('expand!', box['kwargs']['proof_goal'])
        self.assertEqual(box['kwargs']['proof_goal'],
                         'x^{2}-2x+1 = x^{2}-2x+1')

    def test_plain_command_refuses_incompatible_resolved_input(self):
        with mock.patch.object(
                agent_do, 'run_instruction',
                lambda *a, **k: self.fail('outer agent must not run')):
            self.shell.exec('simplify! {expand! x=1}', 1,
                            add_to_history=True)
        self.assertIn('expected expression input, got relation', self._html())

    def test_factor_command_dispatches_and_types_its_input(self):
        box, fake = self._capture_instruction()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('factor! x^3-3x^2+4', 1, add_to_history=True)
        self.assertIn('Factor x^3-3x^2+4', box['instruction'])
        # the typed boundary refuses a shape no factoring works on,
        # before any agent run
        with mock.patch.object(
                agent_do, 'run_instruction',
                lambda *a, **k: self.fail('agent must not run')):
            self.shell.exec('factor! \\{1,2\\}', 2, add_to_history=True)
        self.assertIn('expected expression or relation input, got collection',
                      self._html())

    def test_unregistered_prefix_is_not_a_command(self):
        # `n!` (factorial) must fall through to the math path
        called = []
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: called.append(1)):
            handled = self.shell.dispatch_command('n', '+ 1', 1, False)
        self.assertFalse(handled)
        self.assertEqual(called, [])

    def test_empty_argument_reports_error(self):
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: self.fail('must not run')):
            self.shell.exec('solve!', 1, add_to_history=True)
        self.assertIn('needs an argument', self._html())

    def test_commands_listing_makes_no_api_call(self):
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: self.fail('no agent for list')):
            self.shell.exec('commands!', 1)
        out = self._html()
        self.assertIn('notebook commands', out)
        self.assertIn('int', out)
        self.assertIn('expression -&gt; expression', out)
        self.assertIn('model!', out)

    def test_model_command_sets_configured_routing_for_later_runs(self):
        box, fake = self._capture_instruction()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('model! z-ai/glm-5.2', 1)
            self.shell.exec('do! Hello', 2)
        route = box['kwargs']['route']
        self.assertEqual(route.model, 'z-ai/glm-5.2')
        self.assertEqual(route.providers, ('Cerebras', 'Fireworks'))
        self.assertIn('fallbacks disabled', self._html())

    def test_model_command_provider_arguments_override_config(self):
        self.shell.exec('model! z-ai/glm-5.2, Fireworks, Cerebras, Fireworks',
                        1)
        self.assertEqual(self.shell.model_name, 'z-ai/glm-5.2')
        self.assertEqual(self.shell.model_providers,
                         ('Fireworks', 'Cerebras'))

    def test_model_command_allows_unlisted_model(self):
        self.shell.exec('model! vendor/custom-model', 1)
        self.assertEqual(self.shell.model_name, 'vendor/custom-model')
        self.assertEqual(self.shell.model_providers, ())
        self.assertIn('default provider routing', self._html())

    def test_model_command_without_arguments_guides_to_completion(self):
        original = self.shell.model_name
        self.shell.exec('model!', 1)
        self.assertEqual(self.shell.model_name, original)
        self.assertIn('press <kbd>Tab</kbd>', self._html())

    def test_model_change_notifies_frontend_handler(self):
        changes = []
        self.shell.model_change_handler = (
            lambda model, providers: changes.append((model, providers)))
        self.shell.exec('model! z-ai/glm-5.2', 1)
        self.assertEqual(changes, [
            ('z-ai/glm-5.2', ('Cerebras', 'Fireworks')),
        ])

    def test_model_command_rejects_empty_provider(self):
        original = self.shell.model_name
        self.shell.exec('model! z-ai/glm-5.2,', 1)
        self.assertEqual(self.shell.model_name, original)
        self.assertIn('model! error', self._html())

    def test_command_steps_land_in_shared_ledger(self):
        with mock.patch.object(openrouter_backend, 'build_model',
                               lambda model_name=None: ScriptedModel(
                                   SOLVE_SCRIPT)):
            self.shell.exec('solve! 2x + 3 = 7', 2, add_to_history=True)
        self.assertEqual(len(self.shell.ledger.steps), 2)
        self.assertIn('2', self.shell.resolve_backrefs('[[2]]'))


def _ok(result, goal=None):
    """A well-behaved sub-run: when `goal` is given, the run carries one
    transforming step from the goal to the result (the chain an inline
    expr command now requires); without it the run is step-less."""
    steps = [] if goal is None else [
        {'id': 's1', 'op': 'scripted', 'input': goal, 'result': result}]
    return {'ok': True, 'final_result': result, 'assumptions': [],
            'steps': steps, 'summary': None,
            'final_provenance': {
                'status': 'verified', 'source': 'ledger', 'step': 's1',
                'method': 'test-fixture'}}


def _arg_of(instruction):
    """The argument embedded in a rendered int! instruction (first line:
    'Apply symbolic integration for <ARG>.')."""
    first = instruction.split('\n', 1)[0]
    prefix = 'Apply symbolic integration for '
    assert first.startswith(prefix) and first.endswith('.'), first
    return first[len(prefix):-1]


def _fail(error='no scripted answer'):
    return {'ok': False, 'error': error, 'assumptions': [], 'steps': [],
            'final_result': None}


def _never(*args, **kwargs):
    raise AssertionError('the agent must not be called for direct commands')


class TestExprComposite(unittest.TestCase):
    """Inline composition of expr commands ({diff! {int! x^3}}) — the LLM /
    procedural bridge. Agent mocked; the glue is really oracle-checked."""

    def setUp(self):
        import engine
        self.displays = []
        engine.setHandler(lambda *objs, **kw: self.displays.extend(objs))
        from mathShell import MathShell
        self.shell = MathShell()

    def tearDown(self):
        import engine
        import IPython.display
        engine.setHandler(IPython.display.display)

    def _html(self):
        return ''.join(getattr(d, 'data', str(d)) for d in self.displays)

    def test_nested_resolves_inner_to_outer(self):
        # int! is an LLM tactic (mocked); diff! is a DIRECT command since
        # gen 16, so the outer differentiation costs no agent call and its
        # variable is inferred with the minted C excluded
        calls = []

        def fake(instruction, ledger=None, on_step=None, **kw):
            calls.append(instruction)
            if instruction.startswith('Apply symbolic integration'):
                return _ok('\\frac{x^4}{4} + C', _arg_of(instruction))
            return _fail()
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{diff! {int! x^3}}', 1, add_to_history=True)
        self.assertEqual(len(calls), 1)          # int! only; diff! is direct
        self.assertTrue(calls[0].startswith('Apply symbolic integration'))
        chained = self.shell.resolve_backrefs('[[1]]')
        self.assertEqual(core_tactics.equal_exprs(chained, 'x^3')['verdict'],
                         'yes')
        # ledger: the direct differentiate step + the oracle-checked glue
        ops = [s['op'] for s in self.shell.ledger.steps]
        self.assertEqual(ops, ['differentiate', 'expand'])
        for s in self.shell.ledger.steps:
            self.assertEqual(s['check']['status'], 'agree')

    def test_leibniz_prefixed_definite_integral_is_zero_token(self):
        # the user's exact spelling: the operator names the variable (so
        # inference never sees the free 'd'), the FTC bound rule closes
        # the integral, and no agent runs
        self.shell.exec(
            'diff! \\frac{d}{d x} \\int_0^{x^2} \\sqrt{1+t^2} d t',
            1, add_to_history=True)
        ops = [s['op'] for s in self.shell.ledger.steps]
        self.assertEqual(ops, ['differentiate'])
        step = self.shell.ledger.steps[0]
        self.assertEqual(step['check']['status'], 'agree')
        self.assertEqual(core_tactics.equal_exprs(
            step['result'], '2 x \\sqrt{x^{4}+1}')['verdict'], 'yes')
        texts = [a['text'] for a in step.get('assumptions', [])]
        self.assertTrue(any('continuous' in t for t in texts), texts)

    def test_history_backrefs_do_not_accumulate_index_braces(self):
        # Final do!/composite output is future [[n]] input.  Every old plain
        # writer hop added one transparent group around powers/subscripts.
        source = 'x^{3}+C_{1}'
        for execution_count in range(1, 5):
            sym = self.shell.parser.parse(source)
            shown = self.shell.output(
                sym, self.shell.parsedNotation, execution_count, True)
            self.assertEqual(shown, 'x^{3}+C_{1}')
            source = self.shell.resolve_backrefs(
                f'[[{execution_count}]]')
            self.assertEqual(source, 'x^{3}+C_{1}')

    def test_duplicate_subexpression_memoized(self):
        calls = []

        def fake(instruction, ledger=None, on_step=None, **kw):
            calls.append(instruction)
            return _ok('\\frac{x^3}{3} + C', _arg_of(instruction))
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{int! x^2} + {int! x^2}', 1, add_to_history=True)
        self.assertEqual(len(calls), 1)          # identical arg -> one call
        # glue combined and oracle-checked; the second splice mints its own
        # constant (C + C_{1}), NOT the dishonest 2C of one shared C
        step = self.shell.ledger.steps[-1]
        self.assertEqual(step['op'], 'expand')
        flat = step['result'].replace(' ', '')
        self.assertIn('C', flat)
        self.assertIn('C_{', flat)
        self.assertNotIn('2C', flat)
        self.assertEqual(step['check']['status'], 'agree')

    def test_independent_constants_do_not_cancel(self):
        # {int! f} - {int! f} is an arbitrary constant, not 0: the memoised
        # result is spliced twice but each splice mints its own C
        def fake(instruction, ledger=None, on_step=None, **kw):
            return _ok('\\frac{x^3}{3} + C', _arg_of(instruction))
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{int! x^2} - {int! x^2}', 1, add_to_history=True)
        step = self.shell.ledger.steps[-1]
        flat = step['result'].replace(' ', '')
        self.assertNotEqual(flat, '0')
        self.assertIn('C', flat)
        self.assertIn('C_{', flat)
        self.assertEqual(step['check']['status'], 'agree')

    def test_single_command_keeps_plain_constant(self):
        def fake(instruction, ledger=None, on_step=None, **kw):
            return _ok('\\frac{x^3}{3} + C', _arg_of(instruction))
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{int! x^2}', 1, add_to_history=True)
        chained = self.shell.resolve_backrefs('[[1]]')
        self.assertIn('C', chained)
        self.assertNotIn('C_{', chained)         # no gratuitous renaming

    def test_composite_renders_chain_and_prose_assumptions(self):
        # the live report: a Codex cell showed only its final value — the
        # streamed step displays were lost off the kernel thread, and the
        # assumptions line wrapped whole prose sentences in math mode
        def fake(instruction, ledger=None, on_step=None, **kw):
            arg = _arg_of(instruction)
            run = _ok('\\frac{x^4}{4} + C', goal=f'\\int {arg} \\, dx')
            run['steps'] = [
                {'id': 's1', 'op': 'scripted',
                 'input': f'\\int {arg} \\, dx', 'result': 'F',
                 'assumptions': [], 'check': {'status': 'agree'}},
                {'id': 's2', 'op': 'scripted', 'input': 'F',
                 'result': '\\frac{x^4}{4} + C',
                 'assumptions': [], 'check': {'status': 'agree'}},
            ]
            run['final_provenance']['step'] = 's2'
            run['assumptions'] = [
                {'text': 'x^{3} is continuous on [0, 1]',
                 'display': '$x^{3}$ is continuous on $[0, 1]$'}]
            run['premises'] = [
                {'step': 's1', 'input': f'\\int {arg} \\, dx'}]
            return run
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('int! x^3', 1, add_to_history=True)
        html = self._html()
        # the chain table is rendered on the kernel thread from the run's
        # own records, so the cell shows its ledger evidence even when the
        # per-step streaming was lost
        self.assertIn('<code>s1</code>', html)
        self.assertIn('<code>s2</code>', html)
        # prose stays prose: the display field routes only the math spans
        # to MathJax, never the sentence
        self.assertNotIn('$x^{3} is continuous on [0, 1]$', html)
        self.assertIn('is continuous on', html)
        self.assertIn('$x^{3}$', html)
        # the premises boundary renders too
        self.assertIn('stated premise', html)

    def test_user_constant_never_captured(self):
        # a C the user wrote in the cell must stay distinct from the minted one
        def fake(instruction, ledger=None, on_step=None, **kw):
            return _ok('\\frac{x^3}{3} + C', _arg_of(instruction))
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{int! x^2} + C', 1, add_to_history=True)
        step = self.shell.ledger.steps[-1]
        flat = step['result'].replace(' ', '')
        self.assertIn('C_{', flat)               # the minted C was renamed
        self.assertNotIn('2C', flat)
        self.assertEqual(step['check']['status'], 'agree')

    def test_argument_bound_constant_not_renamed(self):
        # nested {int! {int! x}}: the outer result's C refers to the C in its
        # own argument (the inner result) - bound, not minted, so not renamed
        def fake(instruction, ledger=None, on_step=None, **kw):
            if 'frac' in instruction:  # outer call sees the spliced inner result
                return _ok('\\frac{x^3}{6} + Cx + K', _arg_of(instruction))
            return _ok('\\frac{x^2}{2} + C', _arg_of(instruction))
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{int! {int! x}}', 1, add_to_history=True)
        step = self.shell.ledger.steps[-1]
        flat = step['result'].replace(' ', '')
        self.assertNotIn('C_{', flat)            # arg-bound C keeps its name
        self.assertIn('K', flat)                 # undeclared names untouched
        self.assertEqual(step['check']['status'], 'agree')

    def test_whole_cell_expr_routes_to_composite(self):
        calls = []

        def fake(instruction, ledger=None, on_step=None, **kw):
            calls.append(instruction)
            return _ok('\\frac{x^4}{4} + C', _arg_of(instruction))
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('int! x^3', 1, add_to_history=True)  # no braces
        self.assertEqual(len(calls), 1)
        # whole-cell single command: designated result, no glue step
        self.assertEqual(self.shell.ledger.steps, [])
        self.assertIn('x^{4}', self.shell.resolve_backrefs('[[1]]'))

    def test_composite_agent_run_uses_notebook_model_routing(self):
        calls = []

        def fake(instruction, **kwargs):
            calls.append(kwargs)
            return _ok('\\frac{x^3}{3} + C', _arg_of(instruction))

        self.shell.exec('model! z-ai/glm-5.2', 1)
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('int! x^2', 2, add_to_history=True)
        self.assertEqual(calls[0]['route'].model, 'z-ai/glm-5.2')
        self.assertEqual(calls[0]['route'].providers,
                         ('Cerebras', 'Fireworks'))

    def test_non_expr_command_in_composite_refused(self):
        # solve! is not expr; mixed into a composite it must be refused
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: _ok('x')):
            self.shell.exec('{diff! x} + {solve! x}', 1, add_to_history=True)
        self.assertIn('not an inline expr command', self._html())

    def test_factorial_is_not_a_composite(self):
        # the router must not hijack `n!` (n is not a registered command);
        # only a real expr command routes to the composite evaluator
        self.assertFalse(self.shell.has_expr_command('n! + 1'))
        self.assertTrue(self.shell.has_expr_command('{diff! x}'))

    def test_backref_resolves_in_composite_argument(self):
        self.shell.exec('x^3', 1, add_to_history=True)
        # diff! is direct: the backref resolves before the primitive runs
        # and the whole cell costs zero agent calls
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('{diff! [[1]]}', 2, add_to_history=True)
        chained = self.shell.resolve_backrefs('[[2]]')
        self.assertEqual(core_tactics.equal_exprs(chained, '3x^2')['verdict'],
                         'yes')

    def test_agent_failure_surfaces(self):
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: _fail('boom')):
            self.shell.exec('{int! x^3}', 1, add_to_history=True)
        self.assertIn('boom', self._html())

    def test_unverified_agent_result_cannot_enter_composite(self):
        result = _ok('x=999')
        result['final_provenance'] = {
            'status': 'unverified', 'source': 'query-only',
            'reason': 'no transforming step was recorded'}
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: result):
            self.shell.exec('{int! x^3}', 1, add_to_history=True)
        self.assertIn('unverified final value', self._html())
        self.assertEqual(self.shell.ledger.steps, [])

    def test_resolver_call_cap(self):
        # unit-level: a low cap trips on distinct sub-expressions
        import primitives
        import expr_commands
        from notation import Notation
        cmds = self.shell.commands
        sym, notation = primitives.parse_latex('{int! x} + {int! x^2}')
        r = expr_commands.ExprResolver(
            notation, Notation(), cmds, self.shell.ledger, None,
            lambda instruction, **k: _ok('F', _arg_of(instruction)),
            max_calls=1)
        with self.assertRaises(expr_commands.ExprCommandError):
            r(sym)

    def test_non_closing_subrun_refused_and_summary_surfaced(self):
        # the run's only verified step answers a DIFFERENT question than
        # the cell asked; its result must not become the cell's value
        run = {'ok': True, 'final_result': '0', 'assumptions': [],
               'steps': [{'id': 's1', 'op': 'limit_table',
                          'input': '\\lim_{n \\to \\infty} \\frac{1}{n}',
                          'result': '0'}],
               'summary': 'the tactics do not close this; a telescoping '
                          'move is missing',
               'final_provenance': {'status': 'verified', 'source': 'ledger',
                                    'step': 's1', 'method': 'last-step'}}
        with mock.patch.object(agent_do, 'run_instruction',
                               lambda *a, **k: run):
            self.shell.exec('{int! x^3}', 1, add_to_history=True)
        html = self._html()
        self.assertIn('did not close', html)
        self.assertIn('telescoping move is missing', html)
        self.assertEqual(self.shell.ledger.steps, [])

    def test_operator_wrapped_root_step_accepted(self):
        # agents legitimately restate a bare integrand goal inside \int
        def fake(instruction, ledger=None, on_step=None, **kw):
            arg = _arg_of(instruction)
            run = _ok('\\frac{x^4}{4} + C',
                      goal=f'\\int {arg} \\, dx')
            run['final_provenance']['method'] = 'last-step'
            return run
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{int! x^3}', 1, add_to_history=True)
        # accepted: no goal-chain refusal, and the designated result is
        # chainable (a single command records no glue step)
        self.assertNotIn('did not close', self._html())
        self.assertIn('C', self.shell.resolve_backrefs('[[1]]'))

    def test_whole_cell_lim_ellipsis_closes_via_sum_tactics(self):
        # the original failing notebook cell, end to end through the shell
        # composite path with a scripted agent (offline)
        with mock.patch.object(openrouter_backend, 'build_model',
                               lambda model_name=None: ScriptedModel(
                                   list(LIM_SUM_SCRIPT))):
            self.shell.exec('lim! ' + LIM_SUM_EXPR, 1, add_to_history=True)
        html = self._html()
        self.assertNotIn('do! error', html)
        ops = [s['op'] for s in self.shell.ledger.steps]
        self.assertEqual(ops, ['sum_from_ellipsis', 'sum_telescope',
                               'limit_table'])
        chained = self.shell.resolve_backrefs('[[1]]')
        self.assertEqual(core_tactics.equal_exprs(chained, '1')['verdict'],
                         'yes')
        # the pattern-continuation assumption is surfaced on the cell
        self.assertIn('continues the pattern', html)


class TestLiveLogFolding(unittest.TestCase):
    """The streamed turn log is decoration; the chain table is the record.

    A closed run therefore re-emits its stream collapsed, so the artifact is
    not buried under its own rehearsal (measured: 60 streamed lines above the
    table that restates them). A run that did not close keeps the stream
    expanded — there it is the only account of what happened.
    """

    def setUp(self):
        import engine
        self.displays = []
        # the kernel clears with wait=True, so a fold really does replace
        # everything shown so far; a handler that only appends would hide
        # exactly the duplication these tests are about
        def handler(*objs, **kw):
            if kw.get('clear_output'):
                self.displays.clear()
            self.displays.extend(objs)
        engine.setHandler(handler)
        from mathShell import MathShell
        self.shell = MathShell()

    def tearDown(self):
        import engine
        import IPython.display
        engine.setHandler(IPython.display.display)

    def _html(self):
        return ''.join(getattr(d, 'data', str(d)) for d in self.displays)

    def _steps(self, n):
        return [{'id': f's{i}', 'hash': f'h{i}', 'op': 'expand',
                 'check': {'status': 'agree'}, 'input': f'x^{i}',
                 'result': f'x^{i + 1}', 'assumptions': [], 'continues': True}
                for i in range(1, n + 1)]

    def _run(self, steps, result='x^{4}'):
        """A fake run that streams like a real one before returning."""
        def fake(instruction, ledger=None, on_step=None, **kw):
            for step in steps:
                if on_step is not None:
                    on_step(step)
            return {'ok': True, 'final_result': result, 'assumptions': [],
                    'steps': steps, 'summary': None,
                    'final_provenance': {'status': 'verified',
                                         'source': 'ledger', 'step': 's1',
                                         'method': 'test-fixture'}}
        return fake

    def test_closed_run_folds_the_live_log_into_one_group(self):
        steps = self._steps(4)
        with mock.patch.object(agent_do, 'run_instruction',
                               self._run(steps)):
            self.shell.exec('do! integrate x^3', 1, add_to_history=True)
        html = self._html()
        self.assertEqual(1, html.count('<details'))
        self.assertIn('agent turns', html)
        self.assertIn('4 steps', html)
        self.assertIn('4 checked', html)
        # every streamed line survives, once, inside the group
        head, _, tail = html.partition('</details>')
        for step in steps:
            self.assertEqual(1, html.count(f"{step['id']}#{step['hash']}"),
                             step['id'])
            self.assertIn(f"{step['id']}#{step['hash']}", head)
        # and the record itself is still published, outside the fold
        self.assertIn('verified chain', tail)

    def _chain(self, goal, result, n=4):
        """`n` LINKED steps from `goal` to `result`.

        The composite resolver refuses a final value that no checked chain
        connects to the request, so a fixture of unlinked steps never
        reaches the fold at all — it errors first.
        """
        points = ([goal] + [f'u_{{{i}}}' for i in range(1, n)] + [result])
        return [{'id': f's{i + 1}', 'hash': f'h{i + 1}', 'op': 'expand',
                 'check': {'status': 'agree'}, 'input': points[i],
                 'result': points[i + 1], 'assumptions': [],
                 'continues': True}
                for i in range(n)]

    def test_an_expr_command_cell_folds_too(self):
        """`int!` declares `expr: true`, so a whole-cell `int! ...` is
        resolved by exec_composite, NOT exec_do — the reported cell took
        that path. Gen 67's lesson is that a fix landing only in exec_do
        gets missed here, three times over."""
        result = '\\frac{x^{4}}{4} + C'
        seen = {}

        def fake(instruction, ledger=None, on_step=None, **kw):
            steps = self._chain(_arg_of(instruction), result)
            seen['steps'] = steps
            for step in steps:
                on_step(step)
            return {'ok': True, 'final_result': result, 'assumptions': [],
                    'steps': steps, 'summary': None,
                    'final_provenance': {'status': 'verified',
                                         'source': 'ledger', 'step': 's4',
                                         'method': 'test-fixture'}}
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('int! x^3', 1, add_to_history=True)
        html = self._html()
        self.assertEqual(1, html.count('<details'), html[:400])
        self.assertIn('4 steps', html)
        head, _, tail = html.partition('</details>')
        for step in seen['steps']:
            self.assertIn(f"{step['id']}#{step['hash']}", head)
        self.assertIn('verified chain', tail)

    def test_a_run_that_did_not_close_keeps_its_log_expanded(self):
        steps = self._steps(3)

        def fake(instruction, ledger=None, on_step=None, **kw):
            for step in steps:
                on_step(step)
            return {'ok': True, 'final_result': None, 'assumptions': [],
                    'steps': steps, 'summary': None,
                    'final_provenance': {'source': 'open',
                                         'reason': 'no route closed it'}}
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('do! integrate x^3', 1, add_to_history=True)
        html = self._html()
        self.assertNotIn('<details', html)
        for step in steps:
            self.assertIn(f"{step['id']}#{step['hash']}", html)

    def test_a_handler_that_cannot_clear_never_duplicates_the_log(self):
        """Console and embedding handlers reject the clear keyword. There the
        fold must decline outright, not print the log a second time."""
        import engine
        shown = []

        def console(*objs, **kw):
            if 'clear_output' in kw:
                raise TypeError('unexpected keyword')
            shown.extend(objs)
        engine.setHandler(console)
        log = mathShell._LiveLog(self.shell)
        for step in self._steps(3):
            log(step)
        self.assertEqual(3, len(shown))
        self.assertFalse(log.fold(True))
        self.assertEqual(3, len(shown))

    def test_evidence_published_mid_cell_forfeits_the_fold(self):
        """Resolving inline commands inside another command's argument shows
        the inner chains and only then runs the outer instruction. The swap
        clears the cell wholesale, so it must not fire afterwards."""
        log = mathShell._LiveLog(self.shell)
        for step in self._steps(3):
            log(step)
        log.disable()
        self.assertFalse(log.fold(True))

    def test_one_streamed_line_is_left_alone(self):
        log = mathShell._LiveLog(self.shell)
        log(self._steps(1)[0])
        self.assertFalse(log.fold(True))


class TestChainsToGoal(unittest.TestCase):
    """The goal-coverage gate on inline expr sub-runs."""

    def _steps(self, *triples):
        return [{'id': f's{i}', 'op': 'scripted', 'input': a, 'result': b}
                for i, (a, b) in enumerate(triples, 1)]

    def test_connected_chain_accepted(self):
        import expr_commands as ec
        steps = self._steps(('x^{3}', '3x^{2}'), ('3x^2', '6x'))
        self.assertTrue(ec._chains_to_goal(steps, 's2', 'x^3'))

    def test_disconnected_final_step_rejected(self):
        import expr_commands as ec
        steps = self._steps(('x^{3}', '3x^{2}'), ('y', '1'))
        self.assertFalse(ec._chains_to_goal(steps, 's2', 'x^3'))

    def test_branch_piece_rejected(self):
        # the lim! failure mode: a split produced pieces; a piece result
        # is not an answer to the whole
        import expr_commands as ec
        steps = self._steps(
            ('\\lim_{x \\to 0}(\\frac{\\sin x}{x} + x^2)',
             '\\lim_{x \\to 0} \\frac{\\sin x}{x} + \\lim_{x \\to 0} x^2'),
            ('\\lim_{x \\to 0} x^2', '0'))
        self.assertFalse(ec._chains_to_goal(
            steps, 's2', '\\lim_{x \\to 0}(\\frac{\\sin x}{x} + x^2)'))

    def test_no_steps_rejected(self):
        import expr_commands as ec
        self.assertFalse(ec._chains_to_goal([], 's1', 'x'))

    def test_integral_boundary_hops_accepted(self):
        # the live int! chain shape: rewrite results are \int-wrapped
        # while the next inputs are their bare integrands, and assemble
        # consumes the linearity input. Linkage inherits the ledger's
        # chaining convention; strict spelling identity broke every
        # honest hop across an \int boundary.
        import expr_commands as ec
        steps = self._steps(
            ('\\frac{1}{x^{\\frac{1}{2}}+x^{\\frac{1}{3}}}',
             '\\int \\frac{6u^5}{u^3+u^2} \\, d u'),
            ('\\frac{6u^5}{u^3+u^2}',
             '\\int \\left(6u^2-6u+6-\\frac{6}{u+1}\\right) \\, d u'),
            ('6u^2-6u+6-\\frac{6}{u+1}',
             '2u^{3}-3u^{2}+6u-6 \\ln\\left ((u+1) \\right )+C'),
            ('2u^{3}-3u^{2}+6u-6 \\ln\\left ((u+1) \\right )+C',
             '2x^{\\frac {1} {2}}-3x^{\\frac {1} {3}}+6x^{\\frac {1} {6}}'
             '-6 \\ln\\left ((x^{\\frac {1} {6}}+1) \\right )+C'))
        self.assertTrue(ec._chains_to_goal(
            steps, 's4',
            '\\int\\frac {dx} {(x^{\\frac {1} {2}}+x^{\\frac {1} {3}})}'))
        # a linearity PIECE still cannot stand in for the whole
        self.assertFalse(ec._chains_to_goal(
            self._steps(('6u^2-6u+6-\\frac{6}{u+1}',
                         '\\int 6u^{2} \\, d u - \\int 6u \\, d u '
                         '+ \\int 6 \\, d u - \\int \\frac {6} {u+1} '
                         '\\, d u'),
                        ('6u^2', '2u^{3} + C')),
            's2',
            '\\int\\frac {dx} {(x^{\\frac {1} {2}}+x^{\\frac {1} {3}})}'))

    def test_body_rooted_chain_cannot_establish_a_definite_integral(self):
        # live gen-64 probe: the agent derived F from the bare integrand,
        # substituted one bound, and the cell showed a green "verified"
        # value that was F(upper) alone — right only because F(lower)
        # happened to be 0. On other bounds the same moves admit a wrong
        # number, so a body-rooted chain never establishes the bounded
        # integral.
        import expr_commands as ec
        steps = self._steps(
            ('x^{2}', '\\frac {1} {3}x^{3} + C'),
            ('\\frac {1} {3}x^{3} + C', '\\frac {1} {3}(2)^{3}+C'),
            ('\\frac {1} {3}(2)^{3}+C', '\\frac {1} {3}(2)^{3}'),
            ('\\frac {1} {3}(2)^{3}', '\\frac {8} {3}'))
        self.assertFalse(ec._chains_to_goal(
            steps, 's4', '\\int_1^2 x^{2} \\, dx'))

    def test_body_rooted_chain_cannot_establish_a_limit(self):
        # same binder-family hole: evaluating the body at any point is a
        # recordable chain, and its value must not admit as "the limit"
        import expr_commands as ec
        steps = self._steps(
            ('\\frac{x^2-4}{x-2}', 'x+2'),
            ('x+2', '(2)+2'),
            ('(2)+2', '4'))
        self.assertFalse(ec._chains_to_goal(
            steps, 's3', '\\lim_{x \\to 2} \\frac{x^2-4}{x-2}'))

    def test_integrand_rooted_chain_still_establishes_the_indefinite(self):
        # the honest antiderivative chain roots at its bare integrand and
        # the integrating step itself is derivative-checked; the
        # establishes tightening must not refuse it
        import expr_commands as ec
        steps = self._steps(('x^{2}', '\\frac {1} {3}x^{3} + C'))
        self.assertTrue(ec._chains_to_goal(steps, 's1',
                                           '\\int x^{2} \\, dx'))

    def test_definite_chain_roots_at_the_definite_integral(self):
        # the honest route: integrate_definite consumes the bounded
        # integral itself, so its chain admits directly
        import expr_commands as ec
        steps = self._steps(
            ('x^{2}', '\\frac {1} {3}x^{3} + C'),
            ('\\int_1^2 x^{2} \\, dx',
             '\\left(\\frac {1} {3}(2)^{3}+C\\right) - '
             '\\left(\\frac {1} {3}(1)^{3}+C\\right)'),
            ('\\left(\\frac {1} {3}(2)^{3}+C\\right) - '
             '\\left(\\frac {1} {3}(1)^{3}+C\\right)', '\\frac {7} {3}'))
        self.assertTrue(ec._chains_to_goal(
            steps, 's3', '\\int_1^2 x^{2} \\, dx'))

    def test_bracket_respelling_hop_accepted(self):
        # second live model, same cell: the agent retyped the assemble
        # result WITHOUT its decorative per-piece \left(...\right)
        # wrappers before back-substituting. Bracket respellings of one
        # structure must chain; a binding CHANGE must not.
        import expr_commands as ec
        from ledger import _chain_links
        self.assertTrue(_chain_links(
            '\\left(2u^{3}\\right) - \\left(3u^{2}\\right) + '
            '\\left(6u\\right) - '
            '\\left(6 \\ln\\left ((u+1) \\right )\\right) + C',
            '2u^{3} - 3u^{2} + 6u - 6 \\ln\\left ((u+1) \\right ) + C'))
        # the stripper re-encodes child boundaries: dropping LOAD-BEARING
        # parens is a different expression, not a respelling
        self.assertFalse(_chain_links('(a+b)c', 'a+bc'))
        self.assertFalse(_chain_links('|x|', 'x'))
        steps = self._steps(
            ('\\frac{1}{x^{1/2}+x^{1/3}}',
             '\\int \\left(6u^2 - 6u + 6 - \\frac{6}{u+1}\\right) \\, d u'),
            ('6u^2 - 6u + 6 - \\frac{6}{u+1}',
             '\\left(2u^{3}\\right) - \\left(3u^{2}\\right) + '
             '\\left(6u\\right) - '
             '\\left(6 \\ln\\left ((u+1) \\right )\\right) + C'),
            ('2u^{3} - 3u^{2} + 6u - 6 \\ln\\left ((u+1) \\right ) + C',
             '2x^{\\frac {1} {2}}-3x^{\\frac {1} {3}}+C'
             '+6x^{\\frac {1} {6}}'
             '-6 \\ln\\left ((x^{\\frac {1} {6}}+1) \\right )'))
        self.assertTrue(ec._chains_to_goal(
            steps, 's3',
            '\\int\\frac {dx} {(x^{\\frac {1} {2}}+x^{\\frac {1} {3}})}'))

    def test_cdot_respelling_hop_accepted(self):
        # live int! \int dx/(2\sin x - \cos x + 5): the agent rewrote a
        # bare constant integrand as \frac{\sqrt{5}}{5} \cdot 1, and the
        # explicit-\cdot presentation marking severed the verified chain
        # at the \int-boundary hop — every step green, closure refused.
        # The marking is display-only; linkage must be blind to it.
        from ledger import _chain_links
        self.assertTrue(_chain_links(
            '\\int \\frac{\\sqrt{5}}{5} \\cdot 1 \\, d v',
            '\\frac{\\sqrt{5}}{5} \\cdot 1'))
        self.assertTrue(_chain_links('a \\cdot b', 'a b'))
        # blindness covers the dot marking only, never structure
        self.assertFalse(_chain_links('a \\cdot b', 'a + b'))


class TestDirectCommands(unittest.TestCase):
    """The zero-token tier (gen 16): a `direct: <primitive>` command IS one
    verified primitive - no agent run, ledger step + oracle check for free."""

    def setUp(self):
        import engine
        self.displays = []
        engine.setHandler(lambda *objs, **kw: self.displays.extend(objs))
        from mathShell import MathShell
        self.shell = MathShell()

    def tearDown(self):
        import engine
        import IPython.display
        engine.setHandler(IPython.display.display)

    def _html(self):
        return ''.join(getattr(d, 'data', str(d)) for d in self.displays)

    def test_direct_expand_costs_no_tokens(self):
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('{expand! (1+x)(1-x)} + 2x^2', 1,
                            add_to_history=True)
        ops = [s['op'] for s in self.shell.ledger.steps]
        self.assertEqual(ops, ['expand', 'expand'])  # direct step + glue
        final = self.shell.ledger.steps[-1]
        self.assertEqual(final['result'].replace(' ', ''), 'x^{2}+1')
        for s in self.shell.ledger.steps:
            self.assertEqual(s['check']['status'], 'agree')

    def test_direct_diff_whole_cell(self):
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('diff! x^2', 1, add_to_history=True)  # no braces
        ops = [s['op'] for s in self.shell.ledger.steps]
        self.assertEqual(ops, ['differentiate'])
        self.assertEqual(
            self.shell.ledger.steps[0]['result'].replace(' ', ''), '2x')
        self.assertEqual(
            self.shell.resolve_backrefs('[[1]]').replace(' ', ''), '2x')

    def test_direct_inside_direct(self):
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('{expand! {diff! x^3} x}', 1, add_to_history=True)
        final = self.shell.ledger.steps[-1]
        self.assertEqual(final['result'].replace(' ', ''), '3x^{3}')

    def test_typed_outer_expr_refuses_relation_input_before_agent(self):
        with mock.patch.object(
                agent_do, 'run_instruction',
                lambda *a, **k: self.fail('integration agent must not run')):
            self.shell.exec('{int! {expand! x=1}}', 1,
                            add_to_history=True)
        self.assertIn('expected expression input, got relation', self._html())

    def test_declared_output_type_is_checked_before_splicing(self):
        import expr_commands
        import primitives
        import prompt_commands as pc
        from notation import Notation
        cmd = pc.parse_command(
            '---\nname: ev\ndescription: d\ndirect: evaluate\n'
            'input: expression\noutput: relation\n---\ndocs', 'ev')
        sym, notation = primitives.parse_latex(
            '{ev! 2^{3}}', command_names={'ev': cmd})
        resolver = expr_commands.ExprResolver(
            notation, Notation(), {'ev': cmd}, None, None, _never)
        with self.assertRaisesRegex(
                expr_commands.ExprCommandError,
                'expected relation output, got expression'):
            resolver(sym)

    def test_command_type_distinguishes_judgment_from_relation_list(self):
        # Arithmetic evidence plus an unknown prose-like predicate is not a
        # typed classification judgment.  The command boundary must not call
        # it a relation list merely because it uses \land syntax.
        import expr_commands
        self.assertEqual(
            expr_commands.latex_type(
                r'x=-1 \land \operatorname{maximum}(x)'),
            'list')

    def test_minted_constant_excluded_from_var_inference(self):
        # {diff! {int! x^3}}: the inner splice contains the minted C; the
        # direct differentiate must infer x, not refuse as ambiguous
        def fake(instruction, ledger=None, on_step=None, **kw):
            return _ok('\\frac{x^4}{4} + C', _arg_of(instruction))
        with mock.patch.object(agent_do, 'run_instruction', fake):
            self.shell.exec('{diff! {int! x^3}}', 1, add_to_history=True)
        step = self.shell.ledger.steps[0]
        self.assertEqual(step['op'], 'differentiate')
        self.assertEqual(step['args'].get('var'), 'x')

    def test_user_constant_still_ambiguous(self):
        # a plain user-written second symbol is NOT minted: refuse, don't guess
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('{diff! x^2 + C}', 1, add_to_history=True)
        self.assertIn('ambiguous variable', self._html())
        # the refusal teaches the explicit-parameter syntax
        self.assertIn('[x]', self._html())

    def test_explicit_var_parameter_whole_cell(self):
        # diff! [x] <expr> chooses the variable; no inference needed
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('diff! [x] Cx^{2} + C', 1, add_to_history=True)
        step = self.shell.ledger.steps[0]
        self.assertEqual(step['op'], 'differentiate')
        self.assertEqual(step['args'].get('var'), 'x')

    def test_explicit_var_parameter_inline(self):
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('{diff! [C] x C} + 1', 1, add_to_history=True)
        self.assertEqual(self.shell.ledger.steps[0]['args'].get('var'), 'C')

    def test_ledger_constant_breaks_inference_tie(self):
        # an antiderivative chained via [[n]] carries its minted C into a
        # later cell; the ledger's constant provenance disambiguates
        self.shell.ledger.record(
            integration_tactics.integrate_power_rule('x^2', 'x'))
        self.assertEqual(self.shell.ledger.steps[0].get('constant'), 'C')
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('diff! \\frac{x^{3}}{3} + C', 2,
                            add_to_history=True)
        diff_step = self.shell.ledger.steps[1]
        self.assertEqual(diff_step['op'], 'differentiate')
        self.assertEqual(diff_step['args'].get('var'), 'x')

    def test_var_parameter_on_non_var_command_refused(self):
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('{expand! [x] (x+1)^2}', 3, add_to_history=True)
        self.assertIn('does not take a [var]', self._html())

    def test_var_parameter_must_be_plain_symbol(self):
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('{diff! [x^2] x^3}', 4, add_to_history=True)
        self.assertIn('single plain variable name', self._html())

    def test_domain_differs_renders_amber_not_red(self):
        # collapsing d/dx of a ln-carrying antiderivative to its rational
        # form is a domain extension: conditional, not a failure
        step = {'id': 's1', 'hash': 'h', 'op': 'expand', 'args': {},
                'input': 'i', 'result': 'r', 'assumptions': [],
                'continues': None, 'check': {'status': 'domain-differs'}}
        out = self.shell.render_do_step(step)
        self.assertIn('[D!]', out)
        self.assertIn('#b65c00', out)
        self.assertNotIn('#c00"', out)

    def test_no_variable_refused(self):
        with mock.patch.object(agent_do, 'run_instruction', _never):
            self.shell.exec('{diff! 2 + 3}', 1, add_to_history=True)
        self.assertIn('no variable', self._html())

    def test_primitive_failure_surfaces(self):
        # unit-level: a direct command whose primitive refuses reports the
        # primitive's error through the composite error path
        import primitives
        import expr_commands
        import prompt_commands as pc
        from notation import Notation
        cmds = {'fq': pc.PromptCommand('fq', 'd', '', True, (),
                                       'factor_quadratic')}
        sym, notation = primitives.parse_latex(
            '{fq! x^3 + 1}', command_names=cmds)
        r = expr_commands.ExprResolver(
            notation, Notation(), cmds, None, None, _never)
        with self.assertRaises(expr_commands.ExprCommandError) as ctx:
            r(sym)
        self.assertIn('factor_quadratic', str(ctx.exception))

    def test_direct_evaluate_adapter(self):
        # unit-level: evaluate as a direct command splices a numeric value
        import primitives
        import expr_commands
        import prompt_commands as pc
        from notation import Notation
        cmds = {'ev': pc.PromptCommand('ev', 'd', '', True, (), 'evaluate')}
        sym, notation = primitives.parse_latex(
            '{ev! 2^{10}} + 1', command_names=cmds)
        out = Notation()
        r = expr_commands.ExprResolver(notation, out, cmds, None, None,
                                       _never)
        root = r(sym)
        from LatexWriter import LaTexWriter
        rec = core_tactics.expand(LaTexWriter(out)(root))
        self.assertEqual(rec['result'].replace(' ', ''), '1025')

    def test_unknown_direct_primitive_rejected_at_parse(self):
        import prompt_commands as pc
        with self.assertRaises(ValueError):
            pc.parse_command(
                '---\nname: z\ndescription: d\ndirect: solve\n---\n', 'z')

    def test_direct_body_needs_no_placeholder(self):
        import prompt_commands as pc
        cmd = pc.parse_command(
            '---\nname: z\ndescription: d\ndirect: expand\n---\njust docs',
            'z')
        self.assertEqual(cmd.direct, 'expand')
        self.assertTrue(cmd.expr)

    def test_direct_assumptions_reach_the_cell(self):
        # a direct primitive's recorded assumptions must surface like an
        # agent sub-run's (factor_quadratic records none; use collect on a
        # rational form? expand of 1/x keeps x != 0 out of scope - assert
        # the plumbing instead: direct_records land on the resolver)
        import primitives
        import expr_commands
        import prompt_commands as pc
        from notation import Notation
        cmds = {'ex': pc.PromptCommand('ex', 'd', '', True, (), 'expand')}
        sym, notation = primitives.parse_latex(
            '{ex! (1+x)^2}', command_names=cmds)
        r = expr_commands.ExprResolver(notation, Notation(), cmds, None,
                                       None, _never)
        r(sym)
        self.assertEqual(len(r.direct_records), 1)
        self.assertIn('assumptions', r.direct_records[0])


class TestUnknownToolSteering(unittest.TestCase):
    def test_hallucinated_tool_names_steer_instead_of_killing_the_run(self):
        # live probe: a strong model called the TACTIC name integrate_table
        # as a raw tool and the SDK's ModelBehaviorError aborted a run that
        # already held three green steps. Unknown tool calls must return to
        # the model as one-turn steering; a tactic-shaped name must point
        # at run_tactic and the owning skill.
        captured = []

        class RecordingModel(ScriptedModel):
            async def get_response(self, system_instructions, input,
                                   *args, **kwargs):
                captured.append(input)
                return await super().get_response(
                    system_instructions, input, *args, **kwargs)

        script = [
            [ResponseFunctionToolCall(
                type='function_call', name='integrate_table',
                arguments=json.dumps({'expr': 'x', 'var': 'x'}),
                call_id='b1', id='b1')],
            [ResponseFunctionToolCall(
                type='function_call', name='frobnicate',
                arguments='{}', call_id='b2', id='b2')],
            [tool_call('expand', {'expr': '(x+1)^2'}, 'c3')],
            [message('recovered')],
        ]
        res = run_instruction('misname some tools',
                              model=RecordingModel(script))
        self.assertTrue(res['ok'], res.get('error'))
        self.assertIsNone(res.get('error'))
        self.assertEqual([s['op'] for s in res['steps']], ['expand'])

        second = json.dumps(captured[1], default=str)
        self.assertIn('is a tactic, not a tool', second)
        self.assertIn('run_tactic', second)
        self.assertIn('integration', second)
        third = json.dumps(captured[2], default=str)
        self.assertIn('does not exist', third)
        self.assertIn('load_skill, run_tactic, comment', third)


@unittest.skipUnless(codex_backend.available(),
                     'the pinned Codex runtime is not installed '
                     '(uv pip install ".[codex]")')
class TestCodexToolSetContract(unittest.TestCase):
    """The exact model-visible tool set, captured from the real runtime.

    Offline: the runtime talks to a loopback server that records the
    request and refuses it. No account, no network, no model. This is the
    check the accepted residual-tool exception rests on - a new native
    tool, an inherited MCP helper, or a lifted restriction must fail here
    before any live model runs.
    """

    def setUp(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.captured = {}
        self.seen = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):                       # noqa: N802
                length = int(self.headers.get('Content-Length', '0'))
                try:
                    outer.captured['body'] = json.loads(
                        self.rfile.read(length))
                except ValueError:
                    outer.captured['body'] = {}
                outer.seen.set()
                payload = json.dumps({'error': {
                    'message': 'capture complete',
                    'type': 'invalid_request_error'}}).encode()
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                return

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        self.session = DoSession()
        self.dispatcher = agent_base.ToolDispatcher(
            agent_do.make_tool_bindings(self.session),
            agent_base.CancellationToken())
        self.base = f'http://127.0.0.1:{server.server_address[1]}/v1'
        self.transport = self._start()
        self.thread = codex_transport.CodexThreadRequest(
            instruction='Call run_tactic once, then stop.',
            developer_instructions=codex_backend.role_policy(
                self.dispatcher.names),
            dynamic_tools=tuple(codex_backend.dynamic_tools(
                self.dispatcher)))

    def _start(self, config=None):
        """A runtime against a fresh home, optionally carrying `config`."""
        home = tempfile.mkdtemp(prefix='toymath-codex-home-')
        self.addCleanup(shutil.rmtree, home, True)
        _, workdir = codex_backend.ensure_home(home=home)
        if config is not None:
            # written after the home is ToyMath's own: the realistic way a
            # dedicated home acquires an MCP server is being edited later
            with open(os.path.join(home, codex_backend.CONFIG_FILE), 'w',
                      encoding='utf-8') as handle:
                handle.write(config)
        transport = codex_transport.AppServerTransport(
            binary=codex_backend.runtime_binary(), home=home, cwd=workdir,
            config_overrides=(codex_backend.CONTAINMENT_OVERRIDES
                              + codex_backend.mcp_overrides(home)
                              + codex_backend.capture_config_overrides(
                                  self.base)),
            env={'TOYMATH_CAPTURE_KEY': 'local-dummy'})
        transport.start()
        self.addCleanup(transport.close)
        return transport

    def _outgoing_tools(self, transport):
        self.seen.clear()
        started = transport.request('thread/start',
                                    transport.thread_params(self.thread))
        thread_id = started['thread']['id']
        turn = transport.request('turn/start', {
            'threadId': thread_id,
            'input': [{'type': 'text', 'text': self.thread.instruction}]})
        self.assertTrue(self.seen.wait(30),
                        'the runtime sent no model request')
        transport.interrupt_turn(thread_id, turn['turn']['id'])
        return [tool.get('name') or tool.get('type')
                for tool in (self.captured['body'].get('tools') or [])]

    def test_the_outgoing_tools_are_toymath_plus_the_reviewed_residuals(self):
        tools = self._outgoing_tools(self.transport)
        self.assertEqual(
            sorted(tools),
            sorted(codex_backend.expected_model_tools(self.dispatcher)),
            'the effective Codex tool set changed; review the accepted '
            'residual-tool exception before shipping this runtime')
        body = self.captured['body']
        # the repository's own instructions never join the math agent
        self.assertNotIn('ToyMath is a LaTeX-native symbolic mathematics',
                         json.dumps(body))
        self.assertIn('Use only the client-provided ToyMath dynamic tools',
                      json.dumps(body))

    def test_a_home_carrying_an_mcp_server_still_offers_only_our_tools(self):
        # a clean temporary home cannot catch inherited MCP configuration.
        # Measured here: without the per-server override this home adds
        # mcp__probe, list_mcp_resources, list_mcp_resource_templates, and
        # read_mcp_resource to what the model can call.
        transport = self._start(
            '[mcp_servers.probe]\ncommand = "/bin/echo"\nargs = ["hi"]\n')
        tools = self._outgoing_tools(transport)
        self.assertEqual(
            sorted(tools),
            sorted(codex_backend.expected_model_tools(self.dispatcher)),
            'an MCP server configured in the Codex home reached the model')

    def test_a_refused_turn_ends_promptly_instead_of_hanging(self):
        # the notification loop against the real app-server: a provider
        # error must terminate the turn, not leave the cell waiting
        started = time.monotonic()
        outcome = self.transport.run_thread(
            self.thread,
            on_tool_call=lambda params: codex_backend.tool_call_result(
                self.dispatcher, params))
        self.assertEqual(outcome.status, 'failed')
        self.assertTrue(outcome.error)
        self.assertLess(time.monotonic() - started, 30)

    def test_the_runtime_reports_its_own_models(self):
        models = self.transport.list_models()
        self.assertTrue(models)
        self.assertTrue(all(model.id for model in models))


def _interrupt_after(seconds):
    """Press Jupyter Stop from a background thread."""
    threading.Timer(seconds, _thread.interrupt_main).start()


#: The model every live test runs on, whatever the developer's `.env` says.
#: ToyMath loads `.env` at import, so an unpinned live run measures whichever
#: model, backend, and sandbox configuration that machine happens to carry -
#: and a red live test then says nothing about ToyMath. `TestLiveDefaults`
#: below checks this pinning offline, including that the id is still one of
#: the configured endpoints.
LIVE_MODEL = 'openai/gpt-5.6-luna'
LIVE_CODEX_MODEL = 'gpt-5.6-luna'

#: Routing is explicit for the same reason: `route` carries backend and model
#: together, so neither `OPENROUTER_MODEL` nor `TOYMATH_AGENT_BACKEND` can
#: redirect a live run - a Codex entry in `.env` must not turn "the live
#: OpenRouter test" into a run against someone's ChatGPT allowance.
LIVE_ROUTE = agent_config.AgentRoute(backend=agent_config.OPENROUTER,
                                     model=LIVE_MODEL)
LIVE_CODEX_ROUTE = agent_config.AgentRoute(backend=agent_config.CODEX,
                                           model=LIVE_CODEX_MODEL)

#: The two process-wide switches that have no argument form. Tracing off: a
#: test run is not a derivation anyone wants in Langfuse (the module already
#: forces this at import; the live classes state their own configuration
#: rather than inherit it). Sandboxes off: whether Deno is installed here
#: would otherwise decide whether the live model sees 7 tools or 9, and
#: these derivations are algebra.
#: The base URL is pinned for the same reason as the model: a developer who
#: points ToyMath at a local server must not thereby turn "the live
#: OpenRouter test" into a run against their own machine.
LIVE_ENV = {'TOYMATH_OBSERVABILITY': 'off', 'TOYMATH_SANDBOX': 'off',
            openrouter_backend.BASE_URL_VAR:
                openrouter_backend.OPENROUTER_BASE_URL}


class LiveRun(object):
    """One live class's pinned configuration, held for each test."""

    ROUTE = None

    def setUp(self):
        patcher = mock.patch.dict(os.environ, LIVE_ENV)
        patcher.start()
        self.addCleanup(patcher.stop)

    def live(self, instruction, **kwargs):
        return run_instruction(instruction, route=self.ROUTE, **kwargs)


class TestLiveDefaults(unittest.TestCase):
    """What the live classes pin, checked without spending anything."""

    @mock.patch.dict(os.environ,
                     {'OPENROUTER_MODEL': 'google/gemini-3.6-flash',
                      'TOYMATH_AGENT_BACKEND': 'codex',
                      'TOYMATH_CODEX_MODEL': 'gpt-5.6-terra',
                      openrouter_backend.BASE_URL_VAR:
                          'http://localhost:11434/v1'})
    def test_a_developers_env_cannot_redirect_a_live_run(self):
        for route, name, model in (
                (LIVE_ROUTE, 'openrouter', LIVE_MODEL),
                (LIVE_CODEX_ROUTE, 'codex', LIVE_CODEX_MODEL)):
            backend = agent_do.resolve_backend(backend=route.backend,
                                               model_name=route.model)
            self.assertEqual(backend.name, name)
            self.assertEqual(backend.describe_model(), model)
        # the endpoint travels with the model: a local base URL in .env must
        # not turn the live OpenRouter test into a run against localhost
        self.assertIsNotNone(openrouter_backend.custom_base_url())
        with mock.patch.dict(os.environ, LIVE_ENV):
            self.assertIsNone(openrouter_backend.custom_base_url())

    @mock.patch.dict(os.environ, {'TOYMATH_SANDBOX': 'auto',
                                  'TOYMATH_OBSERVABILITY': 'on'})
    def test_the_pinned_env_switches_figures_and_tracing_off(self):
        with mock.patch.dict(os.environ, LIVE_ENV):
            self.assertIsNone(plot_sandbox.get_backend())
            self.assertIsNone(plot_sandbox.get_tikz_backend())
            self.assertFalse(observability.active())

    def test_the_pinned_model_is_still_a_configured_endpoint(self):
        # a retired id must fail here, offline, not against the provider
        import model_config
        endpoints = model_config.load_model_config()
        self.assertIsNotNone(model_config.find_model(endpoints, LIVE_MODEL))


@unittest.skipUnless(os.environ.get('TOYMATH_LIVE_TESTS') == '1',
                     'set TOYMATH_LIVE_TESTS=1 for a live OpenRouter test')
class TestLiveOpenRouter(LiveRun, unittest.TestCase):
    """Runs on LIVE_MODEL; only the OPEN_ROUTER key comes from `.env`."""

    ROUTE = LIVE_ROUTE

    def test_solve_linear_equation(self):
        ledger = Ledger()
        res = self.live('Solve 2x + 3 = 7 for x, step by step.', ledger=ledger)
        self.assertTrue(res['ok'], res.get('error'))
        self.assertGreaterEqual(len(res['steps']), 2)
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_stop_interrupts_a_live_run(self):
        # cancellation must not be Codex-only: the same Stop has to reach a
        # real OpenRouter run and return promptly
        ledger = Ledger()
        _interrupt_after(3.0)
        started = time.monotonic()
        res = self.live(
            'Integrate x^5 e^{x} dx by parts, one step at a time, showing '
            'every intermediate result.', ledger=ledger)
        self.assertEqual(res['status'], 'interrupted')
        self.assertIsNone(res['final_result'])
        self.assertLess(time.monotonic() - started, 30)
        self.assertEqual(ledger.replay()['status'], 'verified')


@unittest.skipUnless(os.environ.get('TOYMATH_CODEX_LIVE_TESTS') == '1',
                     'set TOYMATH_CODEX_LIVE_TESTS=1 for a live personal '
                     'Codex test (requires an already authenticated account)')
class TestLiveCodex(LiveRun, unittest.TestCase):
    """Requires a signed-in local Codex account; never starts a login.

    Pinned to the same model as the OpenRouter class, under that account's
    own id for it, rather than to whatever `TOYMATH_CODEX_MODEL` holds here.
    """

    ROUTE = LIVE_CODEX_ROUTE
    _sandbox_home = None

    @classmethod
    def setUpClass(cls):
        if not codex_backend.available():
            raise unittest.SkipTest('the Codex extra is not installed')
        # The module runs against a throwaway Codex home so no offline test
        # can read or rewrite the developer's authenticated one. This class
        # is the exception: the account IS what it measures, so without the
        # real home back it skips as "not signed in" on every machine,
        # including a correctly signed-in one. The runtime caches its home,
        # so whatever the sandbox started has to go first.
        codex_backend.close_runtime()
        cls._enter_real_home()
        try:
            # `usable`, not `logged_in`: an API-key or Bedrock account bills
            # a different credential, and this gate leads to spending
            if not codex_backend.account_status().usable:
                raise unittest.SkipTest(
                    'no managed ChatGPT Codex account is signed in; run '
                    'login! first')
            models = {model.id for model in codex_backend.list_models()}
            if LIVE_CODEX_MODEL not in models:
                raise unittest.SkipTest(
                    f'this Codex account does not offer {LIVE_CODEX_MODEL}')
        except BaseException:
            # a skip raised here means tearDownClass never runs, and the
            # rest of the module would keep talking to the real home
            cls._leave_real_home()
            raise

    @classmethod
    def tearDownClass(cls):
        cls._leave_real_home()

    @classmethod
    def _enter_real_home(cls):
        cls._sandbox_home = os.environ.get(codex_backend.HOME_VAR)
        _set_codex_home(_REAL_CODEX_HOME)

    @classmethod
    def _leave_real_home(cls):
        codex_backend.close_runtime()   # never outlive the home it opened
        _set_codex_home(cls._sandbox_home)

    def test_a_short_derivation_is_verified_and_replays(self):
        ledger = Ledger()
        res = self.live(
            'Expand (x+1)^2 with one tactic call, then set_result.',
            ledger=ledger)
        self.assertTrue(res['ok'], res.get('error'))
        self.assertGreaterEqual(len(res['steps']), 1)
        self.assertTrue(res['final_result'])
        self.assertEqual(res['final_provenance']['status'], 'verified')
        self.assertEqual(ledger.replay()['status'], 'verified')
        # only ToyMath tools ran
        provenance = res.get('outcome_metadata') or {}
        self.assertEqual(provenance.get('native_tool_calls', []), [])

    def test_stop_interrupts_a_live_turn_and_chains_nothing(self):
        ledger = Ledger()
        _interrupt_after(4.0)
        res = self.live(
            'Integrate x^5 e^{x} dx by parts, one step at a time, showing '
            'every intermediate result.', ledger=ledger)
        self.assertEqual(res['status'], 'interrupted')
        self.assertIsNone(res['final_result'])
        committed = len(ledger.steps)
        time.sleep(2)                     # nothing may land afterwards
        self.assertEqual(len(ledger.steps), committed)
        if committed:
            self.assertEqual(ledger.replay()['status'], 'verified')


if __name__ == '__main__':
    unittest.main()
