#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Registry, progressive-skill, and CLI compatibility tests."""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import agent_do
import primitives
import tactic_registry
import tactic_skills
import toymath_cli
from ledger import TRANSFORMING_OPS


class TestTacticRegistry(unittest.TestCase):
    def test_tactics_live_in_their_static_subject_modules(self):
        expected = {
            'core': 'tactics.core',
            'differentiation': 'tactics.differentiation',
            'equations': 'tactics.equations',
            'integration': 'tactics.integration',
            'limits': 'tactics.limits',
            'finite_operators': 'tactics.finite_operators',
            'matrices': 'tactics.matrices',
        }
        for spec in tactic_registry.TACTICS:
            self.assertEqual(spec.function.__module__, expected[spec.skill])
            self.assertFalse(hasattr(primitives, spec.function.__name__))

    def test_registry_is_unique_and_every_skill_exists(self):
        self.assertEqual(len(tactic_registry.BY_NAME),
                         len(tactic_registry.TACTICS))
        self.assertEqual(len(tactic_registry.BY_OP),
                         len(tactic_registry.TACTICS))
        self.assertEqual(tactic_skills.validate(), [])

    def test_transforming_ops_are_derived_from_registry(self):
        expected = {spec.op for spec in tactic_registry.TACTICS
                    if spec.transforming}
        self.assertEqual(TRANSFORMING_OPS, expected)

    def test_registry_invocation_and_replay_share_one_spec(self):
        session = agent_do.DoSession()
        record = tactic_registry.invoke_agent(
            'expand', ['(x+1)^2'], session)
        self.assertTrue(record['ok'])
        self.assertEqual(record['step']['id'], 's1')
        replayed = tactic_registry.replay(record['op'], record['args'])
        self.assertTrue(replayed['ok'])
        self.assertEqual(replayed['result'], record['result'])

    def test_subject_tactic_requires_loaded_skill(self):
        session = agent_do.DoSession()
        refused = tactic_registry.invoke_agent(
            'limit_table', [r'\lim_{n \to \infty} \frac{1}{n}'], session)
        self.assertFalse(refused['ok'])
        self.assertIn('load_skill', refused['error'])
        loaded = agent_do.make_api(session)['load_skill']('limits')
        self.assertIn("Loaded skill 'limits'", loaded)
        accepted = tactic_registry.invoke_agent(
            'limit_table', [r'\lim_{n \to \infty} \frac{1}{n}'], session)
        self.assertTrue(accepted['ok'], accepted.get('error'))

    def test_virtual_loader_resolves_subject_alias_and_tactic_owner(self):
        session = agent_do.DoSession()
        api = agent_do.make_api(session)
        by_subject = api['load_skill']('roots')
        self.assertIn("Loaded skill 'equations'", by_subject)
        self.assertIn("resolved from 'roots'", by_subject)
        self.assertIn('quadratic_roots EXPR VAR', by_subject)
        self.assertIn('equations', session.loaded_skills)

        another = agent_do.DoSession()
        by_tactic = agent_do.make_api(another)['load_skill'](
            'quadratic_roots')
        self.assertIn("Loaded skill 'equations'", by_tactic)
        self.assertIn('equations', another.loaded_skills)

    def test_virtual_loader_unknown_subject_lists_canonical_choices(self):
        reply = agent_do.make_api(agent_do.DoSession())['load_skill'](
            'mystery mathematics')
        self.assertIn('Cannot load skill', reply)
        self.assertIn('available subjects', reply)
        self.assertIn('equations', reply)

    def test_runtime_tool_surface_is_constant_and_small(self):
        tools = agent_do.make_tools(agent_do.DoSession())
        self.assertEqual([tool.name for tool in tools], [
            'load_skill', 'run_tactic', 'comment', 'claim', 'conclude',
            'set_result'])
        prompt = agent_do.build_prompt()
        payload_chars = len(prompt)
        payload_chars += sum(len(json.dumps(tool.params_json_schema,
                                            sort_keys=True))
                             + len(tool.description or '')
                             for tool in tools)
        # Gen 26 baseline was 36,001 characters / 33 tools. Keep enough
        # margin for core guidance without silently returning to that shape.
        self.assertLess(payload_chars, 10000)

    def test_loaded_skill_contains_only_its_generated_interface(self):
        limits = tactic_skills.render('limits')
        self.assertIn('limit_squeeze EXPR LOWER UPPER LOWER_STEP UPPER_STEP',
                      limits)
        self.assertNotIn('integrate_by_parts EXPR', limits)
        equations = tactic_skills.render('equations')
        self.assertIn('quadratic_roots EXPR VAR', equations)
        self.assertNotIn('diff EXPR VAR', equations)

    def test_existing_cli_shapes_are_preserved(self):
        parser = toymath_cli.build_parser()
        apply = parser.parse_args(['apply', '2x+3=7', '-', '3'])
        self.assertEqual((apply.cmd, apply.equation, apply.op, apply.arg),
                         ('apply', '2x+3=7', '-', '3'))
        rewrite = parser.parse_args([
            'rewrite', 'x^2-y^2', 'diff_squares',
            '--direction', 'backward'])
        self.assertEqual(rewrite.direction, 'backward')
        self.assertIsNone(rewrite.at)
        at = parser.parse_args([
            'rewrite', '(x^2-1)(x^2-4)', 'diff_squares', '--at', 'x^2-4'])
        self.assertEqual(at.at, 'x^2-4')
        assemble = parser.parse_args([
            'limit_assemble', 'L', '1', '0'])
        self.assertEqual(assemble.values, ['1', '0'])
        roots = parser.parse_args(['quadratic_roots', 'x^2-1', 'x'])
        self.assertEqual((roots.cmd, roots.expr, roots.var),
                         ('quadratic_roots', 'x^2-1', 'x'))
        branch = parser.parse_args([
            'branch', 's2', 'try another route', '--session', 'work.json'])
        self.assertEqual((branch.cmd, branch.from_step, branch.reason),
                         ('branch', 's2', 'try another route'))

    def test_cli_branch_records_and_replays_marker(self):
        from ledger import Ledger
        from tactics import core

        path = os.path.join(tempfile.mkdtemp(), 'branch.json')
        ledger = Ledger(path)
        ledger.record(core.expand('(x+1)^2'))
        ledger.save()
        output = io.StringIO()
        with redirect_stdout(output):
            code = toymath_cli.main([
                'branch', 's1', 'the substitution route stalled',
                '--session', path])
        self.assertEqual(code, 0)
        rec = json.loads(output.getvalue())
        self.assertEqual((rec['op'], rec['from']), ('branch', 's1'))
        loaded = Ledger(path)
        self.assertEqual(loaded.steps[-1]['op'], 'branch')
        self.assertEqual(loaded.replay()['status'], 'verified')

    def test_cli_markdown_show_uses_persisted_selection_to_fold_path(self):
        from ledger import Ledger
        from tactics import core

        path = os.path.join(tempfile.mkdtemp(), 'topology.json')
        ledger = Ledger(path)
        source = ledger.record(core.expand('(x+1)^2'))
        ledger.record(core.substitute(source['result'], 'x', '1'))
        ledger.record_branch(source['id'], 'numeric detour was not the goal')
        resumed = ledger.record(core.factor_quadratic(
            source['result'], 'x'))
        ledger.record_selection(resumed['result'], {
            'status': 'verified', 'source': 'ledger',
            'step': resumed['id'], 'method': 'exact-result',
        })
        ledger.save()

        output = io.StringIO()
        with redirect_stdout(output):
            code = toymath_cli.main([
                'show', '--format', 'md', '--session', path])
        self.assertEqual(code, 0)
        rendered = output.getvalue()
        self.assertIn('Selected final result `r1` from `s4`', rendered)
        self.assertIn('<details>', rendered)
        self.assertIn('numeric detour was not the goal', rendered)
        self.assertIn('**s2**', rendered)
        self.assertEqual(Ledger(path).replay()['status'], 'verified')

    def test_rewrite_at_flows_through_agent_and_replay(self):
        session = agent_do.DoSession()
        record = tactic_registry.invoke_agent(
            'rewrite',
            ['(x^2-1)(x^2-4)', 'diff_squares', 'forward', 'x^2-4'],
            session)
        self.assertTrue(record['ok'], record.get('error'))
        self.assertEqual(record['at'], 'x^{2}-4')
        self.assertEqual(record['args']['at'], 'x^2-4')
        replayed = tactic_registry.replay(record['op'], record['args'])
        self.assertEqual(replayed['result'], record['result'])
        # records from before the selector replay through the default
        legacy = tactic_registry.replay('rewrite', {
            'expr': 'x^2 - 4', 'lemma': 'diff_squares',
            'direction': 'forward'})
        self.assertTrue(legacy['ok'])
        self.assertEqual(legacy['result'], '(x+2)(x-2)')


if __name__ == '__main__':
    unittest.main()
