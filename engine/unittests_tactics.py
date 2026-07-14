#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Registry, progressive-skill, and CLI compatibility tests."""
import json
import unittest

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
            'integration': 'tactics.integration',
            'limits': 'tactics.limits',
            'finite_operators': 'tactics.finite_operators',
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

    def test_existing_cli_shapes_are_preserved(self):
        parser = toymath_cli.build_parser()
        apply = parser.parse_args(['apply', '2x+3=7', '-', '3'])
        self.assertEqual((apply.cmd, apply.equation, apply.op, apply.arg),
                         ('apply', '2x+3=7', '-', '3'))
        rewrite = parser.parse_args([
            'rewrite', 'x^2-y^2', 'diff_squares',
            '--direction', 'backward'])
        self.assertEqual(rewrite.direction, 'backward')
        assemble = parser.parse_args([
            'limit_assemble', 'L', '1', '0'])
        self.assertEqual(assemble.values, ['1', '0'])


if __name__ == '__main__':
    unittest.main()
