#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strategy route records: schema, hybrid extractor, matcher, delivery.

The point of moving routing out of skill prose is that these five things
become testable. Everything here is deterministic; the statistical half of
the evidence (does the record change live closure rates) lives in the
record's own `evidence.observations`, never in a one-run red test.
"""
import os
import tempfile
import unittest
from unittest import mock

import agent_do
import cell_input
import primitives
import strategy_routes
import tactic_registry

FIXTURE = 'indefinite-reduction-boundary-v1'
ROUTE = 'indefinite-reduction-with-boundary-term'


def _corpus(name=FIXTURE):
    return strategy_routes.fixtures()[name]


def _run(instruction):
    """One `run_instruction` against a backend that only records what it was
    handed, so the delivery seam is exercised end to end offline."""
    seen = {}

    class _Handle:
        cancelled = False
        status = 'completed'
        final_text = ''
        metadata = {}

        def cancel(self, reason):
            pass

    class _Backend:
        def start(self, request):
            seen['instructions'] = request.developer_instructions
            seen['trace'] = dict(request.trace_metadata or {})
            return _Handle()

    with mock.patch.object(agent_do, 'resolve_backend',
                           lambda **kwargs: _Backend()), \
            mock.patch.object(agent_do, 'wait_interruptibly',
                              lambda handle, *a, **k: _Handle()):
        return agent_do.run_instruction(instruction), seen


class TestRouteSchema(unittest.TestCase):
    def test_the_committed_file_validates(self):
        self.assertTrue(strategy_routes.validate())
        self.assertTrue(strategy_routes.load())

    def test_every_named_tactic_is_registered_and_owned(self):
        """References are checked against the registry, not copied into the
        record: a tactic that moves subject must never leave a stale
        `load_skill` line behind in committed data."""
        for route in strategy_routes.load():
            for stage in route['stages']:
                names = ([stage['tactic']] if stage.get('tactic')
                         else stage.get('tactics') or [])
                for name in names:
                    self.assertIn(name, tactic_registry.BY_NAME)
            for skill in strategy_routes.required_skills(route):
                self.assertIn(skill, {spec.skill
                                      for spec in tactic_registry.TACTICS})
                self.assertNotEqual(skill, 'core')

    def test_controls_are_the_real_run_level_tools(self):
        """`CONTROLS` is duplicated rather than imported (agent_do imports
        this module); this is what stops the duplicate drifting."""
        tools = {tool.name for tool in
                 agent_do.make_tools(agent_do.DoSession())}
        self.assertTrue(set(strategy_routes.CONTROLS) <= tools)
        self.assertNotIn('run_tactic', strategy_routes.CONTROLS)
        self.assertNotIn('load_skill', strategy_routes.CONTROLS)

    def test_evidence_is_structured_counts_not_an_opaque_score(self):
        for route in strategy_routes.load():
            observations = (route.get('evidence') or {}).get('observations')
            for observation in observations or []:
                self.assertIsInstance(observation['successes'], int)
                self.assertIsInstance(observation['runs'], int)
                self.assertLessEqual(observation['successes'],
                                     observation['runs'])

    def _reject(self, document, fragment):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'routes.yaml')
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write(document)
            with self.assertRaises(strategy_routes.StrategyRouteError) as box:
                strategy_routes.load(path)
        self.assertIn(fragment, str(box.exception))

    def test_an_unknown_feature_is_refused(self):
        self._reject("""
version: 1
routes:
  - id: r
    summary: s
    match:
      all:
        - feature: vibes
          min: 1
    stages:
      - id: only
        action: control
        tool: set_result
""", 'unknown feature')

    def test_an_unregistered_tactic_is_refused(self):
        self._reject("""
version: 1
routes:
  - id: r
    summary: s
    match:
      all:
        - feature: indefinite_integral_count
          min: 1
    stages:
      - id: only
        action: tactic
        tactic: solve
""", 'not a registered tactic')

    def test_a_dangling_consumer_label_is_refused(self):
        """Producer/consumer links are the thing a tactic-NAME route loses,
        so an unresolvable one must not be committable."""
        self._reject("""
version: 1
routes:
  - id: r
    summary: s
    match:
      all:
        - feature: indefinite_integral_count
          min: 1
    stages:
      - id: only
        action: tactic
        tactic: expand
        consumes: nothing-produces-this
""", 'which no earlier stage produces')

    def test_a_bare_yaml_on_key_is_named_not_silently_dropped(self):
        """YAML 1.1 resolves `on:` to the boolean True. The obvious spelling
        of the sub-object field must fail loudly, or a record author debugs a
        key nobody can look up."""
        self._reject("""
version: 1
routes:
  - id: r
    summary: s
    match:
      all:
        - feature: indefinite_integral_count
          min: 1
    stages:
      - id: only
        action: tactic
        tactic: expand
        on: the boundary term
""", 'is `target`')

    def test_an_invented_control_is_refused(self):
        self._reject("""
version: 1
routes:
  - id: r
    summary: s
    match:
      all:
        - feature: indefinite_integral_count
          min: 1
    stages:
      - id: only
        action: control
        tool: prove_it
""", 'tool must be one of')


class TestFeatureExtractor(unittest.TestCase):
    """The hybrid extractor's own golden corpus.

    MEASURED, and the reason the extractor is hybrid at all: the motivating
    instruction is an `aligned` environment that `parse_latex` rejects
    outright, and whose only parseable fragment is the trailing `n > 1`. A
    matcher built on predicates over the notation DAG would have nothing to
    look at.
    """

    def test_the_parser_cannot_read_the_motivating_instruction(self):
        text = _corpus()['positive'][0]
        body = text.split('prove', 1)[1].strip()
        with self.assertRaises(primitives.PrimitiveError):
            primitives.parse_latex(body)
        math = [s['latex'] for s in cell_input.prose_segments(text)
                if s['kind'] == 'math']
        self.assertEqual(math, ['n \\gt 1'])

    def test_integral_signs_are_split_by_boundedness(self):
        features = strategy_routes.features(
            r'\int f dx = \int_0^1 g dx + \int\limits_a^b h dx')
        self.assertEqual(features['indefinite_integral_count'], 1)
        self.assertEqual(features['definite_integral_count'], 2)

    def test_additive_terms_are_top_level_only(self):
        """A `-` inside `^{n-1}` or inside `(a+b\\cos x)` is not a term
        boundary; one shared brace/paren scanner decides that everywhere."""
        features = strategy_routes.features(
            r'\int \frac{dx}{(a+b \cos x)^n} = \frac{A \sin x}'
            r'{(a+b \cos x)^{n-1}} + B \int \frac{dx}{(a+b \cos x)^{n-1}}')
        self.assertTrue(features['relation_has_explicit_nonintegral_term'])
        self.assertEqual(features['shifted_integral_count'], 1)

    def test_a_trailing_sentence_is_not_a_mathematical_term(self):
        """A route match runs over a whole instruction, so prose rides along
        on the last additive term; only real notation counts as a term."""
        features = strategy_routes.features(
            r'\int f dx = \int g dx + something worth checking later')
        self.assertFalse(features['relation_has_explicit_nonintegral_term'])

    def test_the_symbol_ask_stops_at_the_first_non_identifier(self):
        self.assertEqual(strategy_routes.features('Find A,B and C')
                         ['asks_for_symbols'], frozenset('ABC'))
        self.assertEqual(strategy_routes.features('find A, B and C')
                         ['asks_for_symbols'], frozenset('ABC'))
        self.assertEqual(strategy_routes.features(
            'Find the antiderivative of A')['asks_for_symbols'], frozenset())

    def test_the_parsed_leg_reads_the_stated_parameter_domain(self):
        self.assertEqual(strategy_routes.features('holds if n > 1.')
                         ['parameter_constraint_count'], 1)
        # an equality is not a domain constraint, and a closed numeric
        # comparison states nothing about a parameter
        self.assertEqual(strategy_routes.features('with n = 3.')
                         ['parameter_constraint_count'], 0)
        self.assertEqual(strategy_routes.features('note 2 < 3.')
                         ['parameter_constraint_count'], 0)

    def test_extraction_never_raises_on_junk(self):
        for text in ('', 'do!', r'\begin{cases} \\ &', '}{)(' * 40):
            strategy_routes.features(text)


class TestMatcher(unittest.TestCase):
    def test_the_exact_motivating_instruction_matches(self):
        """The design's first gate: the exact cell must match conservatively
        even though the parser cannot read its environment."""
        text = _corpus()['positive'][0]
        self.assertEqual([r['id'] for r in strategy_routes.match(text)],
                         [ROUTE])

    def test_every_committed_positive_matches_its_record(self):
        route = next(r for r in strategy_routes.load() if r['id'] == ROUTE)
        for text in _corpus()['positive']:
            self.assertTrue(
                strategy_routes.matches(route,
                                        strategy_routes.features(text)),
                f'positive fixture did not match: {text[:60]!r}')

    def test_every_committed_hard_negative_is_refused(self):
        route = next(r for r in strategy_routes.load() if r['id'] == ROUTE)
        for text in _corpus()['negative']:
            self.assertFalse(
                strategy_routes.matches(route,
                                        strategy_routes.features(text)),
                f'hard negative matched: {text[:60]!r}')

    def test_the_definite_family_stays_with_its_own_tactic(self):
        """The record tells the agent to AVOID `integrate_reduction`; a
        definite reduction formula is exactly that tactic's own shape and
        must never be steered away from it."""
        definite = _corpus()['negative'][0]
        self.assertEqual(strategy_routes.match(definite), ())
        self.assertGreater(
            strategy_routes.features(definite)['definite_integral_count'], 0)

    def test_a_malformed_route_file_leaves_a_run_unsteered(self):
        """Routing must never be able to take a derivation down."""
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'routes.yaml')
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write('version: 1\nroutes: [{id: broken}]\n')
            self.assertEqual(
                strategy_routes.match(_corpus()['positive'][0], path=path), ())
        self.assertEqual(
            strategy_routes.match('x', path=os.path.join(folder, 'gone')), ())


class TestRendering(unittest.TestCase):
    def setUp(self):
        self.routes = strategy_routes.match(_corpus()['positive'][0])
        self.block = strategy_routes.render(self.routes)

    def test_the_rendered_route_carries_purpose_and_linkage(self):
        """What gen 87 measured a deduplicated tactic-name sequence to have
        lost: which sub-object a stage acts on, why one stage feeds the next,
        and which stage is the provenance-bearing endpoint."""
        self.assertIn('the explicit boundary term', self.block)
        self.assertIn('consuming `derivative-identity`', self.block)
        self.assertIn('produces `coefficient-system`', self.block)
        self.assertIn('provenance-endpoint', self.block)
        self.assertIn('required', self.block)

    def test_load_skill_lines_are_derived_from_the_registry(self):
        self.assertIn('`differentiation`', self.block)
        self.assertIn('`equations`', self.block)
        self.assertNotIn('`core`', self.block)

    def test_the_block_states_that_it_is_steering_not_authority(self):
        self.assertIn('STEERING, not authority', self.block)
        self.assertIn('grants no execution permission', self.block)
        self.assertIn('mechanical check', self.block)

    def test_rendering_is_stable_and_empty_without_a_match(self):
        self.assertEqual(self.block, strategy_routes.render(self.routes))
        self.assertEqual(strategy_routes.render(()), '')


class TestDelivery(unittest.TestCase):
    def test_a_matching_run_gets_the_route_in_its_initial_instructions(self):
        """Delivered up front, not behind `load_skill`: a route meant to
        prevent a premature open outcome cannot depend on the agent already
        having made the right discovery call."""
        routes = strategy_routes.match(_corpus()['positive'][0])
        prompt = agent_do.build_prompt(routes=routes)
        self.assertIn('Recorded strategy route', prompt)
        self.assertIn('system_assemble', prompt)

    def test_the_route_costs_zero_always_on_characters(self):
        base = agent_do.build_prompt()
        self.assertNotIn('Recorded strategy route', base)
        self.assertEqual(agent_do.build_prompt(routes=()), base)
        unmatched = strategy_routes.match('int! \\int x^2 dx')
        self.assertEqual(agent_do.build_prompt(routes=unmatched), base)

    def test_the_matched_ids_reach_the_run_metadata(self):
        """Non-ledger metadata, always present: a later failure can then say
        whether guidance was absent, mismatched, or delivered and ignored."""
        result, seen = _run(_corpus()['positive'][0])
        self.assertEqual(result['strategy_routes'], [ROUTE])
        self.assertEqual(seen['trace']['strategy_routes'], ROUTE)
        self.assertIn('Recorded strategy route', seen['instructions'])
        # never a ledger record: steering leaves no step behind
        self.assertEqual(result['steps'], [])

    def test_a_non_matching_run_reports_an_empty_route_list(self):
        result, seen = _run('expand (x+1)^2')
        self.assertEqual(result['strategy_routes'], [])
        self.assertNotIn('strategy_routes', seen['trace'])
        self.assertNotIn('Recorded strategy route', seen['instructions'])


if __name__ == '__main__':
    unittest.main()
