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

# Route delivery is what this file measures, and the preparation stage would
# add a second model call in front of every `_run` below. It is exercised in
# its own suite; here the lexical matcher must be the only thing speaking.
os.environ['TOYMATH_PREWARM'] = 'off'

import agent_do                                                    # noqa: E402
import cell_input                                                  # noqa: E402
import ledger as ledger_module                                     # noqa: E402
import primitives                                                  # noqa: E402
import strategy_routes                                             # noqa: E402
import tactic_registry                                             # noqa: E402

FIXTURE = 'indefinite-reduction-boundary-v1'
ROUTE = 'indefinite-reduction-with-boundary-term'


def _corpus(name=FIXTURE):
    return strategy_routes.fixtures()[name]


def _matches(text, route=None):
    route = route or next(r for r in strategy_routes.load()
                          if r['id'] == ROUTE)
    return strategy_routes.matches(route, strategy_routes.features(text))


def _run(instruction, calls=(), **kwargs):
    """One `run_instruction` against a backend that records what it was
    handed and then makes the given tool calls, so both the delivery seam and
    a real ledger are exercised end to end offline."""
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
            seen['replies'] = [request.dispatcher.dispatch(name, arguments)
                               for name, arguments in calls]
            return _Handle()

    with mock.patch.object(agent_do, 'resolve_backend',
                           lambda **kwargs_: _Backend()), \
            mock.patch.object(agent_do, 'wait_interruptibly',
                              lambda handle, *a, **k: _Handle()):
        return agent_do.run_instruction(instruction, **kwargs), seen


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

    def test_a_record_may_not_hardcode_the_symbols_it_matches(self):
        """The prohibition is refused BY NAME, like the bare `on:` key above.

        Measured, and the reason the rule exists: the shipped record named
        `symbols: [A, B, C]`, so the same reduction problem posed in
        `\\alpha,\\beta,\\gamma` extracted its unknowns correctly and still
        did not match — and a route that silently stops matching turns a
        measurement arm into an unrouted control.
        """
        self._reject("""
version: 1
routes:
  - id: r
    summary: s
    match:
      all:
        - feature: asked_symbol_count
          symbols: [A, B, C]
    stages:
      - id: only
        action: control
        tool: set_result
""", 'may NOT name the symbols')

    def test_the_retired_symbol_feature_names_its_replacement(self):
        self._reject("""
version: 1
routes:
  - id: r
    summary: s
    match:
      all:
        - feature: asks_for_symbols
          min: 3
    stages:
      - id: only
        action: control
        tool: set_result
""", 'was retired')

    def test_a_deliberate_non_match_must_carry_its_reason(self):
        """`unmatched` records a known limit. Without a `why` it would be
        indistinguishable from a tolerated recall miss."""
        self._reject("""
version: 1
routes: []
fixtures:
  f:
    positive: ['a']
    negative: ['b']
    unmatched:
      - instruction: 'c'
""", 'needs an instruction and a why')


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

    def _asked(self, text):
        return sorted(strategy_routes._asked_symbols(text))

    def test_the_symbol_ask_reads_a_plain_imperative_list(self):
        self.assertEqual(self._asked('Find A,B and C'), ['A', 'B', 'C'])
        self.assertEqual(self._asked('find A, B and C'), ['A', 'B', 'C'])
        self.assertEqual(strategy_routes.features('Find A, B and C')
                         ['asked_symbol_count'], 3)

    def test_a_noun_phrase_between_the_verb_and_the_letters(self):
        """MEASURED miss, and the widest of the three: collection used to
        break at the first non-identifier token, so ANY noun phrase after
        the ask verb emptied the feature outright."""
        for text in ('Find the constants A, B, C',
                     'Find the values of A,B,C',
                     'Calculate the coefficients A, B and C',
                     'Identify the three constants A, B and C',
                     'determine all unknowns A, B, C'):
            self.assertEqual(self._asked(text), ['A', 'B', 'C'], text)

    def test_the_lead_in_vocabulary_is_closed_not_a_prose_strip(self):
        """The tolerance is a CLOSED list of words that introduce unknowns.
        "the area A of the triangle" is prose around a symbol, not an ask
        for a constant, and a general prose strip would read it as one."""
        self.assertEqual(self._asked('Find the area A of the triangle'), [])
        self.assertEqual(self._asked('What are the roots A and B of x^2-1'),
                         [])
        self.assertEqual(self._asked('Find the antiderivative of A'), [])
        self.assertEqual(self._asked('compute \\int f dx'), [])

    def test_a_non_imperative_ask_is_read(self):
        """MEASURED miss: the first vocabulary was imperative-only."""
        self.assertEqual(self._asked('What are A, B and C?'),
                         ['A', 'B', 'C'])
        self.assertEqual(self._asked('Solve for A, B and C'),
                         ['A', 'B', 'C'])
        self.assertEqual(self._asked('What is A?'), ['A'])

    def test_a_trailing_clause_no_longer_swallows_the_last_symbol(self):
        """MEASURED miss, found after the first four: a constraint stated in
        the SAME sentence rode along on the final token, which then failed
        the identifier test and broke collection one symbol short."""
        self.assertEqual(self._asked('determine A, B, C for n > 2'),
                         ['A', 'B', 'C'])
        self.assertEqual(self._asked('Find A, B and C such that the '
                                     'identity holds'), ['A', 'B', 'C'])
        self.assertEqual(self._asked('Find A and B with n > 2'), ['A', 'B'])

    def test_an_identifier_may_be_greek_or_subscripted_but_never_a_macro(self):
        """Symbol-generic extraction, and the reason it is an ALLOWLIST:
        `\\[A-Za-z]+` would read `\\int`, `\\sin` and `\\frac` as unknowns.
        `\\pi` is deliberately absent — a named constant is never what an
        instruction asks the agent to find."""
        self.assertEqual(self._asked('Find \\alpha, \\beta and \\gamma'),
                         ['\\alpha', '\\beta', '\\gamma'])
        self.assertEqual(self._asked('Find the constants C_1, C_2 and C_3'),
                         ['C_1', 'C_2', 'C_3'])
        self.assertEqual(self._asked('Find \\pi'), [])
        self.assertEqual(self._asked('Find \\frac{1}{2}'), [])

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
        """RECALL. The number this corpus exists to hold: 2/11 before the
        recall round, 11/11 after, measured on these exact instructions.

        It is a detector metric in its own right, and it was never measured
        while precision was. Reach can only be measured on runs that MATCH,
        so a paraphrased instruction silently turns a routed arm into an
        unrouted control and reports the difference as noise.
        """
        route = next(r for r in strategy_routes.load() if r['id'] == ROUTE)
        for text in _corpus()['positive']:
            self.assertTrue(
                strategy_routes.matches(route,
                                        strategy_routes.features(text)),
                f'positive fixture did not match: {text[:60]!r}')
        self.assertGreaterEqual(len(_corpus()['positive']), 11)

    def test_every_committed_hard_negative_is_refused(self):
        """PRECISION, the other half, and the design priority: a wrongly
        matched route spends a run's context steering a problem it does not
        fit. 10/10 both before and after the recall round — the corpus grew
        by five negatives at the same time the widenings landed."""
        route = next(r for r in strategy_routes.load() if r['id'] == ROUTE)
        for text in _corpus()['negative']:
            self.assertFalse(
                strategy_routes.matches(route,
                                        strategy_routes.features(text)),
                f'hard negative matched: {text[:60]!r}')
        self.assertGreaterEqual(len(_corpus()['negative']), 10)

    def test_each_widening_is_attacked_by_a_negative_that_fires_it(self):
        """Recall bought with an unmeasured precision loss would be a bad
        trade, so every widening carries its own hard negative — an
        instruction whose ASK fires (2+ symbols extracted, through the very
        tolerance that was added) and whose SHAPE must still refuse it."""
        firing = [text for text in _corpus()['negative']
                  if strategy_routes.features(text)['asked_symbol_count'] >= 2]
        self.assertGreaterEqual(len(firing), 4)
        for text in firing:
            self.assertFalse(_matches(text), f'negative matched: {text[:60]!r}')
        # symbol-generic extraction, a non-imperative ask, and the
        # trailing-clause cut, each on a shape that is not this route's
        self.assertTrue(any('\\alpha' in text for text in firing))
        self.assertTrue(any('What are' in text for text in firing))
        self.assertTrue(any('for which' in text for text in firing))

    def test_the_definite_analogue_is_refused_by_boundedness_alone(self):
        """The sharpest of the new negatives: same identity, same boundary
        term, same shift, same domain, same three unknowns — and DEFINITE
        integrals. After symbol-generic matching, one predicate is all that
        separates it from the positive, so that predicate is load-bearing."""
        route = next(r for r in strategy_routes.load() if r['id'] == ROUTE)
        definite = next(text for text in _corpus()['negative']
                        if '\\int_0^{\\pi/2}' in text)
        vector = strategy_routes.features(definite)
        failing = [p['feature'] for p in route['match']['all']
                   if not strategy_routes._holds(p, vector)]
        self.assertEqual(failing, ['indefinite_integral_count'])
        self.assertGreaterEqual(vector['asked_symbol_count'], 3)

    def test_a_recorded_non_match_stays_unmatched_and_says_why(self):
        """A deliberate limit, committed so it cannot change silently: an
        instruction that DROPS the stated parameter domain is an ablation of
        the shape's mathematical content, not a rewording of its phrasing,
        and the record keeps requiring the domain."""
        for entry in _corpus().get('unmatched') or ():
            self.assertFalse(_matches(entry['instruction']),
                             entry['instruction'][:60])
            self.assertTrue(entry['why'].strip())
        self.assertTrue(_corpus().get('unmatched'))

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

    def test_the_isolate_stage_says_what_to_do_with_coupled_rows(self):
        """gen 89, from the live run: the agent isolated one constant, then
        declared the task open saying it could not eliminate among the
        remaining rows — while `substitute`, already named in this very
        stage, is exactly that elimination. It also burned a `rewrite_as`
        trying to flip a derived `value = unknown` row, which
        `system_assemble` has read in either order since gen 86."""
        self.assertIn('COUPLED', self.block)
        self.assertIn('substituting an already-isolated value back', self.block)
        self.assertIn('read in both orders', self.block)

    def test_the_matching_stage_forbids_a_retyped_identity(self):
        """The same run hand-typed the identity in a fresh variable because
        `match_coefficients` refused `\\cos x`. The refusal is gone; the
        route says not to retype regardless, because a retyped identity is a
        premise no step derived."""
        self.assertIn('never a stand-in variable retyped by hand', self.block)


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
        self.assertEqual(result['strategy_route_stages'], [])
        self.assertNotIn('strategy_routes', seen['trace'])
        self.assertNotIn('Recorded strategy route', seen['instructions'])


class TestRequiredStageReach(unittest.TestCase):
    """Rank 1's own declared metric, computed from the finished ledger.

    It was rendered as the word "required" and evaluated nowhere. The whole
    risk of building it is one step away and forbidden: refusing a
    designation because a required stage went unreached would make a
    heuristic match over instruction TEXT into authority over what the
    ledger admits. The non-gating property is therefore what these tests
    check, not the arithmetic.
    """

    def setUp(self):
        self.route = next(r for r in strategy_routes.load()
                          if r['id'] == ROUTE)

    def test_reach_is_measured_by_the_registry_op_not_the_stage_name(self):
        """`diff` records `differentiate` and `apply` records
        `apply_both_sides`; a route names tactics, a ledger holds ops, and
        the registry is the only mapping between them."""
        self.assertEqual(tactic_registry.BY_NAME['diff'].op, 'differentiate')
        report = strategy_routes.required_stage_reach(
            [self.route], ['differentiate', 'expand'])
        self.assertEqual(report[0]['route'], ROUTE)
        self.assertEqual(report[0]['required'], 3)
        self.assertEqual(report[0]['reached'], 1)
        self.assertEqual([s['stage'] for s in report[0]['stages'] if
                          s['reached']], ['differentiate-boundary'])

    def test_only_required_tactic_stages_are_counted(self):
        """Optional stages are not failures, and a required CONTROL stage
        records no step of its own — whether the run designated anything is
        already in the run's own result."""
        named = {s['stage'] for s in strategy_routes.required_stage_reach(
            [self.route], [])[0]['stages']}
        self.assertEqual(named, {'differentiate-boundary',
                                 'form-coefficient-system',
                                 'assemble-answer'})
        self.assertNotIn('certify-algebra', named)   # optional
        self.assertNotIn('designate', named)         # control

    def test_no_match_no_report(self):
        self.assertEqual(strategy_routes.required_stage_reach((), ['expand']),
                         [])
        self.assertEqual(strategy_routes.required_stage_reach(None, None), [])

    def test_a_run_that_misses_every_required_stage_still_finishes(self):
        """THE property. A matched route whose required stages never ran
        must change NOTHING: the run still designates, the designation is
        still verified, the ledger still replays, and no step of the reach
        report reaches the ledger."""
        ledger = ledger_module.Ledger()
        result, _ = _run(
            _corpus()['positive'][0],
            calls=[('run_tactic', {'tactic': 'expand',
                                   'arguments': ['(x+1)^2']}),
                   ('set_result', {'expr': 'x^2+2x+1'})],
            ledger=ledger)

        self.assertEqual(result['strategy_routes'], [ROUTE])
        report = result['strategy_route_stages']
        self.assertEqual(report[0]['reached'], 0)
        self.assertEqual(report[0]['required'], 3)

        # ... and the run is untouched by that
        self.assertTrue(result['ok'])
        self.assertEqual(result['final_result'], 'x^2+2x+1')
        self.assertEqual(result['final_provenance']['status'], 'verified')
        self.assertEqual(ledger.replay()['status'], 'verified')

    def test_the_reach_report_is_never_ledger_evidence(self):
        """Metadata, not a record: nothing about routing may enter a step's
        stored strings, its hash, or replay."""
        ledger = ledger_module.Ledger()
        result, _ = _run(
            _corpus()['positive'][0],
            calls=[('run_tactic', {'tactic': 'expand',
                                   'arguments': ['(x+1)^2']})],
            ledger=ledger)
        self.assertTrue(result['strategy_route_stages'])
        for step in ledger.steps:
            self.assertNotIn('strategy', repr(step).lower())
            self.assertNotIn('route', repr(step).lower())
        self.assertEqual(ledger.replay()['status'], 'verified')


if __name__ == '__main__':
    unittest.main()
