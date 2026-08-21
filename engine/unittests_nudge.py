#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The stop-reason nudge: retrieval, its guards, and the corpus behind them.

THE CORPUS IS THE DELIVERABLE, in the same sense the route matcher's is. A
heuristic over free prose rots without a golden corpus, and this one has an
unusual property worth stating at the top: **the text does not separate the
classes**. The two highest-scoring stop reasons in it are both cases that must
NOT be nudged - an honest open that recites the route, and a FORCED open by a
run that reached the correct answer and had two tactics refuse it - and both
outscore every real positive. Everything that makes the nudge safe is a guard
around the score, so every guard here has its own hard negative, and each
negative names the shape it defends against.

Every `set_open` reason marked `source: trace` is VERBATIM from a recorded
live run; nothing was tidied. The authored entries are marked as authored.
"""
import os
import unittest

os.environ.setdefault('TOYMATH_OBSERVABILITY', 'off')
os.environ.setdefault('OPENAI_AGENTS_DISABLE_TRACING', 'true')

import stop_nudge                                                  # noqa: E402
import strategy_routes                                             # noqa: E402

ROUTE = next(route for route in strategy_routes.load()
             if route['id'] == 'indefinite-reduction-with-boundary-term')

DIFF = 'differentiate'
REWRITE_AS = 'rewrite_as'
MATCH = 'match_coefficients'
EXPAND = 'expand'
SUBSTITUTE = 'substitute'
APPLY = 'apply_both_sides'
ASSEMBLE = 'system_assemble'


#: name, reason, the ops the run had recorded when it stopped, whether a
#: refusal stood in the way, the stage the nudge must quote (None = none),
#: and why the case is in the corpus.
STOP_REASONS = [
    # ---------------------------------------------------------------- P2
    # The three premature opens this feature exists for. Same cell, same
    # model, same arm; every tool call green; the delivered record's
    # `isolate-assignments` stage refutes each claim in the same prompt.
    dict(name='p2-run6', source='trace', witness=False, ops=(DIFF, MATCH),
         expect='isolate-assignments',
         reason='The derivative and coefficient system were checked, but '
                'the available tactics lack a certified '
                'row-extraction/elimination move to isolate $A$, $B$, and '
                '$C$ from the coupled system.',
         why='8 calls, 3 steps, no refusal anywhere; the claim is false and '
             'the same arm did the elimination in its other runs'),
    dict(name='p2-run10', source='trace', witness=False, ops=(DIFF, REWRITE_AS,
                                                              MATCH),
         expect='isolate-assignments',
         reason='The verified differentiation and coefficient match yield the '
                'coupled system $0=-Abn+Cb^2+2Ab$, $0=2Cab+Aa+Bb$, '
                '$1=Abn+Ca^2-Ab+Ba$. The available tactics provide no checked '
                'row-combination or symbolic linear-system elimination to '
                'isolate $A,B,C$ from it.',
         why='6 calls; names the elimination the stage says is not a separate '
             'move'),
    dict(name='p2-run16', source='trace', witness=False, ops=(DIFF, REWRITE_AS,
                                                              MATCH),
         expect='isolate-assignments',
         reason='The derivative identity was checked under $a^2\\ne b^2$ and '
                '$n>1$, but the available tactics cannot isolate and assemble '
                'the coupled coefficient system $0=-Abn+Cb^2+2Ab,\\ '
                '0=2Cab+Aa+Bb,\\ 1=Abn+Ca^2-Ab+Ba$ into a provenance-bearing '
                'three-constant result.',
         why='names BOTH isolation and assembly, so the highest-scoring stage '
             'is `assemble-answer` while the actionable one is still the '
             'earlier `isolate-assignments` - this case is why the winner is '
             'the earliest candidate, not the best-scoring one'),
    # HELD OUT: recorded LIVE against the shipped feature (gen 100, terra via
    # Codex, unprepared arm). The retrieval was tuned on the three above and
    # had never seen this phrasing; it quoted the same stage at the same
    # score. Kept as the corpus's only case that was not available to the
    # design.
    dict(name='live-gen100', source='trace', witness=False,
         ops=(DIFF, REWRITE_AS, MATCH), expect='isolate-assignments',
         reason='The checked coefficient system is coupled, but the '
                'available tactics provide no row-selection or '
                'row-combination move to isolate $A$, $B$, and $C$; '
                'differentiation and coefficient matching were completed.',
         why='8 tool calls, all green, 3 steps; a fresh instance of the '
             'class, produced after the feature existed and against it'),
    # ------------------------------------------------------------- FORCED
    dict(name='p2-run1-forced', source='trace', witness=True,
         ops=(DIFF, REWRITE_AS, 'rewrite', EXPAND, MATCH, SUBSTITUTE),
         expect=None,
         reason='The checked differentiation and coefficient match produced '
                'the coupled system, but the available tactics have no '
                'row-combination/elimination move to derive separate records '
                '$A=\\cdots$, $B=\\cdots$, and $C=\\cdots$; substitution only '
                'verified a proposed triple and `system_assemble` refused it.',
         why='THE decisive negative. This run reached the CORRECT triple by '
             'substitution and then had `expand` and `system_assemble` refuse '
             'it - closure was mechanically impossible and the open was the '
             'honest end. It outscores every positive on text alone, so only '
             'the refusal witness can refuse it'),
    dict(name='gen86-orientation', source='trace-era', witness=True,
         ops=(DIFF, MATCH, APPLY, EXPAND, SUBSTITUTE), expect=None,
         reason='The available assembly tactic refuses the resulting '
                'right-oriented assignments, so the isolated values cannot be '
                'assembled into one answer.',
         why='the same cell shape, a real engine defect (fixed that round): '
             'system_assemble, rewrite_as and conclude all refused. A nudge '
             'here would have pressed a run to grind against a genuine bug'),
    dict(name='oracle-refused', source='authored', witness=True, ops=(),
         expect=None,
         reason='The numeric oracle returned `disagree` on the derivative '
                'identity, so no derivative step could be recorded and there '
                'is nothing for coefficient matching to consume.',
         why='the archetypal forced open: the FIRST stage was refused, so '
             'every later stage is unreached and scores freely'),
    dict(name='parse-gap', source='authored', witness=True, ops=(),
         expect=None,
         reason="The instruction's `aligned` environment does not parse, so "
                'the identity could not be entered as an expression at all.',
         why='a parse gap surfacing as an open; there is nothing to steer to'),
    # ------------------------------- HONEST, on this shape, all-green tails
    dict(name='recites-the-route', source='authored', witness=False,
         ops=(DIFF, MATCH, APPLY, EXPAND, SUBSTITUTE, ASSEMBLE), expect=None,
         reason='I differentiated the boundary term, certified the algebraic '
                'identity that rewrites that derivative, and matched '
                'coefficients in $\\cos x$, producing a coefficient system '
                'whose rows I then isolated one unknown at a time; the '
                'assembled constants agree with the identity only when $|a| = '
                '|b|$, which the instruction excludes, so the identity as '
                'stated appears false and I will not designate a result.',
         why='THE reach guard\'s hard negative. A reason that RECITES the '
             'route scores higher on text than two of the three positives; '
             'the run reached every stage it names, so the record has '
             'nothing left to offer it'),
    dict(name='domain-honest', source='authored', witness=False,
         ops=(DIFF, MATCH, APPLY, EXPAND, SUBSTITUTE), expect=None,
         reason='Every stage of the derivation is recorded and checked, but '
                'the stated domain $|a| \\neq |b|$ cannot be expressed as an '
                'assumption the oracle restricts its sampling to, so the '
                'coefficient assignments cannot be certified over that '
                'region.',
         why='speaks the isolation vocabulary after isolating: reached, so '
             'not a candidate'),
    dict(name='needs-base-integral', source='authored', witness=False,
         ops=(DIFF,), expect=None,
         reason='Proving the identity would need a closed form for $\\int '
                '\\frac{dx}{a+b\\cos x}$ itself, and no tactic in the library '
                'evaluates that integral.',
         why='a genuinely missing capability on the right shape; no stage '
             'addresses it and none may be quoted at it'),
    dict(name='wrong-subject', source='authored', witness=False, ops=(),
         expect=None,
         reason='The task asks for a numerical plot of the coefficients '
                'against $n$, and no figure backend is configured in this '
                'session.',
         why='shares one word with the record and must stay far below the '
             'threshold'),
    # ------------------------- opens recorded on OTHER problem shapes. v0
    # delivers no route to these instructions at all, so their scores answer
    # the honest question: would retrieval have fired if one HAD been?
    dict(name='i5-integral', source='trace', witness=False,
         ops=(MATCH, APPLY, EXPAND, SUBSTITUTE, ASSEMBLE, 'integrate_rewrite'),
         expect=None,
         reason='The checked derivation reaches the partial-fraction rewrite '
                'and closes its logarithmic pair, but the remaining $(1\\pm '
                'x)^{-2}$ and $(1\\pm x)^{-3}$ pieces still require their '
                'individual substitutions, table/power integrations, and '
                'provenance-aware assemblies.',
         why="the idea pool's own motivating premature open, on the integral "
             'cell. Note its shape: four refusals, all recovered from, with '
             'eleven green calls after the last - which is why the witness '
             'is positional and not "any refusal in the run"'),
    dict(name='gen85-ftc', source='trace-era', witness=False, ops=(),
         expect=None,
         reason='No Leibniz/FTC differentiation move is available; the '
                'variable-bound integral cannot be closed in this skill.',
         why='an open after three calls and zero tactics, on a cell this '
             'record does not cover'),
    dict(name='gen50-onesided', source='trace-era', witness=True,
         ops=('limit_side',), expect=None,
         reason='Both one-sided limits are recorded and agree, but no tactic '
                'combines two one-sided limits into the two-sided limit.',
         why='the missing move was named exactly and was real'),
    dict(name='item42-byparts', source='trace-era', witness=True,
         ops=('integrate_by_parts', 'integrate_table'), expect=None,
         reason='Two by-parts splits and a table integration all verified, '
                'but no move assembles the chain into a single antiderivative '
                'and set_result refuses the hand-written assembly.',
         why='closure really was impossible until the fold tactic shipped; '
             'a refused set_result is a witness exactly like a refused move'),
    dict(name='gen82-fence', source='trace-era', witness=False, ops=(DIFF,),
         expect=None,
         reason='The available integration reduction tactic accepts only a '
                'recurrence between shifted integral families and cannot '
                'record the required boundary term.',
         why='on THIS cell shape, before the record existed. What answers it '
             "is the record's `avoid` entry, not any stage - and `avoid` is "
             'deliberately not retrievable: it names a tactic NOT to use, so '
             'quoting it steers nowhere'),
]

POSITIVES = [case for case in STOP_REASONS if case['expect']]
NEGATIVES = [case for case in STOP_REASONS if not case['expect']]


def nudged(case, routes=(ROUTE,), threshold=stop_nudge.THRESHOLD):
    """The whole predicate for one corpus case: witness, then retrieval."""
    if case['witness']:
        return None
    return stop_nudge.retrieve(case['reason'], routes, case['ops'], threshold)


class CorpusTests(unittest.TestCase):

    def test_every_premature_open_retrieves_the_stage_that_refutes_it(self):
        for case in POSITIVES:
            with self.subTest(case['name']):
                found = nudged(case)
                self.assertIsNotNone(found, case['why'])
                self.assertEqual(found.stage_id, case['expect'], case['why'])
                self.assertEqual(found.route_id, ROUTE['id'])

    def test_no_negative_is_nudged(self):
        for case in NEGATIVES:
            with self.subTest(case['name']):
                found = nudged(case)
                self.assertIsNone(
                    found,
                    f'{case["name"]} would be nudged with '
                    f'{found.stage_id if found else None}: {case["why"]}')

    def test_every_case_carries_its_reason_and_its_why(self):
        # the corpus is only useful if a later round can tell what each entry
        # defends; an entry without a `why` is an entry nobody may re-tune
        seen = set()
        for case in STOP_REASONS:
            self.assertNotIn(case['name'], seen)
            seen.add(case['name'])
            self.assertTrue(case['reason'].strip())
            self.assertTrue(case['why'].strip())
            self.assertIn(case['source'], ('trace', 'trace-era', 'authored'))
        self.assertGreaterEqual(len(NEGATIVES), 3 * len(POSITIVES))


class TextAloneTests(unittest.TestCase):
    """The finding that shapes the whole design: the words do not separate."""

    def _raw_scores(self, reason):
        """Every stage's score, guards ignored."""
        import math
        stages = ROUTE['stages']
        docs = [stop_nudge.stage_terms(stage) for stage in stages]
        frequency = {}
        for doc in docs:
            for term in doc:
                frequency[term] = frequency.get(term, 0) + 1
        want = frozenset(stop_nudge.terms(reason))
        return {stage['id']: sum(math.log(1.0 + len(docs) / frequency[t])
                                 for t in (want & doc))
                for stage, doc in zip(stages, docs)}

    def test_two_negatives_outscore_every_positive_on_text(self):
        # 10.76 for the recitation and 10.50 for the forced open, against a
        # best positive of 9.67. If this ever fails the corpus has changed
        # shape, and the design rests on text NOT being the discriminator.
        best_positive = max(max(self._raw_scores(case['reason']).values())
                            for case in POSITIVES)
        for name in ('recites-the-route', 'p2-run1-forced'):
            with self.subTest(name):
                reason = next(c for c in STOP_REASONS
                              if c['name'] == name)['reason']
                self.assertGreater(
                    max(self._raw_scores(reason).values()), best_positive)

    def test_dropping_the_witness_admits_the_forced_opens(self):
        # measured, not asserted: without the witness the corpus goes from 0
        # false nudges to 3, and all three are runs an engine refusal had
        # already stopped - including one (`item42-byparts`) where the move
        # the run said was missing genuinely did not exist until a later
        # generation shipped it.
        admitted = [case['name'] for case in NEGATIVES
                    if stop_nudge.retrieve(case['reason'], (ROUTE,),
                                           case['ops']) is not None]
        self.assertEqual(admitted, ['p2-run1-forced', 'gen86-orientation',
                                    'item42-byparts'])


class GuardTests(unittest.TestCase):

    def test_a_reached_stage_is_never_quoted(self):
        case = next(c for c in POSITIVES if c['name'] == 'p2-run16')
        # the same reason, from a run that HAS isolated its rows: the
        # earlier stage is spent, so the nudge moves on to the endpoint
        moved = stop_nudge.retrieve(case['reason'], (ROUTE,),
                                    case['ops'] + (SUBSTITUTE, APPLY))
        self.assertEqual(moved.stage_id, 'assemble-answer')
        # and once THAT is reached too, there is nothing left to say
        done = stop_nudge.retrieve(case['reason'], (ROUTE,),
                                   case['ops'] + (SUBSTITUTE, ASSEMBLE))
        self.assertIsNone(done)

    def test_a_control_stage_is_never_quoted(self):
        # `designate` is a set_result stage. Quoting it would be steering a
        # model that just declined to designate toward designating - the one
        # direction that could manufacture a wrong answer.
        reason = ('The certified content is the checked derivative identity '
                  'plus the assembled constants, but no integration step is '
                  'recorded and the ask is the constants; I cannot designate '
                  'a certified result.')
        found = stop_nudge.retrieve(reason, (ROUTE,), (), threshold=0.0)
        self.assertIsNotNone(found)
        self.assertNotEqual(found.stage_id, 'designate')
        for stage in ROUTE['stages']:
            if stage.get('action') == 'control':
                break
        else:                                   # pragma: no cover
            self.fail('the corpus record no longer has a control stage')

    def test_a_stage_that_records_no_step_is_never_quoted(self):
        # `certify-algebra` runs `equal`, which records nothing, so "you have
        # not done this yet" is undecidable from the ledger - and an
        # unfalsifiable claim is exactly what a recitation exploits.
        reason = ('I could not certify the algebraic identity used to rewrite '
                  'that derivative before coefficient matching consumes it.')
        found = stop_nudge.retrieve(reason, (ROUTE,), (), threshold=0.0)
        self.assertNotEqual(found and found.stage_id, 'certify-algebra')

    def test_one_call_of_a_looped_tactic_marks_the_whole_stage_reached(self):
        # A RECORDED COST, not an oversight. `isolate-assignments` loops
        # `apply`/`expand`/`substitute`, so a single unrelated `expand`
        # anywhere in the run spends the stage's candidacy. That is the
        # conservative direction (fewer nudges), and none of the three
        # recorded premature opens ran any of the three tactics - but a live
        # run that did would lose a nudge it might have used.
        case = next(c for c in POSITIVES if c['name'] == 'p2-run6')
        self.assertIsNotNone(nudged(case))
        spent = dict(case, ops=case['ops'] + (EXPAND,))
        self.assertIsNone(nudged(spent))

    def test_no_delivered_route_is_no_retrieval(self):
        case = POSITIVES[0]
        self.assertIsNone(stop_nudge.retrieve(case['reason'], ()))
        self.assertIsNone(stop_nudge.retrieve(case['reason'], None))

    def test_the_threshold_is_not_a_tuned_edge(self):
        # the guards do the discriminating, so every verdict in the corpus is
        # identical across a wide band. A future round that finds itself
        # tuning this number is being told the corpus changed.
        for threshold in (3.0, 4.0, 5.0, 5.5, 6.0, 6.5):
            with self.subTest(threshold=threshold):
                for case in STOP_REASONS:
                    found = nudged(case, threshold=threshold)
                    self.assertEqual(found and found.stage_id, case['expect'],
                                     case['name'])

    def test_retrieval_survives_junk(self):
        for reason in ('', '   ', '$$', '\\int', 'ελληνικά', 'a' * 5000):
            with self.subTest(reason=reason[:12]):
                self.assertIsNone(stop_nudge.retrieve(reason, (ROUTE,)))


class RenderTests(unittest.TestCase):

    def test_the_quote_is_byte_identical_to_the_delivered_line(self):
        case = POSITIVES[0]
        found = nudged(case)
        text = stop_nudge.render(found)
        delivered = strategy_routes.render((ROUTE,))
        line = next(line for line in delivered.splitlines()
                    if '**isolate-assignments**' in line)
        self.assertIn(line, text)
        self.assertIn(ROUTE['id'], text)

    def test_the_voice_is_advisory(self):
        text = stop_nudge.render(nudged(POSITIVES[0]))
        self.assertIn('The open outcome is recorded', text)
        self.assertIn('steering, not a correction', text)
        self.assertIn('confirms the stop', text)
        for forbidden in ('you are wrong', 'incorrect', 'must continue',
                          'do not stop'):
            self.assertNotIn(forbidden, text.lower())

    def test_render_of_nothing_is_nothing(self):
        self.assertEqual(stop_nudge.render(None), '')


if __name__ == '__main__':
    unittest.main()
