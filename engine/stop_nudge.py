#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stop_nudge.py - one advisory reply when a run stops for a reason a delivered
strategy route already answers.

MEASURED SHAPE. Three live runs on the reduction cell stopped at 6-8 tool
calls, every call green, each saying the available tactics offer no
row-combination or elimination move to isolate the constants - while the
strategy-route record delivered in that same prompt says, in its
`isolate-assignments` stage, that substituting an already-isolated value back
into the remaining rows IS the elimination. The information was present and
unread: a premature open is an ATTENTION failure, not an information failure.
So the reply is aimed at the one moment when the model's attention is provably
on the blocker - the sentence it just wrote to justify stopping.

TRUST, inherited unchanged from the route record it quotes: STEERING, NEVER
AUTHORITY. The nudge cannot bar `set_open`. The open outcome is recorded
FIRST, by the ordinary path, and the nudge rides the tool result afterwards;
`set_result` superseding an earlier `set_open` is behaviour this engine has
had and tested since the control shipped, so a run that continues needs no new
semantics. Nothing here touches the ledger, the oracle or replay, and the
record of what fired is run METADATA beside `strategy_routes`.

WHAT THE PROBE MEASURED, and why the guards are not decoration (corpus in
`unittests_nudge.py`; 3 verbatim self-refuting reasons, 13 negatives):

  * TEXT ALONE CANNOT SEPARATE THE CLASSES. The two highest-scoring reasons in
    the corpus are both cases that must NOT be nudged - an honest open that
    recites the route (10.76) and a FORCED open that reached the correct
    triple and had `expand` and `system_assemble` refuse it (10.50) - and both
    outscore every real positive (best 9.67). Any threshold admitting the
    three positives admits them.
  * The REFUSAL WITNESS is what separates the forced ones: a refused move with
    no recorded step after it means an engine refusal stands between the work
    and the open, and pressing such a run to continue is pure harm. With every
    filter but the witness the corpus leaves 3/13 false nudges, all three of
    them refused runs; with it, 0/13. The witness is deliberately "in the
    WAY", not "anywhere in the run": one recorded premature open had four
    refusals, all recovered from, with eleven green calls after the last.
  * A stage the run ALREADY REACHED is never quoted. The measured false
    positive it removes is a reason that RECITES the route ("I differentiated,
    certified, matched coefficients, isolated the rows...") and then stops for
    an unrelated reason; recitation scores high by construction.
  * A CONTROL stage is never quoted. `designate` would be steering toward
    `set_result` a model that has just declined to designate - the one
    direction that could manufacture a wrong answer.
  * A stage whose tactics record no ledger step is never quoted, because
    "you have not done this yet" is then undecidable from the ledger, and an
    unfalsifiable claim is exactly what recitation exploits.

Scores are computed over ALL stages of the delivered routes, so a reason's
score is a stable function of the text and the record; only CANDIDACY depends
on the run.
"""
import math
import re

import strategy_routes
import tactic_registry

__all__ = ['THRESHOLD', 'terms', 'stage_terms', 'retrieve', 'render',
           'Retrieval']

#: A candidate must clear this to be quoted. On the corpus the lowest score
#: actually quoted is 6.70 and the highest surviving candidate on a negative
#: the witness does not already refuse is 2.86, and the verdict is identical
#: anywhere in 3.0-6.5 - the discrimination comes from the candidate filters,
#: not from a tuned edge. Scores are IDF-weighted over the delivered stage
#: set, so this is a bound on shared DISTINCTIVE terms, not on shared words.
THRESHOLD = 5.0

# Words that carry no evidence about WHICH stage a reason is about. The
# engine's own vocabulary is in here on purpose: every stage of every record
# is about tactics, moves, steps and checks, so those terms discriminate
# nothing and only inflate a recitation.
_STOP = frozenset("""
the and that with from this into then than they them there here what when
which while were will would could should have has had been being does did
your you our its it is are was for not but all any one two three each per
some such only also more most other others same both about above after
before between during over under again very much many few own too can may
might must shall upon within without across among because since though
although however therefore thus hence still yet just even ever never
itself themselves whatever whose
available provide provides provided provider using used use uses need needs
needed require requires required call calls called run runs running
note notes tactic tactics move moves
""".split())

_SUFFIX = ('ations', 'ation', 'ising', 'izing', 'ings', 'ing', 'ers', 'er',
           'ies', 'ied', 'ed', 'es', 's')

_MACRO = re.compile(r'\\[A-Za-z]+')
_NON_ALPHA = re.compile(r'[^A-Za-z]+')


class Retrieval(object):
    """One stage a stop reason matched, with the evidence for the match."""

    __slots__ = ('route', 'stage', 'score', 'shared', 'index')

    def __init__(self, route, stage, score, shared, index):
        self.route = route
        self.stage = stage
        self.score = score
        self.shared = tuple(shared)
        self.index = index

    @property
    def route_id(self):
        return self.route['id']

    @property
    def stage_id(self):
        return self.stage['id']

    def metadata(self):
        """The descriptive record of this retrieval. Never ledger content."""
        return {'route': self.route_id, 'stage': self.stage_id,
                'score': round(self.score, 2), 'terms': list(self.shared)}


def _stem(word):
    for suffix in _SUFFIX:
        if len(word) - len(suffix) >= 4 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


def terms(text):
    """Distinct content stems of one text, in first-seen order.

    Deliberately crude and deliberately deterministic: macros go first (a
    reason is full of `$...$` mathematics that says nothing about which stage
    it means), then everything short, common or engine-generic is dropped.
    """
    text = _NON_ALPHA.sub(' ', _MACRO.sub(' ', text or ''))
    out = []
    for word in text.lower().split():
        if len(word) < 4 or word in _STOP:
            continue
        stem = _stem(word)
        if len(stem) >= 3 and stem not in out:
            out.append(stem)
    return out


def stage_terms(stage):
    """The content stems of one stage, over exactly the fields a reader of
    the delivered block sees."""
    parts = [stage.get('target', ''), stage.get('why', ''),
             (stage.get('produces') or '').replace('-', ' '),
             (stage.get('consumes') or '').replace('-', ' '),
             (stage.get('id') or '').replace('-', ' ')]
    if stage.get('tactic'):
        parts.append(stage['tactic'].replace('_', ' '))
    for name in stage.get('tactics') or ():
        parts.append(name.replace('_', ' '))
    if stage.get('tool'):
        parts.append(stage['tool'].replace('_', ' '))
    return frozenset(terms(' '.join(parts)))


def _recording_tactics(stage):
    """The stage's tactics that leave a ledger step behind."""
    if stage.get('action') == 'control':
        return []
    names = ([stage['tactic']] if stage.get('tactic')
             else stage.get('tactics') or [])
    return [name for name in names
            if name in tactic_registry.BY_NAME
            and tactic_registry.BY_NAME[name].op
            in tactic_registry.TRANSFORMING_OPS]


def _reached(stage, ops):
    return any(tactic_registry.BY_NAME[name].op in ops
               for name in _recording_tactics(stage))


def retrieve(reason, routes, ops=(), threshold=THRESHOLD):
    """The stage of a delivered route that addresses this stop reason.

    `ops` are the ledger ops this run recorded. Returns a `Retrieval`, or
    None when nothing is close enough, everything close enough was already
    done, or no route was delivered at all.

    The winner is the EARLIEST surviving candidate, not the highest scoring
    one: a stop reason often names both the move it is stuck before and the
    endpoint it cannot reach (measured - one of the three live reasons names
    isolation AND assembly), and the actionable stage is the earlier one.
    """
    stages = [(route, stage) for route in (routes or ())
              for stage in (route.get('stages') or ())]
    if not stages:
        return None
    docs = [stage_terms(stage) for _, stage in stages]
    total = len(docs)
    frequency = {}
    for doc in docs:
        for term in doc:
            frequency[term] = frequency.get(term, 0) + 1
    want = frozenset(terms(reason))
    ops = frozenset(ops or ())
    best = None
    for index, ((route, stage), doc) in enumerate(zip(stages, docs)):
        shared = want & doc
        score = sum(math.log(1.0 + total / frequency[term])
                    for term in shared)
        if score < threshold:
            continue
        if not _recording_tactics(stage) or _reached(stage, ops):
            continue
        if best is None or index < best.index:
            best = Retrieval(route, stage, score,
                             sorted(shared, key=lambda term: (
                                 frequency[term], term)), index)
    return best


_TEMPLATE = (
    'The open outcome is recorded. One advisory note before you stop: a stage '
    'of the strategy route delivered with this run addresses the blocker you '
    'named, and this run recorded no step from it.\n\n'
    'Route `%(route)s`, quoted verbatim from the block you were given:\n'
    '%(stage)s\n\n'
    'This is steering, not a correction, and nothing about the route is '
    'authority: if the stage does not fit what you actually see, say so in a '
    '`comment` and stop - calling `set_open` again confirms the stop and is '
    'accepted as it stands. If it does fit, continue from the recorded steps '
    'and designate with `set_result` when a value is established.')


def render(retrieval):
    """The one advisory reply, quoting the stage EXACTLY as delivered.

    The quote comes from the same renderer that built the run's prompt, so
    the model is shown the words it already has rather than a paraphrase of
    them - the containment the composed-brief work named as non-negotiable,
    applied to the smaller case.
    """
    if retrieval is None:
        return ''
    stages = retrieval.route.get('stages') or ()
    index = next((position for position, stage in enumerate(stages, start=1)
                  if stage.get('id') == retrieval.stage_id), 1)
    line = strategy_routes.stage_line(index, retrieval.stage)
    return _TEMPLATE % {'route': retrieval.route_id, 'stage': line}
