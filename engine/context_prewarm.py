#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
context_prewarm.py - the run's context is SELECTED before the executor starts.

One model call, same backend and same model as the run itself, reads the
resolved instruction plus a manifest of the library and answers with a
SELECTION: which subject skills to preload, and which recorded strategy route
(if any) fits the shape. The harness then assembles the executor's initial
developer instructions from that selection. Nothing else.

Three lines carry the whole trust story, and none of them is a matter of
prompt wording:

1. **CHOOSE, DON'T WRITE — enforced by the parser.** The reply is matched
   against the CLOSED sets of known skill names and record ids; every other
   character of it is discarded. No text the prewarm model authors can reach
   the executor's instructions, so a prewarmer that hallucinates, argues, or
   is prompt-injected by the cell it reads can only ever pick a different
   validated block, never introduce one.
2. **It ADDS; it never removes.** Route delivery is the UNION of the lexical
   matcher's verdict and the prewarmer's. A record the deterministic matcher
   fired on is delivered even when the prewarmer says `none`, so the offline
   corpus stays the floor and this stage can only be a superset of today's
   behaviour.
3. **It fails open.** An error, a timeout, an unusable backend, an
   unparseable reply — every one of them leaves the run exactly as it is
   today (lexical route match, no preload), with the reason recorded as run
   metadata. A broken or hung prewarm must never break or hang a cell.

The selection is METADATA, in the same class as the `strategy_routes` match
ids: it reaches the result payload and the trace, and it touches no ledger,
no checked content, no stored or hashed string, and no replay. Steering,
never authority - it gates nothing, refuses nothing, and the executor keeps
`load_skill`, so a mis-selection is recoverable in-run.

The manifest is assembled MECHANICALLY from the library: skill names and
descriptions come from the registry-backed catalog, and a record contributes
its id, its summary, and its match criteria rendered from the YAML predicates
(P0's precision lesson: summary-only prompting accepted a reduction identity
whose coefficients were already given, because the discriminator lives in the
criteria). No hand-tuned per-problem hints - a prompt engineered toward the
benchmark corpus would make the benchmark meaningless.
"""
import os
import re
import time

import strategy_routes
import tactic_skills
from agent_backends.base import (COMPLETED, USER, AgentBudget, AgentRequest,
                                 CancellationToken, ToolDispatcher,
                                 wait_interruptibly)

__all__ = ['PREWARM_VAR', 'Selection', 'disabled', 'enabled', 'manifest',
           'parse', 'prewarm', 'preload_markdown', 'union']

#: Off switch. On by default for agent cells; P2's A/B needs exactly this.
PREWARM_VAR = 'TOYMATH_PREWARM'

#: One selection call, bounded. A prewarm that hangs must cost the run this
#: much and no more - past it the run proceeds exactly as it does today.
TIMEOUT_SECONDS = 30.0

#: How many skills the prewarmer may pick. Three is the P0 prompt's bound.
MAX_SKILLS = 3

#: Character budget for preloaded skill bodies. Whole skills are dropped when
#: the budget cannot hold them - never truncated mid-body, which would hand
#: the executor a skill whose interface section is missing while its strategy
#: prose claims the tactic exists.
#:
#: MEASURED, 2026-08-16, against the committed library: integration renders at
#: 14281 characters and equations at 10272, so a budget under ~15k can never
#: preload either - the two subjects the motivating reduction cell needs most.
#: What the budget actually bounds is the cost of a WRONG selection: a correct
#: one moves content the executor would have fetched through `load_skill`
#: anyway, one turn earlier and byte for byte the same. This value admits any
#: single skill plus a small second one, and refuses the two largest together;
#: a drop is reported in run metadata rather than passing silently, because a
#: preload that quietly did nothing is indistinguishable from one that worked.
PRELOAD_BUDGET = 20000

#: `core` is in every run's prompt already; preloading it again would double
#: it. The prewarmer may still name it (its own description says every
#: workflow uses it) - it is simply already satisfied.
ALWAYS_PRESENT = ('core',)

_STATUS_OK = 'ok'
_STATUS_OFF = 'off'
_STATUS_FAILED = 'failed'
#: the reply had the expected shape and deliberately chose nothing
_STATUS_NONE = 'none'
#: the reply had no readable selection at all
_STATUS_UNPARSEABLE = 'unparseable'
_STATUS_CANCELLED = 'cancelled'


class Selection(object):
    """One prewarm verdict, plus how it was reached.

    `skills` and `routes` are always closed-set values (or empty). Everything
    else is descriptive: the status, the reason a fallback happened, and the
    measured latency.
    """

    __slots__ = ('skills', 'routes', 'status', 'error', 'seconds', 'reply')

    def __init__(self, skills=(), routes=(), status=_STATUS_OK, error=None,
                 seconds=None, reply=None):
        self.skills = tuple(skills)
        self.routes = tuple(routes)
        self.status = status
        self.error = error
        self.seconds = seconds
        self.reply = reply

    @property
    def ok(self):
        return self.status == _STATUS_OK

    @property
    def cancelled(self):
        return self.status == _STATUS_CANCELLED

    def metadata(self, preloaded=(), dropped=()):
        """The run-metadata record. Descriptive only; never ledger content."""
        record = {'status': self.status,
                  'skills': list(self.skills),
                  'routes': list(self.routes),
                  'preloaded': list(preloaded)}
        if dropped:
            record['dropped'] = list(dropped)
        if self.seconds is not None:
            record['seconds'] = round(self.seconds, 3)
        if self.error:
            record['error'] = self.error
        return record


def enabled(env=None):
    """Whether the prewarm stage runs. On unless explicitly switched off."""
    env = os.environ if env is None else env
    return (env.get(PREWARM_VAR) or '').strip().lower() not in (
        'off', '0', 'false', 'no')


# ---------------------------------------------------------------------------
# the manifest: the library, rendered mechanically
# ---------------------------------------------------------------------------

_INSTRUCTIONS = """# ToyMath context selection

You prepare the context for a verified-derivation run that has not started
yet. You do NOT solve the problem, plan it, or comment on it. You make one
choice: which of the units below the run ahead should be given up front.

Reply with exactly two lines and nothing else:

skills: <up to {max_skills} names from the skill list, comma separated, or none>
route: <one id from the route list, or none>

Rules:
- Use names and ids EXACTLY as listed. Anything not on the lists is
  discarded, so an invented name simply loses you a choice.
- Pick a route only when EVERY one of its criteria holds for this
  instruction. The criteria are the record's own; a matching surface shape
  with a different question asked is not a match.
- Pick skills the derivation will actually need. The subject on the surface
  is not always the subject of the method: an identity ABOUT an integral may
  be proved by differentiating and solving for constants.
- Choosing nothing is a valid answer; the run still works.
"""

_CRITERIA = {
    'indefinite_integral_count': 'indefinite integral signs',
    'definite_integral_count': 'integral signs carrying limits',
    'shifted_integral_count': (
        'integrals whose exponent is shifted from a parameter (n-1, n-2)'),
    'parameter_constraint_count': (
        'stated inequality constraints on a parameter (n > 1)'),
    'relation_has_explicit_nonintegral_term': (
        'a relation with an explicit non-integral term beside the integrals'),
    'asked_symbol_count': 'named unknown symbols the instruction asks for',
}


def _criterion(predicate):
    """One YAML predicate as a criterion line.

    Mechanical: the bounds are the record's own and the noun comes from the
    feature vocabulary, so no record contributes hand-written matching prose.
    The count goes AFTER the noun (`... signs: at least 3`) because a bound
    of 1 in front of a plural noun reads as a typo in an otherwise careful
    prompt.
    """
    name = predicate['feature']
    noun = _CRITERIA.get(name, name)
    if strategy_routes.FEATURES.get(name) == 'flag':
        return noun if predicate.get('is', True) else f'NOT {noun}'
    bounds = []
    if 'equals' in predicate:
        bounds.append(f"exactly {predicate['equals']}")
    if 'min' in predicate:
        bounds.append(f"at least {predicate['min']}")
    if 'max' in predicate:
        bounds.append(f"at most {predicate['max']}")
    return f"{noun}: {' and '.join(bounds) or 'any number'}"


def _flatten(text):
    return ' '.join((text or '').split())


def manifest(routes=None, root=tactic_skills.SKILL_ROOT):
    """The library block: every skill, every record, with match criteria.

    Assembled from the registry-backed catalog and the committed route file,
    so nothing here can drift from what `load_skill` and the route renderer
    actually deliver.
    """
    lines = ['## Skills', '']
    for record in tactic_skills.catalog_records(root):
        lines.append(f"- {record['name']}: {_flatten(record['description'])}")
    lines.extend(['', '## Strategy route records', ''])
    records = strategy_routes.load() if routes is None else routes
    if not records:
        lines.append('(none recorded)')
    for record in records:
        lines.append(f"- {record['id']}: {_flatten(record['summary'])}")
        criteria = [_criterion(p) for p in record['match']['all']]
        for criterion in criteria:
            lines.append(f'    - requires {criterion}')
    return '\n'.join(lines)


def developer_instructions(routes=None, root=tactic_skills.SKILL_ROOT):
    """The full prewarm prompt: the task, then the library manifest."""
    return (_INSTRUCTIONS.format(max_skills=MAX_SKILLS).rstrip() + '\n\n'
            + manifest(routes=routes, root=root) + '\n')


# ---------------------------------------------------------------------------
# the parser: choose, don't write
# ---------------------------------------------------------------------------

_SKILL_LINE = re.compile(r'^\s*skills?\s*:\s*(.*)$', re.IGNORECASE | re.M)
_ROUTE_LINE = re.compile(r'^\s*routes?\s*:\s*(.*)$', re.IGNORECASE | re.M)
_TOKEN = re.compile(r'[A-Za-z0-9_\-]+')


def parse(reply, skills, route_ids, max_skills=MAX_SKILLS):
    """Read a reply as a selection over two CLOSED sets.

    This is the containment, not a convenience: the return value is built
    only from members of `skills` and `route_ids`, so no character the model
    wrote can reach the executor. Extra prose, an explanation, a refusal, an
    invented skill, a fabricated record id, a whole injected instruction -
    all of it is discarded, and what survives is a (possibly empty) tuple of
    known names.

    Returns `(skills, routes, readable)`; the first two may be empty and
    `None` is never returned - an unreadable reply is an empty selection,
    which is exactly today's behaviour. `readable` says whether the reply
    carried the expected line shape at all, which is what tells a deliberate
    "none" apart from a prewarmer that answered something else entirely.
    """
    text = reply or ''
    known_skills = {name.lower(): name for name in skills}
    known_routes = {name.lower(): name for name in route_ids}

    def _pick(line, table, limit):
        found = []
        if line is None:
            return found
        for token in _TOKEN.findall(line):
            name = table.get(token.lower())
            if name is not None and name not in found:
                found.append(name)
            if limit is not None and len(found) >= limit:
                break
        return found

    skill_match = _SKILL_LINE.search(text)
    route_match = _ROUTE_LINE.search(text)
    chosen_skills = _pick(skill_match.group(1) if skill_match else None,
                          known_skills, max_skills)
    chosen_routes = _pick(route_match.group(1) if route_match else None,
                          known_routes, 1)
    return (tuple(chosen_skills), tuple(chosen_routes),
            skill_match is not None or route_match is not None)


# ---------------------------------------------------------------------------
# preloading: byte-identical to what load_skill returns
# ---------------------------------------------------------------------------

_PRELOAD_HEADER = """
## Preloaded subject skills

A preparation pass selected these subjects for this instruction and they are
already loaded - do not call `load_skill` for them. It is a heuristic
selection, not a restriction: load any other subject you need, and if these
do not fit what you actually see, say so in a `comment` and load your own.
"""


def preload_markdown(skills, budget=PRELOAD_BUDGET,
                     root=tactic_skills.SKILL_ROOT):
    """Render selected skills exactly as `load_skill` would, within budget.

    Returns `(markdown, loaded_names, dropped_names)`. The bodies come from
    `tactic_skills.render` - the same single source `load_skill` calls - so
    preloaded and in-run content cannot drift. A skill that does not fit the
    remaining budget is DROPPED WHOLE and stays available through
    `load_skill`; truncating one would hand the executor a body whose
    generated interface section is missing while its strategy prose still
    promises the tactic. A drop is REPORTED, not swallowed: it is the one
    outcome where the selection was right and the run did not get it.
    """
    blocks, loaded, dropped, spent = [], [], [], 0
    for name in skills:
        if name in ALWAYS_PRESENT:
            continue
        try:
            body = tactic_skills.render(name, root)
        except (ValueError, OSError):
            continue
        block = f'### {name}\n\n{body.strip()}'
        if spent + len(block) > budget:
            dropped.append(name)
            continue
        blocks.append(block)
        loaded.append(name)
        spent += len(block)
    if not blocks:
        return '', (), tuple(dropped)
    return (_PRELOAD_HEADER.strip() + '\n\n' + '\n\n'.join(blocks) + '\n',
            tuple(loaded), tuple(dropped))


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------

def disabled():
    """The selection of a run whose preparation stage is switched off."""
    return Selection(status=_STATUS_OFF)


def prewarm(instruction, backend, cancellation=None, routes=None,
            root=tactic_skills.SKILL_ROOT, timeout=TIMEOUT_SECONDS,
            trace_metadata=None, model=None):
    """Run the selection call, or fall back to today's behaviour.

    A harness stage, not an agent turn: the dispatcher carries NO bindings,
    so the model is offered no tools by either backend and the executor's own
    turn budget is untouched. The run's cancellation propagates INTO the
    stage, so Stop pressed during preparation stops the cell instead of
    preparing a run nobody is waiting for.

    Never raises. Every failure path returns a Selection whose `skills` and
    `routes` are empty and whose status names what went wrong.

    The TOGGLE is not read here. `enabled()` is consulted in exactly one
    place - the run entry point - because two gates on one switch is how an
    explicit `prewarm=True` ends up silently vetoed by an environment
    variable, which is what a measurement arm would then record as "the
    stage does nothing".
    """
    if backend is None:
        return Selection(status=_STATUS_FAILED, error='no backend')
    cancellation = (CancellationToken() if cancellation is None
                    else cancellation)
    if cancellation.cancelled:
        return Selection(status=_STATUS_CANCELLED)
    # LANDMINE: the prewarm's own timeout must NOT be expressed on the run's
    # token. `wait_interruptibly` enforces a wall-clock budget by CANCELLING
    # the token it is given, so a slow preparation call would arrive at the
    # executor as an already-cancelled run - the cell would die of the very
    # timeout that exists to keep it alive. The stage therefore waits on a
    # child token: the run's Stop propagates INTO it, its own budget stop
    # stays inside it.
    local = _child_token(cancellation)
    try:
        records = strategy_routes.load() if routes is None else routes
        catalog = tactic_skills.catalog_records(root)
        prompt = developer_instructions(routes=records, root=root)
    except Exception as exc:                    # a malformed library
        return Selection(status=_STATUS_FAILED,
                         error=f'{type(exc).__name__}: {exc}')
    budget = AgentBudget(max_seconds=timeout)
    metadata = dict(trace_metadata or {})
    metadata['stage'] = 'prewarm'
    request = AgentRequest(
        instruction=instruction,
        developer_instructions=prompt,
        # no tools: this is a selection, and a stage that could call a tactic
        # would be a second agent rather than a preparation step
        dispatcher=ToolDispatcher((), local, budget=budget),
        cancellation=local,
        model=model,
        budget=budget,
        trace_metadata=metadata)
    started = time.monotonic()
    try:
        handle = backend.start(request)
        outcome = wait_interruptibly(handle, local, budget=budget)
    except BaseException as exc:                # never take a cell down
        return Selection(status=_STATUS_FAILED, seconds=_since(started),
                         error=f'{type(exc).__name__}: {exc}')
    seconds = _since(started)
    if cancellation.cancelled:
        return Selection(status=_STATUS_CANCELLED, seconds=seconds)
    if outcome.cancelled:
        # the prewarm's own budget, not the run's: the cell proceeds
        # unprepared rather than dying of a slow preparation
        return Selection(status=_STATUS_FAILED, seconds=seconds,
                         error=(outcome.error
                                or f'prewarm stopped ({outcome.status})'))
    if outcome.status != COMPLETED:
        return Selection(status=_STATUS_FAILED, seconds=seconds,
                         error=outcome.error or outcome.status)
    reply = outcome.final_text or ''
    skills, chosen, readable = parse(
        reply, [record['name'] for record in catalog],
        [record['id'] for record in records])
    if not skills and not chosen:
        # "none" is a legitimate answer, so an empty selection is not a
        # failure - but a prewarmer that always answers nothing looks
        # identical to one that is switched off, so the two are recorded
        # apart, and a reply with no readable selection apart from both.
        return Selection(status=_STATUS_NONE if readable
                         else _STATUS_UNPARSEABLE,
                         seconds=seconds, reply=reply[:200])
    return Selection(skills=skills, routes=chosen, seconds=seconds,
                     reply=reply[:200])


def _child_token(parent):
    """A token the run's cancellation propagates into, but not back out of."""
    child = CancellationToken()
    parent.add_listener(lambda reason: child.cancel(reason or USER))
    return child


def _since(started):
    return time.monotonic() - started


def union(matched, selected, routes=None):
    """Route delivery: the lexical match UNION the prewarmer's pick.

    The non-regression property, expressed as one function so it cannot be
    lost in a refactor: a record the deterministic matcher fired on is
    delivered whatever the prewarmer said. The lexical extractor stays the
    floor; this stage can only ever add.
    """
    records = strategy_routes.load() if routes is None else routes
    by_id = {record['id']: record for record in records}
    delivered = list(matched or ())
    seen = {record['id'] for record in delivered}
    for route_id in selected or ():
        record = by_id.get(route_id)
        if record is not None and record['id'] not in seen:
            delivered.append(record)
            seen.add(record['id'])
    return tuple(delivered)
