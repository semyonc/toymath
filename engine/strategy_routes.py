#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strategy_routes.py - routing as structured, scoped data.

A *route record* answers one question: when a run's instruction has a known
problem SHAPE, which ordered stages have closed it before, and why does each
stage feed the next? Until now that knowledge lived only as paragraphs inside
subject `SKILL.md` files, where a cross-subject route has to be written into
whichever skill the agent happens to load first. That is an O(N^2) hand-written
surface with no test coverage, and deleting one such paragraph measurably
stopped a live derivation from closing.

A record moves the same knowledge into one committed file with a schema, so
five things become testable that were implicit in prose: the schema itself,
the tactic/control references, the matching, the delivery, and the historical
effect. What stays prose is the strategy: `target` names the mathematical
sub-object a stage acts on and `why` says what the stage is for. This format
does not mechanically prove that strategy good — it makes everything around it
checkable.

TRUST: a record is STEERING, never authority. It touches no ledger, no oracle
and no replay, grants no execution permission, and only the tactic registry can
invoke code. A matched record is appended to the run's INITIAL developer
instructions (not gated behind `load_skill`, since a route meant to prevent a
premature open outcome cannot depend on the agent already having made the right
discovery call), and the matched ids are returned as non-ledger run metadata so
a later failure can say whether guidance was absent, mismatched, or ignored.

Matching is a deliberately small typed vocabulary over a HYBRID extractor: the
motivating instruction is an `aligned` environment that `parse_latex` rejects
outright and whose only parseable fragment is the trailing `n > 1`, so
predicates over the notation DAG alone cannot see the integral relation. The
extractor therefore reads brace-balanced structure off the raw command text and
uses the parser for exactly what the parser can see.

Matching has TWO metrics and only one of them is obvious. PRECISION is the
design priority — a wrongly matched record spends a run's context steering a
problem it does not fit — but RECALL is what makes the record measurable at
all: reach can only be measured on runs that MATCH, so an instruction the
matcher misses becomes an unrouted control whose difference reads as noise.
Both are held by the committed corpus in the YAML (`positive` / `negative`,
plus `unmatched` for deliberate limits with their reason), and a record may
not name the symbols it matches — the same problem posed in `\\alpha,\\beta,
\\gamma` is the same problem.
"""
import os
import re

try:
    import yaml
except ImportError:  # pragma: no cover - yaml ships with the kernel env
    yaml = None

import cell_input
import primitives
import tactic_registry
from notation import Notation

__all__ = ['StrategyRouteError', 'FEATURES', 'CONTROLS', 'ROUTES_PATH',
           'load', 'features', 'match', 'render', 'validate', 'fixtures',
           'required_stage_reach']

ROUTES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'strategy_routes.yaml')


class StrategyRouteError(ValueError):
    """A route file that does not satisfy the schema."""


# The run-level controls a stage may name. Deliberately duplicated rather than
# imported from `agent_do` (which imports this module); a test pins this set
# against the real tool surface, so the duplication cannot drift silently.
CONTROLS = ('comment', 'claim', 'conclude', 'set_result', 'set_open')

# The typed predicate vocabulary. Keep it SMALL: a free-form detect language
# would move the O(N^2) prose problem into a second, worse notation.
FEATURES = {
    'indefinite_integral_count': 'count',
    'definite_integral_count': 'count',
    'shifted_integral_count': 'count',
    'parameter_constraint_count': 'count',
    'relation_has_explicit_nonintegral_term': 'flag',
    'asked_symbol_count': 'count',
}

# Spellings a record may not use, each with the reason. Same defensive shape
# as the bare `on:` YAML landmine below: a prohibition nobody can look up is
# a prohibition a future record author rediscovers by debugging.
_RETIRED_FEATURES = {
    'asks_for_symbols': (
        'matching is symbol-generic: count the asked-for symbols with '
        '`asked_symbol_count` instead of naming them'),
}
_PROHIBITED_KEYS = {
    'symbols': (
        'a record may NOT name the symbols it matches - the same problem '
        'posed in \\alpha,\\beta,\\gamma is the same problem, and a record '
        'that hardcodes A,B,C silently stops matching it. Use '
        '`asked_symbol_count` with a bound'),
}

_ACTIONS = ('tactic', 'tactic-loop', 'control')

# ---------------------------------------------------------------------------
# hybrid feature extraction
# ---------------------------------------------------------------------------

_ENVIRONMENT = re.compile(
    r'\\(?:begin|end)\s*\{\s*'
    r'(?:aligned|align\*?|alignat\*?|gather\*?|split|multline\*?|eqnarray\*?)'
    r'\s*\}')
_ROW_BREAK = re.compile(r'\\\\(?:\s*\[[^\]]*\])?')
_INTEGRAL = re.compile(r'\\(?:o|i{1,3})?int(?![A-Za-z])')
_LIMITS = re.compile(r'\A\s*\\(?:no)?limits')
_MACRO = re.compile(r'\\[A-Za-z]+')
_WORD = re.compile(r'[A-Za-z]{3,}')
_SHIFTED_POWER = re.compile(
    r'\^\s*\{\s*(?:[A-Za-z]\s*[-+]\s*\d+|\d+\s*[-+]\s*[A-Za-z])\s*\}')
_ASK = re.compile(
    r'\b(?:find|determine|compute|calculate|obtain|identify|solve\s+for|'
    r'what\s+(?:are|is))\b([^.?!\n]*)', re.IGNORECASE)
# Greek spellings, so the same problem posed in `\alpha,\beta,\gamma` reads
# as the same ask. An ALLOWLIST, not `\\[A-Za-z]+`: the latter would read
# `\int`, `\sin` and `\frac` as unknowns. `\pi`/`\Pi` are deliberately absent
# - a named constant is never what an instruction asks the agent to find.
_GREEK = (r'alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta'
          r'|iota|kappa|lambda|mu|nu|xi|rho|varrho|sigma|varsigma|tau'
          r'|upsilon|phi|varphi|chi|psi|omega'
          r'|Gamma|Delta|Theta|Lambda|Xi|Sigma|Upsilon|Phi|Psi|Omega')
_IDENTIFIER = re.compile(r'\A(?:\\(?:' + _GREEK + r')|[A-Za-z])'
                         r'(?:_\{?\w+\}?)?\Z')
_ASK_SPLIT = re.compile(r'[,;]|\band\b|&', re.IGNORECASE)
# A subordinate clause ends the list of asked-for symbols. Without this cut a
# constraint stated in the SAME sentence ("determine A, B, C for n > 2")
# silently drops the last symbol, because collection breaks at the token that
# carries the clause.
_ASK_CLAUSE = re.compile(
    r'\b(?:for|when|whenever|where|if|given|assuming|provided|with|without'
    r'|such|so|subject|satisfying|using|in|to|that|which|valid|holds)\b',
    re.IGNORECASE)
# A noun phrase between the ask verb and the letters ("the constants", "the
# values of", "the three coefficients"). A CLOSED vocabulary of words that
# introduce a list of unknowns - not "strip any prose", which would let
# "find the area A of the triangle" read as an ask for A.
_ASK_LEADIN = re.compile(
    r'\A(?:(?:the|a|an|all|both|each|every|two|three|four|five|six|real'
    r'|remaining|other|following|values?|constants?|coefficients?|numbers?'
    r'|unknowns?|parameters?|quantit(?:y|ies)|of)\s+)+', re.IGNORECASE)
_CONSTRAINT_OPS = frozenset(('<', '>', '<=', '>=', '\\le', '\\ge', '\\lt',
                             '\\gt', '\\ne', '\\neq', '\\leq', '\\geq'))


def _flatten_alignment(text):
    """Drop alignment scaffolding a route match must see through.

    Read-only and semantics-preserving for the features below: the
    environment delimiters, row breaks and `&` alignment marks carry no
    mathematical content, and an `aligned` row that ends in `+` and resumes
    with `+` on the next row leaves an empty additive term, which
    `_additive_terms` drops.
    """
    text = _ENVIRONMENT.sub(' ', text)
    text = _ROW_BREAK.sub(' ', text)
    return text.replace('&', ' ')


def _scan(text):
    """Yield (index, char, depth) with escapes skipped and depth over
    `{}`/`()`/`[]`. One shared scanner so every split agrees on what
    "top level" means."""
    depth = 0
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == '\\' and index + 1 < length:
            index += 2          # a control sequence or an escaped delimiter
            continue
        if char in '{([':
            depth += 1
        elif char in '})]':
            depth = max(depth - 1, 0)
        else:
            yield index, char, depth
        index += 1


def _relation_sides(text):
    """Split at top-level `=` only. `\\neq`/`\\le`/`\\ge` spell no bare `=`,
    so a plain scan is enough and never splits a comparison macro."""
    cuts = [index for index, char, depth in _scan(text)
            if char == '=' and depth == 0]
    if not cuts:
        return []
    sides, start = [], 0
    for cut in cuts:
        sides.append(text[start:cut])
        start = cut + 1
    sides.append(text[start:])
    return sides


def _additive_terms(text):
    """Top-level additive terms, empty ones dropped."""
    cuts = [index for index, char, depth in _scan(text)
            if char in '+-' and depth == 0]
    parts, start = [], 0
    for cut in cuts:
        parts.append(text[start:cut])
        start = cut + 1
    parts.append(text[start:])
    return [part for part in (p.strip() for p in parts) if part]


def _integral_kinds(text):
    """(indefinite, definite) counts of integral signs in `text`."""
    indefinite = definite = 0
    for hit in _INTEGRAL.finditer(text):
        tail = text[hit.end():]
        limits = _LIMITS.match(tail)
        if limits is not None:
            tail = tail[limits.end():]
        tail = tail.lstrip()
        if tail[:1] in ('_', '^'):
            definite += 1
        else:
            indefinite += 1
    return indefinite, definite


def _is_notation(term):
    """True when a term is notation rather than prose.

    A route match runs over a whole instruction, so a trailing sentence rides
    along on the last additive term. Requiring real notation and no bare word
    keeps prose from counting as a mathematical term of its own.
    """
    if not any(mark in term for mark in ('\\', '^', '_')):
        return False
    return not _WORD.search(_MACRO.sub(' ', term))


def _asked_symbols(text):
    """Identifiers an explicit ask names ("find A, B and C").

    Three widenings over the first version, each measured against its own
    hard negatives, because a wrongly matched route spends a run's context
    steering it wrong: an ask may be non-imperative ("what are A, B and C?"),
    may put a closed-vocabulary noun phrase in front of the letters ("find
    the constants A, B, C"), and may trail a subordinate clause in the same
    sentence ("determine A, B, C for n > 2").

    Still conservative where it counts: collection stops at the first token
    that is not a bare identifier once its lead-in is removed, so "find the
    antiderivative of ..." yields nothing, and an identifier is a single
    letter or an allowlisted Greek macro, never an arbitrary `\\macro`.
    """
    found = []
    for hit in _ASK.finditer(text):
        tail = hit.group(1)
        clause = _ASK_CLAUSE.search(tail)
        if clause is not None:
            tail = tail[:clause.start()]
        for token in _ASK_SPLIT.split(tail):
            token = _ASK_LEADIN.sub('', token.strip().strip('$').strip(),
                                    count=1).strip()
            if not token:
                continue
            if not _IDENTIFIER.match(token):
                break
            if token not in found:
                found.append(token)
    return frozenset(found)


def _parameter_constraints(text):
    """Parseable inequality fragments the instruction states about a
    parameter.

    This is the notation leg of the hybrid, and it is deliberately the only
    one: on the motivating instruction the parser can read exactly one
    fragment (`n > 1`) and nothing else, so a route format that insisted on
    predicates over the DAG would have no detector at all.
    """
    count = 0
    try:
        # None, not (), when the scan finds no formula inside prose
        segments = cell_input.prose_segments(text) or ()
    except Exception:                       # never break a run over a scan
        return 0
    for segment in segments:
        if segment.get('kind') != 'math':
            continue
        latex = segment.get('latex') or ''
        try:
            sym, notation = primitives.parse_latex(latex)
        except Exception:
            continue
        comp = notation.getf(sym, Notation.COMP)
        if comp is None:
            continue
        if comp.sym.props.get('op') not in _CONSTRAINT_OPS:
            continue
        if primitives.free_symbols(sym, notation):
            count += 1
    return count


def features(text):
    """The typed feature vector of one instruction."""
    text = text or ''
    flat = _flatten_alignment(text)
    indefinite, definite = _integral_kinds(flat)
    shifted = 0
    mixed = False
    for side in _relation_sides(flat):
        terms = _additive_terms(side)
        integral_terms = [t for t in terms if _INTEGRAL.search(t)]
        plain_terms = [t for t in terms
                       if not _INTEGRAL.search(t) and _is_notation(t)]
        if integral_terms and plain_terms:
            mixed = True
        shifted += sum(1 for t in integral_terms if _SHIFTED_POWER.search(t))
    return {
        'indefinite_integral_count': indefinite,
        'definite_integral_count': definite,
        'shifted_integral_count': shifted,
        'parameter_constraint_count': _parameter_constraints(text),
        'relation_has_explicit_nonintegral_term': mixed,
        # a COUNT, never the names: a record that hardcoded `[A, B, C]`
        # stopped matching the same problem posed in Greek letters
        'asked_symbol_count': len(_asked_symbols(text)),
    }


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

def _require(condition, message):
    if not condition:
        raise StrategyRouteError(message)


def _validate_predicate(predicate, where):
    _require(isinstance(predicate, dict), f'{where}: predicate must be a map')
    # Prohibited spellings are refused BY NAME and before anything else, so
    # the error says what the rule is rather than "unknown key".
    for key, why in _PROHIBITED_KEYS.items():
        _require(key not in predicate, f'{where}: `{key}` is not allowed - '
                                       f'{why}')
    name = predicate.get('feature')
    retired = _RETIRED_FEATURES.get(name)
    _require(retired is None,
             f'{where}: feature {name!r} was retired - {retired}')
    _require(name in FEATURES, f'{where}: unknown feature {name!r}')
    kind = FEATURES[name]
    keys = set(predicate) - {'feature'}
    allowed = {'count': {'min', 'max', 'equals'}, 'flag': {'is'}}[kind]
    unknown = keys - allowed
    _require(not unknown,
             f'{where}: feature {name!r} takes {sorted(allowed)}, '
             f'got {sorted(unknown)}')
    if kind == 'count':
        _require(keys, f'{where}: feature {name!r} needs a bound')
        for key in keys:
            _require(isinstance(predicate[key], int)
                     and not isinstance(predicate[key], bool),
                     f'{where}: {key} must be an integer')
    else:
        _require(isinstance(predicate.get('is', True), bool),
                 f'{where}: is must be a boolean')


def _validate_stage(stage, index, produced, route_id):
    where = f'route {route_id!r} stage {index + 1}'
    _require(isinstance(stage, dict), f'{where}: stage must be a map')
    stage_id = stage.get('id')
    _require(isinstance(stage_id, str) and stage_id,
             f'{where}: needs a string id')
    where = f'route {route_id!r} stage {stage_id!r}'
    action = stage.get('action')
    _require(action in _ACTIONS,
             f'{where}: action must be one of {list(_ACTIONS)}')
    if action == 'control':
        tool = stage.get('tool')
        _require(tool in CONTROLS,
                 f'{where}: tool must be one of {list(CONTROLS)}')
        _require('tactic' not in stage and 'tactics' not in stage,
                 f'{where}: a control stage names no tactic')
    else:
        names = ([stage['tactic']] if action == 'tactic'
                 else stage.get('tactics'))
        if action == 'tactic':
            _require(isinstance(stage.get('tactic'), str),
                     f'{where}: needs one tactic name')
        else:
            _require(isinstance(names, list) and names
                     and all(isinstance(n, str) for n in names),
                     f'{where}: tactic-loop needs a list of tactic names')
        for name in names:
            _require(name in tactic_registry.BY_NAME,
                     f'{where}: {name!r} is not a registered tactic')
        _require('tool' not in stage, f'{where}: a tactic stage names no tool')
    for key in ('target', 'why', 'consumes', 'produces', 'role'):
        _require(isinstance(stage.get(key, ''), str),
                 f'{where}: {key} must be a string')
    _require(isinstance(stage.get('required', False), bool),
             f'{where}: required must be a boolean')
    consumes = stage.get('consumes')
    if consumes:
        _require(consumes in produced,
                 f'{where}: consumes {consumes!r}, which no earlier stage '
                 f'produces')
    # LANDMINE: the stage field is `target`, not `on`. YAML 1.1 resolves a
    # bare `on:` key to the boolean True, so the obvious spelling silently
    # becomes a key nobody can look up. Name it, rather than let a future
    # record author debug a `unknown keys [True]`.
    _require(True not in stage and False not in stage,
             f'{where}: a bare on/off/yes/no key is YAML 1.1 boolean - the '
             f'field that names the sub-object a stage acts on is `target`')
    unknown = set(stage) - {'id', 'action', 'tactic', 'tactics', 'tool',
                            'target', 'why', 'consumes', 'produces', 'role',
                            'required'}
    _require(not unknown, f'{where}: unknown keys {sorted(unknown)}')
    return stage.get('produces')


def _validate_route(route, seen):
    _require(isinstance(route, dict), 'each route must be a map')
    route_id = route.get('id')
    _require(isinstance(route_id, str) and route_id, 'a route needs a string id')
    _require(route_id not in seen, f'duplicate route id {route_id!r}')
    _require(isinstance(route.get('summary'), str) and route['summary'],
             f'route {route_id!r}: needs a summary')
    unknown = set(route) - {'id', 'summary', 'match', 'avoid', 'stages',
                            'evidence', 'provenance'}
    _require(not unknown, f'route {route_id!r}: unknown keys {sorted(unknown)}')
    matcher = route.get('match') or {}
    _require(isinstance(matcher, dict) and list(matcher) == ['all'],
             f'route {route_id!r}: match takes exactly one `all` list')
    _require(isinstance(matcher['all'], list) and matcher['all'],
             f'route {route_id!r}: match.all must be a non-empty list')
    for predicate in matcher['all']:
        _validate_predicate(predicate, f'route {route_id!r}')
    stages = route.get('stages')
    _require(isinstance(stages, list) and stages,
             f'route {route_id!r}: needs at least one stage')
    produced, ids = set(), set()
    for index, stage in enumerate(stages):
        produces = _validate_stage(stage, index, produced, route_id)
        _require(stage['id'] not in ids,
                 f'route {route_id!r}: duplicate stage id {stage["id"]!r}')
        ids.add(stage['id'])
        if produces:
            produced.add(produces)
    for entry in route.get('avoid') or []:
        _require(isinstance(entry, dict)
                 and entry.get('tactic') in tactic_registry.BY_NAME
                 and isinstance(entry.get('why'), str) and entry['why'],
                 f'route {route_id!r}: each avoid entry needs a registered '
                 f'tactic and a why')
    evidence = route.get('evidence') or {}
    _require(isinstance(evidence, dict),
             f'route {route_id!r}: evidence must be a map')
    for observation in evidence.get('observations') or []:
        _require(isinstance(observation, dict)
                 and isinstance(observation.get('successes'), int)
                 and isinstance(observation.get('runs'), int)
                 and observation['runs'] > 0
                 and 0 <= observation['successes'] <= observation['runs'],
                 f'route {route_id!r}: an observation needs successes/runs '
                 f'as counts, never an opaque score string')
        for key in ('condition', 'metric', 'model', 'date'):
            _require(isinstance(observation.get(key), str)
                     and observation[key],
                     f'route {route_id!r}: observation needs {key}')
    return route_id


def _validate_document(document):
    _require(isinstance(document, dict), 'the route file must be a map')
    _require(document.get('version') == 1, 'route file needs version: 1')
    routes = document.get('routes')
    _require(isinstance(routes, list), 'route file needs a routes list')
    seen = set()
    for route in routes:
        seen.add(_validate_route(route, seen))
    corpus = document.get('fixtures') or {}
    _require(isinstance(corpus, dict), 'fixtures must be a map')
    for name, entry in corpus.items():
        _require(isinstance(entry, dict)
                 and isinstance(entry.get('positive'), list)
                 and entry['positive']
                 and isinstance(entry.get('negative'), list)
                 and entry['negative'],
                 f'fixture {name!r}: needs non-empty positive and negative '
                 f'instruction lists')
        # A deliberate non-match must carry its reason. Without the `why`
        # this list would be indistinguishable from a silently tolerated
        # recall miss, which is the exact thing it exists to prevent.
        for entry_index, skipped in enumerate(entry.get('unmatched') or ()):
            _require(isinstance(skipped, dict)
                     and isinstance(skipped.get('instruction'), str)
                     and skipped['instruction'].strip()
                     and isinstance(skipped.get('why'), str)
                     and skipped['why'].strip(),
                     f'fixture {name!r}: unmatched entry {entry_index + 1} '
                     f'needs an instruction and a why')
    for route in routes:
        fixture = (route.get('evidence') or {}).get('fixture')
        if fixture is not None:
            _require(fixture in corpus,
                     f'route {route["id"]!r}: fixture {fixture!r} is not in '
                     f'this file')
    return document


_CACHE = {}


def load(path=ROUTES_PATH):
    """Parse and schema-validate the committed route file."""
    key = os.path.abspath(path)
    stamp = os.path.getmtime(key) if os.path.exists(key) else None
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    if stamp is None or yaml is None:
        return ()
    with open(key, 'r', encoding='utf-8') as handle:
        document = _validate_document(yaml.safe_load(handle) or {})
    routes = tuple(document.get('routes') or ())
    _CACHE[key] = (stamp, routes)
    return routes


def fixtures(path=ROUTES_PATH):
    """The committed instruction corpus, keyed by fixture id.

    An executable fixture is a copy of the exact instruction, never a
    reference to a notebook cell: notebooks are user-modified and a cell
    index drifts.
    """
    if yaml is None or not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as handle:
        return (yaml.safe_load(handle) or {}).get('fixtures') or {}


def validate(path=ROUTES_PATH):
    """Raise `StrategyRouteError` unless the committed file is well formed."""
    load(path)
    return True


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

def _holds(predicate, vector):
    value = vector[predicate['feature']]
    kind = FEATURES[predicate['feature']]
    if kind == 'count':
        if 'min' in predicate and value < predicate['min']:
            return False
        if 'max' in predicate and value > predicate['max']:
            return False
        if 'equals' in predicate and value != predicate['equals']:
            return False
        return True
    return bool(value) is predicate.get('is', True)


def matches(route, vector):
    """True when every predicate of one record holds on a feature vector."""
    return all(_holds(predicate, vector)
               for predicate in route['match']['all'])


def match(text, routes=None, path=ROUTES_PATH):
    """The records whose shape matcher fires on this instruction.

    Never raises: a malformed or unreadable route file leaves a run with no
    steering, which is exactly the pre-record behaviour. Routing must not be
    able to take a derivation down.
    """
    try:
        routes = load(path) if routes is None else routes
        if not routes:
            return ()
        vector = features(text)
        return tuple(route for route in routes if matches(route, vector))
    except Exception:
        return ()


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

_PREAMBLE = """
## Recorded strategy route

This instruction matches a recorded route for a problem shape that has closed
before. It is STEERING, not authority: it grants no execution permission,
records nothing, and every step still has to pass its own mechanical check. If
what you actually see does not fit the route, say so in a `comment` and choose
your own; a stage named here is never a licence to skip its check, and a tactic
NOT named here is not forbidden.
"""


def required_skills(route):
    """Owning subjects of the route's tactics, from the registry.

    Derived, never copied into the record: a tactic that moves to another
    skill must not leave a stale `load_skill` line behind in committed data.
    """
    skills = []
    for stage in route['stages']:
        names = ([stage['tactic']] if stage.get('tactic')
                 else stage.get('tactics') or [])
        for name in names:
            skill = tactic_registry.BY_NAME[name].skill
            if skill != 'core' and skill not in skills:
                skills.append(skill)
    return skills


def required_stage_reach(routes, ops):
    """Which required stages of each matched route a finished run reached.

    RUN METADATA, and the boundary is the point of the function: nothing
    here may refuse a designation, bar a step, reach the ledger, or change
    replay. A route match is a heuristic over instruction TEXT, and "refuse
    the answer when a required stage was not reached" would turn that
    heuristic into authority - which the record format forbids by design.
    The metric exists so a later failure can be read ("the route was
    delivered and stage 3 never ran"), never so the run can be judged.

    Reach is measured over required TACTIC stages only, by the ledger `op`
    the registry maps each tactic name to. A required CONTROL stage records
    no step of its own; whether the run designated anything is already in
    the run's own result, so counting it here would double-report it.
    """
    seen = set(ops or ())
    report = []
    for route in routes or ():
        stages = []
        for stage in route['stages']:
            if not stage.get('required') or stage.get('action') == 'control':
                continue
            names = ([stage['tactic']] if stage.get('tactic')
                     else stage.get('tactics') or [])
            stages.append({
                'stage': stage['id'],
                'reached': any(tactic_registry.BY_NAME[name].op in seen
                               for name in names if
                               name in tactic_registry.BY_NAME),
            })
        if stages:
            report.append({'route': route['id'], 'stages': stages,
                           'required': len(stages),
                           'reached': sum(1 for s in stages if s['reached'])})
    return report


def _stage_line(index, stage):
    if stage.get('action') == 'control':
        head = f"control `{stage['tool']}`"
    elif stage.get('action') == 'tactic-loop':
        names = ', '.join(f'`{n}`' for n in stage['tactics'])
        head = f'{names} as needed'
    else:
        head = f"`{stage['tactic']}`"
    marks = []
    if stage.get('required'):
        marks.append('required')
    if stage.get('role'):
        marks.append(stage['role'])
    label = f" ({', '.join(marks)})" if marks else ''
    line = f"{index}. **{stage['id']}**{label} - {head}"
    if stage.get('target'):
        line += f" on {stage['target']}"
    if stage.get('consumes'):
        line += f", consuming `{stage['consumes']}`"
    if stage.get('produces'):
        line += f" -> produces `{stage['produces']}`"
    line += '.'
    if stage.get('why'):
        line += f" Why: {stage['why']}."
    return line


def render(routes, loaded=()):
    """The Markdown block appended to a matching run's initial instructions.

    `loaded` names subjects the run already carries, so the "load first" line
    asks only for what is actually missing. Empty by default: with nothing
    preloaded the rendered block is byte-for-byte the one this function has
    always produced.
    """
    routes = [route for route in routes if route]
    if not routes:
        return ''
    present = frozenset(loaded or ())
    blocks = [_PREAMBLE.strip()]
    for route in routes:
        block = [f"### {route['id']}", f"Shape: {route['summary'].strip()}"]
        skills = [skill for skill in required_skills(route)
                  if skill not in present]
        if skills:
            names = ', '.join(f'`{s}`' for s in skills)
            block.append(f'Load first with `load_skill`: {names} - the '
                         f'owning subjects of the tactics below.')
        block.append('Stages:')
        for index, stage in enumerate(route['stages'], start=1):
            block.append(_stage_line(index, stage))
        for entry in route.get('avoid') or []:
            block.append(f"Do not reach for `{entry['tactic']}`: "
                         f"{entry['why'].strip()}")
        blocks.append('\n'.join(block))
    return '\n\n'.join(blocks) + '\n'
