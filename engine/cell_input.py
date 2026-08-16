# -*- coding: utf-8 -*-
"""
cell_input.py - what a notebook cell's raw source means before anything runs.

Two read-only readings of a cell live here:

* `split_lines` - how the dispatcher cuts a cell into statements. Brace-aware,
  so a multi-line `\\frac{...}` stays one statement.
* `preview` - the rendered spelling of the cell input, for the JupyterLab
  extension that shows a formula typeset while the cell is not being edited.

`preview` reads a cell two ways, in order. A cell (or the argument of a
command cell) that parses *whole* as one formula per line renders as that
formula. Otherwise — a do! prompt, which is prose with formulas buried in it
and no `$…$` around them — the prose is scanned for the fragments that are
formulas, and only those are typeset.

The two readings differ in what they can promise, and the difference is worth
keeping in mind. Both check that a rendered formula parses back to the same
expression as the characters it replaces, so neither can show something other
than what the cell runs. But the whole-cell reading is total — the cell is a
formula or it is not — while the prose scan additionally *guesses where a
formula starts and ends*. It is therefore tuned for precision: a formula left
as prose is invisible, whereas prose swallowed into a formula is glaring.

Nothing here touches the ledger, the history, or the live notation graph:
every parse goes into a throwaway `Notation`.
"""

import re

import prompt_commands

# a cell that starts with `name!` may be a command; the caller's registries
# decide, so math like `n! + 1` (factorial) still reads as a formula
_CMD_PREFIX_RE = re.compile(r'^([A-Za-z_]\w*)!')

# a reference to an earlier result. The shell renders one as the formula it
# stands for; until that result exists the reference renders as itself, so a
# cell waiting on a re-run still reads as the command it is
_BACKREF_RE = re.compile(r'\[\[\s*(\d+)\s*\]\]')

# dispatcher built-ins whose argument is a configuration word rather than
# mathematics (`commands!`/`help!` take none at all). `do!` is deliberately
# not here: its argument is prose, which the prose scan reads.
_CONFIG_COMMANDS = prompt_commands.RESERVED - {'do'}

# constructs whose braces legitimately carry a word, so the prose scan must
# not see the word inside them
_WORD_BEARING = re.compile(
    r'\\(?:operatorname|text|textrm|textit|textbf|mathrm|mathbf|mathit'
    r'|mathcal|mathbb|mathsf|mathtt|hbox|mbox|begin|end)\s*\{[^{}]*\}')
_MACRO = re.compile(r'\\[A-Za-z]+')
_WORD = re.compile(r'[A-Za-z]{3,}')

# multi-letter names that are mathematics rather than prose once the macros
# are stripped: `\sin ^3 x` has already lost its backslash by then, and
# differentials (`dx`) are two letters so the scan never sees them
_MATH_WORDS = frozenset((
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan',
    'arccot', 'sinh', 'cosh', 'tanh', 'coth', 'log', 'ln', 'exp', 'lim',
    'sup', 'inf', 'max', 'min', 'det', 'dim', 'deg', 'gcd', 'lcm', 'mod',
    'arg', 'lg'))


def split_lines(code):
    """Cut a cell into statements, keeping braced groups together."""
    bracket = 0
    buffer = ''
    for codePoint in code:
        if codePoint == '{':
            bracket += 1
        elif codePoint == '}':
            bracket -= 1
        elif codePoint == '\n' and bracket == 0:
            yield buffer
            buffer = ''
            continue
        elif codePoint == '\r':
            continue
        buffer += codePoint
    if buffer != '':
        yield buffer


def has_prose(text):
    """True when `text` carries a word no formula would contain.

    The parser cannot answer this: it accepts `the derivative` as a product
    of seven one-letter symbols. So the veto is lexical and comes first —
    strip the macros and the constructs that legitimately brace a word, then
    look for what is left spelling something.
    """
    stripped = _MACRO.sub(' ', _WORD_BEARING.sub(' ', text))
    return any(word.group().lower() not in _MATH_WORDS
               for word in _WORD.finditer(stripped))


def _command_prefix(stripped, command_names, prose_commands):
    """Split a leading `name!` into (label, body, body_may_be_prose).

    A known bang word — a prompt-command (`int!`), the agent endpoint (`do!`),
    or a legacy rewrite action (`mul!`) — labels what follows and renders
    beside it. A bang word no registry knows is not a command at all:
    `n! + 1` is a factorial, so the whole cell is the formula.

    Only a command that hands its argument to the agent may have prose in it.
    A plain cell and a rewrite action are read as one expression: scanning
    those for buried formulas would describe them as something they are not.
    """
    match = _CMD_PREFIX_RE.match(stripped)
    if match is None:
        return None, stripped, False
    name = match.group(1)
    if name in _CONFIG_COMMANDS:
        return None, None, False
    if name not in command_names:
        return None, stripped, False
    return name + '!', stripped[match.end():].strip(), name in prose_commands


# ----------------------------------------------------------------------
# Finding the formulas inside a prose prompt
#
# Two stages, and the order matters. A lexical scan proposes candidate
# fragments, then ToyMath's own parser vetoes them. The parser cannot do the
# first job: it reads `the derivative` as a product of one-letter symbols and
# accepts it, so it can only ever reject, never detect.
# ----------------------------------------------------------------------

# What the user typed -> what the parser reads. Only unambiguous rewrites
# belong here; a character with no entry is not mathematics as far as the scan
# is concerned, so `π` ends a fragment while `\pi` continues one.
_UNICODE_MATH = {
    '⁰': '^0', '¹': '^1', '²': '^2', '³': '^3', '⁴': '^4', '⁵': '^5',
    '⁶': '^6', '⁷': '^7', '⁸': '^8', '⁹': '^9', 'ⁿ': '^n',
    '₀': '_0', '₁': '_1', '₂': '_2', '₃': '_3', '₄': '_4', '₅': '_5',
    '₆': '_6', '₇': '_7', '₈': '_8', '₉': '_9',
    '−': '-', '–': '-', '×': ' \\times ', '÷': ' \\div ', '·': ' \\cdot ',
    '≤': ' \\le ', '≥': ' \\ge ', '≠': ' \\neq ', '∞': ' \\infty ',
    '→': ' \\to ',
}

# Short English words the one-letter-variable rule would otherwise swallow.
# They are refused only at a fragment's edges, so `a = b` still reads as an
# equation while `= 0 as a` gives back the two words it does not need.
_PROSE_WORDS = frozenset((
    'a', 'an', 'as', 'at', 'be', 'by', 'do', 'if', 'in', 'is', 'it', 'of',
    'on', 'or', 'no', 'so', 'to', 'up', 'us', 'we', 'my', 'i'))

_TOKEN_RE = re.compile(r"""
    (?P<backref>\[\[\s*\d+\s*\]\])          # a result reference, never math
  | (?P<macro>\\(?:[A-Za-z]+|[\\{}|,;!\s])) # \frac, \\, \{ ...
  | (?P<ellipsis>\.\.+)                     # `1..20` is a range, not a value
  | (?P<number>\d+(?:\.\d+)?)
  | (?P<word>[A-Za-z]+)
  | (?P<op>[-+*/^_=<>(){}\[\]|&,'!~])
  | (?P<unicode>[^\x00-\x7F])               # ², −, ×, π, — : only some are math
  | (?P<space>[ \t]+)
  | (?P<newline>\n)
  | (?P<other>.)
""", re.X)

# Operators that may not bound a fragment: a trailing `+` is the half of an
# expression whose other half is prose. A bracket bounds a fragment only from
# the side it opens: a leading `}` is debris from a construct the scan could
# not read, and leaving it there blocks the retry on what follows.
_EDGE_OPS = frozenset("-+*/^_=<>,'&|" + '−–×÷·≤≥≠')
_LEAD_OPS = _EDGE_OPS | frozenset('}])')
_TAIL_OPS = _EDGE_OPS | frozenset('{[(')
# Signals that a fragment is mathematics and not a stretch of short words.
_STRONG_OPS = frozenset('^_=<>')
_WORD_EDGE_RE = re.compile(r'^[A-Za-z]+')
_WORD_TAIL_RE = re.compile(r'[A-Za-z]+$')


def _classify(kind, value):
    """Whether a token can sit inside a formula."""
    if kind in ('macro', 'number', 'ellipsis', 'space'):
        return True
    if kind == 'op':
        return True
    if kind == 'unicode':
        return value in _UNICODE_MATH
    if kind == 'word':
        return value.lower() in _MATH_WORDS or len(value) <= 2
    return False


def _seeds(kind, value, previous, adjacent=True):
    """Whether this token proves its fragment is mathematics.

    Returns 'macro' or 'bare', naming which tier the evidence belongs to, so
    the weaker one can be switched off independently.

    Juxtaposition counts only when the two tokens actually touch and the word
    is not an English one: `2x` is a product, `a 2` is an article in front of
    a measurement.
    """
    prev_kind, prev_value = previous
    named = prev_value.lower() in _PROSE_WORDS
    if kind == 'macro':
        return 'macro'
    if kind == 'op' and value in _STRONG_OPS:
        return 'bare'
    if kind == 'unicode' and value in _UNICODE_MATH:
        return 'bare'
    if not adjacent:
        return None
    if kind == 'number' and prev_kind == 'word' and len(prev_value) <= 2 \
            and not named:
        return 'bare'                      # 3x, 2n
    if kind == 'word' and prev_kind == 'number' and len(value) <= 2:
        return 'bare'                      # x2 the other way round
    if kind == 'op' and value == '(' and prev_kind == 'word' \
            and len(prev_value) <= 2 and not named:
        return 'bare'                      # f(x)
    return None


def _next_candidate(text, start, bare_seeds):
    """The next maximal run of formula-eligible tokens carrying a seed."""
    run_start = run_end = None
    seeded = False
    previous, previous_end = (None, ''), -1
    wanted = ('macro', 'bare') if bare_seeds else ('macro',)
    for match in _TOKEN_RE.finditer(text, start):
        kind, value = match.lastgroup, match.group()
        # A formula may wrap across one line break; a blank line ends it.
        newline = kind == 'newline'
        if newline and run_start is not None and \
                not text[match.end():].lstrip(' \t').startswith('\n'):
            previous, previous_end = (None, ''), -1
            continue
        if kind == 'backref' or newline or not _classify(kind, value):
            if seeded:
                return run_start, run_end
            run_start = run_end = None
            seeded = False
            previous, previous_end = (None, ''), -1
            continue
        if run_start is None:
            if kind == 'space':
                continue
            run_start = match.start()
        if _seeds(kind, value, previous,
                  match.start() == previous_end) in wanted:
            seeded = True
        if kind != 'space':
            run_end = match.end()
            previous, previous_end = (kind, value), match.end()
    return (run_start, run_end) if seeded else None


def _trim(text, start, end):
    """Give back the edge tokens that cannot bound a formula."""
    changed = True
    while changed and start < end:
        changed = False
        while start < end and (text[start].isspace()
                               or text[start] in _LEAD_OPS):
            start, changed = start + 1, True
        while end > start and (text[end - 1].isspace()
                               or text[end - 1] in _TAIL_OPS):
            end, changed = end - 1, True
        head = _WORD_EDGE_RE.match(text[start:end])
        if head and head.group().lower() in _PROSE_WORDS:
            start, changed = start + head.end(), True
        tail = _WORD_TAIL_RE.search(text[start:end])
        if tail and tail.group().lower() in _PROSE_WORDS:
            end, changed = start + tail.start(), True
    return start, end


def _normalize(fragment):
    """The parser's spelling of what the user typed."""
    return ''.join(_UNICODE_MATH.get(char, char) for char in fragment)


def _seeded(fragment, bare_seeds):
    """Whether a trimmed fragment still carries its evidence."""
    previous = (None, '')
    wanted = ('macro', 'bare') if bare_seeds else ('macro',)
    for match in _TOKEN_RE.finditer(fragment):
        kind, value = match.lastgroup, match.group()
        if _seeds(kind, value, previous) in wanted:
            return True
        if kind != 'space':
            previous = (kind, value)
    return False


def _is_debris(fragment):
    """What is left when the scan met a construct it could not read.

    `\\begin{aligned}` gives back a bare `\\begin`, and a half-swallowed
    environment gives back `b \\end` — the parser accepts both as opaque
    symbols and the writer spells them back unchanged, so the round-trip
    guard cannot see anything wrong. Two shapes are refused outright: a lone
    macro (which also costs `\\alpha` on its own, and in prose that reads as
    well as it renders), and an environment whose halves do not pair up.
    """
    tokens = [match for match in _TOKEN_RE.finditer(fragment)
              if match.lastgroup != 'space']
    if len(tokens) < 2 and any(match.lastgroup == 'macro'
                               for match in tokens):
        return True
    return fragment.count('\\begin') != fragment.count('\\end')


def _accept(text, start, end, bare_seeds):
    """Validate a candidate, shortening it at prose only.

    Returns `(start, end, latex)`: the bounds are the trimmed ones, because
    what the trim gives back is prose and has to be rendered as prose.
    """
    start, end = _trim(text, start, end)
    if start >= end:
        return None
    fragment = text[start:end]
    if _is_debris(fragment):
        return None
    segment = _math_segment(_normalize(fragment))
    if segment is not None:
        return start, end, segment['latex']
    # The scan reached past the formula into the sentence around it. Retry at
    # the words that are prose to begin with — never mid-expression, so a
    # fragment is not rendered as a silently truncated formula.
    for cut in reversed(_prose_cuts(fragment)):
        head_start, head_end = _trim(text, start, start + cut)
        if head_start >= head_end:
            continue
        head = text[head_start:head_end]
        if not _seeded(head, bare_seeds):
            continue
        segment = _math_segment(_normalize(head))
        if segment is not None:
            return head_start, head_end, segment['latex']
    return None


def _prose_cuts(fragment):
    """Offsets of the words in `fragment` that are prose, not mathematics."""
    cuts = []
    previous = (None, '')
    for match in _TOKEN_RE.finditer(fragment):
        kind, value = match.lastgroup, match.group()
        if kind == 'word' and value.lower() not in _MATH_WORDS \
                and (len(value) > 2 or value.lower() in _PROSE_WORDS):
            cuts.append(match.start())
        if kind != 'space':
            previous = (kind, value)
    return cuts


def formula_spans(text, bare_seeds=True):
    """Every `(start, end, latex)` in `text` that is a formula.

    `bare_seeds` admits fragments whose only evidence is notation rather than
    a `\\macro` — `x³−3x`, `(sin x + cos x)² − 1 = sin(2x)`. That tier finds
    about as much again as the macro tier does, and carries the higher risk,
    which is why it is switchable on its own.
    """
    spans = []
    cursor = 0
    while cursor < len(text):
        candidate = _next_candidate(text, cursor, bare_seeds)
        if candidate is None:
            break
        candidate_start, candidate_end = candidate
        accepted = _accept(text, candidate_start, candidate_end, bare_seeds)
        if accepted is None:
            cursor = max(candidate_end, candidate_start + 1)
            continue
        start, end, latex = accepted
        spans.append((start, end, latex))
        cursor = max(end, start + 1)
    return spans


def sole_formula(text, bare_seeds=True):
    """The one formula in `text`, or None when it holds zero or several.

    Answers a deliberately narrow question for a bare `do!` cell: is there
    exactly ONE thing here that could be the ask? Zero formulas and several
    formulas both give None, because the scan guesses fragment boundaries
    and nothing downstream could pick between two candidates anyway.

    A caller must treat the answer as a HINT about the instruction, never as
    a statement of what the run is required to establish: `formula_spans`
    can mis-split (`I_n=\\int...` comes back as `n=\\int...`) and can pick a
    trailing side condition out of an environment it cannot read (`n \\gt 1`
    out of an `aligned` block). Both shapes were measured on the repository's
    own notebooks.
    """
    spans = formula_spans(text, bare_seeds)
    return spans[0][2] if len(spans) == 1 else None


def _prose_text(segments, text):
    """Append prose, keeping any result reference in it as a reference."""
    if not text:
        return
    position = 0
    for match in _BACKREF_RE.finditer(text):
        head = text[position:match.start()]
        if head:
            segments.append({'kind': 'text', 'text': head})
        segments.append({'kind': 'ref', 'text': f'[[{match.group(1)}]]'})
        position = match.end()
    if text[position:]:
        segments.append({'kind': 'text', 'text': text[position:]})


def prose_segments(text, bare_seeds=True):
    """Segments for prose with formulas in it, or None if it holds none."""
    spans = formula_spans(text, bare_seeds)
    if not spans:
        return None
    segments = []
    position = 0
    for start, end, latex in spans:
        _prose_text(segments, text[position:start])
        segments.append({'kind': 'math', 'latex': latex})
        position = end
    _prose_text(segments, text[position:])
    return segments


def _math_segment(text):
    """One formula segment for `text`, or None if it is not one."""
    import primitives              # deferred: engine import, not a hot path

    try:
        sym, notation = primitives.parse_latex(text, allow_ellipsis=True)
        latex = primitives.write_latex(sym, notation)
    except Exception:
        return None
    # The writer can drop what it has no spelling for — the outer command of
    # `track! {mul! ...}` comes back missing. Show the rendered form only when it
    # parses back to the same expression as the source, so the view can never
    # quietly say something else than the cell runs.
    if not latex.strip() or not primitives.same_expression(text, latex):
        return None
    return {'kind': 'math', 'latex': latex}


def _statement_segments(line):
    """Segments for one statement, or None if any part of it is not math."""
    segments = []
    position = 0
    for match in _BACKREF_RE.finditer(line):
        head = line[position:match.start()]
        if head.strip():
            segment = _math_segment(head)
            if segment is None:
                return None
            segments.append(segment)
        segments.append({'kind': 'ref', 'text': f'[[{match.group(1)}]]'})
        position = match.end()
    tail = line[position:]
    if tail.strip():
        segment = _math_segment(tail)
        if segment is None:
            return None
        segments.append(segment)
    return segments or None


def whole_formula(body):
    """Segments for a body that is a formula per statement, or None."""
    if has_prose(body):
        return None
    segments = []
    for line in split_lines(body):
        if not line.strip():
            continue
        statement = _statement_segments(line)
        if statement is None:
            return None            # all-or-nothing: the cell stays raw
        if segments:
            segments.append({'kind': 'break'})
        segments.extend(statement)
    return segments or None


def preview(code, command_names=(), prose_commands=(), bare_seeds=True):
    """Segments for the rendered view of `code`, or None to keep it raw.

    Returns a list of `{'kind': 'command'|'math'|'ref'|'text'|'break', ...}`
    dicts: JSON the comm can carry unchanged.

    A cell is read as a whole formula first, and only failing that — and only
    where the argument goes to the agent — as prose with formulas in it. So
    `int! \\int x^2 dx` renders as one formula rather than as a sentence that
    happens to contain one.
    """
    stripped = code.strip()
    if not stripped:
        return None
    label, body, prose = _command_prefix(stripped, command_names,
                                         prose_commands)
    if not body:
        return None
    segments = whole_formula(body)
    if segments is None and prose:
        segments = prose_segments(body, bare_seeds)
    if not segments:
        return None
    if label:
        segments.insert(0, {'kind': 'command', 'text': label})
    return segments
