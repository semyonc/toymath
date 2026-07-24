# -*- coding: utf-8 -*-
"""
prompt_commands.py - discoverable, user-defined `do!`-style commands.

A prompt-command is a Markdown file (in the repo `commands/` directory) with
a SKILL-compatible YAML frontmatter and a body that is an instruction
template containing the `$ARGUMENTS` placeholder. Dropping

    ---
    name: int
    description: Apply symbolic integration to the argument
    ---
    Apply symbolic integration for $ARGUMENTS ...

into `commands/int.md` makes an `int!` cell prefix that runs the same do!
agent with the rendered instruction. The command adds no new authority: it
only seeds the agent, which can still only call the oracle-checked
primitives, so a hallucinated step stays structurally impossible.

Discovery mirrors processor.register_actions(): glob a directory, parse each
file, merge into a {name: PromptCommand} registry.
"""

import glob
import logging
import os
import re
from collections import namedtuple

import yaml

logger = logging.getLogger(__name__)

# names that name the dispatcher's own built-ins; a file cannot shadow them
RESERVED = frozenset(('do', 'commands', 'help', 'model'))

_NAME_RE = re.compile(r'^[A-Za-z_]\w*$')
_PLACEHOLDER = '$ARGUMENTS'

_COMMANDS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'commands')

PromptCommand = namedtuple('PromptCommand',
                           ('name', 'description', 'template', 'expr',
                            'fresh', 'direct', 'mode'),
                           defaults=((), None, None))


def _split_frontmatter(text):
    """Return (meta_dict, body). Raises ValueError when the leading
    `---`-delimited YAML frontmatter is missing or malformed."""
    if not text.startswith('---'):
        raise ValueError('missing YAML frontmatter (must start with ---)')
    end = text.find('\n---', 3)
    if end == -1:
        raise ValueError('unterminated YAML frontmatter (no closing ---)')
    meta = yaml.safe_load(text[3:end]) or {}
    if not isinstance(meta, dict):
        raise ValueError('frontmatter is not a mapping')
    body = text[end + len('\n---'):].lstrip('\n')
    return meta, body


def parse_command(text, fallback_name):
    """Parse one command file's text into a PromptCommand. `fallback_name`
    (the filename stem) is used when the frontmatter omits `name`. Raises
    ValueError on any problem so the loader can skip the file with a reason."""
    meta, body = _split_frontmatter(text)
    name = str(meta.get('name') or fallback_name).strip()
    if not _NAME_RE.match(name):
        raise ValueError(f'invalid command name {name!r} '
                         '(letters, digits, underscore; no leading digit)')
    if name in RESERVED:
        raise ValueError(f'{name!r} is a reserved built-in command')
    description = str(meta.get('description', '')).strip()
    if not description:
        raise ValueError('frontmatter is missing a description')
    # expr:true makes the command composable INSIDE an expression
    # ({diff! {int! x^3}}); plain commands stay whole-cell prefixes.
    expr = bool(meta.get('expr', False))
    # direct: <primitive> is the zero-token tier - the command IS one
    # verified primitive, no agent run. Implies expr (inline-composable);
    # the body is documentation, so $ARGUMENTS becomes optional.
    direct = meta.get('direct')
    if direct is not None:
        direct = str(direct).strip()
        import expr_commands
        if direct not in expr_commands.DIRECT_PRIMITIVES:
            raise ValueError(
                f'unknown direct primitive {direct!r} (available: '
                + ', '.join(sorted(expr_commands.DIRECT_PRIMITIVES)) + ')')
        expr = True
    mode = meta.get('mode')
    if mode is not None:
        mode = str(mode).strip().lower()
        if mode != 'prove':
            raise ValueError("mode must be 'prove' when specified")
        if expr:
            raise ValueError('prove-mode commands cannot be expression '
                             'commands')
    if _PLACEHOLDER not in body and direct is None:
        raise ValueError(f'template body does not contain {_PLACEHOLDER}')
    # fresh: symbols the command MINTS (an integration constant C). The
    # composite resolver renames them per splice on collision, so two
    # independent runs never share one constant.
    fresh = meta.get('fresh') or ()
    if isinstance(fresh, str):
        fresh = [p for p in re.split(r'[,\s]+', fresh.strip()) if p]
    if not isinstance(fresh, (list, tuple)):
        raise ValueError('fresh must be a symbol name or a list of names')
    fresh = tuple(str(n).strip() for n in fresh)
    for n in fresh:
        if not _NAME_RE.match(n):
            raise ValueError(f'invalid fresh symbol name {n!r}')
    return PromptCommand(name, description, body.strip(), expr, fresh,
                         direct, mode)


def load_commands(directory=_COMMANDS_DIR):
    """Discover prompt-commands under `directory`. Returns {name: PromptCommand}.
    Malformed files are skipped (a warning is logged), never fatal — one bad
    file must not disable the others or break kernel startup."""
    registry = {}
    if not os.path.isdir(directory):
        return registry
    for path in sorted(glob.glob(os.path.join(directory, '*.md'))):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                cmd = parse_command(fh.read(), stem)
        except (OSError, ValueError, yaml.YAMLError) as e:
            logger.warning('skipping command file %s: %s', path, e)
            continue
        registry[cmd.name] = cmd
    return registry


def render(cmd, arguments):
    """Instantiate a command's template with the cell arguments."""
    return cmd.template.replace(_PLACEHOLDER, arguments)
