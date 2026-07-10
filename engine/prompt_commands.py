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
RESERVED = frozenset(('do', 'commands', 'help'))

_NAME_RE = re.compile(r'^[A-Za-z_]\w*$')
_PLACEHOLDER = '$ARGUMENTS'

_COMMANDS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'commands')

PromptCommand = namedtuple('PromptCommand', ('name', 'description', 'template'))


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
    if _PLACEHOLDER not in body:
        raise ValueError(f'template body does not contain {_PLACEHOLDER}')
    return PromptCommand(name, description, body.strip())


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
