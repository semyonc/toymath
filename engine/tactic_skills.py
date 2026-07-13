#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Progressively disclosed Markdown skills for the tactic registry."""
from dataclasses import dataclass
import glob
import os

import yaml

import tactic_registry


SKILL_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '.claude', 'skills', 'toymath')


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: str
    body: str


def _split(text):
    if not text.startswith('---'):
        raise ValueError('missing YAML frontmatter')
    end = text.find('\n---', 3)
    if end == -1:
        raise ValueError('unterminated YAML frontmatter')
    meta = yaml.safe_load(text[3:end]) or {}
    if not isinstance(meta, dict):
        raise ValueError('frontmatter is not a mapping')
    return meta, text[end + len('\n---'):].lstrip('\n').rstrip()


def discover(root=SKILL_ROOT):
    """Return the committed skill catalog keyed by its short runtime name."""
    paths = [os.path.join(root, 'SKILL.md')]
    paths.extend(sorted(glob.glob(os.path.join(
        root, 'domains', '*', 'SKILL.md'))))
    skills = {}
    for path in paths:
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                meta, body = _split(handle.read())
        except (OSError, ValueError, yaml.YAMLError):
            continue
        name = str(meta.get('key') or meta.get('name') or '').strip()
        description = str(meta.get('description') or '').strip()
        if not name or not description or name in skills:
            continue
        skills[name] = Skill(name, description, path, body)
    return skills


def catalog_records(root=SKILL_ROOT):
    skills = discover(root)
    return [
        {
            'name': skill.name,
            'description': skill.description,
            'tactics': [spec.name for spec in tactic_registry.tactics(
                skill.name)],
        }
        for skill in skills.values()
    ]


def catalog_markdown(root=SKILL_ROOT, include_core=False):
    records = catalog_records(root)
    lines = ['## Available tactic skills', '']
    for record in records:
        if record['name'] == 'core' and not include_core:
            continue
        lines.append(f"- `{record['name']}` — {record['description']}")
    lines.extend([
        '',
        'Call `load_skill` before using a tactic from one of these domains. '
        'Load another skill later if the derivation crosses domains.',
    ])
    return '\n'.join(lines)


def interface_markdown(name):
    specs = tactic_registry.tactics(name)
    if not specs:
        return ''
    lines = [
        '## Tactic interface',
        '',
        'Call `run_tactic` with the tactic name and an ordered list of '
        'string arguments:',
        '',
    ]
    for spec in specs:
        usage = tactic_registry.usage(spec, 'agent')
        lines.append(f'- `{usage}` — {spec.summary}.')
    return '\n'.join(lines)


def render(name, root=SKILL_ROOT):
    """Render one skill body plus its registry-generated exact interface."""
    skill = discover(root).get(name)
    if skill is None:
        available = ', '.join(sorted(discover(root))) or '(none)'
        raise ValueError(f'unknown skill {name!r}; available: {available}')
    interface = interface_markdown(name)
    return skill.body + ('\n\n' + interface if interface else '')


def validate(root=SKILL_ROOT):
    """Return catalog/registry consistency errors."""
    skills = discover(root)
    errors = []
    if 'core' not in skills:
        errors.append('missing core skill')
    registry_skills = {spec.skill for spec in tactic_registry.TACTICS}
    for name in sorted(registry_skills - set(skills)):
        errors.append(f'registry skill {name!r} has no SKILL.md')
    for name in sorted(set(skills) - registry_skills - {'core'}):
        errors.append(f'skill {name!r} owns no tactics')
    return errors
