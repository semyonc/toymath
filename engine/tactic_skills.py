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


# Bounded natural subject vocabulary for the virtual loader.  Executable
# authority remains the static TacticSpec registry; aliases only choose which
# committed strategy document to reveal.
SUBJECT_ALIASES = {
    'algebra': 'core',
    'derivative': 'differentiation',
    'derivatives': 'differentiation',
    'differentiate': 'differentiation',
    'equation': 'equations',
    'equation_solving': 'equations',
    'root': 'equations',
    'roots': 'equations',
    'root_finding': 'equations',
    'integral': 'integration',
    'integrals': 'integration',
    'integrate': 'integration',
    'limit': 'limits',
    'linear_algebra': 'matrices',
    'matrix': 'matrices',
    'determinant': 'matrices',
    'vector': 'matrices',
    'vectors': 'matrices',
    'sum': 'finite_operators',
    'sums': 'finite_operators',
    'series': 'finite_operators',
    'product': 'finite_operators',
    'products': 'finite_operators',
}


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


def _subject_key(value):
    return '_'.join(str(value or '').strip().lower().replace('-', ' ').split())


def resolve(requested, root=SKILL_ROOT):
    """Resolve a subject key, bounded alias, or tactic name to one skill.

    The tactic-name path is derived from the static registry, so a model that
    asks to load ``quadratic_roots`` still receives only its owning equations
    strategy and gains no authority beyond the registry entry.
    """
    skills = discover(root)
    raw = str(requested or '').strip()
    if raw in skills:
        return raw
    key = _subject_key(raw)
    normalized_skills = {_subject_key(name): name for name in skills}
    if key in normalized_skills:
        return normalized_skills[key]
    target = SUBJECT_ALIASES.get(key)
    if target in skills:
        return target
    spec = tactic_registry.BY_NAME.get(raw) or tactic_registry.BY_NAME.get(key)
    if spec is not None and spec.skill in skills:
        return spec.skill
    available = ', '.join(sorted(skills)) or '(none)'
    raise ValueError(
        f'unknown skill or tactic {requested!r}; available subjects: '
        f'{available}')


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
        'Call `load_skill` with the exact backticked subject before using a '
        'tactic from that domain. Load another subject later when the '
        'derivation crosses domains.',
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
