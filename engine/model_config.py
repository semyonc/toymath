# -*- coding: utf-8 -*-
"""Configured OpenRouter models for notebook-local model selection."""

import os
import re
from collections import namedtuple

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover - dotenv ships with the kernel env
    pass


MODEL_VAR = 'OPENROUTER_MODEL'
DEFAULT_MODEL = 'openai/gpt-5.6-luna'
MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'models.yaml')

ModelEndpoint = namedtuple('ModelEndpoint', ('model', 'providers'),
                           defaults=((),))


class ModelConfigError(ValueError):
    """The committed model endpoint configuration is malformed."""


def current_model():
    """Return the environment-selected model, or ToyMath's default."""
    return os.environ.get(MODEL_VAR, DEFAULT_MODEL)


def parse_model_config(text):
    """Parse ``models.yaml`` into an ordered tuple of model endpoints."""
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise ModelConfigError(f'invalid YAML: {e}') from e
    if not isinstance(data, dict):
        raise ModelConfigError('top level must be a mapping')
    rows = data.get('models')
    if not isinstance(rows, list) or not rows:
        raise ModelConfigError('models must be a non-empty list')

    endpoints = []
    seen = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ModelConfigError(f'models[{index}] must be a mapping')
        model = row.get('model')
        if not isinstance(model, str) or not model.strip():
            raise ModelConfigError(
                f'models[{index}].model must be a non-empty string')
        model = model.strip()
        if model in seen:
            raise ModelConfigError(f'duplicate model {model!r}')
        seen.add(model)

        providers = row.get('providers', [])
        if providers is None:
            providers = []
        elif not isinstance(providers, list):
            raise ModelConfigError(
                f'models[{index}].providers must be a list')
        cleaned = []
        for provider in providers:
            if not isinstance(provider, str) or not provider.strip():
                raise ModelConfigError(
                    f'models[{index}].providers must contain only '
                    'non-empty strings')
            provider = provider.strip()
            if provider not in cleaned:
                cleaned.append(provider)
        endpoints.append(ModelEndpoint(model, tuple(cleaned)))
    return tuple(endpoints)


def load_model_config(path=MODELS_PATH):
    """Load configured OpenRouter model endpoints from ``path``."""
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return parse_model_config(fh.read())
    except OSError as e:
        raise ModelConfigError(f'could not read {path}: {e}') from e


def find_model(endpoints, model):
    """Return the configured endpoint named ``model``, if present."""
    return next((endpoint for endpoint in endpoints
                 if endpoint.model == model), None)


_MODEL_COMPLETION_RE = re.compile(r'^[ \t]*model![ \t]+(.*)$')


def complete_model_command(code, cursor_pos, endpoints=None):
    """Return a Jupyter ``complete_reply`` for ``model!`` or ``None``.

    Model ids are completed before the first comma. Once a configured model
    and comma are present, its provider order is completed one provider at a
    time. The returned cursor range replaces only the token at the cursor, so
    the accepted completion leaves a replayable ``model! ...`` cell behind.
    """
    cursor_pos = max(0, min(cursor_pos, len(code)))
    line_start = code.rfind('\n', 0, cursor_pos) + 1
    line = code[line_start:cursor_pos]
    match = _MODEL_COMPLETION_RE.match(line)
    if match is None:
        return None
    if endpoints is None:
        endpoints = load_model_config()

    arguments = match.group(1)
    argument_start = line_start + match.start(1)
    comma = arguments.rfind(',')
    if comma == -1:
        token = arguments
        leading = len(token) - len(token.lstrip())
        matches = [endpoint.model for endpoint in endpoints]
        kind = 'model'
        cursor_start = argument_start + leading
    else:
        model = arguments.split(',', 1)[0].strip()
        endpoint = find_model(endpoints, model)
        token = arguments[comma + 1:]
        leading = len(token) - len(token.lstrip())
        used = {part.strip() for part in arguments[:comma].split(',')[1:]}
        matches = ([] if endpoint is None else
                   [provider for provider in endpoint.providers
                    if provider not in used])
        kind = 'provider'
        cursor_start = argument_start + comma + 1 + leading

    return {
        'matches': matches,
        'cursor_start': cursor_start,
        'cursor_end': cursor_pos,
        'metadata': {
            '_jupyter_types_experimental': [
                {'text': value, 'type': kind} for value in matches
            ],
        },
        'status': 'ok',
    }
