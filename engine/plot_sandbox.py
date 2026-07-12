#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_sandbox.py - sandboxed plotting backends for the do! agent.

The agent writes matplotlib/seaborn code; a backend executes it OUTSIDE
the kernel process and returns rendered figures. Plots are illustrations,
never evidence: they do not enter the ledger and replay ignores them.

v1 backend: Pyodide under Deno. The Python code runs inside the WASM
sandbox (in-memory filesystem); Deno's deny-by-default permission model
grants only what Pyodide needs - read access for the cached npm package,
env probing, and network to the package CDN for the wasm wheels (cached
after first use). No write, no subprocess, no FFI.

Selection: TOYMATH_SANDBOX = auto (default) | pyodide | off.
The seam (`run_plot(code, timeout)`) is backend-agnostic so a Docker /
llm-sandbox backend can be added later.
"""
import json
import os
import shutil
import subprocess

RUNNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'pyodide_runner.mjs')
DEFAULT_TIMEOUT = 180  # first call downloads the wasm wheels (cached after)
MAX_IMAGES = 6


def _find_deno():
    deno = shutil.which('deno')
    if deno:
        return deno
    for candidate in ('/opt/homebrew/bin/deno', '/usr/local/bin/deno',
                      os.path.expanduser('~/.deno/bin/deno')):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _parse_runner_output(stdout, stderr):
    """The result is the last stdout line that parses as a JSON object;
    loader chatter goes to stderr by protocol but stay defensive."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {'ok': False, 'error': 'sandbox produced no result',
            'stdout': stdout[-2000:], 'stderr': stderr[-2000:],
            'images': []}


class PyodideBackend(object):
    name = 'pyodide'

    def __init__(self, deno=None):
        self.deno = deno or _find_deno()
        self._npm_cache = None

    def available(self):
        return self.deno is not None and os.path.isfile(RUNNER_PATH)

    def _flags(self):
        """Deny-by-default permission flags. These are the security
        boundary for the agent's code too (the pyodide `js` bridge
        exposes host APIs, gated by these same grants): read/write only
        Deno's npm cache (wheel caching), network only to the wheel CDN,
        and deliberately NO --allow-env - the agent must not be able to
        read secrets like the OpenRouter key through the bridge."""
        if self._npm_cache is None:
            try:
                info = subprocess.run([self.deno, 'info', '--json'],
                                      capture_output=True, timeout=20)
                self._npm_cache = json.loads(info.stdout)['npmCache']
            except (OSError, ValueError, KeyError,
                    subprocess.TimeoutExpired):
                self._npm_cache = ''
        if not self._npm_cache:
            return None
        # net: wheel CDN + PyPI (micropip resolves pure-python packages
        # such as seaborn from there)
        return ['--quiet', '--no-prompt',
                f'--allow-read={self._npm_cache}',
                f'--allow-write={self._npm_cache}',
                '--allow-net=cdn.jsdelivr.net,pypi.org,'
                'files.pythonhosted.org']

    def run_plot(self, code, timeout=DEFAULT_TIMEOUT):
        """Execute plotting code; returns {ok, stdout, stderr, error?,
        images: [png_base64, ...]}. Never raises on sandbox failure."""
        if not self.available():
            return {'ok': False, 'images': [],
                    'error': 'deno not found - install it (brew install '
                             'deno) to enable plotting'}
        flags = self._flags()
        if flags is None:
            return {'ok': False, 'images': [],
                    'error': 'could not determine the deno npm cache '
                             '(deno info failed)'}
        cmd = [self.deno, 'run'] + flags + [RUNNER_PATH]
        try:
            proc = subprocess.run(
                cmd, input=json.dumps({'code': code}).encode('utf-8'),
                capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {'ok': False, 'images': [],
                    'error': f'sandbox timed out after {timeout}s'}
        except OSError as e:
            return {'ok': False, 'images': [],
                    'error': f'cannot start sandbox: {e}'}
        result = _parse_runner_output(
            proc.stdout.decode('utf-8', 'replace'),
            proc.stderr.decode('utf-8', 'replace'))
        result.setdefault('images', [])
        result['images'] = result['images'][:MAX_IMAGES]
        return result


def get_backend():
    """The configured plotting backend, or None when plotting is off or
    unavailable (the plot tool is simply not registered then)."""
    mode = os.environ.get('TOYMATH_SANDBOX', 'auto').lower()
    if mode == 'off':
        return None
    if mode in ('auto', 'pyodide'):
        backend = PyodideBackend()
        return backend if backend.available() else None
    return None
