#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_sandbox.py - sandboxed figure backends for the do! agent.

The agent writes figure code; a backend executes it OUTSIDE the kernel
process and returns rendered figures. Figures are illustrations, never
evidence: they do not enter the ledger and replay ignores them.

Two backends, deliberately separate processes with different grants:

* `PyodideBackend.run_plot` executes agent-authored *Python*
  (matplotlib/seaborn/plotly) inside Pyodide under Deno. The agent's code
  runs here, and Pyodide's `js` bridge exposes host APIs gated by Deno's
  grants - so this process gets no --allow-env and no write beyond its
  caches.
* `TikzBackend.render` renders agent-authored *TeX* to SVG via
  node-tikzjax. No agent code executes here: TeX runs inside a wasm engine
  over an in-memory filesystem with no bridge to the host. That is why it
  can afford --allow-sys/--allow-env (a transitive Node dependency
  enumerates process.env at import) while needing no network at all.

Both are spawned with a scrubbed environment, so even a widened grant
cannot reach secrets such as the OpenRouter key.

Selection: TOYMATH_SANDBOX = auto (default) | pyodide | off.
The seams (`run_plot(code, timeout)`, `render(code, timeout)`) are
backend-agnostic so a Docker / llm-sandbox backend can be added later.
"""
import json
import os
import re
import shutil
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER_PATH = os.path.join(_HERE, 'pyodide_runner.mjs')
TIKZ_RUNNER_PATH = os.path.join(_HERE, 'tikz_runner.mjs')
# must match the npm: specifier pinned in tikz_runner.mjs
TIKZ_PACKAGE = 'node-tikzjax'
TIKZ_VERSION = '1.0.5'

DEFAULT_TIMEOUT = 180  # first call downloads the wasm wheels (cached after)
TIKZ_TIMEOUT = 120     # TeX is Turing-complete; a macro loop must not hang us
MAX_FIGURES = 6

# Deno needs a little of the ambient environment to run at all; nothing
# else is forwarded. This is what keeps `--allow-env` in the tikz runner
# from being worth anything to an attacker.
_ENV_ALLOWLIST = ('PATH', 'HOME', 'TMPDIR', 'DENO_DIR', 'LANG', 'LC_ALL')


def _child_env():
    return {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}


def _find_deno():
    deno = shutil.which('deno')
    if deno:
        return deno
    for candidate in ('/opt/homebrew/bin/deno', '/usr/local/bin/deno',
                      os.path.expanduser('~/.deno/bin/deno')):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _wheel_cache():
    """Directory for wheels micropip pulls from PyPI. Every run_plot is a
    fresh process with an in-memory filesystem, so without this seaborn
    and plotly re-download on every single call."""
    base = os.environ.get('TOYMATH_WHEEL_CACHE') or os.path.join(
        os.path.expanduser('~'), '.cache', 'toymath', 'wheels')
    try:
        os.makedirs(base, exist_ok=True)
    except OSError:
        return None  # caching is best-effort; fall back to re-downloading
    return base


def _npm_cache_dir(deno):
    try:
        info = subprocess.run([deno, 'info', '--json'], capture_output=True,
                              timeout=20, env=_child_env())
        return json.loads(info.stdout)['npmCache']
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        return ''


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
            'stdout': stdout[-2000:], 'stderr': stderr[-2000:]}


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
        Deno's npm cache and our wheel cache, network only to the wheel
        CDN, and deliberately NO --allow-env - the agent must not be able
        to read secrets like the OpenRouter key through the bridge."""
        if self._npm_cache is None:
            self._npm_cache = _npm_cache_dir(self.deno)
        if not self._npm_cache:
            return None
        paths = [self._npm_cache]
        wheels = _wheel_cache()
        if wheels:
            paths.append(wheels)
        allowed = ','.join(paths)
        # net: wheel CDN + PyPI (micropip resolves pure-python packages
        # such as seaborn and plotly from there)
        return ['--quiet', '--no-prompt',
                f'--allow-read={allowed}',
                f'--allow-write={allowed}',
                '--allow-net=cdn.jsdelivr.net,pypi.org,'
                'files.pythonhosted.org']

    def run_plot(self, code, timeout=DEFAULT_TIMEOUT):
        """Execute figure code; returns {ok, stdout, stderr, error?,
        figures: [{kind, data, height?}, ...], images: [png_base64, ...]}.
        `images` is the PNG subset, kept for callers that only handle
        rasters. Never raises on sandbox failure."""
        if not self.available():
            return {'ok': False, 'figures': [], 'images': [],
                    'error': 'deno not found - install it (brew install '
                             'deno) to enable plotting'}
        flags = self._flags()
        if flags is None:
            return {'ok': False, 'figures': [], 'images': [],
                    'error': 'could not determine the deno npm cache '
                             '(deno info failed)'}
        request = {'code': code, 'wheelCache': _wheel_cache() or ''}
        cmd = [self.deno, 'run'] + flags + [RUNNER_PATH]
        try:
            proc = subprocess.run(
                cmd, input=json.dumps(request).encode('utf-8'),
                capture_output=True, timeout=timeout, env=_child_env())
        except subprocess.TimeoutExpired:
            return {'ok': False, 'figures': [], 'images': [],
                    'error': f'sandbox timed out after {timeout}s'}
        except OSError as e:
            return {'ok': False, 'figures': [], 'images': [],
                    'error': f'cannot start sandbox: {e}'}
        result = _parse_runner_output(
            proc.stdout.decode('utf-8', 'replace'),
            proc.stderr.decode('utf-8', 'replace'))
        figures = result.get('figures')
        if not isinstance(figures, list):
            # a backend that only knows PNGs still satisfies the contract
            figures = [{'kind': 'png', 'data': d}
                       for d in (result.get('images') or [])]
        result['figures'] = figures[:MAX_FIGURES]
        result['images'] = [f['data'] for f in result['figures']
                            if f.get('kind') == 'png']
        return result


#: picture environments a submission may open; `\begin{document}` goes
#: before whichever appears first
_TIKZ_PICTURE = re.compile(r'^[ \t]*\\begin\{(tikzpicture|tikzcd)\}',
                           re.MULTILINE)
#: `\documentclass{x}` with optional [options], possibly spanning lines
_DOCUMENTCLASS = re.compile(
    r'[ \t]*\\document(?:class|style)\s*(?:\[[^\]]*\])?\s*\{[^}]*\}[ \t]*\n?')


def normalize_tikz_source(code):
    """Shape agent TeX into what node-tikzjax actually accepts.

    The engine supplies its own `\\documentclass` but *not* a `document`
    environment, which is an unusual middle ground: a bare `tikzpicture`
    dies with `Missing \\begin{document}`, and a complete document dies
    with `Two \\documentclass commands`. A model writing either
    perfectly-reasonable form gets a TeX error that says nothing about the
    real rule, and spends its turns guessing at scaffolding instead of at
    the drawing - which is exactly what one traced run did, four times.

    So both forms are accepted here. This only ever moves scaffolding: a
    figure is an unverified illustration that never enters the ledger, so
    being forgiving about its wrapper costs no trust.
    """
    if not code:
        return code
    text = _DOCUMENTCLASS.sub('', code, count=1)
    if '\\begin{document}' in text:
        return text
    picture = _TIKZ_PICTURE.search(text)
    if picture is None:
        # nothing recognisable to wrap; let the TeX log speak for itself
        return text
    head, body = text[:picture.start()], text[picture.start():]
    if '\\end{document}' not in body:
        body = body.rstrip() + '\n\\end{document}'
    return f'{head}\\begin{{document}}\n{body}'


class TikzBackend(object):
    name = 'node-tikzjax'

    def __init__(self, deno=None):
        self.deno = deno or _find_deno()
        self._npm_cache = None
        self._module_ready = None

    def available(self):
        return self.deno is not None and os.path.isfile(TIKZ_RUNNER_PATH)

    def _cache_dir(self):
        if self._npm_cache is None:
            self._npm_cache = _npm_cache_dir(self.deno)
        return self._npm_cache

    def _ensure_module(self):
        """Deno resolves `npm:` specifiers over the network the first time
        it sees them. Do that once, explicitly and out of band, so the
        render itself can run with --allow-net absent entirely."""
        if self._module_ready:
            return True
        cache = self._cache_dir()
        if cache:
            pkg = os.path.join(cache, 'registry.npmjs.org', TIKZ_PACKAGE,
                               TIKZ_VERSION)
            if os.path.isdir(pkg):
                self._module_ready = True
                return True
        try:
            proc = subprocess.run(
                [self.deno, 'cache', '--quiet', TIKZ_RUNNER_PATH],
                capture_output=True, timeout=DEFAULT_TIMEOUT,
                env=_child_env())
            self._module_ready = proc.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            self._module_ready = False
        return self._module_ready

    def _flags(self):
        """Read the npm cache (the TeX engine, its package tree and the
        bakoma fonts all ship inside node-tikzjax); --allow-sys=uid,gid
        for the in-memory filesystem; --allow-env because a transitive
        Node dependency enumerates process.env at import time.

        No --allow-net and no --allow-write: TikZ renders fully offline.
        These grants are wider than the plot sandbox's on purpose and are
        only defensible because this process runs no agent-authored code -
        the agent supplies TeX, which the wasm engine reads through an
        in-memory filesystem with no host bridge. Never render agent
        Python here."""
        cache = self._cache_dir()
        if not cache:
            return None
        return ['--quiet', '--no-prompt', f'--allow-read={cache}',
                '--allow-sys=uid,gid', '--allow-env']

    def render(self, code, timeout=TIKZ_TIMEOUT):
        """Render TeX/TikZ to a self-contained SVG; returns
        {ok, svg?, error?}. Never raises on renderer failure."""
        if not self.available():
            return {'ok': False,
                    'error': 'deno not found - install it (brew install '
                             'deno) to enable tikz'}
        flags = self._flags()
        if flags is None:
            return {'ok': False,
                    'error': 'could not determine the deno npm cache '
                             '(deno info failed)'}
        if not self._ensure_module():
            return {'ok': False,
                    'error': f'could not fetch {TIKZ_PACKAGE}@{TIKZ_VERSION}'
                             ' - the first tikz render needs network access'}
        cmd = [self.deno, 'run'] + flags + [TIKZ_RUNNER_PATH]
        try:
            proc = subprocess.run(
                cmd,
                input=json.dumps(
                    {'code': normalize_tikz_source(code)}).encode('utf-8'),
                capture_output=True, timeout=timeout, env=_child_env())
        except subprocess.TimeoutExpired:
            return {'ok': False,
                    'error': f'tikz render timed out after {timeout}s - a '
                             'runaway TeX macro?'}
        except OSError as e:
            return {'ok': False, 'error': f'cannot start tikz renderer: {e}'}
        return _parse_runner_output(
            proc.stdout.decode('utf-8', 'replace'),
            proc.stderr.decode('utf-8', 'replace'))


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


def get_tikz_backend():
    """The configured TikZ backend, or None when figures are off or Deno
    is missing (the tikz tool is simply not registered then)."""
    if os.environ.get('TOYMATH_SANDBOX', 'auto').lower() == 'off':
        return None
    backend = TikzBackend()
    return backend if backend.available() else None
