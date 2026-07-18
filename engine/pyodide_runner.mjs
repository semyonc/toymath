// pyodide_runner.mjs - sandboxed matplotlib/seaborn/plotly execution for
// the toymath `do!` plot tool.
//
// Protocol: one JSON object {code: string, wheelCache?: string} on stdin;
// one JSON line {ok, stdout, stderr, error?, figures: [{kind, data,
// height?}], images: [png_base64, ...]} as the LAST stdout line.
// Everything else (pyodide loader chatter) goes to stderr.
//
// Run under Deno with deny-by-default permissions; the launcher
// (engine/plot_sandbox.py) grants only read/write to the two caches plus
// network to the wheel CDNs, and deliberately no --allow-env. The agent's
// Python runs inside the WASM sandbox with an in-memory filesystem, so
// those grants are the boundary for the agent's code too.

import { loadPyodide } from "npm:pyodide@0.28.3";

const HARNESS = `
import base64 as _b64
import importlib.util as _ilu
import io as _io
import json as _json
import os as _os
import sys as _sys
import traceback as _tb
from pyodide.code import find_imports as _find_imports

_os.environ["MPLBACKEND"] = "Agg"

_result = {"ok": True, "error": None}

# Packages outside the pyodide distribution (seaborn, plotly) resolve from
# PyPI via micropip. Two traps here: find_imports reports submodules as
# well as top-level names ("plotly.graph_objects"), but micropip installs
# distributions; and find_spec RAISES rather than returning None when a
# dotted name's parent is absent, which used to kill the whole sandbox.
# Reduce to deduped top-level names first, and never let find_spec throw.
_wanted = []
for _name in _find_imports(USER_CODE):
    _top = _name.split(".")[0]
    if _top and _top not in _wanted:
        _wanted.append(_top)

_missing = []
for _top in _wanted:
    try:
        if _ilu.find_spec(_top) is None:
            _missing.append(_top)
    except BaseException:
        _missing.append(_top)

if _missing:
    _failed = []
    try:
        import pyodide_js as _pjs
        await _pjs.loadPackage("micropip")
        import micropip as _mp
        for _top in _missing:
            # one at a time: a single unresolvable name must not stop the
            # others from installing
            try:
                await _mp.install(_top)
            except BaseException as _e:
                _failed.append(_top + " (" + str(_e)[:160] + ")")
    except BaseException:
        _result["ok"] = False
        _result["error"] = "micropip unavailable:\\n" + _tb.format_exc(limit=3)
    if _failed and _result["ok"]:
        _result["ok"] = False
        _result["error"] = "cannot install " + ", ".join(_failed)

_out, _err = _io.StringIO(), _io.StringIO()
_old = _sys.stdout, _sys.stderr
_sys.stdout, _sys.stderr = _out, _err
_ns = {"__name__": "__main__"}
if _result["ok"]:
    try:
        exec(compile(USER_CODE, "<plot>", "exec"), _ns)
    except BaseException:
        _result["ok"] = False
        _result["error"] = _tb.format_exc(limit=8)
_sys.stdout, _sys.stderr = _old

_figures = []

try:
    import matplotlib.pyplot as _plt
    for _num in _plt.get_fignums():
        _buf = _io.BytesIO()
        _plt.figure(_num).savefig(_buf, format="png", dpi=110,
                                  bbox_inches="tight")
        _figures.append({
            "kind": "png",
            "data": _b64.b64encode(_buf.getvalue()).decode("ascii"),
        })
    _plt.close("all")
except BaseException:
    pass

# Plotly cannot rasterise in here: to_image() needs kaleido, a native
# Chrome binary with no wasm build. Its figures therefore travel as
# self-contained HTML and the kernel iframes them. Only look when the code
# actually imported plotly - checking sys.modules never force-loads it.
if "plotly" in _sys.modules:
    try:
        from plotly.basedatatypes import BaseFigure as _BaseFigure
        _seen = set()
        for _value in list(_ns.values()):
            if not isinstance(_value, _BaseFigure) or id(_value) in _seen:
                continue
            _seen.add(id(_value))
            try:
                _height = int(_value.layout.height or 0)
            except BaseException:
                _height = 0
            _figures.append({
                "kind": "html",
                "data": _value.to_html(include_plotlyjs="cdn",
                                       full_html=True),
                "height": (_height + 40) if _height else 520,
            })
    except BaseException:
        pass

_result["stdout"] = _out.getvalue()
_result["stderr"] = _err.getvalue()
_result["figures"] = _figures
_result["images"] = [_f["data"] for _f in _figures if _f["kind"] == "png"]
_json.dumps(_result)
`;

function fail(message) {
  console.log(JSON.stringify({ ok: false, error: message, stdout: "",
                               stderr: "", figures: [], images: [] }));
  Deno.exit(0); // protocol errors are reported in-band
}

let request;
try {
  const raw = await new Response(Deno.stdin.readable).text();
  request = JSON.parse(raw);
} catch (e) {
  fail(`bad request: ${e.message}`);
}
if (typeof request.code !== "string" || !request.code.trim()) {
  fail("request.code must be a non-empty string");
}

// Every run is a fresh process with an in-memory filesystem, so without a
// disk cache each call re-downloads every wheel that is not bundled in the
// pyodide npm package - scipy and pandas from jsdelivr, seaborn and plotly
// from PyPI. Wheel URLs are versioned and content-addressed, so a hit can
// never be stale; PyPI's JSON metadata is deliberately NOT cached, so
// resolution still sees new releases.
const wheelCache = typeof request.wheelCache === "string"
  ? request.wheelCache
  : "";
if (wheelCache) {
  const nativeFetch = globalThis.fetch;
  const cachePath = (url) => {
    let h = 0;
    for (let i = 0; i < url.length; i++) {
      h = (Math.imul(31, h) + url.charCodeAt(i)) | 0;
    }
    const base = (url.split("/").pop() || "wheel").split("?")[0]
      .replace(/[^A-Za-z0-9._-]/g, "_").slice(-90);
    return `${wheelCache}/${(h >>> 0).toString(36)}-${base}`;
  };
  globalThis.fetch = async (input, init) => {
    const url = typeof input === "string"
      ? input
      : (input && input.url) || String(input);
    if (!/\.whl(\?|$)/i.test(url)) return nativeFetch(input, init);
    const path = cachePath(url);
    try {
      const hit = await Deno.readFile(path);
      return new Response(hit, {
        status: 200,
        headers: { "content-type": "application/zip" },
      });
    } catch {
      // cache miss: fall through and fetch
    }
    const resp = await nativeFetch(input, init);
    if (!resp.ok) return resp;
    const body = new Uint8Array(await resp.arrayBuffer());
    try {
      await Deno.writeFile(path, body);
    } catch {
      // caching is best-effort; a failed write must not fail the plot
    }
    return new Response(body, { status: 200, headers: resp.headers });
  };
}

try {
  const pyodide = await loadPyodide({
    stdout: (s) => console.error(s),
    stderr: (s) => console.error(s),
  });
  // matplotlib is always needed for capture; everything else the code
  // imports (numpy, pandas, seaborn, scipy, plotly, ...) resolves from
  // imports, falling back to micropip inside the harness
  await pyodide.loadPackage("matplotlib", {
    messageCallback: (s) => console.error(s),
  });
  await pyodide.loadPackagesFromImports(request.code, {
    messageCallback: (s) => console.error(s),
  });
  pyodide.globals.set("USER_CODE", request.code);
  const result = await pyodide.runPythonAsync(HARNESS);
  console.log(result); // already a JSON string
} catch (e) {
  fail(`pyodide failure: ${e.message}`);
}
