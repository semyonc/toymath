// pyodide_runner.mjs - sandboxed matplotlib/seaborn execution for the
// toymath `do!` plot tool.
//
// Protocol: one JSON object {code: string} on stdin; one JSON line
// {ok, stdout, stderr, error?, images: [png_base64, ...]} as the LAST
// stdout line. Everything else (pyodide loader chatter) goes to stderr.
//
// Run under Deno with deny-by-default permissions; the launcher
// (engine/plot_sandbox.py) grants only read/env plus network to the
// Pyodide package CDN. The agent's Python runs inside the WASM sandbox
// with an in-memory filesystem.

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

# pure-python packages outside the pyodide distribution (e.g. seaborn)
# resolve from PyPI via micropip
_missing = [_m for _m in _find_imports(USER_CODE)
            if _ilu.find_spec(_m) is None]
if _missing:
    try:
        import pyodide_js as _pjs
        await _pjs.loadPackage("micropip")
        import micropip as _mp
        await _mp.install(_missing)
    except Exception:
        _result["ok"] = False
        _result["error"] = ("cannot install " + ", ".join(_missing)
                            + ":\\n" + _tb.format_exc(limit=3))

_out, _err = _io.StringIO(), _io.StringIO()
_old = _sys.stdout, _sys.stderr
_sys.stdout, _sys.stderr = _out, _err
if _result["ok"]:
    try:
        exec(compile(USER_CODE, "<plot>", "exec"), {"__name__": "__main__"})
    except BaseException:
        _result["ok"] = False
        _result["error"] = _tb.format_exc(limit=8)
_sys.stdout, _sys.stderr = _old

_images = []
try:
    import matplotlib.pyplot as _plt
    for _num in _plt.get_fignums():
        _buf = _io.BytesIO()
        _plt.figure(_num).savefig(_buf, format="png", dpi=110,
                                  bbox_inches="tight")
        _images.append(_b64.b64encode(_buf.getvalue()).decode("ascii"))
    _plt.close("all")
except Exception:
    pass

_result["stdout"] = _out.getvalue()
_result["stderr"] = _err.getvalue()
_result["images"] = _images
_json.dumps(_result)
`;

function fail(message) {
  console.log(JSON.stringify({ ok: false, error: message, stdout: "",
                               stderr: "", images: [] }));
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

try {
  const pyodide = await loadPyodide({
    stdout: (s) => console.error(s),
    stderr: (s) => console.error(s),
  });
  // matplotlib is always needed for capture; everything else the code
  // imports (numpy, pandas, seaborn, scipy, ...) resolves from imports
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
