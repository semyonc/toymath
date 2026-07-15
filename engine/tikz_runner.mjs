// tikz_runner.mjs - offline TikZ -> SVG rendering for the toymath `tikz`
// tool.
//
// Protocol: one JSON object {code: string} on stdin; one JSON line
// {ok, svg?, error?} as the LAST stdout line. Everything else (TeX engine
// chatter) goes to stderr.
//
// Unlike pyodide_runner.mjs, this process never executes agent-authored
// code: the agent supplies TeX, which a wasm TeX engine reads through an
// in-memory filesystem with no bridge to the host. That is the whole
// reason it can afford --allow-env (a transitive Node dependency
// enumerates process.env at import time) where the plot sandbox
// deliberately cannot. It runs with no network and no write access: the
// TeX engine, its package tree and the bakoma fonts all ship inside the
// node-tikzjax package, so a rendered figure needs nothing at view time.

import * as tikzjax from "npm:node-tikzjax@1.0.5";

const PKG = "npm:node-tikzjax@1.0.5";

// CJS/ESM interop: under Deno the callable lands on .default.default
const tex2svg = typeof tikzjax.default === "function"
  ? tikzjax.default
  : tikzjax.default.default;

function base64(bytes) {
  let s = "";
  const chunk = 0x8000; // fromCharCode(...) blows the stack on big inputs
  for (let i = 0; i < bytes.length; i += chunk) {
    s += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(s);
}

/** Strip anything executable. dvi2html only emits drawing ops and text,
 *  but the TeX source is agent-authored and the SVG lands in the user's
 *  notebook, so this is the one place that boundary is enforced. */
function sanitize(svg) {
  return svg
    .replace(/<\s*script[\s\S]*?<\s*\/\s*script\s*>/gi, "")
    .replace(/<\s*script\b[^>]*\/\s*>/gi, "")
    .replace(/<\s*foreignObject[\s\S]*?<\s*\/\s*foreignObject\s*>/gi, "")
    .replace(/\son\w+\s*=\s*"[^"]*"/gi, "")
    .replace(/\son\w+\s*=\s*'[^']*'/gi, "")
    .replace(/(xlink:href|href)\s*=\s*(["'])\s*javascript:[^"']*\2/gi, "");
}

/** TeX font encodings put glyphs at high code points - cmsy10 keeps the
 *  minus sign at U+00A1 - so the text payload is non-ASCII, and it is read
 *  through a symbol font where a mis-decode does not degrade, it silently
 *  draws different glyphs ("-2" came out as "BZ"). Emitting those as
 *  numeric character references makes the figure independent of whatever
 *  encoding the host page happens to declare. The base64 font blobs and
 *  path data are pure ASCII, so this only ever touches text. */
function asciify(svg) {
  return svg.replace(/[^\x00-\x7F]/g, (c) => `&#${c.charCodeAt(0)};`);
}

/** Inline only the fonts this figure actually references, as data URIs.
 *  The full bakoma set is ~3.7MB; a typical figure touches three faces
 *  (~84KB), and inlining them is what keeps the SVG self-contained -
 *  no CSS fetch when the notebook is reopened offline. */
async function inlineFonts(svg) {
  const names = new Set();
  for (const m of svg.matchAll(/font-family="([^"]+)"/g)) names.add(m[1]);
  const faces = [];
  for (const name of names) {
    if (!/^[A-Za-z0-9_-]+$/.test(name)) continue; // never build a path from loose text
    try {
      const url = import.meta.resolve(`${PKG}/css/bakoma/ttf/${name}.ttf`);
      const buf = await Deno.readFile(new URL(url));
      faces.push(`@font-face{font-family:"${name}";`
        + `src:url(data:font/ttf;base64,${base64(buf)}) format("truetype");}`);
    } catch {
      // unknown face: the browser falls back, metrics drift but the
      // figure still reads. Never fail a render over a font.
    }
  }
  if (!faces.length) return svg;
  const defs = `<defs><style type="text/css">${faces.join("")}</style></defs>`;
  return svg.replace(/(<svg\b[^>]*>)/, `$1${defs}`);
}

/** The TeX log is the agent's only feedback channel, but node-tikzjax
 *  prints it with console.log, which would corrupt the stdout protocol.
 *  Capture it, and surface the `!` error lines - those name the actual
 *  mistake ("! Undefined control sequence"). */
function texDetail(logs) {
  const errors = logs.filter((l) => /^!|^l\.\d+/.test(l.trim()));
  const detail = (errors.length ? errors : logs.slice(-14)).join("\n");
  return detail.slice(0, 1500);
}

// captured before the TeX engine's chatter is redirected below, so the
// result line always reaches real stdout
const realLog = console.log;

function emit(obj) {
  realLog(JSON.stringify(obj));
}

let request;
try {
  const raw = await new Response(Deno.stdin.readable).text();
  request = JSON.parse(raw);
} catch (e) {
  emit({ ok: false, error: `bad request: ${e.message}` });
  Deno.exit(0);
}
if (typeof request.code !== "string" || !request.code.trim()) {
  emit({ ok: false, error: "request.code must be a non-empty string" });
  Deno.exit(0);
}

const logs = [];
console.log = (...a) => {
  const line = a.map((x) => typeof x === "string" ? x : String(x)).join(" ");
  logs.push(line);
  console.error(line);
};
try {
  let svg = await tex2svg(request.code, { showConsole: true });
  console.log = realLog;
  svg = asciify(await inlineFonts(sanitize(svg)));
  emit({ ok: true, svg });
} catch (e) {
  console.log = realLog;
  const detail = texDetail(logs);
  emit({
    ok: false,
    error: `TikZ render failed: ${String(e && e.message).slice(0, 300)}`
      + (detail ? `\nTeX log:\n${detail}` : ""),
  });
}
Deno.exit(0);
