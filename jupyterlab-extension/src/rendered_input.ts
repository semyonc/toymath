/**
 * The rendered view of a ToyMath cell's input.
 *
 * A ToyMath cell holds LaTeX, and a dense formula is unreadable as source.
 * This is the markdown-cell bargain applied to code cells: the source while
 * the cell is being edited, the typeset formula the rest of the time.
 *
 * The kernel decides what a cell renders as — it owns the parser, so the
 * rendered formula is what the *engine* understood, not a second reading of
 * the source by the frontend. A cell the kernel does not read as a formula
 * (a do! prompt, a `model!` line, anything that fails to parse) answers with
 * no segments and keeps showing its source.
 *
 * Kept free of JupyterLab imports so it is testable without a browser.
 */

/** The `int!` a cell runs: a command name, not part of the mathematics. */
export interface ICommandSegment {
  readonly kind: 'command';
  readonly text: string;
}

/** Prose, from a do! prompt the kernel found formulas inside. */
export interface ITextSegment {
  readonly kind: 'text';
  readonly text: string;
}

export interface IMathSegment {
  readonly kind: 'math';
  readonly latex: string;
}

/**
 * A `[[n]]` reference to an earlier result that has no formula yet.
 *
 * Once that result exists the kernel sends the formula instead, so this is
 * what a cell waiting on a re-run shows — still legible as the command it is,
 * rather than dropping back to raw source.
 */
export interface IRefSegment {
  readonly kind: 'ref';
  readonly text: string;
}

export interface IBreakSegment {
  readonly kind: 'break';
}

export type PreviewSegment =
  | ICommandSegment
  | ITextSegment
  | IMathSegment
  | IRefSegment
  | IBreakSegment;

/** The reply payload the kernel sends back over the render comm. */
export interface IPreviewReply {
  id?: unknown;
  segments?: unknown;
}

/**
 * Validate one comm reply into segments, or null to keep the source.
 *
 * A reply that carries nothing but a label renders nothing: an input area
 * showing only `int!` would hide the cell instead of explaining it.
 */
export function readSegments(data: IPreviewReply): PreviewSegment[] | null {
  if (!Array.isArray(data?.segments)) {
    return null;
  }
  const segments: PreviewSegment[] = [];
  for (const raw of data.segments) {
    const kind = (raw as { kind?: unknown })?.kind;
    const text = (raw as { text?: unknown })?.text;
    const latex = (raw as { latex?: unknown })?.latex;
    if (
      (kind === 'command' || kind === 'text' || kind === 'ref') &&
      typeof text === 'string'
    ) {
      segments.push({ kind, text });
    } else if (kind === 'math' && typeof latex === 'string') {
      segments.push({ kind, latex });
    } else if (kind === 'break') {
      segments.push({ kind });
    } else {
      return null;
    }
  }
  const carries = (segment: PreviewSegment): boolean =>
    segment.kind === 'math' || segment.kind === 'ref';
  return segments.some(carries) ? segments : null;
}

/**
 * The MathJax spelling of a rendered formula.
 *
 * Display style rather than `$$…$$`: a cell's formula wants full-size
 * integral and sum limits, but it stays left-aligned where its source was.
 */
export function displayMath(latex: string): string {
  return `$\\displaystyle ${latex}$`;
}

export interface IRenderState {
  /** The notebook-local toggle. */
  readonly enabled: boolean;
  /** Only ToyMath's own kernel can answer a preview request. */
  readonly isToyMath: boolean;
  /** Markdown and raw cells already have their own rendered view. */
  readonly isCodeCell: boolean;
  readonly isActive: boolean;
  /** The notebook is in edit mode. */
  readonly editing: boolean;
}

/**
 * Whether a cell should show its rendered view.
 *
 * Rendering is the resting state; the editor comes back for the one cell
 * being edited. This mirrors what `Notebook.setMode` does for markdown, which
 * it does not do for code cells.
 */
export function shouldRender(state: IRenderState): boolean {
  if (!state.enabled || !state.isToyMath || !state.isCodeCell) {
    return false;
  }
  return !(state.isActive && state.editing);
}

export interface IPreviewTransport {
  send(payload: { id: string; code: string }): void;
}

interface IPendingPreview {
  readonly code: string;
  readonly cacheable: boolean;
  readonly settle: (segments: PreviewSegment[] | null) => void;
}

/**
 * Preview requests in flight, and the answers already known.
 *
 * Several cells ask at once — a freshly opened notebook asks for all of
 * them — so replies are matched by request id rather than by arrival order.
 */
export class PreviewRequests {
  constructor(transport: IPreviewTransport, limit = 256) {
    this._transport = transport;
    this._limit = limit;
  }

  /**
   * Ask what `code` renders as. Resolves to null when it renders as itself.
   *
   * A source carrying a `[[n]]` backreference is never cached: it renders as
   * the formula that reference stands for, and that follows the notebook's
   * history rather than the cell's own text.
   */
  request(code: string): Promise<PreviewSegment[] | null> {
    const cacheable = !code.includes('[[');
    if (cacheable && this._cache.has(code)) {
      return Promise.resolve(this._cache.get(code)!);
    }
    const id = `${++this._counter}`;
    return new Promise(resolve => {
      this._pending.set(id, { code, cacheable, settle: resolve });
      try {
        this._transport.send({ id, code });
      } catch (error) {
        this._pending.delete(id);
        resolve(null);
      }
    });
  }

  /** Settle the request a reply belongs to. Unknown ids are ignored. */
  resolve(data: IPreviewReply): void {
    const id = typeof data?.id === 'string' ? data.id : null;
    if (id === null) {
      return;
    }
    const pending = this._pending.get(id);
    if (!pending) {
      return;
    }
    this._pending.delete(id);
    const segments = readSegments(data);
    if (pending.cacheable) {
      this._remember(pending.code, segments);
    }
    pending.settle(segments);
  }

  /**
   * Drop everything known and settle what is in flight.
   *
   * Called when the kernel changes: no reply is coming from the old one, and
   * a new kernel has its own history, so a cached backreference-free answer
   * is the only thing worth keeping — and not worth the exception.
   */
  reset(): void {
    const pending = Array.from(this._pending.values());
    this._pending.clear();
    this._cache.clear();
    // Settled without an answer, so nothing is learned: a request abandoned
    // here must not leave "renders as nothing" behind in the cache.
    pending.forEach(entry => entry.settle(null));
  }

  private _remember(code: string, segments: PreviewSegment[] | null): void {
    if (this._cache.size >= this._limit) {
      const oldest = this._cache.keys().next();
      if (!oldest.done) {
        this._cache.delete(oldest.value);
      }
    }
    this._cache.set(code, segments);
  }

  private _transport: IPreviewTransport;
  private _limit: number;
  private _counter = 0;
  private _pending = new Map<string, IPendingPreview>();
  private _cache = new Map<string, PreviewSegment[] | null>();
}
