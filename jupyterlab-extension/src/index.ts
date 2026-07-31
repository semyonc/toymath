import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { ICommandPalette } from '@jupyterlab/apputils';
import { Cell } from '@jupyterlab/cells';
import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { Kernel, KernelMessage } from '@jupyterlab/services';
import { IDisposable } from '@lumino/disposable';
import { Widget } from '@lumino/widgets';

import { closeCommSafely } from './comm_lifecycle';
import { shouldInvokeCompletion } from './completion';
import {
  applyModelTitle,
  findAddedKernelNameItem,
  IItemNode,
  KERNEL_NAME_ITEM
} from './model_title';
import {
  displayMath,
  IPreviewReply,
  PreviewRequests,
  PreviewSegment,
  shouldRender
} from './rendered_input';

const COMM_TARGET = 'toymath.model';
const RENDER_COMM_TARGET = 'toymath.render';
const TOYMATH_KERNEL = 'toymath';
const TOGGLE_COMMAND = 'toymath:toggle-rendered-input';
const RENDERED_CLASS = 'tm-mod-rendered';
const RENDERED_INPUT_CLASS = 'tm-RenderedInput';

interface IModelState {
  model: string;
  backend: string | null;
  providers: string[];
}

/** What a cell currently shows instead of its editor. */
interface IRenderedCell {
  source: string;
  key: string;
  widget: Widget;
}

interface IPanelState {
  comm: Kernel.IComm | null;
  model: IModelState | null;
  activeCell: Cell | null;
  completionTimer: number | null;
  disposed: boolean;
  kernelNameItem: HTMLElement | null;
  itemObserver: MutationObserver | null;
  toolbarObserver: MutationObserver | null;
  renderComm: Kernel.IComm | null;
  previews: PreviewRequests | null;
  renderEnabled: boolean;
  rendered: WeakMap<Cell, IRenderedCell>;
}

/** The state of every attached panel, for the notebook-local toggle. */
const panelStates = new WeakMap<NotebookPanel, IPanelState>();

function isToyMath(panel: NotebookPanel): boolean {
  return panel.sessionContext.session?.kernel?.name === TOYMATH_KERNEL;
}

function paintKernelTitle(panel: NotebookPanel, state: IPanelState): void {
  // KernelNameComponent has no dynamic-label API beyond kernelChanged, so keep
  // JupyterLab's own button and click handler and rewrite only its label. The
  // write is idempotent because it runs again on every mutation of the item.
  if (panel.isDisposed || !state.kernelNameItem) {
    return;
  }
  const routing = state.model && isToyMath(panel) ? state.model : null;
  applyModelTitle(
    // The DOM node satisfies the structural subset the painter needs.
    state.kernelNameItem as unknown as IItemNode,
    routing ? routing.model : null,
    panel.sessionContext.kernelDisplayName,
    routing ? routing.backend : null
  );
}

/**
 * Track the kernel-name toolbar item of `panel` and paint the current model.
 *
 * `item` is the node just added to the toolbar, when the caller saw it; the
 * responsive toolbar can park the item in its document-level overflow popup,
 * where a re-query of the toolbar would no longer find it.
 */
function bindKernelName(
  panel: NotebookPanel,
  state: IPanelState,
  item: HTMLElement | null = null
): void {
  const found =
    item ?? panel.toolbar.node.querySelector<HTMLElement>(KERNEL_NAME_ITEM);
  if (found && found !== state.kernelNameItem) {
    state.itemObserver?.disconnect();
    state.kernelNameItem = found;
    // React remounts the button whenever the item is re-parented between the
    // toolbar and the overflow popup, which drops the model from the label.
    const observer = new MutationObserver(() => paintKernelTitle(panel, state));
    observer.observe(found, {
      childList: true,
      subtree: true,
      characterData: true
    });
    state.itemObserver = observer;
  }
  paintKernelTitle(panel, state);
}

function readModelMessage(msg: KernelMessage.ICommMsgMsg): IModelState | null {
  const data = msg.content.data as Partial<IModelState>;
  if (typeof data.model !== 'string' || !data.model) {
    return null;
  }
  return {
    model: data.model,
    backend: typeof data.backend === 'string' ? data.backend : null,
    providers: Array.isArray(data.providers)
      ? data.providers.filter((value): value is string =>
          typeof value === 'string'
        )
      : []
  };
}

function connectPanel(panel: NotebookPanel, state: IPanelState): void {
  closeCommSafely(state.comm);
  state.comm = null;
  state.model = null;

  const kernel = panel.sessionContext.session?.kernel;
  if (!kernel || kernel.name !== TOYMATH_KERNEL) {
    paintKernelTitle(panel, state);
    return;
  }

  const comm = kernel.createComm(COMM_TARGET);
  state.comm = comm;
  comm.onMsg = msg => {
    const model = readModelMessage(msg);
    if (!model || state.disposed || state.comm !== comm) {
      return;
    }
    state.model = model;
    // A background tab restored by the workspace has no toolbar items until it
    // is first revealed, so the item may still be missing at this point.
    bindKernelName(panel, state);
  };
  comm.onClose = () => {
    if (state.comm === comm) {
      state.comm = null;
    }
  };
  comm.open({});
}

/**
 * The input area a cell shows in place of its editor.
 *
 * Everything that is not a formula carries MathJax's ignore class — the same
 * guard the kernel already puts on agent prose. A do! prompt in particular
 * may contain stray `$` or backslashes that the kernel did *not* read as
 * mathematics, and those must stay literal rather than being typeset by a
 * second, dumber reader.
 */
function previewWidget(segments: PreviewSegment[]): Widget {
  const node = document.createElement('div');
  node.className = RENDERED_INPUT_CLASS;
  for (const segment of segments) {
    if (segment.kind === 'break') {
      node.appendChild(document.createElement('br'));
      continue;
    }
    const span = document.createElement('span');
    if (segment.kind === 'math') {
      span.className = `${RENDERED_INPUT_CLASS}-math`;
      span.textContent = displayMath(segment.latex);
    } else {
      span.className =
        `${RENDERED_INPUT_CLASS}-${segment.kind} tex2jax_ignore`;
      span.textContent = segment.text;
    }
    node.appendChild(span);
  }
  return new Widget({ node });
}

function renderState(
  panel: NotebookPanel,
  state: IPanelState,
  cell: Cell
): boolean {
  return shouldRender({
    enabled: state.renderEnabled,
    isToyMath: isToyMath(panel),
    isCodeCell: cell.model.type === 'code',
    isActive: panel.content.activeCell === cell,
    editing: panel.content.mode === 'edit'
  });
}

function unrenderCell(state: IPanelState, cell: Cell): void {
  const current = state.rendered.get(cell);
  if (!current) {
    return;
  }
  state.rendered.delete(cell);
  if (!cell.isDisposed) {
    cell.removeClass(RENDERED_CLASS);
    cell.inputArea?.showEditor();
  }
  current.widget.dispose();
}

function renderCell(
  panel: NotebookPanel,
  state: IPanelState,
  cell: Cell,
  source: string,
  key: string,
  segments: PreviewSegment[]
): void {
  const input = cell.inputArea;
  if (!input) {
    return;                       // a cell without its input area yet
  }
  const widget = previewWidget(segments);
  // A single click, not markdown's double-click: this widget stands where a
  // code cell's editor stands, and clicking into a code cell to type is what
  // the notebook has taught. The click that reveals the source is also the
  // one that places the cursor.
  widget.node.addEventListener('click', () => {
    const index = panel.content.widgets.indexOf(cell);
    if (index >= 0) {
      panel.content.activeCellIndex = index;
    }
    panel.content.mode = 'edit';
  });
  input.renderInput(widget);
  cell.addClass(RENDERED_CLASS);
  state.rendered.set(cell, { source, key, widget });
  void panel.content.rendermime.latexTypesetter?.typeset(widget.node);
}

/**
 * Bring one cell to the view its state calls for.
 *
 * `refresh` re-asks about a source holding a `[[n]]` backreference: what it
 * renders as follows the notebook's history rather than the cell's own text,
 * so running a cell can change what another one shows. The rendered widget is
 * replaced only when the answer actually changed, so re-asking costs no
 * flicker — but typing is not a reason to ask, which is why this is a flag
 * and not the default.
 */
function syncCell(
  panel: NotebookPanel,
  state: IPanelState,
  cell: Cell,
  refresh = false
): void {
  if (state.disposed || cell.isDisposed) {
    return;
  }
  if (!renderState(panel, state, cell)) {
    unrenderCell(state, cell);
    return;
  }
  const source = cell.model.sharedModel.getSource();
  const current = state.rendered.get(cell);
  const stale = refresh && source.includes('[[');
  if (current && current.source === source && !stale) {
    return;
  }
  const previews = state.previews;
  if (!previews) {
    return;
  }
  void previews.request(source).then(segments => {
    if (state.disposed || cell.isDisposed || state.previews !== previews) {
      return;
    }
    if (cell.model.sharedModel.getSource() !== source) {
      return;
    }
    if (!renderState(panel, state, cell)) {
      return;
    }
    const key = segments ? JSON.stringify(segments) : '';
    const shown = state.rendered.get(cell);
    if (shown && shown.key === key) {
      shown.source = source;
      return;
    }
    unrenderCell(state, cell);
    if (segments) {
      renderCell(panel, state, cell, source, key, segments);
    }
  });
}

function syncCells(
  panel: NotebookPanel,
  state: IPanelState,
  refresh = false
): void {
  for (const cell of panel.content.widgets) {
    syncCell(panel, state, cell, refresh);
  }
}

function unrenderAll(panel: NotebookPanel, state: IPanelState): void {
  for (const cell of panel.content.widgets) {
    unrenderCell(state, cell);
  }
}

/**
 * (Re)connect the comm that answers what a cell renders as.
 *
 * The kernel owns the parser, so it is the only thing that can say what a
 * cell means. Without it — no kernel, another kernel, a restart in flight —
 * every cell falls back to showing its source.
 */
function connectRender(panel: NotebookPanel, state: IPanelState): void {
  closeCommSafely(state.renderComm);
  state.renderComm = null;
  state.previews?.reset();
  state.previews = null;

  const kernel = panel.sessionContext.session?.kernel;
  if (!kernel || kernel.name !== TOYMATH_KERNEL) {
    unrenderAll(panel, state);
    return;
  }
  const comm = kernel.createComm(RENDER_COMM_TARGET);
  const previews = new PreviewRequests({
    send: payload => comm.send(payload as unknown as Record<string, string>)
  });
  comm.onMsg = msg => {
    if (state.previews === previews) {
      previews.resolve(msg.content.data as IPreviewReply);
    }
  };
  comm.onClose = () => {
    if (state.renderComm === comm) {
      state.renderComm = null;
      previews.reset();
    }
  };
  comm.open({});
  state.renderComm = comm;
  state.previews = previews;
  syncCells(panel, state);
}

function attachPanel(
  app: JupyterFrontEnd,
  notebooks: INotebookTracker,
  panel: NotebookPanel
): IDisposable {
  const state: IPanelState = {
    comm: null,
    model: null,
    activeCell: null,
    completionTimer: null,
    disposed: false,
    kernelNameItem: null,
    itemObserver: null,
    toolbarObserver: null,
    renderComm: null,
    previews: null,
    renderEnabled: true,
    rendered: new WeakMap<Cell, IRenderedCell>()
  };
  panelStates.set(panel, state);
  const onKernelChanged = (): void => {
    connectPanel(panel, state);
    connectRender(panel, state);
  };
  const onCellContentChanged = (): void => {
    if (state.completionTimer !== null) {
      window.clearTimeout(state.completionTimer);
    }
    state.completionTimer = window.setTimeout(() => {
      state.completionTimer = null;
      const cell = state.activeCell;
      if (
        state.disposed ||
        !cell ||
        notebooks.currentWidget !== panel ||
        !isToyMath(panel) ||
        !cell.editor
      ) {
        return;
      }
      const code = cell.model.sharedModel.getSource();
      const cursor = cell.editor.getOffsetAt(cell.editor.getCursorPosition());
      if (shouldInvokeCompletion(code, cursor)) {
        void app.commands.execute('completer:invoke-notebook');
      }
    }, 0);
  };
  const onActiveCellChanged = (
    _notebook: NotebookPanel['content'],
    cell: Cell | null
  ): void => {
    state.activeCell?.model.contentChanged.disconnect(onCellContentChanged);
    state.activeCell = cell;
    state.activeCell?.model.contentChanged.connect(onCellContentChanged);
    // The cell just left renders again; the one arriving keeps its editor
    // only while the notebook is in edit mode. Moving on from a cell is also
    // where a just-run cell lands, so this is the moment a `[[n]]` elsewhere
    // can have gained its meaning.
    syncCells(panel, state, true);
  };
  // `setMode` emits this before it hands focus to the editor, so the editor a
  // cell is about to be edited in is back on screen by the time focus lands.
  const onNotebookState = (
    _notebook: NotebookPanel['content'],
    args: { name: string }
  ): void => {
    if (args.name === 'mode') {
      syncCells(panel, state);
    }
  };
  // Cells added, removed, or edited from outside the active editor (paste,
  // find-and-replace, a collaborator).
  const onModelContentChanged = (): void => syncCells(panel, state);
  // A windowed notebook detaches the cells it scrolls past; a formula that
  // was typeset while detached is measured again when it comes back.
  const onViewportChanged = (cell: Cell, inViewport: boolean): void => {
    if (!inViewport || state.disposed) {
      return;
    }
    const current = state.rendered.get(cell);
    if (current) {
      void panel.content.rendermime.latexTypesetter?.typeset(
        current.widget.node
      );
    } else {
      syncCell(panel, state, cell);
    }
  };
  const bound = new WeakSet<Cell>();
  const onCellsChanged = (): void => {
    for (const cell of panel.content.widgets) {
      if (!bound.has(cell)) {
        bound.add(cell);
        cell.inViewportChanged.connect(onViewportChanged);
      }
    }
  };

  panel.sessionContext.kernelChanged.connect(onKernelChanged);
  panel.content.activeCellChanged.connect(onActiveCellChanged);
  panel.content.stateChanged.connect(onNotebookState);
  panel.content.modelContentChanged.connect(onModelContentChanged);
  panel.content.modelContentChanged.connect(onCellsChanged);
  onActiveCellChanged(panel.content, panel.content.activeCell);
  onCellsChanged();

  // The toolbar of a workspace-restored panel is populated only when the tab is
  // first revealed, and its items move in and out of the overflow popup as the
  // panel is shown, hidden, or resized. Follow the kernel-name item across all
  // of that instead of painting the title once.
  const toolbarObserver = new MutationObserver(records => {
    const added = records.flatMap(record =>
      Array.from(record.addedNodes).filter(
        (node): node is Element => node.nodeType === Node.ELEMENT_NODE
      )
    );
    bindKernelName(
      panel,
      state,
      findAddedKernelNameItem(added) as HTMLElement | null
    );
  });
  toolbarObserver.observe(panel.toolbar.node, { childList: true });
  state.toolbarObserver = toolbarObserver;
  bindKernelName(panel, state);

  void panel.sessionContext.ready.then(() => {
    connectPanel(panel, state);
    connectRender(panel, state);
  });

  return {
    get isDisposed(): boolean {
      return state.disposed;
    },
    dispose: (): void => {
      if (state.disposed) {
        return;
      }
      // Give every cell its editor back before the listeners go: a disposed
      // extension must not leave a notebook that cannot be typed into.
      unrenderAll(panel, state);
      state.disposed = true;
      state.toolbarObserver?.disconnect();
      state.toolbarObserver = null;
      state.itemObserver?.disconnect();
      state.itemObserver = null;
      state.kernelNameItem = null;
      panel.sessionContext.kernelChanged.disconnect(onKernelChanged);
      panel.content.activeCellChanged.disconnect(onActiveCellChanged);
      panel.content.stateChanged.disconnect(onNotebookState);
      panel.content.modelContentChanged.disconnect(onModelContentChanged);
      panel.content.modelContentChanged.disconnect(onCellsChanged);
      for (const cell of panel.content.widgets) {
        cell.inViewportChanged.disconnect(onViewportChanged);
      }
      state.activeCell?.model.contentChanged.disconnect(onCellContentChanged);
      state.activeCell = null;
      if (state.completionTimer !== null) {
        window.clearTimeout(state.completionTimer);
        state.completionTimer = null;
      }
      closeCommSafely(state.comm);
      state.comm = null;
      closeCommSafely(state.renderComm);
      state.renderComm = null;
      state.previews?.reset();
      state.previews = null;
    }
  };
}

const plugin: JupyterFrontEndPlugin<void> = {
  id: '@toymath/model-ui:plugin',
  description:
    'Show the notebook-local ToyMath agent model, and ToyMath cell input ' +
    'as rendered mathematics.',
  autoStart: true,
  requires: [INotebookTracker],
  optional: [ICommandPalette],
  activate: (
    app: JupyterFrontEnd,
    notebooks: INotebookTracker,
    palette: ICommandPalette | null
  ): void => {
    const attached = new WeakSet<NotebookPanel>();
    const attach = (panel: NotebookPanel): void => {
      if (attached.has(panel)) {
        return;
      }
      attached.add(panel);
      const attachment = attachPanel(app, notebooks, panel);
      panel.disposed.connect(() => attachment.dispose());
    };
    notebooks.forEach(attach);
    notebooks.widgetAdded.connect((_sender, panel) => {
      attach(panel);
    });

    // Rendering changes how a cell is edited, so it stays switchable — and
    // notebook-local, like the model and backend it sits beside.
    const current = (): [NotebookPanel, IPanelState] | null => {
      const panel = notebooks.currentWidget;
      const state = panel ? panelStates.get(panel) : undefined;
      return panel && state && isToyMath(panel) ? [panel, state] : null;
    };
    app.commands.addCommand(TOGGLE_COMMAND, {
      label: 'Render ToyMath Cell Input',
      isToggleable: true,
      isToggled: () => current()?.[1].renderEnabled ?? false,
      isEnabled: () => current() !== null,
      execute: () => {
        const found = current();
        if (!found) {
          return;
        }
        const [panel, state] = found;
        state.renderEnabled = !state.renderEnabled;
        syncCells(panel, state);
      }
    });
    palette?.addItem({ command: TOGGLE_COMMAND, category: 'ToyMath' });
  }
};

export default plugin;
