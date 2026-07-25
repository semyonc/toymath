import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { Cell } from '@jupyterlab/cells';
import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { Kernel, KernelMessage } from '@jupyterlab/services';
import { IDisposable } from '@lumino/disposable';

import { closeCommSafely } from './comm_lifecycle';
import {
  applyModelTitle,
  findAddedKernelNameItem,
  IItemNode,
  KERNEL_NAME_ITEM
} from './model_title';

const COMM_TARGET = 'toymath.model';
const TOYMATH_KERNEL = 'toymath';

interface IModelState {
  model: string;
  providers: string[];
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
}

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
  const model = state.model && isToyMath(panel) ? state.model.model : null;
  applyModelTitle(
    // The DOM node satisfies the structural subset the painter needs.
    state.kernelNameItem as unknown as IItemNode,
    model,
    panel.sessionContext.kernelDisplayName
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

function shouldInvokeModelCompletion(code: string, cursor: number): boolean {
  const lineStart = code.lastIndexOf('\n', cursor - 1) + 1;
  const line = code.slice(lineStart, cursor);
  return (
    /^[ \t]*model![ \t]+$/.test(line) ||
    /^[ \t]*model![ \t]+[^,\n]+(?:,[^,\n]+)*,[ \t]*$/.test(line)
  );
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
    toolbarObserver: null
  };
  const onKernelChanged = (): void => connectPanel(panel, state);
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
      if (shouldInvokeModelCompletion(code, cursor)) {
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
  };

  panel.sessionContext.kernelChanged.connect(onKernelChanged);
  panel.content.activeCellChanged.connect(onActiveCellChanged);
  onActiveCellChanged(panel.content, panel.content.activeCell);

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

  void panel.sessionContext.ready.then(() => connectPanel(panel, state));

  return {
    get isDisposed(): boolean {
      return state.disposed;
    },
    dispose: (): void => {
      if (state.disposed) {
        return;
      }
      state.disposed = true;
      state.toolbarObserver?.disconnect();
      state.toolbarObserver = null;
      state.itemObserver?.disconnect();
      state.itemObserver = null;
      state.kernelNameItem = null;
      panel.sessionContext.kernelChanged.disconnect(onKernelChanged);
      panel.content.activeCellChanged.disconnect(onActiveCellChanged);
      state.activeCell?.model.contentChanged.disconnect(onCellContentChanged);
      state.activeCell = null;
      if (state.completionTimer !== null) {
        window.clearTimeout(state.completionTimer);
        state.completionTimer = null;
      }
      closeCommSafely(state.comm);
      state.comm = null;
    }
  };
}

const plugin: JupyterFrontEndPlugin<void> = {
  id: '@toymath/model-ui:plugin',
  description: 'Show the notebook-local ToyMath agent model in JupyterLab.',
  autoStart: true,
  requires: [INotebookTracker],
  activate: (app: JupyterFrontEnd, notebooks: INotebookTracker): void => {
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
  }
};

export default plugin;
