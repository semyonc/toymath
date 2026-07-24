import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { Cell } from '@jupyterlab/cells';
import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { Kernel, KernelMessage } from '@jupyterlab/services';
import { IDisposable } from '@lumino/disposable';

import { closeCommSafely } from './comm_lifecycle';

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
}

function isToyMath(panel: NotebookPanel): boolean {
  return panel.sessionContext.session?.kernel?.name === TOYMATH_KERNEL;
}

function setKernelTitle(panel: NotebookPanel, model: string | null): void {
  // KernelNameComponent has no dynamic-label API beyond kernelChanged. Keep
  // its existing button/click handler and change only this notebook panel's
  // rendered label after React has handled any kernel-change signal.
  requestAnimationFrame(() => {
    if (panel.isDisposed) {
      return;
    }
    const button = panel.toolbar.node.querySelector<HTMLElement>(
      '.jp-Toolbar-kernelName'
    );
    if (!button) {
      return;
    }
    const label = button.querySelector<HTMLElement>(
      '.jp-ToolbarButtonComponent-label'
    );
    if (!label) {
      return;
    }
    if (model && isToyMath(panel)) {
      label.textContent = `Toy Math · ${model}`;
      button.dataset.toymathModel = 'true';
      button.title = `ToyMath agent model: ${model}. Click to switch kernel.`;
      button.setAttribute('aria-label', `Toy Math agent model ${model}`);
    } else {
      label.textContent = panel.sessionContext.kernelDisplayName;
      delete button.dataset.toymathModel;
      button.title = 'Switch kernel';
      button.setAttribute('aria-label', panel.sessionContext.kernelDisplayName);
    }
  });
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
    setKernelTitle(panel, null);
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
    setKernelTitle(panel, model.model);
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
    disposed: false
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
