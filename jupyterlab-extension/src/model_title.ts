/**
 * Paint the notebook-local ToyMath model into JupyterLab's kernel-name button.
 *
 * The button belongs to a React component JupyterLab owns, so a single write
 * does not survive: the responsive toolbar re-parents that item into a
 * document-level overflow popup (and back) whenever a panel is shown, hidden,
 * or resized, and each move unmounts and remounts React, restoring the plain
 * kernel name. A workspace-restored background tab has no toolbar items at all
 * until it is first revealed. Callers therefore repaint on every mutation, and
 * everything here stays idempotent so repainting cannot loop against itself.
 *
 * The interfaces are the DOM subset used, kept structural so the painting is
 * testable without a browser.
 */

/** Toolbar item holding the kernel-name button (`Toolbar.createKernelNameItem`). */
export const KERNEL_NAME_ITEM_CLASS = 'jp-KernelName';
export const KERNEL_NAME_ITEM = `.${KERNEL_NAME_ITEM_CLASS}`;
const KERNEL_NAME_BUTTON = '.jp-Toolbar-kernelName';
const BUTTON_LABEL = '.jp-ToolbarButtonComponent-label';

export interface ILabelNode {
  textContent: string | null;
}

export interface IButtonNode {
  title: string;
  readonly dataset: { toymathModel?: string };
  querySelector(selector: string): ILabelNode | null;
  setAttribute(name: string, value: string): void;
}

export interface IItemNode {
  querySelector(selector: string): IButtonNode | null;
}

/** The backend whose model ids need no label: the historical default. */
const DEFAULT_BACKEND = 'openrouter';

/**
 * Write `model` into the kernel-name button of `item`, or restore
 * `kernelDisplayName` when there is no ToyMath model to show.
 *
 * `backend` names the agent backend the notebook runs on. It is shown in the
 * label only when it is not the default one - an OpenRouter model id already
 * implies its backend, while `codex` is an opt-in experiment worth seeing at
 * a glance. The tooltip always names both.
 *
 * Returns whether the button was rewritten: `false` means either React has not
 * (re)rendered the button yet or it already reads what it should.
 */
export function applyModelTitle(
  item: IItemNode | null,
  model: string | null,
  kernelDisplayName: string,
  backend: string | null = null
): boolean {
  const button = item?.querySelector(KERNEL_NAME_BUTTON);
  const label = button?.querySelector(BUTTON_LABEL);
  if (!button || !label) {
    return false;
  }
  const badge = backend && backend !== DEFAULT_BACKEND ? `${backend} · ` : '';
  const text = model ? `Toy Math · ${badge}${model}` : kernelDisplayName;
  const marked = button.dataset.toymathModel === 'true';
  if (label.textContent === text && marked === !!model) {
    return false;
  }
  label.textContent = text;
  if (model) {
    const routing = backend ? `${backend} · ${model}` : model;
    button.dataset.toymathModel = 'true';
    button.title = `ToyMath agent: ${routing}. Click to switch kernel.`;
    button.setAttribute('aria-label', `Toy Math agent model ${model}`);
  } else {
    delete button.dataset.toymathModel;
    button.title = 'Switch kernel';
    button.setAttribute('aria-label', kernelDisplayName);
  }
  return true;
}

export interface IElementNode {
  classList: { contains(token: string): boolean };
  querySelector(selector: string): IElementNode | null;
}

/**
 * The kernel-name item among freshly added toolbar nodes.
 *
 * The item can be created and moved into the overflow popup within a single
 * mutation batch, so re-querying the toolbar afterwards would miss it entirely.
 */
export function findAddedKernelNameItem(
  added: Iterable<IElementNode>
): IElementNode | null {
  for (const node of added) {
    if (node.classList.contains(KERNEL_NAME_ITEM_CLASS)) {
      return node;
    }
    const nested = node.querySelector(KERNEL_NAME_ITEM);
    if (nested) {
      return nested;
    }
  }
  return null;
}
