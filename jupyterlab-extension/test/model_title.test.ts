import assert from 'node:assert/strict';
import test from 'node:test';

import {
  applyModelTitle,
  findAddedKernelNameItem
} from '../src/model_title.ts';
import type { IElementNode, IItemNode } from '../src/model_title.ts';

/** The kernel-name button as JupyterLab's React component renders it. */
class FakeButton {
  title = 'Switch kernel';
  dataset: { toymathModel?: string } = {};
  label = { textContent: 'Toy Math' };
  attributes: Record<string, string> = { 'aria-label': 'Toy Math' };

  querySelector(selector: string): { textContent: string | null } | null {
    return selector === '.jp-ToolbarButtonComponent-label' ? this.label : null;
  }

  setAttribute(name: string, value: string): void {
    this.attributes[name] = value;
  }
}

class FakeItem {
  button: FakeButton | null;

  constructor(button: FakeButton | null) {
    this.button = button;
  }

  querySelector(selector: string): FakeButton | null {
    return selector === '.jp-Toolbar-kernelName' ? this.button : null;
  }
}

function item(button: FakeButton | null): IItemNode {
  return new FakeItem(button) as unknown as IItemNode;
}

test('the model is painted into the kernel-name button', () => {
  const button = new FakeButton();

  const painted = applyModelTitle(item(button), 'x-ai/grok-5', 'Toy Math');

  assert.equal(painted, true);
  assert.equal(button.label.textContent, 'Toy Math · x-ai/grok-5');
  assert.equal(button.dataset.toymathModel, 'true');
  assert.equal(
    button.attributes['aria-label'],
    'Toy Math agent model x-ai/grok-5'
  );
});

test('repainting an unchanged title writes nothing', () => {
  const button = new FakeButton();
  applyModelTitle(item(button), 'x-ai/grok-5', 'Toy Math');

  assert.equal(applyModelTitle(item(button), 'x-ai/grok-5', 'Toy Math'), false);
});

test('a remounted button is repainted', () => {
  const button = new FakeButton();
  applyModelTitle(item(button), 'x-ai/grok-5', 'Toy Math');
  // React remounting the toolbar item restores the plain kernel name.
  const remounted = new FakeButton();

  assert.equal(applyModelTitle(item(remounted), 'x-ai/grok-5', 'Toy Math'), true);
  assert.equal(remounted.label.textContent, 'Toy Math · x-ai/grok-5');
});

test('a button JupyterLab has not rendered yet is left alone', () => {
  assert.equal(applyModelTitle(item(null), 'x-ai/grok-5', 'Toy Math'), false);
  assert.equal(applyModelTitle(null, 'x-ai/grok-5', 'Toy Math'), false);
});

test('dropping the model restores the kernel display name', () => {
  const button = new FakeButton();
  applyModelTitle(item(button), 'x-ai/grok-5', 'Toy Math');

  const painted = applyModelTitle(item(button), null, 'Python 3 (ipykernel)');

  assert.equal(painted, true);
  assert.equal(button.label.textContent, 'Python 3 (ipykernel)');
  assert.equal(button.dataset.toymathModel, undefined);
  assert.equal(button.title, 'Switch kernel');
});

class FakeNode implements IElementNode {
  classes: string[];
  children: FakeNode[];

  constructor(classes: string[], children: FakeNode[] = []) {
    this.classes = classes;
    this.children = children;
  }

  classList = {
    contains: (token: string): boolean => this.classes.includes(token)
  };

  querySelector(selector: string): IElementNode | null {
    const wanted = selector.replace('.', '');
    return (
      this.children.find(child => child.classList.contains(wanted)) ?? null
    );
  }
}

test('the kernel-name item is found among added toolbar nodes', () => {
  const kernelName = new FakeNode(['jp-KernelName', 'jp-Toolbar-item']);
  const added = [new FakeNode(['jp-Toolbar-item']), kernelName];

  assert.equal(findAddedKernelNameItem(added), kernelName);
});

test('the kernel-name item is found when added inside a subtree', () => {
  const kernelName = new FakeNode(['jp-KernelName']);
  const added = [new FakeNode(['jp-Toolbar'], [kernelName])];

  assert.equal(findAddedKernelNameItem(added), kernelName);
});

test('unrelated toolbar additions find nothing', () => {
  const added = [new FakeNode(['jp-Toolbar-item'], [new FakeNode(['jp-icon'])])];

  assert.equal(findAddedKernelNameItem(added), null);
});
