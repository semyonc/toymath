import assert from 'node:assert/strict';
import test from 'node:test';

import {
  displayMath,
  PreviewRequests,
  readSegments,
  shouldRender
} from '../src/rendered_input.ts';
import type { IPreviewTransport } from '../src/rendered_input.ts';

/** A transport that records what was sent and answers on demand. */
class FakeTransport implements IPreviewTransport {
  sent: { id: string; code: string }[] = [];
  failing = false;

  send(payload: { id: string; code: string }): void {
    if (this.failing) {
      throw new Error('kernel is gone');
    }
    this.sent.push(payload);
  }
}

const MATH = { kind: 'math', latex: 'x^{2}' };
const LABEL = { kind: 'command', text: 'int!' };

test('a reply is read into segments', () => {
  const segments = readSegments({ id: '1', segments: [LABEL, MATH] });
  assert.deepEqual(segments, [
    { kind: 'command', text: 'int!' },
    { kind: 'math', latex: 'x^{2}' }
  ]);
});

test('a do! prompt arrives as prose around its formulas', () => {
  const segments = readSegments({
    id: '1',
    segments: [
      { kind: 'command', text: 'do!' },
      { kind: 'text', text: 'differentiate ' },
      { kind: 'math', latex: 'x^{3}-3x' },
      { kind: 'text', text: ' and plot it' }
    ]
  });
  assert.equal(segments?.length, 4);
  assert.deepEqual(segments?.[1], { kind: 'text', text: 'differentiate ' });
});

test('a cell that renders as nothing keeps its source', () => {
  assert.equal(readSegments({ id: '1', segments: null }), null);
  assert.equal(readSegments({ id: '1' }), null);
});

test('a reply with only a label renders nothing', () => {
  // an input area showing only `int!` would hide the cell, not explain it
  assert.equal(readSegments({ id: '1', segments: [LABEL] }), null);
});

test('a reference with no result yet is enough to render', () => {
  // `simplify! [[1]]` before cell 1 has run: still legible as the command it
  // is, rather than dropping back to raw source beside its rendered siblings
  assert.deepEqual(
    readSegments({
      id: '1',
      segments: [LABEL, { kind: 'ref', text: '[[1]]' }]
    }),
    [
      { kind: 'command', text: 'int!' },
      { kind: 'ref', text: '[[1]]' }
    ]
  );
});

test('a malformed segment rejects the whole reply', () => {
  assert.equal(readSegments({ id: '1', segments: [MATH, { kind: 'x' }] }), null);
  assert.equal(
    readSegments({ id: '1', segments: [MATH, { kind: 'math' }] }),
    null
  );
});

test('a formula is typeset in display style', () => {
  assert.equal(displayMath('\\int x'), '$\\displaystyle \\int x$');
});

const BASE = {
  enabled: true,
  isToyMath: true,
  isCodeCell: true,
  isActive: false,
  editing: false
};

test('rendering is the resting state of a ToyMath code cell', () => {
  assert.equal(shouldRender(BASE), true);
  assert.equal(shouldRender({ ...BASE, isActive: true }), true);
  // another cell being edited does not un-render this one
  assert.equal(shouldRender({ ...BASE, editing: true }), true);
});

test('the cell being edited shows its editor', () => {
  assert.equal(shouldRender({ ...BASE, isActive: true, editing: true }), false);
});

test('rendering needs the toggle, ToyMath, and a code cell', () => {
  assert.equal(shouldRender({ ...BASE, enabled: false }), false);
  assert.equal(shouldRender({ ...BASE, isToyMath: false }), false);
  assert.equal(shouldRender({ ...BASE, isCodeCell: false }), false);
});

test('replies are matched by request id, not by arrival order', async () => {
  const transport = new FakeTransport();
  const requests = new PreviewRequests(transport);
  const first = requests.request('x^2');
  const second = requests.request('y^3');
  const [idA, idB] = transport.sent.map(payload => payload.id);

  requests.resolve({ id: idB, segments: [{ kind: 'math', latex: 'y^{3}' }] });
  requests.resolve({ id: idA, segments: [MATH] });

  assert.deepEqual(await first, [{ kind: 'math', latex: 'x^{2}' }]);
  assert.deepEqual(await second, [{ kind: 'math', latex: 'y^{3}' }]);
});

test('an unknown id is ignored', async () => {
  const transport = new FakeTransport();
  const requests = new PreviewRequests(transport);
  const pending = requests.request('x^2');
  requests.resolve({ id: 'not-a-request', segments: [MATH] });
  requests.resolve({ segments: [MATH] });
  requests.resolve({ id: transport.sent[0].id, segments: [MATH] });
  assert.deepEqual(await pending, [{ kind: 'math', latex: 'x^{2}' }]);
});

test('an answered source is not asked about twice', async () => {
  const transport = new FakeTransport();
  const requests = new PreviewRequests(transport);
  const pending = requests.request('x^2');
  requests.resolve({ id: transport.sent[0].id, segments: [MATH] });
  await pending;

  assert.deepEqual(await requests.request('x^2'), [
    { kind: 'math', latex: 'x^{2}' }
  ]);
  assert.equal(transport.sent.length, 1);
});

test('a backreference is asked about every time', async () => {
  // `[[1]]` renders as the formula it stands for, and that follows the
  // notebook's history rather than the cell's own text
  const transport = new FakeTransport();
  const requests = new PreviewRequests(transport);
  const pending = requests.request('int! [[1]]');
  requests.resolve({ id: transport.sent[0].id, segments: [LABEL, MATH] });
  await pending;

  void requests.request('int! [[1]]');
  assert.equal(transport.sent.length, 2);
});

test('a kernel change settles what is in flight', async () => {
  const transport = new FakeTransport();
  const requests = new PreviewRequests(transport);
  const pending = requests.request('x^2');
  requests.reset();
  assert.equal(await pending, null);

  // the cache went with it: the new kernel has its own history
  void requests.request('x^2');
  assert.equal(transport.sent.length, 2);
});

test('a dead kernel renders nothing rather than hanging', async () => {
  const transport = new FakeTransport();
  transport.failing = true;
  const requests = new PreviewRequests(transport);
  assert.equal(await requests.request('x^2'), null);
});

test('the cache does not grow without bound', async () => {
  const transport = new FakeTransport();
  const requests = new PreviewRequests(transport, 2);
  for (const code of ['a', 'b', 'c']) {
    const pending = requests.request(code);
    requests.resolve({
      id: transport.sent[transport.sent.length - 1].id,
      segments: [MATH]
    });
    await pending;
  }
  void requests.request('a'); // evicted, so it is asked about again
  assert.equal(transport.sent.length, 4);
  void requests.request('c'); // still known
  assert.equal(transport.sent.length, 4);
});
