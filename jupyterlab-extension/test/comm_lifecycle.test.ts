import assert from 'node:assert/strict';
import test from 'node:test';

import { closeCommSafely } from '../src/comm_lifecycle.ts';

class FakeComm {
  isDisposed = false;
  closeCalls = 0;
  disposeCalls = 0;
  throwOnClose = false;

  close(): void {
    this.closeCalls += 1;
    if (this.throwOnClose) {
      throw new Error('Cannot close');
    }
    this.isDisposed = true;
  }

  dispose(): void {
    this.disposeCalls += 1;
    this.isDisposed = true;
  }
}

test('an already-disposed restart Comm is an idempotent no-op', () => {
  const comm = new FakeComm();
  comm.isDisposed = true;

  closeCommSafely(comm);

  assert.equal(comm.closeCalls, 0);
  assert.equal(comm.disposeCalls, 0);
});

test('a live Comm is closed normally', () => {
  const comm = new FakeComm();

  closeCommSafely(comm);

  assert.equal(comm.closeCalls, 1);
  assert.equal(comm.disposeCalls, 0);
});

test('a Comm disposed concurrently is locally cleaned up', () => {
  const comm = new FakeComm();
  comm.throwOnClose = true;

  closeCommSafely(comm);

  assert.equal(comm.closeCalls, 1);
  assert.equal(comm.disposeCalls, 1);
});
