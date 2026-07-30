import assert from 'node:assert/strict';
import test from 'node:test';

import { shouldInvokeCompletion } from '../src/completion.ts';

/** Invoke with the cursor at the end of `code`, as typing leaves it. */
function atEnd(code: string): boolean {
  return shouldInvokeCompletion(code, code.length);
}

test('the popup opens as soon as a word command can take an argument', () => {
  assert.equal(atEnd('backend! '), true);
  assert.equal(atEnd('login! '), true);
  assert.equal(atEnd('model! '), true);
});

test('the popup stays shut until the separating space is typed', () => {
  assert.equal(atEnd('backend!'), false);
  assert.equal(atEnd('login!'), false);
  assert.equal(atEnd('model!'), false);
});

test('a half-typed argument does not reopen the popup', () => {
  // otherwise it would fire again on every keystroke
  assert.equal(atEnd('backend! cod'), false);
  assert.equal(atEnd('login! dev'), false);
  assert.equal(atEnd('model! google/gem'), false);
});

test('model! also completes each provider after a comma', () => {
  assert.equal(atEnd('model! z-ai/glm-5.2,'), true);
  assert.equal(atEnd('model! z-ai/glm-5.2, '), true);
  assert.equal(atEnd('model! z-ai/glm-5.2, Cerebras,'), true);
  assert.equal(atEnd('model! z-ai/glm-5.2, Cere'), false);
});

test('the other commands take one word, so no comma trigger', () => {
  assert.equal(atEnd('backend! codex,'), false);
  assert.equal(atEnd('login! status,'), false);
});

test('leading whitespace is allowed, other leading text is not', () => {
  assert.equal(atEnd('   backend! '), true);
  assert.equal(atEnd('\t login! '), true);
  // ordinary math mentioning a command must never pop the completer
  assert.equal(atEnd('x + backend! '), false);
  assert.equal(atEnd('$login! '), false);
});

test('only the line holding the cursor is considered', () => {
  assert.equal(atEnd('x^2\nbackend! '), true);
  assert.equal(atEnd('backend! \nx^2'), false);
});

test('the cursor position, not the end of the cell, decides', () => {
  const code = 'login! status';
  assert.equal(shouldInvokeCompletion(code, 'login! '.length), true);
  assert.equal(shouldInvokeCompletion(code, code.length), false);
});

test('an empty cell asks for nothing', () => {
  assert.equal(shouldInvokeCompletion('', 0), false);
});
