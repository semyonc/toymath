/**
 * When to pop the completer open by itself inside a ToyMath cell.
 *
 * JupyterLab only invokes the completer on an explicit Tab. The ToyMath
 * configuration commands all take a small closed vocabulary the kernel can
 * enumerate — backends, login options, model ids, provider names — so the
 * list is worth offering unprompted the moment the cell can accept one.
 *
 * The rule is deliberately narrow: fire only where the cursor sits exactly at
 * an argument position and the user has typed nothing of it yet. Firing while
 * a token is half-typed would reopen the popup on every keystroke, and firing
 * on a line that merely contains `model!` somewhere would interrupt ordinary
 * math. Kept free of JupyterLab imports so it is testable without a browser.
 */

/**
 * Cell prefixes whose whole argument is a single word from a fixed list, so
 * the completer opens as soon as the separating space is typed.
 *
 * Mirrors the kernel: `agent_config.complete_backend_command` and
 * `complete_login_command`.
 */
const WORD_COMMANDS = ['backend', 'login'];

/**
 * `model!` is not in that list because its argument is comma-separated: the
 * model id first, then provider names. It therefore has two trigger points —
 * after the command, and after each comma.
 */
const MODEL_ID = /^[ \t]*model![ \t]+$/;
const MODEL_PROVIDER = /^[ \t]*model![ \t]+[^,\n]+(?:,[^,\n]+)*,[ \t]*$/;

const WORD_COMMAND = new RegExp(
  `^[ \\t]*(?:${WORD_COMMANDS.join('|')})![ \\t]+$`
);

/**
 * Whether the completer should be invoked for `code` with the cursor at
 * `cursor`, an offset into `code`.
 */
export function shouldInvokeCompletion(code: string, cursor: number): boolean {
  const lineStart = code.lastIndexOf('\n', cursor - 1) + 1;
  const line = code.slice(lineStart, cursor);
  return (
    WORD_COMMAND.test(line) || MODEL_ID.test(line) || MODEL_PROVIDER.test(line)
  );
}
