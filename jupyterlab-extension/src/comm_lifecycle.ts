/**
 * Close a frontend Comm when it is still attached to a live kernel.
 *
 * JupyterLab disposes every Comm while restarting a kernel. The notebook's
 * later `kernelChanged` signal can therefore find an already-disposed Comm;
 * calling `close()` on it throws instead of behaving like an idempotent close.
 */
export interface IClosableComm {
  readonly isDisposed: boolean;
  close(): unknown;
  dispose(): void;
}

export function closeCommSafely(comm: IClosableComm | null): void {
  if (!comm || comm.isDisposed) {
    return;
  }
  try {
    comm.close();
  } catch {
    // A kernel can disappear between the isDisposed check and close().
    comm.dispose();
  }
}
