from LatexWriter import LaTexWriter
import sys

stdwrite = sys.stdout.write

def attach_debugger(breakpoint=False):
    import debugpy
    debugpy.listen(5678)  # noqa
    print("Waiting for debugger attach on port 5678...")
    debugpy.wait_for_client()
    print("Debugger attached")
    if breakpoint:
        debugpy.breakpoint()
    sys.stdout.flush()

def trace_notation(notation, sym, tag=None):
    writer = LaTexWriter(notation)
    str = writer(sym)
    print(tag, str, file=sys.stderr)