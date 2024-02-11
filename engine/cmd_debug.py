from engine import display
from helpers import attach_debugger
from notation import Notation

class Debug(object):
    arity = 0

    def exec(self, processor, sym, f):
        if f.args[0] is not None:
            raise AttributeError(f'The dump command does not define any attributes')
        attach_debugger()
        return Notation.NONE
    
def create_actions():
    return {'debug': Debug()}