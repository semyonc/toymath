from notation import Notation
from engine import get_mathshell


class Clear(object):
    arity = 0

    def exec(self, processor, sym, f):
        if f.args[0] is not None:
            raise AttributeError(f'The dump command does not define any attributes')
        get_mathshell().clear()
        return Notation.NONE


def create_actions():
    return {'clear': Clear()}
