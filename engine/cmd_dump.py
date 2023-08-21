from IPython.display import HTML
from engine import display
from engine import get_mathshell


def dump(outsym, notation):
    res = '<table>'
    for index, sym in enumerate(iter(notation.rel)):
        if sym == outsym:
            res += f'<tr><td><b>{sym.__repr__()}</b></td><td>{notation.rel[sym]}</td></tr>'
        else:
            res += f'<tr><td>{sym.__repr__()}</td><td>{notation.rel[sym]}</td></tr>'
    res += "</table>"
    return res


class DumpNotation(object):
    arity = 1

    def exec(self, processor, sym, f):
        if f.args[0] is not None:
            raise AttributeError(f'The dump command does not define any attributes')
        get_mathshell().set_show_quotes(True)
        sym = processor.enter_subformula(f.args[1][0])
        display(HTML(dump(sym, processor.output_notation)))
        return sym


def create_actions():
    return {'dump': DumpNotation()}
