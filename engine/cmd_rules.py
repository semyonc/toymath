from IPython.display import HTML
from engine import display
from notation import Notation


def dump(model):
    res = '<table>'
    count = 0
    for sym in model.rules:
        if sym.name in model.callbacks:
            continue
        rules = model.rules[sym]
        for rule in reversed(rules):
            res += f'<tr><td>$${rule.__repr__()}$$</td></tr>'
            count += 1
    res += "</table>"
    res += f'<p><i>{count} rules(s) in database</i></p>'
    return res


class PrintRules(object):
    arity = 0

    def exec(self, processor, sym, f):
        if f.args[0] is not None:
            raise AttributeError(f'The dump command does not define any attributes')
        display(HTML(dump(processor.prologModel)))
        return Notation.NONE


def create_actions():
    return {'rules': PrintRules()}
