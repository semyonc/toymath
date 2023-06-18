from IPython.display import HTML
from engine import display
from notation import Notation


def dump(rules):
    res = '<table>'
    for rule in rules:
        res += f'<tr><td>${rule.__repr__()}$</td></tr>'
    res += "</table>"
    res += f'<p><i>{len(rules)} rules(s) in database</i></p>'
    return res


class PrintRules(object):
    arity = 0

    def exec(self, processor, sym, f):
        if f.args[0] is not None:
            raise AttributeError(f'The dump command does not define any attributes')
        display(HTML(dump(processor.prologModel.rules)))
        return Notation.NONE


def create_actions():
    return {'rules': PrintRules()}
