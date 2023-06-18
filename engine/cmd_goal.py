from IPython.display import HTML
from engine import display
from notation import Symbol, Notation
from prolog import Term
from LatexWriter import LaTexWriter
from engine import get_mathshell

def printenv(env, notation):
    writer = LaTexWriter(notation)
    res = '<table style="border:1px solid;">'
    for key in env:
        skey = key.replace('#', "\\#")
        res += f'<tr><td style="border:1px solid;">${skey}$</td><td style="border:1px solid;">${writer(env[key])}$</td></tr>'
    res += "</table>"
    return res


class ExecuteGoal(object):
    arity = 1

    def exec(self, processor, sym, f):
        if f.args[0] is not None:
            raise AttributeError(f'The dump command does not define any attributes')
        sym = processor.enter_subformula(f.args[1][0])
        term = Term(sym=sym, notation=processor.output_notation)
        flag = False
        for env, notation in processor.prologModel.search(term, trace=get_mathshell().trace):
            flag = True
            display(HTML(printenv(env, notation)))
        if flag:
            return processor.output_notation.setf(Symbol('\\textit'), (str(True),))
        return Notation.NONE


def create_actions():
    return {'goal': ExecuteGoal()}
