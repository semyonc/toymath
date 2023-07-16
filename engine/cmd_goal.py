from IPython.display import HTML
from engine import display
from notation import Symbol, Notation
from engine.processor import MathProcessor
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

    def exec(self, processor, _, f):
        if f.args[0] is not None:
            raise AttributeError(f'The dump command does not define any attributes')
        sym = processor.enter_subformula(f.args[1][0])
        goals = processor.prologModel.parse_goals(sym, processor.output_notation)
        if goals:
            flag = False
            for env, notation in processor.prologModel.search(goals, trace=get_mathshell().trace):
                flag = True
                display(HTML(printenv(env, notation)))
            if flag:
                return MathProcessor.create_true(processor.output_notation)
        return Notation.NONE


def create_actions():
    return {'goal': ExecuteGoal()}
