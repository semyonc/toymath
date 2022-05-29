from engine import get_mathshell, display
from notation import Symbol, Notation
from comparer import NotationComparer
from IPython.display import HTML
from LatexWriter import LaTexWriter


def boolean_result(procssor, value):
    return procssor.output_notation.setf(Symbol('\\textit'), (str(value),))


class RunMatch(object):
    arity = 2

    def add_argument(self, sym, notation, params):
        if not isinstance(sym, Symbol) or notation.get(sym) is not None:
            raise AttributeError(f'The match command should use only symbol arguments')
        params.append(sym.name)

    def exec(self, processor, sym, f):
        get_mathshell().set_echo(current_echo=True)
        params = []
        if f.args[0] is not None:
            writer = LaTexWriter(processor.output_notation)
            list_f = processor.notation.getf(f.args[0], Notation.C_LIST)
            if list_f is None:
                self.add_argument(f.args[0], processor.notation, params)
            else:
                for arg in list_f.args:
                    self.add_argument(arg, processor.notation, params)
        sym1 = processor.enter_subformula(f.args[1][0])
        sym2 = processor.enter_subformula(f.args[1][1])
        comparer = NotationComparer(sym2, processor.output_notation, params)
        subst = comparer.match(sym1, processor.output_notation)
        if subst is None:
            return boolean_result(processor, 'false')
        else:
            for p in params:
                output = f'{p}\\,=\\space '
                outsym = subst[p]
                output += writer(outsym)
                display(HTML('$' + output + '$'))
            return boolean_result(processor, 'true')


def create_actions():
    return {'match': RunMatch()}
