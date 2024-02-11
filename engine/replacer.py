from replicator import Replicator
from notation import Notation, Func


class Replacer(Replicator):
    """Replacer"""

    crosstab = (
        (Notation.P_LIST, Notation.S_LIST),
        (Notation.P_LIST, Notation.C_LIST),
        (Notation.S_LIST, Notation.C_LIST),
        (Notation.P_LIST, Notation.PLUS),
        (Notation.P_LIST, Notation.MINUS),
        (Notation.S_LIST, Notation.CMD),
        (Notation.PLUS, Notation.PLUS),
        (Notation.PLUS, Notation.MINUS),
        (Notation.MINUS, Notation.PLUS),
        (Notation.MINUS, Notation.MINUS),
        (Notation.PLUS, Notation.S_LIST),
        (Notation.MINUS, Notation.S_LIST),
    )

    def escape(self, sym, br, linked_sym):
        return self.output_notation.repf(
            self.mapsym(sym), Func(Notation.GROUP, (linked_sym,), br=br)
        )

    def subst(self, sym, new_sym, ctx):
        f = self.output_notation.get(new_sym)
        if f is not None:
            f_sym = f.sym
            if 'command' in f.sym.props:
                f_sym = Notation.CMD
            if ctx is not None:
                if ctx.name in Notation.oper:
                    return self.escape(sym, "{}", new_sym)
                if ctx == Notation.INDEX or (ctx, f_sym) in self.crosstab:
                    return self.escape(sym, "()", new_sym)
        return new_sym
