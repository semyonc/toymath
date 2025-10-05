from replicator import Replicator
from replacer import Replacer
from notation import NOTATION, Symbol
from typing import Dict, Any
from comparer import isVariable

class PrologReplicator(Replicator):
    def __init__(self, notation: NOTATION, output_notation: NOTATION):
        super().__init__(notation, output_notation)

    def mapsym(self, sym):
        return Symbol()
    
class SymbolReplacer(Replacer):
    """SymbolReplacer"""

    def __init__(self, notation: NOTATION, output_notation: NOTATION,
                 outer_notation: NOTATION, mapping: Dict[str, Any]):
        super().__init__(notation, output_notation)
        self.outer_notation = outer_notation
        self.mapping = mapping

    def enter_symbol(self, sym):
        if isVariable(sym) and sym.name in self.mapping:
            replacer = self.mapping[sym.name]
            if isinstance(replacer, Symbol) and self.output_notation.get(replacer) is None:
                replacer = Replicator(self.outer_notation, self.output_notation)(replacer)
            return self.subst(sym, replacer, self.context())
        return super().enter_symbol(sym)
