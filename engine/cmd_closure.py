from replicator import Replicator


class Closure(object):

    arity = 1

    @staticmethod
    def exec(processor, sym, f):
        if f.args[0] is not None:
            raise AttributeError(f'The closure command does not define any attributes')
        repl = Replicator(processor.notation, processor.output_notation)
        return repl.enter_command(sym, f)


def create_actions():
    return {'closure': Closure()}
