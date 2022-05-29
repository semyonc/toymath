from engine import get_mathshell


class Track(object):
    arity = 1

    @staticmethod
    def exec(processor, sym, f):
        if f.args[0] is not None:
            raise AttributeError(f'The lock command does not define any attributes')
        get_mathshell().set_trace(True)
        get_mathshell().set_echo(current_echo=True)
        return processor.enter_subformula(f.args[1][0])


def create_actions():
    return {'track': Track()}
