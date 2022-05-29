from notation import Notation
from engine import get_mathshell


class SetEcho(object):
    arity = 0

    def __init__(self, flag):
        self.flag = flag

    def exec(self, processor, sym, f):
        get_mathshell().set_echo(echo_mode=self.flag)
        return Notation.NONE


def create_actions():
    return {'echo-on': SetEcho(True), 'echo-off': SetEcho(False)}
