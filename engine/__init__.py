from os.path import dirname, realpath
import sys
import IPython.display

sys.path.append(dirname(realpath(__file__)))

math_shell = None

handler = IPython.display.display


def get_mathshell():
    return math_shell


def setShell(shell):
    global math_shell
    math_shell = shell


def display(*objs, **kwargs):
    handler(*objs, **kwargs)


def setHandler(new_handler):
    global handler
    handler = new_handler

