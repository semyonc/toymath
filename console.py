#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 12:30:32 2021

@author: semyonc
"""

from engine import setShell, setHandler
from engine.mathShell import MathShell

from colored import fg, attr
import IPython

prompt = f'{fg(2)}{{}}>{attr(0)} '


def display(*objs, **kwargs):
    for item in objs:
        if isinstance(item, IPython.core.display.HTML):
            print(item.__html__(), **kwargs)
        else:
            print(repr(item), **kwargs)


if __name__ == "__main__":
    shell = MathShell()
    shell.trace_mode = True
    execution_count = 1
    setHandler(display)
    setShell(shell)
    while True:
        try:
            expr = input(prompt.format(execution_count))
            shell.exec(expr, execution_count, add_to_history=True)
            execution_count += 1
        except EOFError:
            break
