import time
from typing import Union


_tic_stack: list[float]


def init():
    global _tic_stack

    _tic_stack = []


def tic(message: str):
    global _tic_stack

    print(len(_tic_stack) * "\t" + message + "...")
    _tic_stack.append(time.time())


def toc(message: Union[str, None]=None):
    global _tic_stack

    toc_ = time.time()
    depth = len(_tic_stack)

    if depth == 0:
        print("Error - no debug tic to toc")
        return

    if message is None:
        message = ""
    else:
        message = ". " + message

    print((depth - 1) * "\t" + "> Done; took {:.3f}s".format(toc_ - _tic_stack.pop()) + message)


def get_indent() -> str:
    global _tic_stack
    return len(_tic_stack) * "\t"