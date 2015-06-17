import numpy as np
from contextlib import contextmanager
import os
import sys


def unique_rows(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]


def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


@contextmanager
def stdout_redirect(to=os.devnull,mode='w'):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    # assert that Python and C stdio write using the same file descriptor
    # assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()   # + implicit flush()
        os.dup2(to.fileno(), fd)   # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, mode)   # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, mode) as file:
            _redirect_stdout(to=file)
        try:
            yield   # allow code to be run with the redirected stdout
        finally:
            # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different
            _redirect_stdout(to=old_stdout)
