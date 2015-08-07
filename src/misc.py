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
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.flush() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        #sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)


# class HideOutput(object):
#     '''
#     A context manager that block stdout for its scope, usage:

#     with HideOutput():
#         os.system('ls -l')
#     '''

#     def __init__(self, *args, **kw):
#         sys.stdout.flush()
#         self._origstdout = sys.stdout
#         self._oldstdout_fno = os.dup(sys.stdout.fileno())
#         self._devnull = os.open(os.devnull, os.O_WRONLY)

#     def __enter__(self):
#         self._newstdout = os.dup(1)
#         os.dup2(self._devnull, 1)
#         os.close(self._devnull)
#         sys.stdout = os.fdopen(self._newstdout, 'w')

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         sys.stdout = self._origstdout
#         sys.stdout.flush()
#         os.dup2(self._oldstdout_fno, 1)