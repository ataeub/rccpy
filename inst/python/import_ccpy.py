import os
import sys

from contextlib import contextmanager


@contextmanager
def stderr_redirected(to=os.devnull):
    """

    Suppresses the output of:
    - QFileSystemWatcher: Removable drive notification will not work if there is no QCoreApplication instance.
    - JsonRPCPlugin::JsonRPCPlugin
    everytime CloudComPy is imported.

    Taken from https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python/17954769#17954769

    """

    fd = sys.stderr.fileno()

    # assert that Python and C stdio write using the same file descriptor
    # assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stderr")) == fd == 1

    def _redirect_stderr(to):
        sys.stderr.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stderr = os.fdopen(fd, 'w')  # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stderr:
        with open(to, 'w') as file:
            _redirect_stderr(to=file)
        try:
            yield  # allow code to be run with the redirected stderr
        finally:
            _redirect_stderr(to=old_stderr)  # restore stderr.
            # buffering and flags such as
            # CLOEXEC may be different


with stderr_redirected():
    import cloudComPy as cc
    # from cloudpy import cloudpy

cc.initCC()
