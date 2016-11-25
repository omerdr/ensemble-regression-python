from concurrent.futures import Executor, Future
from threading import Lock
import traceback
import sys

class DummyPoolExecutor(Executor):
    def __init__(self, max_workers=1):
        self._shutdown = False
        self._shutdownLock = Lock()

    def submit(self, fn, *args, **kwargs):
        with self._shutdownLock:
            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')

            f = Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                f.set_exception(e)
                print('Traceback:\n', file=sys.stderr)
                traceback.print_tb(e.__traceback__)
                print(e, file=sys.stderr)
                raise e
            else:
                f.set_result(result)

            return f

    def shutdown(self, wait=True):
        with self._shutdownLock:
            self._shutdown = True
