

import signal
import time
from multiprocessing import Pool

class SignalInterrupt(Exception):
    """Raised when a signal handler was invoked"""
    def __init__(self,signal,frame):
        print("Received Signal",signal)
        self.signal = signal
        self.frame  = frame
        raise self
    pass


def fn(i):
    signal.signal(signal.SIGALRM,SignalInterrupt)
    signal.alarm(3)
    try:
        while True:
            print("waiting for a signal")
            time.sleep(1)
    except SignalInterrupt as e:
        print(e.signal)
        return f"valuable information: {i}"

with Pool(processes=4) as pool:
    print(pool.map(fn, range(4)))
