
import signal
import time

class SignalInterrupt(Exception):
    """Raised when a signal handler was invoked"""
    def __init__(self,signal,frame):
        print("Received Signal",signal)
        self.signal = signal
        self.frame  = frame
        raise self
    pass

signal.signal(signal.SIGALRM,SignalInterrupt)

def fn():
    try:
        while True:
            print("waiting for a signal")
            time.sleep(1)
    except SignalInterrupt as e:
        print(e.signal)

signal.alarm(3)
fn()
